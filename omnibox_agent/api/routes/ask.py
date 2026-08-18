"""/v1/ask/stream endpoint — streaming Q&A route with LLM-based complexity routing.

记忆系统接线（MEMORY_HARNESS_INTEGRATION_DESIGN.md §4.5，V2.0）：
  - 记忆的初始化/持久化/收尾三钩子已整体平移进 MemoryManager（harness.memory_manager），
    本路由只留调用：setup_request / persist_event / teardown。
  - 请求开始：append_user（status=pending，request_id 幂等）+ 会话上下文重建
    （request_input["session_context"]，三条管线统一消费）+ QU/judge 历史源切换
    + 长期记忆 recall（request_input["long_term"]，管线 suffix 只读此字段）。
  - done 事件：append_assistant + 同步压缩（压缩顺带提取长期记忆候选）；
    clarify 事件：澄清提问上树；客户端中断/异常：mark_interrupted（best-effort）。
  - manager 为 None（两开关 env 全关）时全部跳过——与改造前逐字节等价（回归基线）。
  - done 帧 metadata.memory 只读可观测字段由 _annotate_done_memory 追加（展示层职责，留在本文件）。
"""

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Query, Request
from starlette.responses import JSONResponse, StreamingResponse

from omnibox_agent.agent.context import AgentContext
from omnibox_agent.models.ask import AskRequest
from omnibox_agent.services.ai_config_store import get_user_ai_config
from omnibox_agent.api.lifecycle import get_harness

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["ask"])


def _annotate_done_memory(event_line: str, memory_state: dict) -> str:
    """done 帧追加只读可观测字段 metadata.memory（§8；前端不消费时静默忽略）。"""
    if not memory_state.get("user_entry_id"):
        return event_line
    try:
        ev = json.loads(event_line)
    except (ValueError, TypeError):
        return event_line
    if ev.get("event") != "done":
        return event_line
    data = ev.setdefault("data", {})
    meta = data.setdefault("metadata", {})
    meta["memory"] = {
        "summary": bool(memory_state.get("summary")),
        "recent_turns": memory_state.get("recent_turns", 0),
        "compacted": bool(memory_state.get("summary")),
        # 长期记忆命中可观测（§19.3 冒烟 1）：注入 profile/记忆片段或 L3 命中任一 → True
        "long_term_hit": bool(
            (memory_state.get("long_term") or {}).get("profile_text")
            or (memory_state.get("long_term") or {}).get("memory_text")
            or (memory_state.get("long_term") or {}).get("l3_hit") or []
        ),
    }
    return json.dumps(ev, ensure_ascii=False) + "\n"


def _resolve_request_id(request: AskRequest, raw_request: Request) -> str:
    """按契约解析 request_id：body.requestId（后端生成透传）→ 请求头 → 自生成。

    §3.1：requestId 由后端 AiController 生成（req_ + 32hex）并经 body 透传；
    Agent 优先使用，缺省（老调用方/直连调试）时自生成并回传。
    """
    from omnibox_agent.core.trace_recorder import normalize_request_id, new_request_id

    raw = request.request_id
    if raw is None:
        raw = raw_request.headers.get("X-Request-Id") or raw_request.headers.get("Request-Id")
    rid = normalize_request_id(raw)
    if rid is None:
        rid = new_request_id()
    return rid


def _finalize(
    response: Any,
    *,
    status: str = "done",
    error_msg: str | None = None,
) -> Any:
    """结束追踪（发 ask.done / ask.error + 落库）并原样返回响应。

    必须在每个 return 分支前调用一次（收尾会清理 contextvars）。
    """
    from omnibox_agent.core.trace_recorder import get_llm_calls, get_recorder, trace_event
    from omnibox_agent.core.trace_store import persist_trace_end

    recorder = get_recorder()
    if recorder is None:
        return response

    # 流结束时清理 cancel registry 登记（防 _active_tasks 字典泄漏）。
    # register_task 在 ask_stream 入口调用，所有 return 分支都经 _finalize，
    # 在这里统一 unregister 保证不泄漏。
    from omnibox_agent.api.routes.task import unregister_task
    unregister_task(recorder.request_id)

    # §5.2 回传契约：响应体携带 request_id
    try:
        if response is not None and hasattr(response, "request_id"):
            response.request_id = recorder.request_id
    except Exception:
        pass

    text = getattr(response, "text", None) or ""
    if status == "error":
        trace_event("ask.error", phase="ask", level="error",
                    message=error_msg or "request failed",
                    data={"status": "error", "answer_length": len(text)})
        persist_trace_end(recorder, status="error", error_msg=error_msg or "request failed")
    else:
        trace_event("ask.done", phase="ask",
                    data={
                        "status": "done",
                        "answer_length": len(text),
                        "llm_calls": get_llm_calls(),
                    })
        persist_trace_end(recorder, status="done")
    return response


# ---- Streaming Ask endpoint ----

@router.post("/ask/stream")
async def ask_stream(request: AskRequest, raw_request: Request):
    """Streaming Ask endpoint: returns NDJSON event stream.

    Same logic as the non-streaming Q&A path but produces a streaming response suitable for
    real-time token-by-token display in the frontend. Returns
    Content-Type: application/x-ndjson with one JSON event per line.
    """
    from omnibox_agent.core.config import get_config
    from omnibox_agent.core.tracing import set_trace_id

    cfg = get_config()
    trace_id = set_trace_id()
    log.info("Ask stream request: userId=%s, query='%s' trace=%s",
             request.user_id, request.query[:50], trace_id)

    # Ask 追踪：解析 request_id（§3.1）并开启请求追踪
    from omnibox_agent.core.trace_recorder import begin_trace, trace_event
    from omnibox_agent.core.trace_store import persist_trace_start, set_complexity, set_route
    request_id = _resolve_request_id(request, raw_request)

    # 注册到 cancel registry（POST /v1/task/cancel 据此判定 registered=True/False）。
    # 首次 ask 与 resume 流共用同一端点，统一注册——
    # 之前 register_task 定义了但从未被调用，导致 cancel API 永远 registered=False，
    # 客户端断开后 Agent 无法收到取消信号，仍跑完 60s+ 的 Creative DAG。
    from omnibox_agent.api.routes.task import register_task
    register_task(request_id)

    recorder = begin_trace(
        request_id=request_id,
        trace_id=trace_id,
        user_code=request.user_id or "",
        session_id=request.session_id,
        query=request.query or "",
    )
    trace_event("ask.received", phase="ask",
                data={
                    "user_code": request.user_id or "",
                    "query": (request.query or "")[:200],
                    "has_history": bool(request.history),
                })
    persist_trace_start(recorder)

    if not request.query or len(request.query.strip()) < cfg.pipeline.min_query_length:
        # Return a single-line NDJSON error stream
        async def _short_query():
            yield json.dumps({"event": "error", "data": {
                "reason": f"问题太短，请至少输入{cfg.pipeline.min_query_length}个字符",
                "code": "short_query",
            }}, ensure_ascii=False) + "\n"
        return _finalize(
            StreamingResponse(
                _short_query(),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            ),
            status="error",
            error_msg="short_query",
        )

    # Resolve AI config: DB first, request fallback
    ai_config = request.ai_config or {}
    db_ai_config = get_user_ai_config(request.user_id)
    if db_ai_config:
        ai_config = {**db_ai_config, **ai_config}

    if not ai_config.get("apiKey"):
        async def _no_key():
            yield json.dumps({"event": "error", "data": {
                "reason": "用户没有提供 API Key，无法完成任务。请先在设置中接入你的模型服务（API Key / Base URL / 模型名）后再使用问答功能",
                "code": "no_api_key",
            }}, ensure_ascii=False) + "\n"
        return _finalize(
            StreamingResponse(
                _no_key(),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            ),
            status="error",
            error_msg="no_api_key",
        )

    # 澄清启用开关：后端经 X-Clarify-Enabled 请求头透传，body 兜底（灰度开关）
    clarify_header = raw_request.headers.get("X-Clarify-Enabled")
    if clarify_header is not None:
        clarify_enabled = clarify_header.strip().lower() in ("1", "true", "yes", "on")
    else:
        clarify_enabled = bool(request.clarify_enabled) if request.clarify_enabled is not None else True
    # 若 Agent 侧配置关闭，则整体关闭
    if not get_config().clarify.enabled:
        clarify_enabled = False

    # Build request input
    request_input = {
        "query": request.query.strip(),
        "user_id": request.user_id,
        "session_id": request.session_id,
        "favorite_only": request.favorite_only,
        "scope": request.scope,
        "history": request.history or [],
        "ai_config": ai_config,
        "clarify_enabled": clarify_enabled,
        "clarify_count": request.clarify_count if request.clarify_count is not None else 0,
    }

    # 把 clarify_session_id 注入下游（Agent 侧计数器以它为 key 维护 Stream 内计数权威）
    # 无论是否为 resume，都透传（首次请求也可能在之后被分配 session，由后端在 clarify 帧
    # 返回后回填，因此首发时为 None 属正常；None 不会被缓存计数）。
    request_input["clarify_session_id"] = request.clarify_session_id

    # resume（澄清恢复）：透传后端缓存的上下文快照与用户回答元信息
    if request.clarify_session_id:
        resume_context = request.resume_context or {}
        request_input["resume_context"] = resume_context
        request_input["answer_type"] = request.answer_type
        request_input["answer_key"] = request.answer_key
        request_input["answer_text"] = request.answer_text or request.query
        request_input["original_query"] = resume_context.get("original_query", "")

    harness = get_harness()
    ask_agent = harness.get("ask")
    if ask_agent is None:
        async def _no_agent():
            yield json.dumps({"event": "error", "data": {
                "reason": "Ask agent not available",
                "code": "agent_unavailable",
            }}, ensure_ascii=False) + "\n"
        return _finalize(
            StreamingResponse(
                _no_agent(),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            ),
            status="error",
            error_msg="agent_unavailable",
        )

    # SKILL：显式注入 SkillManager（来自 harness，§5.4），供 skill 节点消费
    skill_manager = getattr(harness, "skill_manager", None)

    # MEMORY：消费 MemoryManager（§4.5）。None → 记忆整体关闭（两开关 env 全关），天然降级
    memory = getattr(harness, "memory_manager", None)

    ctx = AgentContext(
        input=request_input,
        session_id=request.session_id,
        trace_id=trace_id,
    )

    # 记忆系统初始化（best-effort）：append_user + session_context + QU 历史源切换
    # + 长期记忆 recall（MemoryManager.setup_request，§4.5）。
    # 放在复杂度判定之前，使 classifier 也能吃到 Session Tree 历史。
    memory_state = (await memory.setup_request(request_input, ai_config, request_id)
                    if memory else {})

    # R7/R9/R10：会话内指代查询的路由主判据 = **是否发生检索**。
    #   - 纯回顾（referential=True 且 need_retrieval=False，如「总结上面的内容」）：
    #     答案在会话记忆里，不检索 → 走 stream_conversation_pipeline（单步会话作答）。
    #   - 需要检索（referential=True 且 need_retrieval=True，如「第三点有哪些推荐的」
    #     「根据第三点的收藏内容进行分析」）：**按正常流程走复杂度判断**
    #     （simple QA 或 complex DAG），会话上下文只是检索输入的一环——
    #     复杂度由 LLM 语义判定（R10），不因"涉及会话"强制压成单步。
    # R9 语义判定：规则正则作零成本短路（命中直接判 referential=True），未命中
    # 模糊地带交给 LLM judge（judge_conversation_referential）。
    # resume 请求（澄清回答）不参与本判定——路由上 resume 恒优先，judge 是纯浪费。
    if request.clarify_session_id:
        is_conversation_referential = False
        conv_need_retrieval = False
        conv_resolved_query = ""
        log.info("Conversation-referential judge skipped (resume request)")
    else:
        from omnibox_agent.services.clarify import judge_conversation_referential
        conv_judge = await judge_conversation_referential(
            request.query.strip() if request.query else "",
            request_input.get("history") or [],
            ai_config,
        )
        is_conversation_referential = conv_judge["referential"]
        # R10：LLM judge 判定的会话指代（含编号/指代，R6 正则覆盖不到的）——
        # 即使 need_retrieval=True 走 simple 检索路径，也不应弹澄清气泡
        # （graph_qa._maybe_clarify_qa 消费此标记跳过澄清）。
        if is_conversation_referential:
            request_input["_conv_referential"] = True
            if conv_judge.get("source") == "llm":
                # LLM 判定：referential/need_retrieval/resolved_query 均可信
                conv_need_retrieval = conv_judge["need_retrieval"]
                conv_resolved_query = conv_judge.get("resolved_query") or ""
            else:
                # 规则短路（正则命中）：只负责 referential——检索与否不能硬编码 True
                # （「上面说的什么」是纯回顾，不应检索），用 R8 规则按提问+会话记忆判断。
                from omnibox_agent.agent.stream_pipeline import _should_retrieve_for_conversation
                _recent = request_input.get("session_context", {}).get("recent") \
                    or request_input.get("history") or []
                conv_need_retrieval = _should_retrieve_for_conversation(
                    request.query.strip() if request.query else "", _recent)
                conv_resolved_query = ""
        else:
            conv_need_retrieval = False
            conv_resolved_query = ""
        # 需要检索时：把 resolved_query 注入下游（simple/complex 检索词优先用它，
        # 解决「第三点」这类编号/指代在 QU 里不一定能消解的问题）
        if conv_need_retrieval and conv_resolved_query:
            request_input["_conv_resolved_query"] = conv_resolved_query
        log.info("Conversation-referential judge: referential=%s need_retrieval=%s source=%s resolved_query=%r",
                 is_conversation_referential, conv_need_retrieval,
                 conv_judge.get("source") if is_conversation_referential else "n/a",
                 conv_resolved_query[:50])
        trace_event("ask.conv_judge", phase="ask", data={
            "referential": is_conversation_referential,
            "need_retrieval": conv_need_retrieval,
            "source": conv_judge.get("source") if is_conversation_referential else "n/a",
            "resolved_query": conv_resolved_query[:80],
        })

    # R10：纯回顾（referential 且不检索）走会话记忆单步管线，语义等效 simple；
    # 需要检索的会话指代查询走正常复杂度判断（不在此标记）。
    if is_conversation_referential and not conv_need_retrieval:
        trace_event("ask.classify", phase="ask", data={
            "type": "simple",
            "reason": "会话内指代纯回顾，会话记忆单步作答",
        })
        set_complexity(recorder, "simple")

    # --- Complexity classification (only when DAG path is enabled) ---
    # R10：仅对"纯回顾"跳过复杂度分类；需要检索的会话指代查询正常走复杂度判断。
    creative_cfg = cfg.creative
    if creative_cfg.mode != "off" and not (is_conversation_referential and not conv_need_retrieval):
        from omnibox_agent.services.complexity_classifier import classify_complexity
        try:
            complexity = await classify_complexity(
                query=request.query.strip(),
                history=request_input.get("history"),
                ai_config=ai_config,
            )
            ctx.artifacts["complexity"] = complexity
            log.info("Complexity (stream): type=%s reason=%s", complexity.type, complexity.reason)
            # Ask 追踪：复杂度判定（§4.1 ask.classify）
            trace_event("ask.classify", phase="ask", data={
                "type": complexity.type,
                "reason": (complexity.reason or "")[:200],
            })
            set_complexity(recorder,
                           "complex" if complexity.type == "complex" else "simple")
        except Exception as e:
            log.warning("Complexity classifier failed (stream): %s, defaulting to simple QA", e)
            trace_event("ask.classify.fallback", phase="ask", level="warn",
                        message=str(e)[:200])

    # --- Route ---
    from omnibox_agent.agent.orchestration.router import ComplexityRouter
    route = ComplexityRouter().route(ctx)
    if is_conversation_referential and not conv_need_retrieval:
        # R10：纯回顾会话指代查询走会话记忆管线，路由标记为 conversation 便于追踪；
        # 需要检索的会话指代查询走正常 simple/complex 流程（route 由复杂度决定）。
        route = "conversation"
    log.info("Route decision (stream): %s", route)
    # Ask 追踪：路由决策（§4.1 ask.route）
    trace_event("ask.route", phase="ask", data={"route": route})
    set_route(recorder, route)

    async def event_generator():
        """Generate NDJSON events based on complexity route.

        首帧 meta 帧携带 requestId（§5.2 回传契约）；流式结束统一收尾追踪。
        客户端断开支持：每轮事件前检查 is_disconnected，断开即向 pipeline
        注入 asyncio.CancelledError 级联取消（可中断挂起的 httpx/LLM 调用与
        gather 子任务），并停止后续生成，避免空耗 LLM 调用。
        """
        final_status = "done"
        final_error = None
        try:
            # §5.2: 首帧 meta 帧回传 request_id + clarifySupported
            yield json.dumps({"event": "meta", "data": {
                "requestId": request_id,
                "route": route,
                "clarifySupported": clarify_enabled,
            }}, ensure_ascii=False) + "\n"

            # resume：澄清恢复，走 stream_resume_pipeline（跳过 Parse/Retrieve）
            if request.clarify_session_id:
                from omnibox_agent.agent.stream_pipeline import stream_resume_pipeline
                source = stream_resume_pipeline(ctx, request_input, skill_manager=skill_manager)
            elif is_conversation_referential and not conv_need_retrieval:
                # R10：纯回顾会话指代（不检索）→ 会话记忆单步管线
                from omnibox_agent.agent.stream_pipeline import stream_conversation_pipeline
                source = stream_conversation_pipeline(ctx, request_input, skill_manager=skill_manager)
            elif route == "dag" and creative_cfg.mode != "off":
                from omnibox_agent.agent.stream_pipeline import stream_creative_pipeline
                source = stream_creative_pipeline(ctx, request_input, skill_manager=skill_manager)
            else:
                from omnibox_agent.agent.stream_pipeline import stream_qa_pipeline
                source = stream_qa_pipeline(ctx, request_input, skill_manager=skill_manager)

            async for event_line in source:
                if await raw_request.is_disconnected():
                    # 客户端已断开：级联取消 pipeline（httpx 等 await 点可被
                    # CancelledError 中断），随后结束本流
                    try:
                        await source.athrow(asyncio.CancelledError())
                    except (asyncio.CancelledError, GeneratorExit, StopAsyncIteration):
                        pass
                    raise asyncio.CancelledError()
                # done 帧追加 memory 可观测字段（§8，仅记忆启用时）
                if memory_state:
                    event_line = _annotate_done_memory(event_line, memory_state)
                # 记忆持久化（含同步压缩 + 长期记忆提取）在发帧**之前**执行：
                # 客户端收到 done 时，assistant 已落库、压缩已完成，树处于一致状态
                # （§4.2 同步语义；MemoryManager.persist_event，§4.5）。
                if memory:
                    await memory.persist_event(memory_state, event_line, ai_config, request_id)
                yield event_line

        except asyncio.CancelledError:
            # 客户端断开 → 标记中断（仅用于追踪；连接已断，不再发送事件）
            final_status = "error"
            final_error = "client_disconnected"
            raise
        except GeneratorExit:
            final_status = "error"
            final_error = "client_disconnected"
            return
        except Exception as e:
            log.exception("Stream pipeline unexpected error")
            final_status = "error"
            final_error = "stream_error"
            yield json.dumps({"event": "error", "data": {
                "reason": "处理失败，请稍后重试",
                "code": "internal_error",
            }}, ensure_ascii=False) + "\n"

        finally:
            # 记忆收尾：未完成（中断/异常）的 user 节点标记 interrupted（§4.6；§4.5 manager）
            if memory:
                await memory.teardown(memory_state)
            # Ask 追踪：流式请求结束并落库
            if final_status == "error":
                _finalize(None, status="error", error_msg=final_error)
            else:
                _finalize(None, status="done")

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, user_id: str = Query(..., alias="userId")):
    """级联删除会话树（归属校验，§3.4）+ 长期记忆降权钩子（§15）。

    供后端 OmniHub_server 在删除 ask_session 时联动调用：
      DELETE /v1/session/{sessionId}?userId={user_code}
    best-effort：表不存在/无权限等仅返回 ok=false，不影响主流程。
    """
    from omnibox_agent.services import session_store

    if not session_id or not user_id:
        return JSONResponse(status_code=400, content={"ok": False, "reason": "missing sessionId/userId"})
    try:
        ok = await session_store.delete_session_tree(session_id, user_id)
    except Exception as e:
        log.warning("delete_session failed (best-effort): %s", e)
        return JSONResponse(status_code=500, content={"ok": False, "reason": "delete failed"})

    # 长期记忆降权钩子（§15）：源自该会话的情景记忆**降权**而非删除
    # （偏好可能已独立成立，不因会话删除而失忆）。不改写 delete_session_tree
    # 本体（保持存储层边界）；按 meta.source_session_id 匹配，MySQL/Chroma 双写。
    try:
        from omnibox_agent.services import long_term_store
        n = await asyncio.to_thread(
            long_term_store.downgrade_by_session_sync, session_id, user_id)
        if n:
            log.info("LT downgrade by session %s: %s memories", session_id, n)
    except Exception as e:
        log.warning("LT downgrade_by_session failed (best-effort): %s", e)

    return {"ok": bool(ok)}
