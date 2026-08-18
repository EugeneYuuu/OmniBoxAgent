"""Streaming pipeline for the Ask endpoint.

Provides:
  - ndjson(): format an NDJSON event line
  - stream_qa_pipeline(): streaming QA pipeline (Parse → Guard → Retrieve →
    Gate → Reason → Act(streaming LLM) → done)
  - stream_creative_pipeline(): streaming DAG pipeline (Plan → Solve →
    Reflect → Synthesize(streaming LLM) → done)

Each event is a single NDJSON line:
  {"event":"thinking","data":{"phase":"...","message":"..."}}
  {"event":"references","data":{"items":[...]}}
  {"event":"token","data":{"delta":"..."}}
  {"event":"done","data":{"sessionId":"...","text":"...","metadata":{...}}}
  {"event":"error","data":{"reason":"...","code":"..."}}
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncGenerator

from omnibox_agent.agent.context import (
    AgentContext,
    RetrievalOutput,
)
from omnibox_agent.models.ask import RefItem
from omnibox_agent.core.config import get_config

log = logging.getLogger(__name__)

# ---- NDJSON event formatting ----

def ndjson(event: str, data: dict) -> str:
    """Format a single NDJSON event line."""
    return json.dumps({"event": event, "data": data}, ensure_ascii=False) + "\n"


# ---- Reference extraction helper ----

def _build_ref_items(retrieval: RetrievalOutput) -> list[dict]:
    """Extract reference items from retrieval output for the references event."""
    cfg = get_config()
    refs = []
    for item in retrieval.fused_items[:cfg.retrieval.reference_display_limit]:
        cid = item.get("content_id")
        detail = retrieval.content_map.get(cid, {})
        refs.append({
            "id": cid,
            "title": detail.get("title") or item.get("title"),
            "cover": detail.get("cover") or item.get("cover"),
            "platformName": detail.get("platform_name") or item.get("platform_name"),
            "authorName": detail.get("author_name") or item.get("author_name"),
            "originalUrl": detail.get("original_url") or item.get("original_url"),
        })
    return refs


# ---- Streaming QA pipeline ----

async def stream_qa_pipeline(
    ctx: AgentContext,
    request_input: dict,
    skill_manager: Any = None,
) -> AsyncGenerator[str, None]:
    """Streaming version of the 7-step QA pipeline, driven by LangGraph.

    QA 子图（graph_qa.run_qa_graph）以 LangGraph 编排 Parse→Guard→Retrieve→
    Gate→Reason→Act(流式)→done；各节点经 progress_cb/references_cb/token_cb 把
    thinking/references/token 事件实时入队，本生成器边收边 yield（豆包式）。
    clarify/error/done 事件在 handler 层组装。

    事件顺序（与旧手写驱动一致）：
      thinking(parsing) → thinking(checking) → thinking(retrieving) →
      thinking(filtering) → [references] → thinking(reasoning) →
      [clarify | thinking(generating) → token* → done/error]
    """
    import asyncio
    from omnibox_agent.agent.graph_qa import run_qa_graph
    from omnibox_agent.services.clarify import ClarifySignal, ClarifySessionCounter
    from omnibox_agent.core.trace_recorder import trace_event

    t_start = time.monotonic()
    q: asyncio.Queue = asyncio.Queue()

    # Agent 内部计数权威：以 clarify_session_id 为 key，Stream 内累计。
    # 请求里传过来的 clarify_count 在此被忽略/覆盖，避免后端跨会话错算。
    clarify_session_id = request_input.get("clarify_session_id")
    # v2（D2）：无 resume_context 的请求视为新提问，重置该链路计数——
    # 即使后端复用同一 clarify_session_id，也不让上一个提问的澄清跨问题累计。
    if not request_input.get("resume_context"):
        ClarifySessionCounter.reset(clarify_session_id)
    internal_clarify_count = ClarifySessionCounter.get(clarify_session_id)
    request_input["clarify_count"] = internal_clarify_count

    async def _progress_cb(phase, message):
        await q.put(("thinking", {"phase": phase, "message": message}))

    async def _token_cb(tok):
        await q.put(("token", {"delta": tok}))

    async def _references_cb(retrieval):
        refs = _build_ref_items(retrieval) if retrieval else []
        if refs:
            await q.put(("references", {"items": refs}))

    async def _clarify_cb(decision):
        raise ClarifySignal(decision, "qa", decision.context)

    task = asyncio.ensure_future(run_qa_graph(
        ctx,
        config=get_config(),
        request_input=request_input,
        progress_cb=_progress_cb,
        token_cb=_token_cb,
        references_cb=_references_cb,
        clarify_cb=_clarify_cb,
        skill_manager=skill_manager,
    ))

    full_text = ""
    while True:
        try:
            kind, data = await asyncio.wait_for(q.get(), timeout=0.2)
            if kind == "token":
                full_text += data["delta"]
            yield ndjson(kind, data)
            continue
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            task.cancel()
            raise
        if task.done():
            break

    if task.cancelled():
        return

    exc = task.exception()
    if exc is not None:
        if isinstance(exc, ClarifySignal):
            decision = exc.decision
            # v2（R3）：计数已在判定时经 try_incr 原子占位（decision._reserved_snapshot）；
            # 旧路径（无占位）兜底再 incr，保证审计字段不缺失、不重复计数。
            new_snap = getattr(decision, "_reserved_snapshot", None) \
                or ClarifySessionCounter.incr(clarify_session_id, phase=exc.phase)
            new_total = new_snap["total"]
            phase_this = new_snap["phase_counts"].get(exc.phase, 1)
            trace_event("qa.clarify", phase="qa", data={
                "importance": decision.importance, "question": decision.question,
                "clarify_total": new_total, "clarify_phase_qa": phase_this})
            event_data = decision.to_event_data()
            # 把最新计数附带进澄清帧 context，供后端审计 + 前端调试（不改动现有契约字段）
            ctx_dict = event_data.setdefault("context", {})
            ctx_dict["_clarify_total"] = new_total
            ctx_dict["_clarify_phase"] = exc.phase
            ctx_dict["_clarify_phase_count"] = phase_this
            yield ndjson("clarify", event_data)
            return
        log.warning("QA graph failed: %s", exc)
        yield ndjson("error", {"reason": "服务暂不可用，请稍后重试", "code": "error"})
        return

    # ---- 图正常结束，组装 done/error/fallback ----

    # 1. critical 中止（如 guard 无账号）→ error
    err = ctx.artifacts.get("error")
    if err:
        yield ndjson("error", {
            "reason": _map_abort_reason(err.get("code")),
            "code": err.get("code"),
        })
        return

    # 2. LLM 流式异常且无任何输出 → error
    if ctx.flags.get("llm_stream_error") and not full_text:
        yield ndjson("error", {
            "reason": "模型服务暂不可用，请稍后重试",
            "code": "llm_error",
        })
        return

    # 3. LLM 返回空流 → fallback
    if not full_text:
        retrieval_r = ctx.artifacts.get("retrieval", RetrievalOutput())
        from omnibox_agent.agent.ask_agent import _fallback_answer
        full_text = _fallback_answer(
            qu_result=ctx.artifacts.get("perception"),
            top_items=retrieval_r.fused_items,
            content_map=retrieval_r.content_map,
            total_count=retrieval_r.total_count,
        )

    elapsed = round(time.monotonic() - t_start, 2)
    metadata = {
        "confidence": "normal",
        "llm_calls": ctx.llm_call_count,
        "elapsed_s": elapsed,
    }
    if ctx.flags.get("gate_degraded"):
        metadata["gate_degraded"] = True
        metadata["confidence"] = "low"
    if ctx.flags.get("low_confidence"):
        metadata["low_confidence"] = True
        metadata["confidence"] = "low"

    # §5.6：技能命中可观测性字段（append-only，前端未消费时静默忽略）
    _add_skill_metadata(metadata, ctx.artifacts.get("skills"))

    yield ndjson("done", {
        "sessionId": request_input.get("session_id"),
        "text": full_text,
        "metadata": metadata,
    })



# ---- Streaming conversation-memory pipeline (R7/R8) ----

# R8 条件检索判断：
#   - 纯回顾类提问（总结/概括/复述/重复/说了什么/什么意思…）→ 不检索，纯会话作答
#   - 引用细节类提问（在哪/地址/多少钱/明细/具体…）→ 检索会话中引用的 content://N 条目
# 判断依据 = 提问 + 会话记忆（会话里有没有可检索的引用锚点）。
_PURE_RECAP_RE = re.compile(
    r"总结|概括|摘要|复述|重复|再说一遍|重新说|说了什么|说的什么|什么意思|是什么意思|"
    r"回顾|继续|然后呢|还有吗|就这些"
)
_DETAIL_ASK_RE = re.compile(
    r"在哪|哪里|哪儿|地址|多少钱|价格|花费|费用|明细|详情|具体|怎么去|怎么走|"
    r"几点|营业|开门|推荐|链接|打开|那家|这家|那篇|这篇|上面提到|刚才提到"
)


def _extract_content_ids_from_messages(messages: list[dict],
                                       max_ids: int = 8) -> list[int]:
    """从会话消息里抽 content://N 引用（assistant 回答常带收藏条目链接）。

    「上面说的」指**紧邻的上一轮 assistant 回答**——只从最近一条含引用的
    assistant 消息抽取（倒序找第一条），且上限 max_ids 条，避免把整段会话
    的所有引用都当作检索锚点（references 噪音 + prompt 膨胀）。
    """
    ids: list[int] = []
    seen: set[int] = set()
    for m in reversed(messages or []):
        if m.get("role") != "assistant":
            continue
        for cid in re.findall(r"content://(\d+)", str(m.get("content") or "")):
            c = int(cid)
            if c not in seen:
                seen.add(c)
                ids.append(c)
        if ids:
            break  # 只取最近一条含引用的 assistant 回答
        if len(seen) >= max_ids * 3:  # 防御：单条引用过多时不无限收集
            break
    return ids[:max_ids]


def _dedup_queries(queries: list[str], drop_index_refs: bool = True) -> list[str]:
    """去重并丢弃空/过短/含编号指代原文的检索词（R9.1 多路召回用）。

    丢弃含「第X点」的 query——编号在向量检索里没有语义，只会召回噪音
    （LLM 消解失败时保留的原文，如「第三点相关收藏推荐」）。

    drop_index_refs=False 时不做该过滤：**带会话上下文**的检索词（规则
    消解词/逐条主题词）即使含编号指代也有语义（主题词主导 embedding），
    不应被误杀——修复：原实现把「AI/职场 第三点有哪些推荐」这类带前缀的
    兜底词也一起丢了，导致 LLM 消解失败时兜底路 100% 失效。
    """
    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        q = (q or "").strip()
        if not q or len(q) < 2 or q in seen:
            continue
        if drop_index_refs and re.search(r"第[一二三四五六七八九十\d]+点", q):
            continue
        seen.add(q)
        out.append(q)
    return out


def _conversation_topic_queries(
    messages: list[dict], max_queries: int = 3, max_chars: int = 60,
) -> list[str]:
    """从最近对话逐条提取主题检索词（R9.1 兜底路）。

    替代旧的"整段会话摘要混合块"（_conversation_brief 仍保留给 system
    prompt 用）：每条消息 _clean_for_embedding 清理后**独立成词**——语义
    干净，避免混合块稀释 embedding 相似度（与评论向量同源问题，多条主题
    混拼会让向量既不像 A 也不像 B）。倒序取最近 max_queries 条，去重，
    丢弃过短/空。
    """
    from omnibox_agent.services.query_understanding import _clean_for_embedding

    out: list[str] = []
    seen: set[str] = set()
    for m in reversed(messages or []):
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        cleaned = _clean_for_embedding(content)[:max_chars].strip()
        if len(cleaned) < 2 or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) >= max_queries:
            break
    return out


def _conversation_brief(messages: list[dict], max_chars: int = 600) -> str:
    """把最近对话压缩成简短摘要（R9.1 内容型查询用）。

    保留 role 前缀与内容，单条截断 150 字、总计 ≤ max_chars，时间正序。
    用途：内容型查询的 system prompt 里放摘要而非逐条消息，避免历史里过时的
    否定表述（「还没有X内容」）压过消息流末尾的检索条目详情。
    """
    if not messages:
        return ""
    picked: list[str] = []
    total = 0
    for m in messages:
        role = m.get("role") if m.get("role") in ("user", "assistant") else "user"
        content = str(m.get("content") or "").replace("\n", " ").strip()[:150]
        if not content:
            continue
        line = f"- {role}: {content}"
        if total + len(line) > max_chars:
            break
        picked.append(line)
        total += len(line)
    return "\n".join(picked)


def _resolve_query_with_memory(query: str, recent: list[dict]) -> str:
    """R8：细节追问但会话无引用锚点时，用会话记忆消解查询做向量兜底。

    「上面说的那家店」→ 取最近一条 user 提问的关键词前缀 + 当前查询，
    复用 QU 的指代消解思路（query_understanding._resolve_context）。
    """
    q = (query or "").strip()
    if not q:
        return q
    for m in reversed(recent):
        if m.get("role") == "user":
            prev = str(m.get("content") or "").strip()
            if prev and prev != q:
                from omnibox_agent.services.query_understanding import _clean_for_embedding
                kw = _clean_for_embedding(prev)
                if kw:
                    return f"{kw} {q}"
            break
    return q


def _should_retrieve_for_conversation(query: str, recent: list[dict]) -> bool:
    """R8：会话内指代查询是否需要结合检索（提问 + 会话记忆共同判断）。

    - 纯回顾类提问 → 不检索（答案就在会话里，检索反而会引入噪音/带偏）
    - 引用细节类提问 + 会话有 content://N 锚点 → 检索（按 ID 精确取详情）
    - 细节追问但会话无锚点 → 语义向量兜底（消解查询后检索）
    - 其他（无细节意图也无锚点）→ 不检索
    """
    q = (query or "").strip()
    if not q or not recent:
        return False
    if _PURE_RECAP_RE.search(q):
        return False
    if _extract_content_ids_from_messages(recent):
        return True
    if _DETAIL_ASK_RE.search(q):
        return True
    return False


async def _conditional_retrieve_for_conversation(
    query: str, recent: list[dict], request_input: dict,
) -> tuple[list[dict], list[dict]]:
    """R8/R9.1：会话指代查询的条件检索（best-effort，失败返回空）。

    Retrieval query 优先级：
      1. content://N 锚点（会话里已引用的条目，按 ID 精确取详情）
      2. R9.1 LLM 消解的 resolved_query（编号/指代 → 可检索词，如
         「第三点有哪些推荐的」→「AI/职场 相关收藏推荐」）
      3. 规则消解 _resolve_query_with_memory 兜底

    Returns: (items, refs) — items 为条目详情（含 summary），refs 为 references 帧格式。
    """
    from omnibox_agent.core.trace_recorder import trace_event
    try:
        from omnibox_agent.services.retrieval_store import (
            get_account_ids, get_content_by_ids,
        )
        from omnibox_agent.services.vector_search import vector_search, full_candidate_budget
        from omnibox_agent.agent.loop import run_blocking

        user_id = request_input.get("user_id", "")
        account_ids = await run_blocking(get_account_ids, user_id)
        if not account_ids:
            trace_event("qa.conv_retrieve", phase="qa", level="warn",
                        message="无可用账号，跳过检索", data={"user_id": user_id})
            return [], []

        items: list[dict] = []
        ref_ids = _extract_content_ids_from_messages(recent)
        if ref_ids:
            trace_event("qa.conv_retrieve", phase="qa",
                        message=f"命中会话内 content:// 锚点 {len(ref_ids)} 个",
                        data={"anchor_ids": ref_ids[:20]})
            items = await run_blocking(get_content_by_ids, ref_ids, account_ids)
        if not items:
            # 无锚点 → 向量检索多路召回：
            #   1) R9.1 LLM 消解词（精准，如「AI/职场 相关收藏推荐」）
            #   2) 最近对话摘要词（兜底：LLM 消解失败/保留编号时，摘要里的
            #      主题词仍能召回）
            # 并集去重，按序合并（LLM 词优先）。
            resolved = (request_input.get("_conv_resolved_query") or "").strip() \
                or _resolve_query_with_memory(query, recent)
            candidate_ids: list[int] = []
            # 放开 n_results=5（用户指令：默认不限制 top-k）——按用户库全量
            # 召回预算取所有命中；candidate_ids[:10] 仍保留（展示层限制：
            # references 胶囊只需前 10 条详情）。
            n_results = full_candidate_budget(user_id)
            # 兜底路改为逐条主题提取（每条消息独立成词，语义干净，替代旧的
            # 整段摘要混合块）。"第X点"过滤只在规则消解无进展（resolved 就是
            # 原始 query，即新会话无上下文）时生效——带主题前缀的规则消解词
            # 和逐条主题词即使含编号也要保留（修复：兜底路被误杀 100% 失效）。
            drop_index_refs = (resolved == query)
            candidates = [resolved] + _conversation_topic_queries(recent, max_queries=3)
            for rq in _dedup_queries(candidates, drop_index_refs=drop_index_refs):
                if not rq:
                    continue
                trace_event("qa.conv_retrieve", phase="qa",
                            message=f"向量检索: {rq[:60]}",
                            data={"query": rq[:100]})
                hits = await run_blocking(vector_search, rq, user_id, n_results=n_results)
                hits_ids = [h.get("content_id") for h in hits if h.get("content_id") is not None]
                trace_event("qa.conv_retrieve", phase="qa",
                            message=f"召回 {len(hits)} 条({len(hits_ids)} 去重候选)",
                            data={"hit_ids": hits_ids[:20]})
                for h in hits:
                    cid = h.get("content_id")
                    if cid is not None and cid not in candidate_ids:
                        candidate_ids.append(cid)
            if candidate_ids:
                items = await run_blocking(get_content_by_ids, candidate_ids[:10], account_ids)

        if not items:
            trace_event("qa.conv_retrieve", phase="qa", level="warn",
                        message="检索未命中条目", data={"candidate_count": len(candidate_ids) if 'candidate_ids' in locals() else 0})
            return [], []
        trace_event("qa.conv_retrieve", phase="qa",
                    message=f"检索命中 {len(items)} 条收藏条目",
                    data={"item_ids": [it.get("id") for it in items if it.get("id") is not None]})
        refs = [{
            "id": it.get("id"),
            "title": it.get("title") or "",
            "cover": it.get("cover") or "",
            "platformName": it.get("platform_name") or "",
            "authorName": it.get("author_name") or "",
            "originalUrl": it.get("original_url") or "",
        } for it in items if it.get("id") is not None]
        return items, refs
    except Exception as e:
        log.warning("Conversation conditional retrieval failed (best-effort): %s", e)
        return [], []


async def stream_conversation_pipeline(
    ctx: AgentContext,
    request_input: dict,
    skill_manager: Any = None,
) -> AsyncGenerator[str, None]:
    """流式回答「会话内指代」查询（如「上面说的什么」「总结上面的内容」）。

    R7 语义：这类问题的答案在**当前会话的记忆**里，不在收藏库检索里——
    用户问的是"上一轮回答/最近对话"的内容，答案必须来自会话历史。
    因此本管线以会话历史（session_context.recent 优先，注入 history 兜底）为
    主 prompt，流式作答（不做澄清）。这是对 R6（仅跳过澄清判定）的补全：
    R6 只保证不弹澄清气泡，R7 保证回答本身基于会话记忆而非收藏库。

    R8 语义（条件检索）：根据**提问 + 会话记忆**判断是否需要结合检索——
    - 纯回顾类（总结/说了什么/什么意思）→ 不检索，纯会话作答；
    - 引用细节类（上面说的那家店在哪/攻略花费明细）→ 抽取会话中的 content://N
      引用按 ID 取详情（精确、零噪音），无锚点时用消解查询做向量兜底；
      检索结果作为补充信息注入 prompt（明确"仅用于回答用户问到的具体条目细节，
      不要把检索条目当作『上面的内容』来总结"），并随 references 事件下发。

    R9 语义判定：referential/need_retrieval 由 ask.py 的
    judge_conversation_referential（LLM-as-a-Judge）给出，替代关键词枚举；
    本管线消费 request_input["_conv_need_retrieval"]，缺失时回退规则判断。

    R9.1 编号/指代消解：need_retrieval=True 时 judge 输出 resolved_query
    （如「第三点有哪些推荐的」→「AI/职场 相关收藏推荐」），本管线优先用它
    做向量检索（request_input["_conv_resolved_query"]），无则回退规则消解。

    事件顺序：
      thinking(generating) → [references] → token* → done/error
    """
    from omnibox_agent.core.trace_recorder import trace_event

    t_start = time.monotonic()
    query = request_input.get("query", "")
    ai_config = request_input.get("ai_config", {})

    # 会话历史来源（服务端权威优先）：session_context.recent → 注入 history 兜底
    session_context = request_input.get("session_context") or {}
    recent = list(session_context.get("recent") or [])
    if not recent:
        recent = [
            m for m in (request_input.get("history") or [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and (m.get("content") or "").strip()
        ]

    # 去掉 recent 末尾的当前 query（append_user 已把本轮 query 挂进树，避免重复喂给 LLM）
    if recent and recent[-1].get("role") == "user" \
            and (recent[-1].get("content") or "").strip() == query.strip():
        recent = recent[:-1]

    # 兜底：fresh session 时 session_context.recent 只含当前 query（去重后为空），
    # 而注入 history 因 recent 非空被跳过 → 此时回退注入 history，避免会话线索丢失
    if not recent:
        recent = [
            m for m in (request_input.get("history") or [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and (m.get("content") or "").strip()
        ]

    # R8/R9/R9.1：条件检索（提问 + 会话记忆判断）。
    # R9 语义判定优先：ask.py 的 judge_conversation_referential 仅在 LLM 判定时
    # 写入 _conv_need_retrieval（规则短路只判 referential，检索与否仍走规则）；
    # 缺失时回退 _should_retrieve_for_conversation 规则判断。
    # R9.1：LLM 消解的 resolved_query（_conv_resolved_query）供向量检索使用，
    # 处理编号/指代消解（「第三点有哪些推荐的」→「AI/职场 相关收藏推荐」）。
    judge_need = request_input.get("_conv_need_retrieval")
    should_retrieve = (
        bool(judge_need) if judge_need is not None
        else _should_retrieve_for_conversation(query, recent)
    )
    items: list[dict] = []
    refs: list[dict] = []
    if should_retrieve:
        items, refs = await _conditional_retrieve_for_conversation(
            query, recent, request_input)

    trace_event("qa.conversation_memory", phase="qa", data={
        "recent_turns": len(recent),
        "source": "session_context" if session_context.get("recent") else "history",
        "query": query[:50],
        "retrieved": len(items),
        "should_retrieve": should_retrieve,
    }, message="读取会话记忆" + (f"，条件检索命中 {len(items)} 条" if should_retrieve else "，纯会话作答"))

    # R9.1 prompt 分流：检索是否为主角取决于查询类型。
    #   - 推荐/索要内容型（need_retrieval=True，如「第三点有哪些推荐的」）：
    #     检索到的条目是**用户收藏里该主题的真实内容**，是回答主角。
    #     关键结构：条目详情作为**独立 system 消息放在消息流末尾**（query 之前），
    #     权重最高——历史消息里的否定表述（"目前还没有X内容"）可能过时，不能压过
    #     检索到的真实条目；最近对话压缩为摘要，避免逐条否定干扰。
    #   - 纯回顾型（need_retrieval=False，如「总结上面的内容」）：检索仅作补充，
    #     禁止把检索条目当作「上面的内容」总结，总结以最近对话为准。
    is_content_ask = bool(should_retrieve)
    system_prompt = (
        "你是 OmniHub Ask。用户刚才的问题是询问**当前会话里之前的内容**"
        "（含回顾、复述、总结上文、编号/指代回指等）。\n"
    )

    messages: list[dict] = []

    if is_content_ask and items:
        # 最近对话压缩为摘要（保留主题线索，削弱过时否定表述）
        system_prompt += (
            "请结合**最近对话**理解用户指的是哪部分内容，然后以**消息末尾「条目详情」"
            "中检索到的用户收藏条目为准**来回答/推荐——这些条目是用户收藏里该主题的"
            "真实内容，应优先引用（`[标题](content://id)` 链接）。\n"
            "注意：若最近对话里曾提到「该主题没有内容」，但条目详情中已列出相关条目，"
            "**以条目详情为准**（会话里的说法可能是过时的）。\n\n"
        )
        brief = _conversation_brief(recent, max_chars=600)
        if brief:
            system_prompt += f"【最近对话摘要】\n{brief}\n"
        messages.append({"role": "system", "content": system_prompt})
        messages.extend(recent)
        # 条目详情独立 system 消息，放 query 前最后位置（权重最高）
        detail = "【条目详情（用户收藏里该主题的真实内容，据此回答/推荐）】\n"
        for i, it in enumerate(items, 1):
            title = str(it.get("title") or "无标题")
            cid = it.get("id")
            summary = str(it.get("summary") or "")[:300]
            detail += f"{i}. {title} (content://{cid})\n   摘要: {summary}\n"
        messages.append({"role": "system", "content": detail})
        messages.append({"role": "user", "content": query})
    else:
        system_prompt += (
            "请**首要**基于下面给出的**最近对话**来回答/总结；"
            "如果最近对话里没有相关信息，请如实说明。\n"
        )
        if items:
            system_prompt += (
                "【补充信息】用户问到的具体收藏条目详情见下方「条目详情」，"
                "**仅用于回答用户问到的具体条目的细节/地址/价格等**（可引用 "
                "`[标题](content://id)` 链接）；**禁止**把检索到的条目当作『上面的内容』"
                "来总结，总结仍以最近对话为准。\n\n"
                "【条目详情】\n"
            )
            for i, it in enumerate(items, 1):
                title = str(it.get("title") or "无标题")
                cid = it.get("id")
                summary = str(it.get("summary") or "")[:300]
                system_prompt += f"{i}. {title} (content://{cid})\n   摘要: {summary}\n"
        system_prompt += "\n【最近对话】\n"
        messages.append({"role": "system", "content": system_prompt})
        messages.extend(recent)
        messages.append({"role": "user", "content": query})

    yield ndjson("thinking", {"phase": "generating", "message": "正在回顾刚才的对话..."})
    if refs:
        yield ndjson("references", {"items": refs})
    trace_event("qa.step.start", phase="qa",
                data={"step": "ConversationMemoryActStep", "recent_turns": len(recent), "retrieved": len(items)},
                message="基于会话记忆流式作答")

    full_text = ""
    try:
        from omnibox_agent.services.llm_service import stream_chat
        async for token in stream_chat(
            messages,
            ai_config=ai_config,
            temperature=0.5,
            max_tokens=4096,
            no_thinking=True,
        ):
            full_text += token
            yield ndjson("token", {"delta": token})
        trace_event("qa.step.done", phase="qa",
                    data={"step": "ConversationMemoryActStep", "ok": True,
                          "duration_ms": int((time.monotonic() - t_start) * 1000)},
                    message=f"会话作答完成，耗时 {(time.monotonic() - t_start):.1f}s")
    except Exception as e:
        log.error("Conversation memory pipeline LLM streaming failed: %s", e)
        trace_event("qa.step.done", phase="qa", level="error",
                    data={"step": "ConversationMemoryActStep", "ok": False})
        if not full_text:
            yield ndjson("error", {
                "reason": "模型服务暂不可用，请稍后重试",
                "code": "llm_error",
            })
            return

    if not full_text:
        full_text = "抱歉，暂时无法基于上面的对话回答，请重新描述。"

    elapsed = round(time.monotonic() - t_start, 2)
    yield ndjson("done", {
        "sessionId": request_input.get("session_id"),
        "text": full_text,
        "metadata": {
            "confidence": "normal",
            "llm_calls": ctx.llm_call_count,
            "elapsed_s": elapsed,
            "conversation_memory": True,
            "retrieved_items": len(items),
        },
    })


# ---- Streaming creative/DAG pipeline ----

class _CreativeFallback(Exception):
    """DAG 管线需要回退（返回 None / 异常）时抛出的哨兵异常，由调用方决定回退路径。"""


async def _stream_creative_run(
    query: str,
    ctx: AgentContext,
    request_input: dict,
    allow_clarify: bool = True,
    skill_manager: Any = None,
) -> AsyncGenerator[str, None]:
    """以队列实时运行 DAG 并产出 thinking/token/done/clarify 事件。

    由于 handle_creative_query 是阻塞式 await，将其放入 asyncio.Task，通过
    progress_cb / token_cb 把阶段进度与最终答案 token 实时放入队列，本生成器
    边收边 yield，实现豆包式"边算边输出"。DAG 各阶段不再长时间静默。

    结束语义：
      - 澄清信号（allow_clarify=True 时）→ yield clarify 事件后结束本流
      - 返回 None / 异常 → raise _CreativeFallback（调用方走 QA / simple resume 回退）

    allow_clarify：DAG 运行期间是否允许澄清。普通 DAG 路径为 True；
    v2（D1/D3）澄清 resume 也传 True——同一问题内允许在后续节点继续澄清，
    次数由 per-node + total 上限封顶（此前传 False 导致"只澄清一次"）。
    """
    import asyncio
    from omnibox_agent.services.creative_orchestrator import handle_creative_query
    from omnibox_agent.services.clarify import ClarifySignal, ClarifySessionCounter
    from omnibox_agent.core.trace_recorder import trace_event

    q: asyncio.Queue = asyncio.Queue()

    # Agent 内部计数权威：同 QA 管线处理，覆盖请求透传字段
    clarify_session_id = request_input.get("clarify_session_id")
    # v2（D2）：无 resume_context 的请求视为新提问，重置该链路计数（防跨问题累计）
    if not request_input.get("resume_context"):
        ClarifySessionCounter.reset(clarify_session_id)
    internal_clarify_count = ClarifySessionCounter.get(clarify_session_id)
    request_input["clarify_count"] = internal_clarify_count

    async def _clarify_cb(decision, phase, context):
        raise ClarifySignal(decision, phase, context)

    async def _progress_cb(phase, message):
        await q.put(("thinking", {"phase": phase, "message": message}))

    async def _token_cb(tok):
        await q.put(("token", {"delta": tok}))

    task = asyncio.ensure_future(handle_creative_query(
        query, ctx,
        clarify_cb=_clarify_cb if allow_clarify else None,
        clarify_count=request_input.get("clarify_count", 0) or 0,
        clarify_enabled=request_input.get("clarify_enabled", True) and allow_clarify,
        progress_cb=_progress_cb,
        token_cb=_token_cb,
        skill_manager=skill_manager,
    ))

    full_text = ""
    while True:
        try:
            kind, data = await asyncio.wait_for(q.get(), timeout=0.2)
            if kind == "token":
                full_text += data["delta"]
            yield ndjson(kind, data)
            continue
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            task.cancel()
            raise
        if task.done():
            break

    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        if isinstance(exc, ClarifySignal):
            # v2（R3）：计数已在判定时经 try_incr 原子占位；旧路径兜底 incr（不重复计数）
            new_snap = getattr(exc.decision, "_reserved_snapshot", None) \
                or ClarifySessionCounter.incr(clarify_session_id, phase=exc.phase)
            new_total = new_snap["total"]
            phase_this = new_snap["phase_counts"].get(exc.phase, 1)
            trace_event("creative.clarify", phase="creative", data={
                "phase": exc.phase, "importance": exc.decision.importance,
                "clarify_total": new_total, "clarify_phase_count": phase_this})
            event_data = exc.decision.to_event_data()
            ctx_dict = event_data.setdefault("context", {})
            ctx_dict["_clarify_total"] = new_total
            ctx_dict["_clarify_phase"] = exc.phase
            ctx_dict["_clarify_phase_count"] = phase_this
            yield ndjson("clarify", event_data)
            return
        log.warning("Creative streaming run failed, falling back: %s", exc)
        raise _CreativeFallback()
    response = task.result()
    if response is None:
        raise _CreativeFallback()

    if not full_text:
        full_text = response.get("answer", "")
    metadata = {
        "pipeline": "creative",
        "confidence": response.get("confidence", "normal"),
        "llm_calls": ctx.llm_call_count,
        "elapsed_s": round(ctx.elapsed(), 2),
        "creative_rounds": ctx.metrics.get("creative_rounds", 0),
    }
    missing = response.get("missing") or []
    if missing:
        metadata["missing"] = missing
    if response.get("partial"):
        metadata["partial"] = True
    if request_input.get("resume_context"):
        metadata["resumed_from_clarify"] = True

    # §5.6：技能命中可观测性字段（append-only）
    _add_skill_metadata(metadata, ctx.artifacts.get("skills"))

    yield ndjson("done", {
        "sessionId": request_input.get("session_id"),
        "text": full_text,
        "metadata": metadata,
    })


def _add_skill_metadata(metadata: dict, skills) -> None:
    """把 SkillResolution 的技能可观测性字段追加进 done.metadata（§5.6）。"""
    if skills is None:
        return
    selected = getattr(skills, "selected", None)
    if selected:
        metadata["skills"] = [s.name for s in selected]
        metadata["skill_matched_by"] = getattr(skills, "matched_by", None)
        if getattr(skills, "candidates", None):
            metadata["skill_candidates"] = skills.candidates
        if getattr(skills, "match_score", None) is not None:
            metadata["skill_score"] = skills.match_score
    if getattr(skills, "degraded", False):
        metadata["skill_level1"] = "degraded"
    else:
        metadata["skill_level1"] = "ok"
    if getattr(skills, "resources_injected", None):
        metadata["skill_resources"] = skills.resources_injected


async def stream_creative_pipeline(
    ctx: AgentContext,
    request_input: dict,
    skill_manager: Any = None,
) -> AsyncGenerator[str, None]:
    """Streaming version of the creative Plan-Solve-Reflect-Synthesize pipeline.

    DAG 四阶段：PLAN → SOLVE → REFLECT → SYNTHESIZE。
    现在通过 _stream_creative_run 实现：SOLVE/REFLECT 阶段实时下发阶段进度
    thinking 事件，SYNTHESIZE 阶段逐 token 流式输出最终答案（豆包式）。
    """
    query = request_input.get("query", "")
    # R10：会话指代查询若被判定为 complex 走 DAG，plan 的拆解 query 优先用 LLM
    # 消解的 resolved_query（「根据第三点的收藏内容进行分析」→「AI/职场 相关收藏
    # 推荐」，DAG 才能拆出 AI/职场子任务而非「第三点」），无则用原始 query。
    conv_resolved = (request_input.get("_conv_resolved_query") or "").strip()
    if conv_resolved:
        query = conv_resolved

    yield ndjson("thinking", {
        "phase": "planning",
        "message": "复杂问题，正在拆解子任务...",
    })

    try:
        async for event in _stream_creative_run(query, ctx, request_input, skill_manager=skill_manager):
            yield event
    except _CreativeFallback:
        # Creative pipeline failed — fall back to QA
        ctx.counters["creative_fallback"] = ctx.counters.get("creative_fallback", 0) + 1
        async for event in stream_qa_pipeline(ctx, request_input, skill_manager=skill_manager):
            yield event


# ---- Resume pipeline（澄清后恢复作答） ----

async def stream_resume_pipeline(
    ctx: AgentContext,
    request_input: dict,
    skill_manager: Any = None,
) -> AsyncGenerator[str, None]:
    """澄清 resume：跳过 Parse/Retrieve，直接用后端回传的上下文快照重建 prompt，
    拼装 augmented query（原始问题 + 用户补充）后流式作答。

    resume_context 由后端在澄清时缓存（见 clarify.build_clarify_context）：
      { top_items, content_map, qu }
    """
    from omnibox_agent.core.trace_recorder import trace_event
    from omnibox_agent.models.query import QueryUnderstandingResult
    from omnibox_agent.services.ask_orchestrator import _build_system_prompt
    from omnibox_agent.agent.context import RetrievalOutput

    t_start = time.monotonic()
    ai_config = request_input.get("ai_config", {})
    query = request_input.get("query", "")
    resume_context = request_input.get("resume_context") or {}

    # ── DAG 澄清 resume：以澄清后的 augmented query 重跑完整 DAG（§4.1.1 回退/重规划） ──
    if resume_context.get("dag"):
        from omnibox_agent.core.trace_recorder import trace_event
        trace_event("qa.step.start", phase="qa", data={"step": "ResumeDagStep"})
        yield ndjson("thinking", {"phase": "planning", "message": "正在结合你的补充重新规划..."})

        # v3.1 augmented query：DAG resume 此前直接用 resume 请求的 query 重跑，
        # 而 resume 的 query 常是用户对澄清的简短作答（如选项 label），重新 plan
        # 的拆解质量残缺。对齐 _simple_resume：原始问题 + 澄清回答拼接后再进 DAG。
        original_query = (resume_context.get("original_query")
                          or resume_context.get("query") or query)
        answer_text = (request_input.get("answer_text") or "").strip()
        dag_query = original_query
        if answer_text:
            dag_query = f"{original_query}\n\n【用户澄清补充】{answer_text}"

        # v3.1 增量 resume：reflect 强制澄清的答案按意图映射为局部重做/直接合成
        # （"直接用当前版本"/裁决类不再全量重跑）；不可映射（token 失效/意图未知/
        # 非 reflect 强制澄清）自动回退全量重跑（augmented query 路径）
        incremental = None
        try:
            from omnibox_agent.services.clarify import (
                build_incremental_resume,
                get_dag_resume_state,
            )
            token = resume_context.get("_dag_resume_token")
            resume_entry = get_dag_resume_state(token) if token else None
            incremental = build_incremental_resume(
                resume_entry,
                request_input.get("answer_type"),
                request_input.get("answer_key"),
                answer_text,
            )
        except Exception as e:
            log.warning("Incremental resume resolution failed, full rerun: %s", e)
        if incremental is not None:
            request_input["incremental_dag"] = incremental
            trace_event("clarify.incremental_resume", phase="creative", data={
                "skip_to": incremental.get("skip_to"),
                "restored_results": len(incremental.get("results") or {}),
            })
            yield ndjson("thinking", {
                "phase": "planning",
                "message": "已结合你的选择，继续完善内容...",
            })

        try:
            # 队列流式运行 DAG：阶段进度 + Synthesize 逐 token 输出（豆包式）。
            # v2（D1/D3）：resume 允许在后续节点（reflect/synthesize 等）继续澄清，
            # 次数由 per-node + total 上限封顶；用户若再补充，会再次进入本 resume 流。
            async for ev in _stream_creative_run(dag_query, ctx, request_input, allow_clarify=True,
                                                 skill_manager=skill_manager):
                yield ev
        except _CreativeFallback:
            # 兜底：走简单 resume 作答
            async for ev in _simple_resume(ctx, request_input):
                yield ev
            return
        # 可观测性：DAG 澄清被回答并恢复重跑
        trace_event("clarify.answered", phase="creative", data={
            "type": "dag",
            "phase": resume_context.get("phase"),
            "answer_type": request_input.get("answer_type"),
            "answer_key": request_input.get("answer_key"),
            "elapsed_ms": int(ctx.elapsed() * 1000),
        })
        return

    # 简单 QA 澄清恢复
    async for ev in _simple_resume(ctx, request_input):
        yield ev


async def _simple_resume(
    ctx: AgentContext,
    request_input: dict,
) -> AsyncGenerator[str, None]:
    """simple QA 澄清恢复：用后端回传的上下文快照重建 prompt 后流式作答。"""
    from omnibox_agent.core.trace_recorder import trace_event
    from omnibox_agent.models.query import QueryUnderstandingResult
    from omnibox_agent.services.ask_orchestrator import _build_system_prompt
    from omnibox_agent.agent.context import RetrievalOutput

    t_start = time.monotonic()
    ai_config = request_input.get("ai_config", {})
    query = request_input.get("query", "")
    resume_context = request_input.get("resume_context") or {}

    trace_event("qa.step.start", phase="qa", data={"step": "ResumeStep"})

    # ── 从上下文快照重建 retrieval + qu ──
    top_items = resume_context.get("top_items") or []
    content_map = resume_context.get("content_map") or {}
    content_map = {int(k): v for k, v in content_map.items()}
    qu_dict = resume_context.get("qu") or {}
    qu_result = QueryUnderstandingResult(
        resolved_query=qu_dict.get("resolved_query", "") or query,
        explicit_limit=bool(qu_dict.get("explicit_limit", False)),
        want_classify=bool(qu_dict.get("want_classify", False)),
        classify_by=qu_dict.get("classify_by"),
    )
    retrieval = RetrievalOutput(
        fused_items=top_items,
        content_map=content_map,
        total_count=len(top_items),
    )

    # 发送 references（澄清前已检索的内容）
    if top_items:
        refs = _build_ref_items(retrieval)
        if refs:
            yield ndjson("references", {"items": refs})

    # ── v2（D1/D3）：resume 也允许继续澄清（同一问题内多次） ──
    # 受 per-phase(qa≤3) + total(≤5) 上限约束；judge 携带用户澄清答案
    # （supplement），防对同一歧义反复追问。need 则发 clarify 事件结束本流，
    # 用户再答 → 再次 resume → 再次判定，直到上限或歧义消解。
    if request_input.get("clarify_enabled", True) and get_config().clarify.enabled:
        from omnibox_agent.services.clarify import (
            judge_need_clarification,
            ClarifySessionCounter,
            build_resume_supplement,
        )
        clarify_session_id = request_input.get("clarify_session_id")
        snap = ClarifySessionCounter.get_state(clarify_session_id)
        cfg_clarify = get_config().clarify
        decision = await judge_need_clarification(
            query=query,
            qu_result=qu_result,
            retrieval=retrieval,
            history=request_input.get("history", []) or [],
            ai_config=ai_config,
            phase="qa",
            total_count=snap["total"],
            phase_count=snap["phase_counts"].get("qa", 0),
            max_total_per_stream=cfg_clarify.effective_max_total(),
            max_per_phase=cfg_clarify.max_per_phase_qa,
            enabled=True,
            is_resume=True,
            supplement=build_resume_supplement(request_input),
        )
        if decision is not None and decision.need:
            # v2（R3）：发出前原子占位；若并发已占满上限则放弃二次澄清、直接作答
            reserved = ClarifySessionCounter.try_incr(
                clarify_session_id, phase="qa",
                max_total=cfg_clarify.effective_max_total(),
                max_phase=cfg_clarify.max_per_phase_qa,
            )
            if reserved is not None:
                # 复用当前 resume_context 作为下一轮澄清上下文（已含 top_items/content_map/qu/skills）
                decision.context = dict(resume_context or {})
                new_snap = reserved
                new_total = new_snap["total"]
                phase_this = new_snap["phase_counts"].get("qa", 1)
                trace_event("qa.clarify", phase="qa", data={
                    "importance": decision.importance, "question": decision.question,
                    "clarify_total": new_total, "clarify_phase_qa": phase_this,
                    "resume_re_clarify": True,
                })
                event_data = decision.to_event_data()
                ctx_dict = event_data.setdefault("context", {})
                ctx_dict["_clarify_total"] = new_total
                ctx_dict["_clarify_phase"] = "qa"
                ctx_dict["_clarify_phase_count"] = phase_this
                yield ndjson("clarify", event_data)
                return
            log.info("Resume clarify reservation rejected (cap reached), answering directly")

    # ── 构建 system prompt + messages ──
    system_prompt = _build_system_prompt(
        qu_result=qu_result,
        top_items=top_items,
        content_map=content_map,
        total_count=len(top_items),
        platform_dist={},
    )

    # §5.5：从 resume_context 恢复技能指令并注入（simple resume 路径）
    skills_snap = resume_context.get("skills")
    if skills_snap and skills_snap.get("instructions"):
        from omnibox_agent.agent.graph_skill import build_skill_instructions
        system_prompt += build_skill_instructions(skills_snap["instructions"])

    # 追加澄清回答的标注（供 Agent 上下文理解用户补充）
    answer_text = request_input.get("answer_text") or ""
    answer_type = request_input.get("answer_type") or "custom"
    answer_key = request_input.get("answer_key")
    original_query = request_input.get("original_query") or ""
    augment = (
        f"用户澄清回答：{answer_text}"
        + (f"（对应选项 key={answer_key}）" if answer_key else "（自由输入）")
    )
    system_prompt += (
        f"\n\n【澄清补充】你之前因 {original_query} 存在歧义而向用户追问。"
        f"{augment}。请结合该补充与已检索内容给出最终回答。"
    )

    # 记忆系统：会话摘要 + 近期消息注入（§4.3，三条管线统一消费；未启用时无影响）
    session_context = request_input.get("session_context")
    if session_context:
        from omnibox_agent.services.session_store import session_memory_suffix
        system_prompt += session_memory_suffix(session_context)

    # 长期记忆：L1 画像 + L2/L3 召回注入（§12.2；未启用时 long_term 为空，行为不变）
    long_term = request_input.get("long_term")
    if long_term:
        from omnibox_agent.services.memory_manager import (
            user_profile_suffix, recalled_memories_suffix)
        system_prompt += user_profile_suffix(long_term)
        system_prompt += recalled_memories_suffix(long_term)

    messages = [{"role": "system", "content": system_prompt}]
    if session_context:
        messages.extend(session_context.get("recent") or [])
    messages.append({"role": "user", "content": query})

    yield ndjson("thinking", {"phase": "generating", "message": "正在组织回答..."})
    trace_event("qa.step.start", phase="qa", data={"step": "ResumeActStep"})

    full_text = ""
    try:
        from omnibox_agent.services.llm_service import stream_chat

        async for token in stream_chat(
            messages,
            ai_config=ai_config,
            temperature=0.7,
            max_tokens=4096,
            no_thinking=True,
        ):
            full_text += token
            yield ndjson("token", {"delta": token})
    except Exception as e:
        log.error("Resume pipeline LLM streaming failed: %s", e)
        if not full_text:
            yield ndjson("error", {
                "reason": "模型服务暂不可用，请稍后重试",
                "code": "llm_error",
            })
            return

    if not full_text:
        full_text = "抱歉，暂时无法基于你的补充继续回答，请重新描述。"

    elapsed = round(time.monotonic() - t_start, 2)
    metadata = {
        "confidence": "normal",
        "llm_calls": ctx.llm_call_count,
        "elapsed_s": elapsed,
        "resumed_from_clarify": True,
    }
    # 可观测性：澄清被回答（simple QA resume）
    trace_event("clarify.answered", phase="qa", data={
        "type": "simple",
        "answer_type": request_input.get("answer_type"),
        "answer_key": request_input.get("answer_key"),
        "elapsed_ms": int(elapsed * 1000),
    })
    yield ndjson("done", {
        "sessionId": request_input.get("session_id"),
        "text": full_text,
        "metadata": metadata,
    })


# ---- Helpers ----

def _map_abort_reason(code: str) -> str:
    return {
        "error": "处理失败，请稍后重试",
        "busy": "服务繁忙，请稍后重试",
        "guard": "尚未授权任何账号",
    }.get(code, "处理失败，请稍后重试")
