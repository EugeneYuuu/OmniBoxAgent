"""LangGraph QA subgraph: 流式 QA 编排（Parse→Guard→Retrieve→Gate→Reason→Act→done）.

节点经 progress_cb/references_cb/token_cb 实时产出 NDJSON 事件，
clarify/error/done 事件在 stream_qa_pipeline handler 组装。

Design notes (see docs/omnibox-agent-langchain-langgraph-migration.md §4):
  - GraphState 是 TypedDict，包装 mutable AgentContext + 流式回调。
  - 非 critical 降级在节点内 try/except 处理，LangGraph 只管状态流转。
  - Critical abort (GuardStep → PipelineAborted) 由 run_qa_graph 捕获写入 error。
  - 无 checkpoint saver：澄清用 ClarifySignal 中断，State 不需序列化。
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from omnibox_agent.agent.ask_agent import (
    GateStep,
    GuardStep,
    ParseStep,
    ReasonStep,
    RetrieveStep,
)
from omnibox_agent.agent.context import AgentContext, RetrievalOutput
from omnibox_agent.agent.loop import ExecutorBusyError, PipelineAborted
from omnibox_agent.core.config import Config, get_config
from omnibox_agent.core.trace_recorder import trace_event

log = logging.getLogger(__name__)


class GraphState(TypedDict):
    """LangGraph state: 整个 AgentContext + 流式回调作为字段传递。"""
    ctx: AgentContext
    request_input: dict
    progress_cb: Any | None         # async (phase, message) -> None
    token_cb: Any | None            # async (token) -> None
    clarify_cb: Any | None          # async (decision) -> None，内部 raise ClarifySignal
    references_cb: Any | None       # async (retrieval) -> None
    skill_manager: Any | None       # SkillManager | None（显式注入，§5.4）


# ── Node wrappers (reuse existing step implementations) ─────────────────

async def _run_step(ctx: AgentContext, step: Any, name: str) -> None:
    """在节点内执行一个 step，并保留原 AgentLoop 的降级/计时/追踪语义。

    节点直接调用 Step.execute 会绕开 AgentLoop.run，因此把原来写在
    AgentLoop.run 里的非 critical 降级、`{step.name}_ms` 指标和
    `qa.step.start/done` 事件下沉到节点内（方案 §4.8）。

    降级语义（与 AgentLoop.run 一致）：
      - critical 步骤失败 → 抛 PipelineAborted（中止整图，run_qa_graph 捕获）
      - 非 critical 步骤失败 → 记 ctx.errors，继续下一节点
    """
    start = time.monotonic()
    ok = False
    trace_event("qa.step.start", phase="qa",
                data={"step": name, "critical": getattr(step, "critical", False)})
    try:
        await step.execute(ctx)
        ok = True
    except PipelineAborted:
        # critical 中止（如 guard 无账号）→ 向上传播，由 run_qa_graph 映射
        raise
    except ExecutorBusyError:
        if getattr(step, "critical", False):
            raise PipelineAborted("Service busy, please retry later", code="busy") from None
        log.warning("Executor busy at step %s", name)
        ctx.errors.append(f"{name}: executor busy")
    except Exception as e:
        if getattr(step, "critical", False):
            raise PipelineAborted(f"Critical step '{name}' failed: {e}", code="error") from e
        log.exception("Step %s failed (degraded): %s", name, e)
        ctx.errors.append(f"{name}: {e}")
    finally:
        ms = int((time.monotonic() - start) * 1000)
        ctx.metrics[f"{name}_ms"] = ms
        trace_event("qa.step.done", phase="qa",
                    data={"step": name, "ok": ok, "duration_ms": ms})


async def _progress(state: GraphState, phase: str, message: str) -> None:
    """节点间阶段进度（thinking 事件），经 progress_cb 入队（方案 §4.6）。"""
    cb = state.get("progress_cb")
    if cb is not None:
        await cb(phase, message)


async def parse_node(state: GraphState) -> GraphState:
    await _progress(state, "parsing", "正在理解问题...")
    await _run_step(state["ctx"], ParseStep(), "ParseStep")
    return state


async def guard_node(state: GraphState) -> GraphState:
    await _progress(state, "checking", "正在校验权限...")
    await _run_step(state["ctx"], GuardStep(), "GuardStep")
    return state


async def retrieve_node(state: GraphState) -> GraphState:
    await _progress(state, "retrieving", "正在检索你的收藏内容...")
    await _run_step(state["ctx"], RetrieveStep(config=get_config()), "RetrieveStep")
    return state


async def gate_node(state: GraphState) -> GraphState:
    await _progress(state, "filtering", "正在筛选相关内容...")
    await _run_step(state["ctx"], GateStep(config=get_config()), "GateStep")
    # 流式：Gate 后提前下发 references（与 stream_qa_pipeline 一致）
    ref_cb = state.get("references_cb")
    if ref_cb is not None:
        await ref_cb(state["ctx"].artifacts.get("retrieval"))
    return state


async def skill_node(state: GraphState) -> GraphState:
    """SKILL 渐进式匹配（非 critical）。匹配 query 取 perception.resolved_query。"""
    from omnibox_agent.agent.graph_skill import skill_node as _skill_node

    ctx = state["ctx"]
    perception = ctx.artifacts.get("perception")
    query = (getattr(perception, "resolved_query", "") or "") or state["request_input"].get("query", "")
    await _skill_node(ctx, query, state.get("skill_manager"), _progress_cb_for(state))
    return state


def _progress_cb_for(state: GraphState):
    cb = state.get("progress_cb")
    if cb is None:
        return None

    async def wrapper(phase, message):
        await cb(phase, message)
    return wrapper


async def reason_node(state: GraphState) -> GraphState:
    await _progress(state, "reasoning", "正在构建回答策略...")
    await _run_step(state["ctx"], ReasonStep(), "ReasonStep")
    # 流式：检索后主路径澄清判定（§4.2 晚·检索后）
    await _maybe_clarify_qa(state)
    return state


async def act_node(state: GraphState) -> GraphState:
    """Act 节点：全流式，经 token_cb 逐 token 输出；无 token_cb 时 no-op（eval 不消费）。"""
    await _progress(state, "generating", "正在组织回答...")
    ctx = state["ctx"]

    if state.get("token_cb") is None:
        # 无 token_cb（如 eval 场景）不生成——done 事件由 handler 组装。
        return state

    query = state["request_input"].get("query", "")
    ai_config = state["request_input"].get("ai_config", {})
    reasoning = ctx.artifacts.get("reasoning")
    messages = list(reasoning.messages) if reasoning and reasoning.messages \
        else [{"role": "user", "content": query}]

    trace_event("qa.step.start", phase="qa", data={"step": "ActStep"})
    try:
        from omnibox_agent.services.llm_service import stream_chat
        async for token in stream_chat(
            messages,
            ai_config=ai_config,
            temperature=0.7,
            max_tokens=4096,
            no_thinking=True,
        ):
            await state["token_cb"](token)
        trace_event("qa.step.done", phase="qa", data={"step": "ActStep", "ok": True})
    except Exception as e:
        log.error("Stream pipeline LLM streaming failed: %s", e)
        ctx.flags["llm_stream_error"] = True
        trace_event("qa.step.done", phase="qa", level="error",
                    data={"step": "ActStep", "ok": False})
    return state


async def build_node(state: GraphState) -> GraphState:
    """Build 节点：no-op（流式 handler 组装 done 事件）。"""
    return state


async def _maybe_clarify_qa(state: GraphState) -> None:
    """检索后的澄清判定：需要时通过 clarify_cb 中断图，由 handler 转 clarify 事件。"""
    if state.get("clarify_cb") is None:
        return
    request_input = state["request_input"]
    if not request_input.get("clarify_enabled", True):
        return

    # R10：会话内指代查询（LLM judge 判定 referential=True，即使走了 simple 检索路径）
    # 不弹澄清气泡——用户问的是会话里已有内容，澄清无意义（R6 正则只能覆盖部分
    # 表达，LLM 判定的编号/指代类需在此兜底跳过）。
    if request_input.get("_conv_referential"):
        log.info("Clarify skipped: conversation-referential query (LLM judge)")
        return

    from omnibox_agent.services.clarify import (
        build_clarify_context,
        judge_need_clarification,
        build_resume_supplement,
        ClarifySessionCounter,
    )
    from omnibox_agent.core.trace_recorder import trace_event

    ctx = state["ctx"]
    cfg = get_config().clarify
    query = request_input.get("query", "")
    ai_config = request_input.get("ai_config", {})

    # Agent 内部双维计数权威：查 clarify_session_id 的 total + per phase("qa")
    clarify_session_id = request_input.get("clarify_session_id")
    snap = ClarifySessionCounter.get_state(clarify_session_id)
    total_count = snap["total"]
    phase_count = snap["phase_counts"].get("qa", 0)

    decision = await judge_need_clarification(
        query=query,
        qu_result=ctx.artifacts.get("perception"),
        retrieval=ctx.artifacts.get("retrieval", RetrievalOutput()),
        history=request_input.get("history", []) or [],
        ai_config=ai_config,
        phase="qa",
        total_count=total_count,
        phase_count=phase_count,
        max_total_per_stream=cfg.effective_max_total(),
        max_per_phase=cfg.max_per_phase_qa,
        enabled=True,
        # v2（D1/D3）：resume 请求跳过 already_clarified 守卫；携带用户澄清答案防重复追问
        is_resume=bool(request_input.get("resume_context")),
        supplement=build_resume_supplement(request_input),
    )
    if decision is None or not decision.need:
        return

    # v2（R3/TOCTOU）：判定通过后在发出前原子占位；若并发同 id 请求已占满上限则放弃本次澄清
    reserved = ClarifySessionCounter.try_incr(
        clarify_session_id, phase="qa",
        max_total=cfg.effective_max_total(),
        max_phase=cfg.max_per_phase_qa,
    )
    if reserved is None:
        log.info("Clarify reservation rejected (cap reached), skipping clarify")
        return
    decision._reserved_snapshot = reserved

    retrieval = ctx.artifacts.get("retrieval", RetrievalOutput())
    qu_result = ctx.artifacts.get("perception")
    decision.context = build_clarify_context(retrieval=retrieval, qu_result=qu_result)
    # §5.5：技能命中快照并入澄清 context，供 resume 时恢复注入
    skills = ctx.artifacts.get("skills")
    if skills is not None and getattr(skills, "to_snapshot", None):
        decision.context["skills"] = skills.to_snapshot()
    trace_event("qa.step.done", phase="qa", data={
        "step": "ClarifyJudge", "ok": True, "need": True,
        "importance": decision.importance})
    trace_event("clarify.asked", phase="qa", data={
        "type": "simple",
        "question": decision.question,
        "options_count": len(decision.options),
        "importance": decision.importance,
        "recommendedKey": decision.recommended_key,
        "clarify_total": total_count,
        "clarify_phase_count": phase_count,
    })
    await state["clarify_cb"](decision)  # 内部 raise ClarifySignal → 中断图


# ── Graph construction ──────────────────────────────────────────────────

# 节点是无状态的，编译好的图可复用；避免每次 run_qa_graph 都重新 compile。
_graph_cache: Any | None = None


def build_qa_graph(config: Config | None = None) -> Any:
    """Build the QA subgraph: Parse→Guard→Retrieve→Gate→Reason→Act→Build."""
    global _graph_cache
    if _graph_cache is not None:
        return _graph_cache

    g = StateGraph(GraphState)
    g.add_node("parse", parse_node)
    g.add_node("guard", guard_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("gate", gate_node)
    g.add_node("skill", skill_node)
    g.add_node("reason", reason_node)
    g.add_node("act", act_node)
    g.add_node("build", build_node)

    g.set_entry_point("parse")
    g.add_edge("parse", "guard")
    g.add_edge("guard", "retrieve")
    g.add_edge("retrieve", "gate")
    g.add_edge("gate", "skill")
    g.add_edge("skill", "reason")
    g.add_edge("reason", "act")
    g.add_edge("act", "build")
    g.add_edge("build", END)

    _graph_cache = g.compile()
    return _graph_cache


async def run_qa_graph(
    ctx: AgentContext,
    config: Config | None = None,
    *,
    request_input: dict | None = None,
    progress_cb: Any | None = None,
    token_cb: Any | None = None,
    clarify_cb: Any | None = None,
    references_cb: Any | None = None,
    skill_manager: Any | None = None,
) -> AgentContext:
    """Run the QA subgraph: Parse→Guard→Retrieve→Gate→Reason→Act(流式)→done.

    节点经 progress_cb/references_cb/token_cb 实时产出 thinking/references/token
    事件。clarify/error/done 事件在调用方 handler 组装。

    Critical aborts (PipelineAborted) → `ctx.artifacts["error"]`。
    ClarifySignal 原样向上传播，由 handler 转成 clarify 事件。
    """
    graph = build_qa_graph(config)
    initial: GraphState = {
        "ctx": ctx,
        "request_input": request_input or {},
        "progress_cb": progress_cb,
        "token_cb": token_cb,
        "clarify_cb": clarify_cb,
        "references_cb": references_cb,
        "skill_manager": skill_manager,
    }
    try:
        await graph.ainvoke(initial)
    except PipelineAborted as e:
        ctx.artifacts["error"] = {"code": e.code, "message": str(e)}
    return ctx


__all__ = [
    "GraphState",
    "build_qa_graph",
    "run_qa_graph",
]