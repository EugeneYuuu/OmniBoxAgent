"""LangGraph Creative subgraph (migration unit two).

Maps the hand-written 6-state machine in
`services/creative_orchestrator.py::handle_creative_query` onto a
LangGraph StateGraph, while reusing the existing stage implementations
(plan / solve_phase / reflect / synthesize) verbatim. `solve` and `replan`
keep ALL their internal logic inside a single node (wave-based parallel
scheduler and Strategy 1/2/3 re-plan are intentionally NOT split into
LangGraph Send subgraphs — see docs §4.4).

Design notes (see docs/omnibox-agent-langchain-langgraph-migration.md §4.4):
  - CreativeGraphState carries the mutable loop locals (results, plan_output,
    reflect_result, shared_state, round_num, max_rounds) in addition to ctx,
    so nodes can read/write them across transitions.
  - Conditional edges reproduce the 9 transitions exactly:
      PLAN -> SOLVE | DONE(fallback)
      SOLVE -> REFLECT
      REFLECT -> SYNTHESIZE | REPLAN
      REPLAN -> SOLVE
      SYNTHESIZE -> DONE
  - Clarification / progress / token streaming stay at the node level via the
    callbacks (same semantics as the orchestrator). ClarifySignal propagates
    out of ainvoke and is re-raised by run_creative_graph for the stream layer
    to convert into a `clarify` event.
  - No checkpoint saver: creative resumes use the existing draft-cache
    mechanism, so State need not be serializable.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from omnibox_agent.agent.context import AgentContext
from omnibox_agent.core.config import get_config
from omnibox_agent.core.trace_recorder import trace_event
from omnibox_agent.models.note import PlanOutput, ReflectResult, SubResult
from omnibox_agent.services.creative_planner import (
    compute_max_rounds,
    plan,
    route_task,
)
from omnibox_agent.services.creative_reflect import reflect
from omnibox_agent.services.creative_solver import solve_phase
from omnibox_agent.services.creative_synthesize import (
    build_short_circuit_response,
    should_short_circuit,
    synthesize,
)
from omnibox_agent.services.clarify import ClarifySignal

log = logging.getLogger(__name__)


class CreativeGraphState(TypedDict):
    """LangGraph state: creative 状态机全部局部变量作为字段传递。"""
    query: str
    ctx: AgentContext
    clarify_cb: Any | None
    clarify_count: int
    clarify_enabled: bool
    progress_cb: Any | None
    token_cb: Any | None
    results: dict[str, SubResult]
    plan_output: PlanOutput | None
    reflect_result: ReflectResult | None
    shared_state: dict[str, Any]
    round_num: int
    max_rounds: int
    fallback: bool            # PLAN 失败回退 QA 时置 True
    go_synthesize: bool       # REPLAN 全量重规划封顶时跳过 SOLVE 直达 SYNTHESIZE
    response: dict[str, Any] | None
    skill_manager: Any | None  # SkillManager | None（显式注入，§5.4）
    incremental: bool         # v3.1：增量 resume 运行（抑制 reflect 澄清点）
    skip_to: str              # v3.1：增量入口路由 "solve" | "synthesize"


# ── Shared helpers (镜像 orchestrator 的 _progress / _maybe_clarify) ──────

def _plan_summary(plan_output: PlanOutput | None) -> str:
    """Plan 阶段澄清用的子任务概要。"""
    if plan_output is None or not getattr(plan_output, "tasks", None):
        return ""
    return "\n".join(
        f"- {t.id} [{t.type}]: {t.query}" for t in plan_output.tasks
    )


def _result_summary(results: dict | None) -> str:
    """Reflect/Synthesize 阶段澄清用的子任务结果概要。"""
    if not results:
        return ""
    lines = []
    for tid, r in results.items():
        status = getattr(r, "coverage_status", "unknown")
        text = getattr(r, "section_text", "") or ""
        lines.append(f"- {tid}: {status}" + (f" ({len(text)}字)" if text else ""))
    return "\n".join(lines)


def _extract_topic_name(query: str, max_len: int = 25) -> str:
    """从 task query 提取简短、用户可读的主题名。

    task query 通常包含 augmented 内容（如「我收藏的穿搭建议...【用户澄清补充】...」），
    直接展示给用户冗余且混乱。此函数：
    1. 去除【...】/（...）包裹的补充标记
    2. 去掉『我收藏的/我的』等引导前缀
    3. 按标点截断（取第一个分句）
    4. 截断到 max_len 字符
    """
    q = (query or "").strip()
    if not q:
        return ""
    q = re.sub(r"[【\[［][^】\]］]*[】\]］]", "", q)
    q = re.sub(r"[（(][^）)]*[）)]", "", q)
    q = re.sub(r"\s+", " ", q).strip()
    # 去掉常见的引导前缀，只保留核心主题（长项在前，避免只匹配到"我"）
    q = re.sub(
        r"^(?:我收藏的|用户收藏的|关于|针对|我的|我|请|帮我)?"
        r"(?:做一个|做一份|整理|总结|分析|推荐|规划|生成)?"
        r"(?:一下|一份|一个|一遍|一回)?",
        "", q,
    ).strip()
    for sep in ["。", "，", ",", "、", "；", ";", "！", "？"]:
        if sep in q:
            q = q.split(sep)[0].strip()
            break
    if not q:
        return ""
    if len(q) > max_len:
        q = q[:max_len].rstrip() + "…"
    return q


def _detect_forced_clarify(
    reflect_result,
    plan_output: PlanOutput | None,
    round_num: int,
    max_rounds: int,
    base_rounds: int = 0,
    results: dict | None = None,
) -> dict | None:
    """v3 Reflect 强制澄清信号：replan 多轮仍有 poor/conflicts 时返回结构化信号。

    触发条件（规则先行，不经过 confidence 门控）：
      • round_num + base_rounds ≥ min_rounds —— round N 的 reflect 发生在 N-1 次
        replan 之后，默认 3 即严格"已重规划 2 轮"；与 max_rounds 钳制
        （max_rounds=2 → 最后一轮 reflect 也触发，带着伤去 synthesize 不如问一次；
        =1 → 永不触发，零次 replan 不追问），下限 2。
        base_rounds（v3/跨澄清累计）：resume 重跑时从澄清快照恢复的此前已运行
        reflect 轮数，避免 resume 后 round_num 从 0 重置导致强制阈值不可达。
      • 覆盖类问题（sparse/empty）要求单个子任务 source==0 才列入问题任务；
        quality=poor / violation / hallucination 不受 source 数量限制。
      • v3.4：**所有相关子任务全空才澄清**——如果任一 section 子任务有内容
        （sources > 0），说明收藏里确实有相关内容，无需用户定方向，交给普通
        replan 改善召回即可。仅当所有 section 子任务都 source==0（真真空）
        时，才需要用户帮助。
      • conflicts 独立触发，不受 source 检查影响。

    刻意不看 has_fixable：convergence stop（连续 patience 轮无改进）会把它清 False
    直接路由 synthesize，而"连续多轮无改进"恰是最强的卡死信号，必须仍能触发。
    conflicts 是 replan 修不了的（Re-Plan Action Matrix 将其交 Synthesize 仲裁），
    用户裁决显著优于盲仲裁兜底。
    """
    cfg = get_config().clarify
    if not cfg.reflect_force_enabled:
        return None
    min_rounds = max(2, min(cfg.reflect_force_min_rounds, max_rounds))
    if round_num + base_rounds < min_rounds:
        return None

    reasons: list[str] = []
    poor_tasks: list[dict] = []
    conflict_list: list[dict] = []

    if cfg.reflect_force_on_poor:
        task_map = {t.id: t for t in plan_output.tasks} if plan_output is not None else {}
        # 持久问题任务 = get_problem_task_ids()（sparse / empty-with-action /
        # violation / poor / hallucination 全覆盖）∪ quality=poor。
        problem_ids = set(reflect_result.get_problem_task_ids() or [])
        problem_ids |= {
            tid for tid, q in (reflect_result.quality or {}).items() if q == "poor"
        }
        for tid in sorted(problem_ids):
            # v3.4：覆盖类问题（sparse/empty）仅当 source==0 才列入问题任务。
            # 有内容但召回不足（sources≥1）→ 普通 replan 换角度重检索即可。
            r = (results or {}).get(tid)
            n_sources = len(getattr(r, "sources", None) or []) if r is not None else 0
            coverage = (reflect_result.coverage or {}).get(tid)
            is_quality_poor = (reflect_result.quality or {}).get(tid) == "poor"
            is_violation = (reflect_result.compliance or {}).get(tid) == "violation"
            has_hallucination = bool(
                (reflect_result.hallucinations or {}).get(tid)
                and reflect_result.hallucinations[tid].has_hallucination
            )
            if (
                coverage in ("sparse", "empty")
                and n_sources > 0
                and not is_quality_poor
                and not is_violation
                and not has_hallucination
            ):
                continue
            raw_q = getattr(task_map.get(tid), "query", "") or ""
            poor_tasks.append({
                "id": tid,
                "query": raw_q,
                "topic": _extract_topic_name(raw_q) or tid,
                "sources": n_sources,
            })

        # v3.5：只要任一 section 子任务能召回内容（source>0），reflect 就强制
        # 澄清。无论是覆盖类（sparse/empty source==0）还是质量类（poor/empty-
        # apology/violation/hallucination），只要收藏里确实有相关内容，就交给
        # 普通 replan 换角度重检索/重新生成改善，不打扰用户。只有当所有 section
        # 子任务都无内容（source==0，真真空）时，才需要用户定方向。
        if poor_tasks and plan_output is not None:
            any_has_content = False
            for t in plan_output.tasks:
                if t.type != "section":
                    continue
                r = (results or {}).get(t.id)
                if r is not None:
                    n = len(getattr(r, "sources", None) or [])
                    if n > 0:
                        any_has_content = True
                        break
            if any_has_content:
                log.info(
                    "Creative: forced clarify skipped — some section tasks have "
                    "content (sources>0), only %d task(s) are problematic. "
                    "Let normal replan handle it.",
                    len(poor_tasks),
                )
                poor_tasks.clear()
            else:
                reasons.append("poor")
        elif poor_tasks:
            reasons.append("poor")

    if cfg.reflect_force_on_conflicts and reflect_result.conflicts:
        reasons.append("conflicts")
        for c in reflect_result.conflicts[:5]:
            conflict_list.append({
                "sections": list(getattr(c, "sections", []) or []),
                "issue": getattr(c, "issue", "") or "",
            })

    if not reasons:
        return None
    return {
        "reasons": reasons,
        "round_num": round_num,
        "poor_tasks": poor_tasks[:8],
        "conflicts": conflict_list,
    }


async def _maybe_clarify(
    state: CreativeGraphState,
    phase: str,
    plan_output: PlanOutput | None = None,
    results: dict | None = None,
    variant_pool: list | None = None,
    forced_signals: dict | None = None,
) -> None:
    """DAG 澄清点：判定需要澄清时调用 clarify_cb（调用方 raise ClarifySignal 暂停流）。

    双维计数：total（整流全链路最多 5）+ per phase（Plan/Reflect/Synthesize 各节点最多 2）。
    forced_signals（v3）：reflect 强制澄清信号非空时走 judge 强制模式
    （绕过 confidence 门控，LLM 只负责措辞；计数上限仍生效，cap 满则降级不澄清）。
    """
    ctx = state["ctx"]
    if not state["clarify_cb"] or not state["clarify_enabled"]:
        return
    from omnibox_agent.services.clarify import (
        build_dag_clarify_context,
        judge_dag_clarification,
        build_resume_supplement,
        ClarifySessionCounter,
        save_dag_resume_state,
    )
    cfg = get_config().clarify

    ctx_input = ctx.input if hasattr(ctx, "input") else {}

    # Agent 内部双维计数权威
    clarify_session_id = ctx_input.get("clarify_session_id")
    snap = ClarifySessionCounter.get_state(clarify_session_id)
    total_count = snap["total"]
    phase_count = snap["phase_counts"].get(phase, 0)

    decision = await judge_dag_clarification(
        query=state["query"], phase=phase,
        history=ctx_input.get("history", []) or [],
        ai_config=ctx_input.get("ai_config"),
        total_count=total_count,
        phase_count=phase_count,
        max_total_per_stream=cfg.effective_max_total(),
        max_per_phase=cfg.max_per_phase_dag,
        enabled=state["clarify_enabled"],
        plan_summary=_plan_summary(plan_output),
        result_summary=_result_summary(results),
        # v2（D1/D3）：DAG resume 重跑时允许继续澄清（后续节点不再被第一次澄清阻塞）；
        # 携带用户澄清答案防对同一歧义反复追问
        is_resume=bool(ctx_input.get("resume_context")),
        supplement=build_resume_supplement(ctx_input),
        forced_signals=forced_signals,
        # v3.2 否决权收紧：仅当本次 resume 的上一次澄清就发生在 reflect（同一
        # 问题问过、用户答过、仍失败）时，才允许 supplement 否决强制澄清——
        # 防对同一问题反复追问。plan 阶段的偏好型答案（如"综合以上因素"）
        # 对质量/覆盖类持续失败无修复作用，无权否决，否则多轮失败被静默吞掉。
        veto_allowed=(ctx_input.get("resume_context") or {}).get("phase") == "reflect",
    )
    if decision is not None and decision.need:
        # v2（R3/TOCTOU）：判定通过后在发出前原子占位；并发已占满上限则放弃本次澄清
        reserved = ClarifySessionCounter.try_incr(
            clarify_session_id, phase=phase,
            max_total=cfg.effective_max_total(),
            max_phase=cfg.max_per_phase_dag,
        )
        if reserved is None:
            log.info("DAG clarify reservation rejected (cap reached), skipping clarify @%s", phase)
            return
        decision._reserved_snapshot = reserved
        context = build_dag_clarify_context(
            phase, state["query"], plan_output, results, variant_pool,
            # v3：记录澄清触发时已累计的 reflect 轮数（resume 场景含跨澄清累计的
            # creative_rounds_base），供下次 resume 重跑时作为强制澄清轮次基准，
            # 避免 resume 后 round 从 0 重置导致"已重规划 N 轮"累计丢失
            replan_rounds_so_far=ctx.metrics.get(
                "creative_rounds_base", 0) or ctx.metrics.get("creative_rounds", 0),
        )
        # §5.5：技能命中快照并入澄清 context，供 resume 时恢复注入
        skills = ctx.artifacts.get("skills")
        if skills is not None and getattr(skills, "to_snapshot", None):
            context["skills"] = skills.to_snapshot()
        # v3.1 增量 resume：agent 侧缓存完整 plan/results（context 快照正文截断
        # 不可还原），token 随 context 回传，resume 时映射为增量 DAG 初始状态
        context["_dag_resume_token"] = save_dag_resume_state(
            phase=phase, query=state["query"], plan_output=plan_output,
            results=results, forced_signals=forced_signals,
            option_intents=getattr(decision, "option_intents", None),
        )
        decision.context = context
        # 可观测性：DAG 澄清下发（Phase 澄清点）
        trace_event("clarify.asked", phase="creative", data={
            "type": "dag",
            "phase": phase,
            "question": decision.question,
            "options_count": len(decision.options),
            "importance": decision.importance,
            "recommendedKey": decision.recommended_key,
            "clarify_total": total_count,
            "clarify_phase_count": phase_count,
        })
        await state["clarify_cb"](decision, phase, context)  # raises ClarifySignal


async def _progress(state: CreativeGraphState, phase: str, message: str) -> None:
    if state["progress_cb"]:
        await state["progress_cb"](phase, message)


# ── Nodes ───────────────────────────────────────────────────────────────

async def skill_node(state: CreativeGraphState) -> CreativeGraphState:
    """SKILL 渐进式匹配（非 critical）。匹配 query 取 perception.resolved_query。"""
    from omnibox_agent.agent.graph_skill import skill_node as _skill_node

    ctx = state["ctx"]
    perception = ctx.artifacts.get("perception")
    query = (getattr(perception, "resolved_query", "") or "") or state["query"]
    try:
        await _skill_node(ctx, query, state.get("skill_manager"),
                          (lambda p, m: _progress(state, p, m)) if state.get("progress_cb") else None)
    except Exception as e:
        ctx.artifacts["skills"] = None
        log.warning("Creative skill_node degraded: %s", e)
    return state


async def plan_node(state: CreativeGraphState) -> CreativeGraphState:
    """PLAN：拆解子任务 + 路由判定 + 澄清点①。失败则置 fallback 回退 QA。"""
    ctx = state["ctx"]
    query = state["query"]

    # 初始化 creative 上下文（原 orchestrator 在进入循环前初始化）
    ctx.flags.setdefault("variant_pool_done", False)
    ctx.variant_pool = []
    state["go_synthesize"] = False

    await _progress(state, "planning", "正在拆解子任务...")
    trace_event("creative.plan.start", phase="creative")
    plan_output = await plan(query, ctx)

    route = route_task(query, plan_output, ctx)
    if route == "qa":
        plan_err = ""
        if plan_output is None:
            plan_err = "no plan output"
        elif not plan_output.valid:
            plan_err = str(plan_output.error or "plan invalid")[:300]
        elif not plan_output.tasks:
            plan_err = "empty plan tasks"
        log.info("Creative: route_task → qa (fallback): %s", plan_err)
        trace_event("creative.plan.fallback", phase="creative", level="warn",
                    data={"reason": plan_err})
        state["fallback"] = True
        return state

    max_rounds = compute_max_rounds(plan_output)
    log.info("Creative: plan has %d tasks, max_rounds=%d",
             len(plan_output), max_rounds)

    await _maybe_clarify(state, "plan", plan_output=plan_output)

    trace_event("creative.plan.done", phase="creative", data={
        "task_count": len(plan_output),
        "max_rounds": max_rounds,
        "task_types": [t.type for t in plan_output],
    })
    state["plan_output"] = plan_output
    state["max_rounds"] = max_rounds
    return state


async def solve_node(state: CreativeGraphState) -> CreativeGraphState:
    """SOLVE：wave-based 并行求解（内部逻辑整体保留，不拆子图）。"""
    ctx = state["ctx"]
    plan_output = state["plan_output"]
    round_num = state["round_num"]
    results = state["results"]
    shared_state = state["shared_state"]

    await _progress(state, "solving", f"正在生成第 {round_num + 1} 部分内容...")

    # 无时间上限 —— pipeline 不受 time budget 约束
    replan_overrides = None
    reflect_result = state["reflect_result"]
    if reflect_result and reflect_result.replan_actions:
        for tid in reflect_result.replan_actions:
            if tid in results:
                del results[tid]
        replan_overrides = reflect_result.replan_actions

    results = await solve_phase(
        list(plan_output), shared_state, results, ctx,
        replan_overrides,
    )

    trace_event("creative.solve.round", phase="creative", data={
        "round_num": round_num,
        "settled": sum(
            1 for t in plan_output
            if t.id in results and results[t.id].section_text
        ),
        "total_tasks": len([t for t in plan_output if t.type == "section"]),
    })

    # 全部 settled 与否都进 REFLECT（原实现两分支均转 REFLECT，含 deadlock 兜底）
    all_settled = all(
        tid in results for tid in
        [t.id for t in plan_output if t.type == "section"]
    )
    if not all_settled:
        log.warning("Creative: not all tasks settled, forcing REFLECT")

    state["results"] = results
    state["go_synthesize"] = False
    return state


async def reflect_node(state: CreativeGraphState) -> CreativeGraphState:
    """REFLECT：四维评估 + 记录指标 + 澄清点②。条件边决定下一状态。"""
    ctx = state["ctx"]
    plan_output = state["plan_output"]
    results = state["results"]
    round_num = state["round_num"] + 1
    max_rounds = state["max_rounds"]

    await _progress(state, "reflecting", "正在检查内容质量...")

    reflect_result = await reflect(
        list(plan_output), results, ctx, round_num, max_rounds,
    )

    ctx.metrics["creative_rounds"] = round_num
    ctx.metrics["creative_all_pass"] = reflect_result.all_pass
    ctx.metrics["creative_all_empty"] = reflect_result.all_empty
    ctx.metrics["creative_has_fixable"] = reflect_result.has_fixable
    ctx.metrics["creative_conflicts"] = len(reflect_result.conflicts)
    ctx.metrics["creative_replan_actions"] = len(reflect_result.replan_actions)

    trace_event("creative.reflect", phase="creative", data={
        "round_num": round_num,
        "all_pass": reflect_result.all_pass,
        "all_empty": reflect_result.all_empty,
        "has_fixable": reflect_result.has_fixable,
        "conflicts": len(reflect_result.conflicts),
        "replan_actions": len(reflect_result.replan_actions),
    })

    # v3 Reflect 强制澄清：replan 多轮（round≥3，与 max_rounds 钳制）仍有
    # poor/conflicts 时构造结构化信号，judge 走强制模式（LLM 只负责措辞）。
    # base_rounds：resume 重跑时从澄清快照恢复此前已跑轮次，跨澄清累计强制阈值，
    # 否则 resume 后 round 从 0 起算，多轮仍未解决的强制澄清永远不可达。
    # v3.1 增量 resume：本轮为澄清后的单次把关轮（轮次已预置必转 synthesize），
    # 抑制 reflect 澄清点（含强制澄清），防止对同一问题反复追问。
    # v3.3 不重复：resume 且上次澄清已在 reflect 阶段（用户已就这批内容给过处理
    # 意见，如"就用现在的"），resume 后不再重复触发强制澄清——同类问题已问过、答过。
    resume_ctx = (ctx.input or {}).get("resume_context") or {}
    prev_phase = (resume_ctx or {}).get("phase")
    if state.get("incremental") or prev_phase == "reflect":
        log.info("Creative: reflect clarify suppressed "
                 "(incremental=%s, prev_phase=%s)", state.get("incremental"), prev_phase)
    else:
        base_rounds = 0
        if resume_ctx.get("dag"):
            base_rounds = int(resume_ctx.get("creative_rounds_so_far", 0) or 0)
        # 累计轮次写入 metrics：供本轮/后续澄清快照记录，使跨多次 resume 的累计不丢失
        ctx.metrics["creative_rounds_base"] = base_rounds
        forced_signals = _detect_forced_clarify(
            reflect_result, plan_output, round_num, max_rounds,
            base_rounds=base_rounds, results=results,
        )
        if forced_signals:
            ctx.metrics["creative_forced_clarify"] = {
                "reasons": forced_signals["reasons"],
                "round_num": round_num,
                "poor_count": len(forced_signals["poor_tasks"]),
                "conflict_count": len(reflect_result.conflicts),
            }
            trace_event("clarify.forced", phase="creative", data={
                "round_num": round_num,
                "reasons": forced_signals["reasons"],
                "poor_task_ids": [t["id"] for t in forced_signals["poor_tasks"]],
                "conflict_count": len(reflect_result.conflicts),
            })

        await _maybe_clarify(
            state, "reflect",
            plan_output=plan_output, results=results,
            forced_signals=forced_signals,
        )

    state["reflect_result"] = reflect_result
    state["round_num"] = round_num
    return state


async def replan_node(state: CreativeGraphState) -> CreativeGraphState:
    """REPLAN：Strategy 1/2 普通重规划，或 Strategy 3 全量重规划（内部逻辑保留）。"""
    ctx = state["ctx"]
    query = state["query"]
    results = state["results"]
    shared_state = state["shared_state"]
    plan_output = state["plan_output"]

    # Strategy 3: Full re-plan — freeze completed nodes, rebuild DAG
    if ctx.metrics.get("need_full_replan"):
        full_replan_count = ctx.metrics.get("full_replan_count", 0)
        if full_replan_count >= 1:
            log.warning(
                "Creative: full replan limit reached (%d), "
                "synthesizing with current results",
                full_replan_count,
            )
            ctx.metrics["need_full_replan"] = False
            state["go_synthesize"] = True
            return state

        ctx.metrics["full_replan_count"] = full_replan_count + 1
        feedback = ctx.metrics.get("replan_feedback", "")
        log.info("Creative: REPLAN → Strategy 3 (full re-plan #%d): %s",
                 full_replan_count + 1, feedback[:80])

        completed_ids = {
            tid for tid, r in results.items()
            if r.coverage_status == "sufficient" and not r.degraded_reason
        }
        log.info("Creative: freezing %d completed tasks: %s",
                 len(completed_ids), completed_ids)

        # Cleanup stale state from old plan
        for t in plan_output:
            if t.id not in completed_ids:
                for key in t.produces:
                    shared_state.pop(key, None)
        stale_tids = [tid for tid in results if tid not in completed_ids]
        for tid in stale_tids:
            del results[tid]
        log.info("Creative: cleaned %d stale results, kept %d completed",
                 len(stale_tids), len(results))

        replan_query = query
        if feedback:
            replan_query = (
                f"{query}\n\n"
                f"[规划反馈] 上一版方案存在问题: {feedback}. "
                f"请重新规划，确保query保留用户原始查询的核心词。"
            )

        new_plan = await plan(replan_query, ctx)

        if new_plan.valid and new_plan.tasks:
            new_plan.tasks = [
                t for t in new_plan.tasks
                if t.id not in completed_ids
            ]
            if new_plan.tasks:
                plan_output = new_plan
                max_rounds = compute_max_rounds(plan_output)
                log.info("Creative: re-planned %d remaining tasks, "
                         "max_rounds=%d", len(plan_output), max_rounds)
                state["max_rounds"] = max_rounds
            else:
                log.info("Creative: all tasks already completed after re-plan")
        else:
            log.warning("Creative: re-plan failed, keeping old plan")

        # Reset pipeline state for the new plan
        ctx.metrics["need_full_replan"] = False
        ctx.flags["variant_pool_done"] = False
        ctx.metrics["replanned_tasks"] = set()
        ctx.metrics.pop("reflect_problem_history", None)
        state["reflect_result"] = None
        state["round_num"] = 0
        state["plan_output"] = plan_output
        state["results"] = results
        return state

    # Strategy 1/2: Normal replan — just loop back to SOLVE
    log.info("Creative: REPLAN → SOLVE (strategy 1/2 overrides)")
    state["plan_output"] = plan_output
    state["results"] = results
    return state


async def synthesize_node(state: CreativeGraphState) -> CreativeGraphState:
    """SYNTHESIZE：澄清点③ + all-empty 短路或完整合成。"""
    ctx = state["ctx"]
    plan_output = state["plan_output"]
    results = state["results"]
    reflect_result = state["reflect_result"]

    await _progress(state, "generating", "正在组织最终回答...")

    await _maybe_clarify(state, "synthesize", plan_output=plan_output,
                         results=results, variant_pool=ctx.variant_pool)

    if should_short_circuit(results):
        log.info("Creative: short circuit (all empty)")
        response = build_short_circuit_response(results)
        ctx.metrics["creative_short_circuit"] = True
        ctx.metrics["creative_confidence"] = "empty"
        state["response"] = response
        return state

    response = await synthesize(
        list(plan_output), results, ctx.variant_pool,
        reflect_result.conflicts if reflect_result else [],
        ctx,
        token_cb=state["token_cb"],
    )

    ctx.metrics["creative_confidence"] = response["confidence"]
    ctx.metrics["creative_partial"] = response["partial"]
    ctx.metrics["creative_missing"] = response["missing"]
    ctx.metrics["creative_llm_calls"] = ctx.llm_call_count
    ctx.metrics["creative_elapsed"] = round(ctx.elapsed(), 2)

    trace_event("creative.synthesize", phase="creative", data={
        "confidence": response["confidence"],
        "partial": response["partial"],
        "missing_count": len(response.get("missing", [])),
        "source_count": len(response.get("sources", [])),
    })

    state["response"] = response
    return state


# ── Conditional edge routers ────────────────────────────────────────────

def _route_after_plan(state: CreativeGraphState) -> str:
    """PLAN → SOLVE（正常） | END（回退 QA）。"""
    if state.get("fallback"):
        return "__fallback__"
    return "solve"


def _route_after_reflect(state: CreativeGraphState) -> str:
    """REFLECT → SYNTHESIZE | REPLAN（镜像 orchestrator 转移逻辑）。"""
    reflect_result = state["reflect_result"]
    if reflect_result.all_empty:
        log.info("Creative: all empty → short circuit SYNTHESIZE")
        return "synthesize"
    if reflect_result.all_pass or not reflect_result.has_fixable:
        log.info("Creative: all_pass=%s → SYNTHESIZE", reflect_result.all_pass)
        return "synthesize"
    if reflect_result.has_fixable:
        log.info("Creative: has fixable → REPLAN (%d actions)",
                 len(reflect_result.replan_actions))
        return "replan"
    return "synthesize"


def _route_after_replan(state: CreativeGraphState) -> str:
    """REPLAN → SOLVE（重规划后再求解） | SYNTHESIZE（全量重规划封顶）。"""
    if state.get("go_synthesize"):
        return "synthesize"
    return "solve"


# ── Graph construction ──────────────────────────────────────────────────

_graph_cache: Any | None = None


def build_creative_graph() -> Any:
    """Build the Creative subgraph: PLAN→SOLVE→REFLECT→{SYNTHESIZE|REPLAN→SOLVE}→DONE.

    注意：Strategy 3 全量重规划在 replan_node 内部完成（计划重规划后仍转 SOLVE），
    因此 replan → solve 为固定边，与设计 §4.4 一致。
    """
    global _graph_cache
    if _graph_cache is not None:
        return _graph_cache

    g = StateGraph(CreativeGraphState)
    g.add_node("skill", skill_node)
    g.add_node("plan", plan_node)
    g.add_node("solve", solve_node)
    g.add_node("reflect", reflect_node)
    g.add_node("replan", replan_node)
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("skill")
    g.add_edge("skill", "plan")
    g.add_conditional_edges("plan", _route_after_plan, {
        "solve": "solve",
        "__fallback__": END,
    })
    g.add_edge("solve", "reflect")
    g.add_conditional_edges("reflect", _route_after_reflect, {
        "synthesize": "synthesize",
        "replan": "replan",
    })
    g.add_conditional_edges("replan", _route_after_replan, {
        "solve": "solve",
        "synthesize": "synthesize",
    })
    g.add_edge("synthesize", END)

    _graph_cache = g.compile()
    return _graph_cache


# ── v3.1 增量 resume 子图 ────────────────────────────────────────────────

_graph_cache_incremental: Any | None = None


def _route_incremental_entry(state: CreativeGraphState) -> str:
    """增量 resume 入口路由：skill → {solve（局部重做）| synthesize（直接合成）}。"""
    return "solve" if state.get("skip_to") == "solve" else "synthesize"


def build_creative_graph_incremental() -> Any:
    """增量 resume 子图：复用全部现有节点，跳过 PLAN。

    skip_to=solve：SOLVE（仅重做 override 任务）→ REFLECT（单次把关，轮次预置使
      必转 SYNTHESIZE，不再 replan/二次澄清）→ SYNTHESIZE → END
    skip_to=synthesize：直接合成（矛盾经合成 reflect_result.conflicts 注入裁决）→ END
    """
    global _graph_cache_incremental
    if _graph_cache_incremental is not None:
        return _graph_cache_incremental

    g = StateGraph(CreativeGraphState)
    g.add_node("skill", skill_node)
    g.add_node("solve", solve_node)
    g.add_node("reflect", reflect_node)
    g.add_node("replan", replan_node)
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("skill")
    g.add_conditional_edges("skill", _route_incremental_entry, {
        "solve": "solve",
        "synthesize": "synthesize",
    })
    g.add_edge("solve", "reflect")
    g.add_conditional_edges("reflect", _route_after_reflect, {
        "synthesize": "synthesize",
        "replan": "replan",
    })
    g.add_conditional_edges("replan", _route_after_replan, {
        "solve": "solve",
        "synthesize": "synthesize",
    })
    g.add_edge("synthesize", END)

    _graph_cache_incremental = g.compile()
    return _graph_cache_incremental


async def run_creative_graph(
    query: str,
    ctx: AgentContext,
    clarify_cb: Any = None,
    clarify_count: int = 0,
    clarify_enabled: bool = True,
    progress_cb: Any = None,
    token_cb: Any = None,
    skill_manager: Any | None = None,
) -> dict[str, Any] | None:
    """Run the Creative subgraph on a prepared AgentContext.

    Mirrors `handle_creative_query` exception semantics:
      - ClarifySignal → re-raise (stream layer converts to `clarify` event)
      - generic Exception → log + set creative_error/creative_fallback, return None
    """
    graph = build_creative_graph()

    initial: CreativeGraphState = {
        "query": query,
        "ctx": ctx,
        "clarify_cb": clarify_cb,
        "clarify_count": clarify_count,
        "clarify_enabled": clarify_enabled,
        "progress_cb": progress_cb,
        "token_cb": token_cb,
        "results": {},
        "plan_output": None,
        "reflect_result": None,
        "shared_state": {},
        "round_num": 0,
        "max_rounds": 1,
        "fallback": False,
        "go_synthesize": False,
        "response": None,
        "skill_manager": skill_manager,
        "incremental": False,
        "skip_to": "",
    }

    # v3.1 增量 resume：stream 层据澄清答案映射的增量初始状态（详见
    # clarify.build_incremental_resume）。恢复 plan/results + 合成 reflect_result
    # （replan_actions=局部重做通道 / conflicts=用户裁决通道），跳过 PLAN。
    incremental = (ctx.input or {}).get("incremental_dag") if hasattr(ctx, "input") else None
    if incremental:
        inc_plan = incremental.get("plan_output")
        inc_results = incremental.get("results") or {}
        max_rounds = compute_max_rounds(inc_plan) if inc_plan is not None else 1
        # 初始化 creative 上下文（正常路径由 plan_node 完成；增量跳过 plan 故需补齐，
        # 否则 synthesize 访问 ctx.variant_pool 会 AttributeError）
        ctx.flags.setdefault("variant_pool_done", False)
        ctx.variant_pool = []
        initial.update({
            "plan_output": inc_plan,
            "results": inc_results,
            "max_rounds": max_rounds,
            "reflect_result": incremental.get("reflect_result"),
            "incremental": True,
            "skip_to": incremental.get("skip_to") or "synthesize",
        })
        if initial["skip_to"] == "solve":
            # 预置轮次使 reflect 单次把关后必转 synthesize：
            # reflect round = round_num+1 = max_rounds → has_fixable=False（不再 replan）
            initial["round_num"] = max(0, max_rounds - 1)
        graph = build_creative_graph_incremental()
        ctx.metrics["incremental_resume"] = {
            "skip_to": initial["skip_to"],
            "restored_results": len(inc_results),
            "max_rounds": max_rounds,
        }
        log.info("Creative: incremental resume (skip_to=%s, %d restored results)",
                 initial["skip_to"], len(inc_results))

    try:
        final = await graph.ainvoke(initial)
    except ClarifySignal:
        # DAG 澄清信号：向上传播，由 stream 层转成 clarify 事件
        raise
    except Exception as e:
        log.exception("Creative pipeline failed: %s — falling back to QA", e)
        ctx.metrics["creative_error"] = str(e)
        ctx.metrics["creative_fallback"] = True
        return None

    round_num = final.get("round_num", 0)

    if final.get("fallback"):
        return None

    # DONE：Ask 追踪 + 主记录轮数
    from omnibox_agent.core.trace_recorder import get_recorder
    from omnibox_agent.core.trace_store import set_rounds
    log.info("Creative: DONE (rounds=%d, elapsed=%.1fs, llm_calls=%d)",
             round_num, ctx.elapsed(), ctx.llm_call_count)
    trace_event("creative.done", phase="creative", data={
        "rounds": round_num,
        "elapsed_ms": int(ctx.elapsed() * 1000),
        "llm_calls": ctx.llm_call_count,
    })
    set_rounds(get_recorder(), round_num)

    return final.get("response")


__all__ = [
    "CreativeGraphState",
    "build_creative_graph",
    "run_creative_graph",
]
