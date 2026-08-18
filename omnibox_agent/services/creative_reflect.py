"""v4.1 §9.4-9.5: Creative Reflect — four-dimensional evaluation.

Four dimensions per Re-Plan round:
  ① Coverage (rule-based, zero LLM): sufficient / sparse / empty
  ② Compliance (batch LLM, JSON contract): compliant / violation
  ③ Quality (batch LLM ∪ hallucination rules): good / poor
  ④ Consistency (LLM, outputs conflict pairs): consistent / conflict

Per-round cost: 3 batch LLM calls + 0 LLM for hallucination (rule-based)
  + 1 LLM call for strategy classification (empty tasks only).

Re-Plan action matrix (§9.5):
  empty     → LLM classifies into Strategy 1/2/3 (see below)
  sparse    → re-retrieve (Strategy 1: rewritten query)
  violation → regenerate (Strategy 1: reuse retrieval, with feedback)
  poor/hall → regenerate (Strategy 1: adjusted prompt + hallucination report)
  conflict  → no re-dispatch (conflict pairs go to Synthesize)

Three replan strategies (LLM-decided for empty tasks):
  Strategy 1 (仅改执行策略): Task query too narrow or wrong search source,
          but task goal is correct. → Override query, keep produces/requires
          contract unchanged. Downstream B naturally waits for A to re-run
          and picks up new produces via stale detection.
  Strategy 2 (改任务目标): Task goal itself is wrong (e.g. narrowed to a
          sub-type instead of a dimension). → Override query with corrected
          goal + CASCADE: modify A's task definition (produces_remap if the
          output structure changes), walk the DAG downward, judge each
          downstream task — if A's new output still satisfies its input
          (requires keys untouched by the remap), B stays unchanged (stale
          detection re-runs it); if not (requires key remapped away), remap
          its requires and add it to the replan queue. Deeper levels recurse
          naturally: when B re-runs and its produces change, C stale-detects.
  Strategy 3 (整体重建): Multiple tasks empty, plan fundamentally broken.
          → Freeze completed nodes, call planner to rebuild remaining DAG.

Key invariant: ALL strategies preserve the produces/requires contract.
  The replan task always writes the same produces keys. Downstream tasks
  detect the changed values via _needs_run() stale detection and re-run
  automatically. No explicit cascade override is needed.

Dynamic rounds + convergence stop (§9.6):
  max_rounds = compute_max_rounds(plan)
  convergence_patience = 2 (consecutive rounds with no improvement → stop)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from omnibox_agent.core.config import get_config
from omnibox_agent.models.note import (
    SubTask, SubResult, SubTaskOverride,
    HallucinationReport, ConflictPair, ReflectResult,
)
from omnibox_agent.services.llm_service import generate

log = logging.getLogger(__name__)


# ── Reflect Entry Point ─────────────────────────────────────────────────

async def reflect(
    plan_tasks: list[SubTask],
    results: dict[str, SubResult],
    ctx: Any,
    round_num: int,
    max_rounds: int,
) -> ReflectResult:
    """Execute four-dimensional evaluation on all sub-task results.

    §9.4: 3 batch LLM calls + rule-based hallucination detection.

    Returns:
        ReflectResult with per-task assessments, conflicts, and Re-Plan actions.
    """
    # Only evaluate tasks that are in the current plan.
    # After Strategy 3 full replan, results may contain frozen completed
    # tasks from the old plan — their task definitions are gone from
    # plan_tasks, so they should not be re-evaluated (they're frozen).
    plan_task_ids = {t.id for t in plan_tasks}
    section_results = {
        tid: r for tid, r in results.items()
        if tid in plan_task_ids
        and (r.section_text or r.coverage_status == "empty" or r.degraded_reason)
    }

    if not section_results:
        return ReflectResult(all_pass=True, all_empty=True)

    # ① Coverage (rule-based, zero LLM)
    coverage = _assess_all_coverage(plan_tasks, results)

    # ② Compliance (batch LLM)
    compliance = await _batch_check_compliance(section_results, plan_tasks, ctx)

    # ③ Quality (batch LLM ∪ hallucination rules)
    quality_llm = await _batch_check_quality(section_results, ctx)
    hallucinations = _detect_all_hallucinations(section_results, results)
    quality = _merge_quality(quality_llm, hallucinations)

    # ③b Empty-apology rule: if a section's text is basically "no related
    # content in collection" (e.g. coverage deemed sufficient by doc count
    # but the sub-agent actually found nothing on-topic), force quality=poor
    # so Reflect re-dispatches it instead of freezing it as "completed".
    # Symptom of the classic bad case: query narrowed to a sub-type (e.g.
    # "本帮菜") that the user's collection doesn't contain → gate rewrites
    # query → 20+ docs come back (all off-topic) → coverage=sufficient →
    # sub-agent honestly writes "收藏里没有相关内容" → but the task gets
    # frozen. This rule catches that and sends it back to replan.
    for tid, r in section_results.items():
        if r.section_text and _is_empty_apology(r.section_text):
            quality[tid] = "poor"
            log.info("Reflect: task %s empty-apology text → quality=poor", tid)

    # ④ Consistency (LLM, outputs conflict pairs)
    conflicts = await _check_consistency(section_results, plan_tasks, ctx)

    # Build Re-Plan actions (async — may call LLM for strategy classification)
    replan_actions = await _build_replan_actions(
        coverage, compliance, quality, hallucinations, conflicts, results, ctx,
        plan_tasks,
    )

    # Overall assessment
    all_empty = all(c == "empty" for c in coverage.values())
    has_fixable = len(replan_actions) > 0 and not all_empty
    all_pass = (
        not has_fixable
        and not all_empty
        and all(c == "sufficient" for c in coverage.values())
        and all(v == "compliant" for v in compliance.values())
        and all(q == "good" for q in quality.values())
        and not conflicts
    )

    # Convergence check
    converged = _check_convergence(ctx, has_fixable, round_num)

    result = ReflectResult(
        coverage=coverage,
        compliance=compliance,
        quality=quality,
        hallucinations=hallucinations,
        conflicts=conflicts,
        replan_actions=replan_actions if has_fixable and not converged and round_num < max_rounds else {},
        all_pass=all_pass,
        all_empty=all_empty,
        has_fixable=has_fixable and not converged and round_num < max_rounds,
    )

    log.info("Reflect round %d: all_pass=%s, all_empty=%s, has_fixable=%s, conflicts=%d, actions=%d",
             round_num, all_pass, all_empty, result.has_fixable,
             len(conflicts), len(result.replan_actions))

    return result


# ── ① Coverage (rule-based, zero LLM) ───────────────────────────────────

def _assess_all_coverage(
    plan_tasks: list[SubTask],
    results: dict[str, SubResult],
) -> dict[str, str]:
    """Assess coverage for all tasks based on source count and coverage_status."""
    coverage = {}
    for t in plan_tasks:
        r = results.get(t.id)
        if r is None:
            coverage[t.id] = "empty"
        elif r.degraded_reason:
            coverage[t.id] = "empty"
        elif r.coverage_status == "empty":
            coverage[t.id] = "empty"
        elif len(r.sources) < 2:
            coverage[t.id] = "sparse"
        else:
            coverage[t.id] = "sufficient"
    return coverage


# ── ② Compliance (batch LLM) ────────────────────────────────────────────

async def _batch_check_compliance(
    section_results: dict[str, SubResult],
    plan_tasks: list[SubTask],
    ctx: Any,
) -> dict[str, str]:
    """Batch LLM: check if each section complies with its constraints.

    Single LLM call for all sections. JSON output.
    """
    task_map = {t.id: t for t in plan_tasks}

    lines = []
    task_ids = []
    for tid, r in section_results.items():
        task = task_map.get(tid)
        constraints = task.constraints if task else {}
        lines.append(f'[{len(task_ids)}] 任务ID: {tid}\n约束: {json.dumps(constraints, ensure_ascii=False)}\n内容: {r.section_text[:500]}')
        task_ids.append(tid)

    prompt = (
        "你是一个合规检查器。逐条检查每个章节是否满足其约束条件。\n"
        "输出 JSON 数组, 每个元素: {\"i\": <序号>, \"compliant\": true/false}\n"
        f"数组长度必须等于 {len(task_ids)}。\n\n"
        + "\n".join(lines)
    )

    ctx.llm_call_count += 1
    try:
        raw = await generate(
            [{"role": "system", "content": "只输出JSON数组。"},
             {"role": "user", "content": prompt}],
            ai_config=ctx.input.get("ai_config"),
            temperature=0.0, max_tokens=2048, timeout=None,
            no_thinking=True,  # 结构化 JSON 输出，关闭思考（同 planner，防 v4-flash 思考占满 max_tokens 导致 content 空）
        )
        verdicts = _parse_indexed_json(raw, len(task_ids), "compliant")
        return {task_ids[i]: ("compliant" if verdicts[i] else "violation") for i in range(len(task_ids))}
    except Exception as e:
        log.warning("Compliance check failed: %r (fail-open: all compliant)", e)
        return {tid: "compliant" for tid in task_ids}


# ── ③ Quality (batch LLM ∪ hallucination rules) ────────────────────────

async def _batch_check_quality(
    section_results: dict[str, SubResult],
    ctx: Any,
) -> dict[str, str]:
    """Batch LLM: assess if each section is "good" (substantive, relevant)."""
    lines = []
    task_ids = []
    for tid, r in section_results.items():
        if not r.section_text:
            continue
        lines.append(f'[{len(task_ids)}] {r.section_text[:400]}')
        task_ids.append(tid)

    if not task_ids:
        return {}

    prompt = (
        "你是一个质量评估器。逐条判断每个章节是否言之有物、内容充实。\n"
        "输出 JSON 数组: [{\"i\": <序号>, \"good\": true/false}]\n"
        f"数组长度必须等于 {len(task_ids)}。\n\n"
        + "\n".join(lines)
    )

    ctx.llm_call_count += 1
    try:
        raw = await generate(
            [{"role": "system", "content": "只输出JSON数组。"},
             {"role": "user", "content": prompt}],
            ai_config=ctx.input.get("ai_config"),
            temperature=0.0, max_tokens=2048, timeout=None,
            no_thinking=True,  # 结构化 JSON 输出，关闭思考（同 planner，防 v4-flash 思考占满 max_tokens 导致 content 空）
        )
        verdicts = _parse_indexed_json(raw, len(task_ids), "good")
        return {task_ids[i]: ("good" if verdicts[i] else "poor") for i in range(len(task_ids))}
    except Exception as e:
        log.warning("Quality check failed: %r (fail-open: all good)", e)
        return {tid: "good" for tid in task_ids}


def _detect_all_hallucinations(
    section_results: dict[str, SubResult],
    all_results: dict[str, SubResult],
) -> dict[str, HallucinationReport]:
    """Run rule-based hallucination detection on all sections (zero LLM)."""
    reports = {}
    for tid, r in section_results.items():
        if not r.section_text:
            continue
        # Build source text from the task's sources
        source_text = _build_source_text(tid, all_results)
        reports[tid] = detect_hallucination(r.section_text, source_text)
    return reports


def detect_hallucination(section_text: str, source_text: str) -> HallucinationReport:
    """§9.4: Rule-based hallucination detection (zero LLM).

    1. Entity extraction: NER/regex for store names, brands, prices, addresses, numbers
    2. Source cross-check: entities not in source text → unsupported
    3. Number validation: generated numbers vs source numbers, >10% deviation → hallucination
    4. Unsupported sentence detection: sentences with no source entities → suspected fabrication
    """
    # Extract entities
    entities = _extract_entities(section_text)
    numbers = _extract_numbers(section_text)

    # Cross-check entities
    unsupported_entities = [
        e for e in entities
        if e.lower() not in source_text.lower() and len(e) >= 2
    ]

    # Cross-check numbers
    source_numbers = _extract_numbers(source_text)
    number_mismatches = []
    for n in numbers:
        matched = any(
            abs(n - sn) <= max(n * 0.1, 1.0)
            for sn in source_numbers
        )
        if not matched and n > 0:
            number_mismatches.append(str(n))

    # Unsupported sentences
    sentences = _split_sentences_simple(section_text)
    unsupported_sentences = []
    for s in sentences:
        s_entities = _extract_entities(s)
        if s_entities and not any(e.lower() in source_text.lower() for e in s_entities):
            unsupported_sentences.append(s[:50])

    has_hall = bool(unsupported_entities or number_mismatches or unsupported_sentences)

    return HallucinationReport(
        has_hallucination=has_hall,
        unsupported_entities=unsupported_entities[:10],
        number_mismatches=number_mismatches[:10],
        unsupported_sentences=unsupported_sentences[:5],
    )


def _merge_quality(
    quality_llm: dict[str, str],
    hallucinations: dict[str, HallucinationReport],
) -> dict[str, str]:
    """Merge LLM quality + hallucination rules (union: either poor → poor)."""
    all_ids = set(quality_llm.keys()) | set(hallucinations.keys())
    result = {}
    for tid in all_ids:
        llm_verdict = quality_llm.get(tid, "good")
        hall = hallucinations.get(tid)
        if llm_verdict == "poor" or (hall and hall.has_hallucination):
            result[tid] = "poor"
        else:
            result[tid] = "good"
    return result


def _is_empty_apology(text: str) -> bool:
    """Rule-based detection: is this section text basically "no related
    content in collection" (empty apology)?

    Covers the phrasing the sub-agent LLM uses when it retrieved docs but
    none were actually on-topic (query narrowed to a sub-type the user's
    collection doesn't contain). Examples:
      "我的收藏里暂时没有专门对应这份主题的清单"
      "这部分在你的收藏中暂时没有相关内容"
      "关于XX，收藏中暂时没有相关内容"
    """
    if not text:
        return False
    patterns = [
        "没有相关",        # 没有相关内容 / 没有相关笔记
        "暂时没有",        # 暂时没有对应 / 暂时没有专门
        "没有对应",        # 没有专门对应 / 没有对应主题
        "暂无",            # 暂无相关内容
        "收藏中没有",      # 收藏中没有相关
        "收藏里没有",      # 收藏里没有相关
        "没有找到",        # 没有找到相关
        "没有专门",        # 没有专门对应这份主题
        "有点遗憾",        # 结尾的遗憾话术
    ]
    return any(p in text for p in patterns)


# ── ④ Consistency (LLM, conflict pairs) ─────────────────────────────────

async def _check_consistency(
    section_results: dict[str, SubResult],
    plan_tasks: list[SubTask],
    ctx: Any,
) -> list[ConflictPair]:
    """LLM: detect inter-section conflicts, output conflict pairs with arbitration."""
    task_map = {t.id: t for t in plan_tasks}

    lines = []
    for tid, r in section_results.items():
        if not r.section_text:
            continue
        lines.append(f'章节[{tid}]: {r.section_text[:300]}')

    if len(lines) < 2:
        return []

    prompt = (
        "你是一致性检查器。检查各章节间是否存在矛盾(如地点、价格、时间冲突)。\n"
        "输出 JSON 数组, 每个冲突: {\"sections\": [\"id1\",\"id2\"], \"issue\": \"描述\", \"arbitrate\": \"以哪节为准\"}\n"
        "如无冲突输出 []。\n\n"
        + "\n".join(lines)
    )

    ctx.llm_call_count += 1
    try:
        raw = await generate(
            [{"role": "system", "content": "只输出JSON数组。"},
             {"role": "user", "content": prompt}],
            ai_config=ctx.input.get("ai_config"),
            temperature=0.0, max_tokens=2048, timeout=None,
            no_thinking=True,  # 结构化 JSON 输出，关闭思考（同 planner，防 v4-flash 思考占满 max_tokens 导致 content 空）
        )
        return _parse_conflicts(raw)
    except Exception as e:
        log.warning("Consistency check failed: %r (fail-open: no conflicts)", e)
        return []


def _parse_conflicts(raw: str) -> list[ConflictPair]:
    """Parse LLM conflict output into ConflictPair list."""
    if not raw or not raw.strip():
        return []
    try:
        text = raw.strip()
        arr_match = re.search(r'\[.*\]', text, re.DOTALL)
        if arr_match:
            text = arr_match.group(0)
        data = json.loads(text)
        conflicts = []
        for item in data:
            conflicts.append(ConflictPair(
                sections=item.get("sections", []),
                issue=item.get("issue", ""),
                arbitrate=item.get("arbitrate", ""),
            ))
        return conflicts
    except (json.JSONDecodeError, TypeError) as e:
        log.debug("Parse conflicts failed: %s", e)
        return []


# ── Re-Plan Action Matrix (§9.5) ────────────────────────────────────────

async def _build_replan_actions(
    coverage: dict[str, str],
    compliance: dict[str, str],
    quality: dict[str, str],
    hallucinations: dict[str, HallucinationReport],
    conflicts: list[ConflictPair],
    results: dict[str, SubResult],
    ctx: Any = None,
    plan_tasks: list[SubTask] | None = None,
) -> dict[str, SubTaskOverride]:
    """§9.5: Build Re-Plan actions based on assessment results.

    For empty tasks: LLM classifies each into Strategy 1/2/3, then generates
    the appropriate override. All strategies preserve the produces/requires
    contract — downstream tasks re-run automatically via stale detection.

    For sparse/violation/poor: Rule-based Strategy 1 (re_retrieve/regenerate).

    Returns:
        Dict of task_id → SubTaskOverride.
    """
    actions: dict[str, SubTaskOverride] = {}

    # Track which tasks have already been replanned (prevent loops)
    replanned_set: set[str] = set()
    if ctx:
        replanned_set = ctx.metrics.setdefault("replanned_tasks", set())

    original_query = ""
    if ctx:
        original_query = ctx.input.get("query", "")

    # Tasks in conflicts don't get re-dispatched
    conflicted_tasks = set()
    for c in conflicts:
        conflicted_tasks.update(c.sections)

    # Collect empty tasks for LLM strategy classification
    empty_task_ids: list[str] = []

    for tid, cov in coverage.items():
        if tid in conflicted_tasks:
            continue  # Conflict → Synthesize handles

        if cov == "empty":
            # Empty tasks are classified by LLM into Strategy 1/2/3
            if tid not in replanned_set:
                empty_task_ids.append(tid)
            # If already replanned, leave as empty (truly not fixable)
            continue

        if cov == "sparse":
            actions[tid] = SubTaskOverride(
                mode="re_retrieve",
                # 内部重检索指令：不得泄漏为"关键词"等检索机制表述
                feedback="检索结果不足，请从不同角度重述查询以扩大召回范围（内部指令，不要向用户提及检索过程）",
                strategy=1,
            )
            continue

        # Check compliance
        if compliance.get(tid) == "violation":
            actions[tid] = SubTaskOverride(
                mode="regenerate",
                feedback="内容违反约束条件，请修正后重新生成",
                strategy=1,
            )
            continue

        # Check quality / hallucination
        if quality.get(tid) == "poor":
            hall = hallucinations.get(tid)
            result = results.get(tid)
            # Empty-apology: sub-agent wrote "收藏里没有相关内容" — the task
            # query was narrowed to a sub-type the collection doesn't have.
            # Regenerating with the same narrow query won't help — replan with
            # the original broad query instead (Strategy 1: broaden query).
            if result and result.section_text and _is_empty_apology(result.section_text):
                actions[tid] = SubTaskOverride(
                    mode="replan",
                    query=original_query,  # 宽泛的原始查询，替代窄化子类型词
                    feedback=(
                        "上一版输出为'收藏中无相关内容'，说明查询词被窄化成了收藏中"
                        "不存在的子类型。请用宽泛的原始查询重新检索（内部指令，"
                        "不要向用户提及检索过程）"
                    ),
                    strategy=1,
                )
                replanned_set.add(tid)
                log.info("Reflect: task %s empty-apology → replan with original query",
                         tid)
                continue
            feedback = "内容质量不足，请补充具体信息"
            if hall and hall.has_hallucination:
                feedback = (
                    f"检测到幻觉: 未支撑实体={hall.unsupported_entities[:3]}, "
                    f"数值不匹配={hall.number_mismatches[:3]}. "
                    "请仅基于检索来源重新生成，不要编造信息"
                )
            actions[tid] = SubTaskOverride(
                mode="regenerate",
                feedback=feedback,
                strategy=1,
            )

    # LLM-based strategy classification for empty tasks
    if empty_task_ids and original_query:
        task_map = {t.id: t for t in plan_tasks} if plan_tasks else {}

        classifications = await _classify_replan_strategy(
            empty_task_ids, task_map, results, original_query, ctx,
        )

        for tid, cls in classifications.items():
            strategy = cls.get("strategy", 1)
            new_query = cls.get("new_query", original_query)
            reason = cls.get("reason", "")

            if strategy == 3:
                # Strategy 3: Major restructure → flag for full re-plan
                # Still create an override so orchestrator deletes the result
                ctx.metrics["need_full_replan"] = True
                ctx.metrics["replan_feedback"] = reason
                actions[tid] = SubTaskOverride(
                    mode="replan",
                    query=new_query or original_query,
                    feedback=f"整体方案需重建: {reason}",
                    strategy=3,
                )
                replanned_set.add(tid)
                log.info("Reflect: task %s → Strategy 3 (full re-plan): %s",
                         tid, reason[:80])
                continue

            # ── Strategy 1: execution-only — A re-runs with the new query,
            #    contract unchanged, downstream untouched (stale detection
            #    re-runs them automatically if A's produces values change).
            # ── Strategy 2: goal/output structure changed — cascade.
            #    1. Modify A's task definition (query override + produces remap)
            #    2. Walk the DAG downward, find all tasks depending on A's keys
            #    3. Judge each downstream: if A's new output still satisfies
            #       its input (requires keys untouched by remap) → B unchanged;
            #       if not (requires key remapped away) → remap its requires and
            #       add it to the replan queue
            #    4. Deeper levels recurse naturally: when B re-runs and its
            #       produces change, C stale-detects and re-runs.
            mode_label = "broaden query" if strategy == 1 else "change goal"
            remap = cls.get("produces_remap") or {}
            actions[tid] = SubTaskOverride(
                mode="replan",
                query=new_query,
                feedback=f"{'执行策略调整' if strategy == 1 else '任务目标调整'}: {reason}",
                strategy=strategy,
            )
            replanned_set.add(tid)
            log.info("Reflect: task %s → Strategy %d (%s): query=%s, reason=%s",
                     tid, strategy, mode_label, new_query[:50], reason[:80])

            if strategy == 2 and remap and plan_tasks:
                task_a = task_map.get(tid)
                # Step 1: modify A's task definition (produces keys remapped).
                # This is an intentional in-place mutation of the SubTask object
                # in plan_tasks. It is safe because:
                #   a) The cascade loop below remaps ALL downstream requires
                #      BEFORE the next SOLVE wave runs.
                #   b) Downstream stale detection in _needs_run() uses B's
                #      (already-remapped) requires vs its old dep_snapshot
                #      keys — the key mismatch triggers a correct stale rerun.
                #   c) The same task object is not reused across re-plan
                #      iterations (results are deleted for replan tasks).
                if task_a and task_a.produces:
                    old_produces = list(task_a.produces)
                    task_a.produces = [remap.get(k, k) for k in task_a.produces]
                    log.info("Reflect: Strategy 2 — A[%s] produces remapped %s → %s",
                             tid, old_produces, task_a.produces)

                # Steps 2-4: walk the DAG, judge every downstream consumer.
                # NOTE: requires remap is UNCONDITIONAL — even if the
                # downstream task already has its own override (e.g. it was
                # sparse → re_retrieve, or itself empty → its own replan),
                # its requires must still be remapped to the new keys, or it
                # will keep waiting for the old key A no longer produces.
                # Only creating a NEW cascade override is skipped when the
                # task already has one (its own override takes precedence).
                cascaded = 0
                for t in plan_tasks:
                    if t.id == tid:
                        continue
                    hit = False
                    for i, k in enumerate(t.requires):
                        if k in remap:
                            t.requires[i] = remap[k]
                            hit = True
                    if hit and t.id not in actions:
                        actions[t.id] = SubTaskOverride(
                            mode="replan",
                            query=t.query,  # B keeps its own query
                            feedback=(
                                f"上游任务({tid})目标/输出已调整: {reason}。"
                                f"你的输入依赖已自动更新为新的产出键，"
                                f"请基于新的上游产出重新生成"
                            ),
                            strategy=2,
                        )
                        cascaded += 1
                        log.info("Reflect: Strategy 2 cascade → %s (requires remapped)",
                                 t.id)
                if cascaded:
                    log.info("Reflect: Strategy 2 cascaded %d downstream task(s) for %s",
                             cascaded, tid)

    elif empty_task_ids and not original_query:
        # Fallback: no original query → Strategy 1 with task's own query
        for tid in empty_task_ids:
            if tid not in replanned_set:
                actions[tid] = SubTaskOverride(
                    mode="replan",
                    query="",
                    feedback="原查询方向过窄，请用更宽泛的角度重新检索",
                    strategy=1,
                )
                replanned_set.add(tid)

    return actions


# ── LLM Strategy Classification ─────────────────────────────────────────

async def _classify_replan_strategy(
    empty_task_ids: list[str],
    task_map: dict[str, SubTask],
    results: dict[str, SubResult],
    original_query: str,
    ctx: Any,
) -> dict[str, dict]:
    """LLM-based strategy classification for empty sub-tasks.

    For each empty task, the LLM analyzes why it returned empty and selects
    one of three replan strategies:
      1 (仅改执行策略): Query too narrow, but task goal is correct.
      2 (改任务目标): Task goal itself is wrong (e.g. narrowed to sub-type).
      3 (整体重建): Multiple tasks empty, plan fundamentally broken.

    Returns:
        Dict of task_id → {"strategy": int, "new_query": str, "reason": str}
    """
    if not empty_task_ids:
        return {}

    # Build prompt with empty task details + plan context
    lines = []
    for i, tid in enumerate(empty_task_ids):
        task = task_map.get(tid)
        query = task.query if task else ""
        produces = task.produces if task else []
        requires = task.requires if task else []
        result = results.get(tid)
        degraded = result.degraded_reason if result else ""

        line = (
            f"[{i}] 任务ID: {tid}\n"
            f"    查询: {query}\n"
            f"    产出键(produces): {produces}\n"
            f"    依赖键(requires): {requires}"
        )
        if degraded:
            line += f"\n    降级原因: {degraded}"
        line += "\n    状态: empty (无检索结果)"
        lines.append(line)

    # Brief plan structure for context — include full DAG contract so the
    # LLM can reason about downstream impact and produce accurate
    # produces_remap key mappings (it must know A's old produces keys).
    plan_summary = "; ".join(
        f"{t.id}(q={t.query[:30]}, prod={t.produces}, req={t.requires})"
        for t in task_map.values()
    ) if task_map else ""

    prompt = (
        f"用户原始查询: {original_query}\n\n"
        f"当前计划结构: {plan_summary}\n\n"
        f"以下子任务检索结果为空:\n"
        + "\n".join(lines) + "\n\n"
        "请分析每个空任务的原因，并选择修复策略:\n"
        "策略1（仅改执行策略）: 任务目标正确但查询方向过窄或检索源不对。"
        "→ 提供更宽泛的查询词，输出契约不变，下游任务不用动。\n"
        "策略2（改任务目标）: 任务目标本身有误（如窄化为主题子类型而非维度）。"
        "→ 提供修正后的查询词，下游任务会自动感知变化并重跑。\n"
        "策略3（整体重建）: 多个任务都空，整个方案可能有根本性问题。"
        "→ 标记为需要重新规划。\n\n"
        "输出JSON数组: [{\"i\": 序号, \"strategy\": 1或2或3, \"new_query\": \"新查询词\", \"reason\": \"简短原因\"}]\n"
        "注意: new_query必须保留用户原始查询的核心词，只能追加维度修饰或调整方向。\n"
        "如果选择策略3，new_query可以留空。\n"
        "如果选择策略2且任务A的输出键(produces)需要改变，请额外输出 "
        "produces_remap: {\"旧键\": \"新键\"}（当任务的目标从全集窄化到子集、"
        "或输出结构变化导致旧产出键不再适用时，给出新旧键映射；键名不变则省略该字段）。"
    )

    ctx.llm_call_count += 1
    try:
        raw = await generate(
            [{"role": "system", "content": "你是一个任务修复策略分析器。只输出JSON数组。"},
             {"role": "user", "content": prompt}],
            ai_config=ctx.input.get("ai_config"),
            temperature=0.0, max_tokens=2048, timeout=None,
            no_thinking=True,  # 结构化 JSON 输出，关闭思考（同 planner，防 v4-flash 思考占满 max_tokens 导致 content 空）
        )
        return _parse_strategy_results(raw, empty_task_ids, original_query)
    except Exception as e:
        log.warning("Strategy classification failed: %r (fallback: Strategy 1)", e)
        return {
            tid: {"strategy": 1, "new_query": original_query,
                  "reason": "LLM分类失败，默认策略1", "produces_remap": {}}
            for tid in empty_task_ids
        }


def _parse_strategy_results(
    raw: str,
    task_ids: list[str],
    original_query: str,
) -> dict[str, dict]:
    """Parse LLM strategy classification output.

    Expected format: [{"i": 0, "strategy": 1, "new_query": "...", "reason": "..."}]
    """
    if not raw or not raw.strip():
        return {tid: {"strategy": 1, "new_query": original_query,
                      "reason": "解析失败", "produces_remap": {}}
                for tid in task_ids}

    try:
        text = raw.strip()
        arr_match = re.search(r'\[.*\]', text, re.DOTALL)
        if arr_match:
            text = arr_match.group(0)
        data = json.loads(text)

        if isinstance(data, dict):
            for k in ("results", "data", "items"):
                if k in data and isinstance(data[k], list):
                    data = data[k]
                    break

        results: dict[str, dict] = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                idx = item.get("i", item.get("index", -1))
                if 0 <= idx < len(task_ids):
                    tid = task_ids[idx]
                    strategy = int(item.get("strategy", 1))
                    strategy = max(1, min(3, strategy))  # Clamp to 1-3
                    new_query = item.get("new_query", "").strip() or original_query
                    reason = item.get("reason", "")
                    remap_raw = item.get("produces_remap") or {}
                    if isinstance(remap_raw, dict):
                        remap = {
                            str(k): str(v) for k, v in remap_raw.items()
                            if str(k) and str(v) and str(k) != str(v)
                        }
                    else:
                        remap = {}
                    results[tid] = {
                        "strategy": strategy,
                        "new_query": new_query,
                        "reason": reason,
                        "produces_remap": remap,
                    }
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        log.warning("Parse strategy results failed: %s", e)

    # Fill in any missing tasks with fallback
    for tid in task_ids:
        if tid not in results:
            results[tid] = {
                "strategy": 1,
                "new_query": original_query,
                "reason": "未分类，默认策略1",
                "produces_remap": {},
            }

    return results


# ── Convergence Stop (§9.6) ─────────────────────────────────────────────

def _check_convergence(ctx: Any, has_fixable: bool, round_num: int) -> bool:
    """§9.6: Convergence patience — consecutive rounds with no improvement → stop.

    Tracks problem count history in ctx.metrics.
    """
    cfg = get_config()
    patience = cfg.creative.convergence_patience

    history = ctx.metrics.setdefault("reflect_problem_history", [])
    current_problems = 1 if has_fixable else 0
    history.append(current_problems)

    # Check if last `patience` rounds had no decrease in problems
    if len(history) >= patience + 1:
        recent = history[-(patience + 1):]
        if all(p >= recent[0] for p in recent[1:]) and recent[0] > 0:
            log.info("Convergence stop: %d rounds no improvement", patience)
            return True

    return False


# ── Entity / Number Extraction (rule-based) ─────────────────────────────

def _extract_entities(text: str) -> list[str]:
    """Extract potential entities: capitalized words, quoted strings, numbers with units."""
    entities = []

    # Quoted strings (Chinese quotes and regular)
    entities.extend(re.findall(r'["\u201c\u201d\'\u2018\u2019](.*?)["\u201c\u201d\'\u2018\u2019]', text))

    # Chinese entity patterns — broad suffix list covering multiple domains
    # (retail/food, tech, fitness, education, etc.)
    entities.extend(re.findall(
        r'[\u4e00-\u9fa5]{2,8}(?:'
        '店|餐厅|酒店|商场|品牌|'
        '框架|语言|工具|库|组件|插件|'
        '课程|教程|书籍|频道|博主|'
        '动作|训练|计划|周期|'
        'App|应用|平台|服务'
        ')', text))

    # English capitalized words (covers tech terms: React, Python, TensorFlow, etc.)
    entities.extend(re.findall(r'\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*\b', text))
    # English all-caps acronyms (API, CSS, HTTP, etc.)
    entities.extend(re.findall(r'\b[A-Z]{2,8}\b', text))

    # Numeric values with units (price, time, measurements, versions, etc.)
    entities.extend(re.findall(r'\d+(?:\.\d+)?(?:元|块|万|秒|分|小时|天|周|月|年|kg|km|米|斤|次|组|个|版|v)', text))

    return [e for e in entities if e and len(e) >= 2]


def _extract_numbers(text: str) -> list[float]:
    """Extract numeric values from text."""
    numbers = []
    # Match numbers including decimals
    for match in re.finditer(r'\d+(?:\.\d+)?', text):
        try:
            numbers.append(float(match.group()))
        except ValueError:
            pass
    return numbers


def _split_sentences_simple(text: str) -> list[str]:
    """Split text into sentences for unsupported sentence detection."""
    parts = re.split(r'(?<=[。！？；.!?;])\s*', text)
    return [p.strip() for p in parts if p.strip()]


def _build_source_text(task_id: str, all_results: dict[str, SubResult]) -> str:
    """Build source text from a task's sources for hallucination cross-check."""
    r = all_results.get(task_id)
    if not r or not r.sources:
        return ""
    # In a full implementation, we'd load the actual content from sources.
    # For now, return the section_text itself as a proxy (hallucination
    # detection will be more effective once source content is available).
    return r.section_text or ""


def _parse_indexed_json(raw: str, expected: int, key: str) -> list[bool]:
    """Parse indexed JSON array like [{"i": 0, "compliant": true}, ...]."""
    if not raw:
        return [True] * expected
    try:
        text = raw.strip()
        arr_match = re.search(r'\[.*\]', text, re.DOTALL)
        if arr_match:
            text = arr_match.group(0)
        data = json.loads(text)

        if isinstance(data, dict):
            for k in ("results", "data", "items"):
                if k in data and isinstance(data[k], list):
                    data = data[k]
                    break

        if isinstance(data, list):
            verdicts = [True] * expected
            for item in data:
                if isinstance(item, dict):
                    idx = item.get("i", item.get("index", -1))
                    val = item.get(key, item.get("relevant", True))
                    if 0 <= idx < expected:
                        verdicts[idx] = bool(val)
            return verdicts
    except (json.JSONDecodeError, TypeError):
        pass
    return [True] * expected  # Fail-open
