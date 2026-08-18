"""v4.1 §9.3: Creative Solve — wave-based scheduler with cascade failure
and dependency degradation (no time-based deadlines — not time-limited).

Core components:
  - sub_agent_solve: One sub-task execution (retrieve → gate → refine → generate → write produces)
  - run_variant_group: Shared retrieval for all variants → background pool
  - solve_phase: Wave-based scheduler (parallel within wave, dependency between waves)
  - needs_run: Stale detection (dep_snapshot comparison)
  - deps_permanently_missing: Deadlock prevention (upstream done but key missing → degrade)

Key invariants:
  - RERUN_CAP=2: Single task max executions (including stale reruns) to prevent ping-pong
  - Lock-free LLM: All LLM calls happen outside per-note locks
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import Any

from omnibox_agent.core.config import get_config
from omnibox_agent.models.note import SubTask, SubResult, SubTaskOverride
from omnibox_agent.services.llm_service import generate
from omnibox_agent.services.quality_gate import quality_gate, refine_docs, fit_budget
from omnibox_agent.services.filter_utils import max_pool_by_note_id
from omnibox_agent.services.creative_planner import parse_produces_block

log = logging.getLogger(__name__)


# ── Sub-agent Solve (§9.3) ──────────────────────────────────────────────

async def sub_agent_solve(
    task: SubTask,
    shared_state: dict[str, Any],
    ctx: Any,
    override: SubTaskOverride | None = None,
) -> SubResult:
    """Execute one sub-task: retrieve → gate → refine → generate → write produces.

    §9.3: Records dep_snapshot for cascade failure detection.
    Gate context shares the parent request's global deadline clock.

    Args:
        task: The SubTask to execute
        shared_state: Central blackboard for inter-task communication
        ctx: Parent AgentContext (shares deadline clock)
        override: Optional Re-Plan override (rewritten query or feedback)

    Returns:
        SubResult with section_text, confidence, coverage_status, sources, dep_snapshot
    """
    # Replan tasks keep their output contract (produces/requires) unchanged.
    # All three strategies preserve the produces/requires contract —
    # downstream tasks detect changed produces via stale detection and
    # re-run automatically. No special detachment is needed.
    #
    # Record dependency snapshot (for stale detection)
    # Replan tasks still read upstream deps — they may need upstream context
    # to generate a better result with the new query.
    dep_snapshot = {k: shared_state.get(k) for k in task.requires}

    # Determine query (override or original)
    query = override.query if override and override.query else task.query
    feedback = override.feedback if override else ""

    cfg = get_config()

    # 1. Retrieve (main + media vectors)
    retrieval = await _retrieve_for_subtask(task, query, ctx)
    docs = retrieval.fused_items if retrieval else []

    # 1b. Comment vector fallback — when main+media retrieval is empty,
    # search comment vectors to find content whose comments discuss the
    # query topic. This catches cases where the post itself doesn't match
    # but the comments contain relevant info (e.g. a general food post
    # whose comments mention specific Shanghai dishes).
    if not docs:
        comment_docs = await _retrieve_comments_fallback(task, query, ctx)
        if comment_docs:
            log.info("sub_agent_solve[%s]: comment fallback found %d docs",
                     task.id, len(comment_docs))
            docs = comment_docs
            # Build a retrieval output so the gate step can process it
            from omnibox_agent.agent.context import RetrievalOutput
            retrieval = RetrievalOutput(fused_items=comment_docs)

    # Enrich with comments — per-item, per-sub-task.
    # This replaces the old global comment_fallback_threshold approach.
    # The LLM in _build_section_prompt decides whether to use comment data.
    docs = await _supplement_docs_with_comments(docs, ctx)

    # 2. Gate (shares parent deadline clock)
    gate_ctx = ctx.sub(f"gate_{task.id}")
    gate_ctx.artifacts["retrieval"] = retrieval
    gate_ctx.input["query"] = query

    try:
        await quality_gate(gate_ctx)
    except Exception as e:
        log.warning("sub_agent_solve[%s]: gate failed: %s", task.id, e)
        gate_ctx.flags["gate_degraded"] = True

    gated = gate_ctx.artifacts.get("retrieval", retrieval)
    relevant = gated.fused_items if gated else []

    # 3. Refine (>300 char docs)
    try:
        relevant = await refine_docs(relevant, query, gate_ctx)
        # DAG 用独立 creative_context_token_budget（15000），配合 top_k_complete=15
        # 保证前 15 条完整注入；不碰 simple QA 的 context_token_budget(3000)。
        budget = cfg.generation.creative_context_token_budget
        relevant = fit_budget(relevant, budget, top_k_complete=15)
    except Exception as e:
        log.warning("sub_agent_solve[%s]: refine failed: %s", task.id, e)

    # 4. Coverage assessment (rule-based, zero LLM)
    coverage = _assess_coverage(relevant)

    # 5. Generate section text (if not empty)
    if coverage == "empty" or not relevant:
        return SubResult(
            sub_task_id=task.id,
            section_text="",
            confidence="empty",
            coverage_status="empty",
            sources=[],
            dep_snapshot=dep_snapshot,
        )

    # Build generation prompt
    # Replan tasks keep their [PRODUCES] block — output contract is unchanged
    # (Strategy 1: only the query changes, not the output structure).
    section_prompt = _build_section_prompt(task, relevant, dep_snapshot, feedback)
    ctx.llm_call_count += 1
    try:
        # Generate under the user's own API key (NOT evaluator/Zhipu);
        # only embedding should use Zhipu. Falls back to evaluator if no key.
        raw_output = await generate(
            section_prompt,
            ai_config=ctx.input.get("ai_config"),
            temperature=0.7,
            max_tokens=20480,  # 规范：solve 保留 thinking + max_tokens=20480（创作型章节）
            timeout=None,  # user directive: sub-task answers are NOT time-limited
        )
    except Exception as e:
        log.warning("sub_agent_solve[%s]: generation failed: %r", task.id, e)
        return SubResult(
            sub_task_id=task.id,
            section_text="",
            confidence="low",
            coverage_status=coverage,
            sources=[str(d.get("content_id", "")) for d in relevant],
            dep_snapshot=dep_snapshot,
        )

    # 6. Parse produces block and write to shared_state
    #    All tasks (including replan) keep their produces contract.
    #    Strategy 1: output contract unchanged — replan task produces same keys.
    #    When the task re-runs with a new query, it writes new produce values.
    #    Downstream tasks detect the stale dep and re-run automatically.
    section_text, produces_data = parse_produces_block(raw_output, task.produces)
    for key in task.produces:
        if key in produces_data:
            shared_state[key] = produces_data[key]
            log.debug("sub_agent_solve[%s]: wrote produces %s=%s", task.id, key, produces_data[key])

    # 7. Compute confidence
    gate_degraded = gate_ctx.flags.get("gate_degraded", False)
    confidence = "low" if gate_degraded else "normal"

    return SubResult(
        sub_task_id=task.id,
        section_text=section_text,
        confidence=confidence,
        coverage_status=coverage,
        sources=[str(d.get("content_id", "")) for d in relevant],
        dep_snapshot=dep_snapshot,
    )


# ── Variant Group (§9.3) ────────────────────────────────────────────────

async def run_variant_group(
    variants: list[SubTask],
    main_query: str,
    ctx: Any,
) -> list[dict]:
    """§9.3: Execute all variant sub-tasks with shared retrieval.

    All variants + main query retrieve in parallel, results merged via
    union RRF. Only main query goes through quality gate.
    Gated results → background pool for Synthesize.

    Returns:
        Deduplicated list of retrieval result dicts (background pool).
    """
    if not variants:
        return []

    # Collect all queries: main + variants
    queries = [main_query] + [v.query for v in variants]

    # Parallel retrieval
    from omnibox_agent.services.qa_complex import shared_retrieval
    merged = await shared_retrieval(
        queries, {}, ctx.input.get("account_ids", []), ctx, None
    )

    # Gate on main query only
    from omnibox_agent.agent.context import RetrievalOutput
    gate_ctx = ctx.sub("variant_gate")
    gate_ctx.artifacts["retrieval"] = RetrievalOutput(fused_items=merged)
    gate_ctx.input["query"] = main_query

    try:
        await quality_gate(gate_ctx)
    except Exception as e:
        log.warning("variant_group: gate failed: %s", e)
        gate_ctx.flags["gate_degraded"] = True

    gated = gate_ctx.artifacts.get("retrieval", RetrievalOutput(fused_items=merged))

    # Deduplicate by note_id
    return max_pool_by_note_id(gated.fused_items) if gated else []


# ── Solve Phase (§9.3: wave-based scheduler) ────────────────────────────

async def solve_phase(
    plan_tasks: list[SubTask],
    shared_state: dict[str, Any],
    results: dict[str, SubResult],
    ctx: Any,
    replan_overrides: dict[str, SubTaskOverride] | None = None,
) -> dict[str, SubResult]:
    """Wave-based scheduler: parallel within wave, dependency between waves.

    §9.3:
      1. Deadline check each wave (remaining < SYNTH_RESERVE → partial break)
      2. Variant group runs first (shared retrieval → background pool)
      3. Wave loop: ready tasks (deps satisfied) → parallel solve
      4. Stale detection: upstream produces changed → downstream rerun
      5. Dependency missing: upstream done but key absent → degrade
      6. RERUN_CAP: stale rerun cap to prevent ping-pong

    Args:
        plan_tasks: All SubTasks from the planner
        shared_state: Central blackboard
        results: Existing results (from previous rounds — supports Re-Plan)
        ctx: Parent AgentContext
        replan_overrides: Re-Plan overrides keyed by task_id

    Returns:
        Updated results dict.
    """
    cfg = get_config()
    rerun_cap = cfg.creative.rerun_cap

    variants = [t for t in plan_tasks if t.type == "retrieval_variant"]
    sections = [t for t in plan_tasks if t.type == "section"]

    # Variant group: shared retrieval → background pool
    if variants and not ctx.flags.get("variant_pool_done"):
        ctx.variant_pool = await run_variant_group(variants, ctx.input.get("query", ""), ctx)
        ctx.flags["variant_pool_done"] = True
        log.info("solve_phase: variant pool ready (%d docs)", len(ctx.variant_pool))

    # Stale rerun counter (only counts stale-triggered reruns)
    stale_reruns: Counter = Counter()

    overrides = replan_overrides or {}

    while True:
        # No deadline check — pipelines are not time-limited.

        # ① Find tasks that need to run
        todo = [
            t for t in sections
            if _needs_run(t, results, shared_state, stale_reruns, rerun_cap)
        ]
        if not todo:
            break

        # ③ Check readiness (dependencies satisfied)
        # Replan tasks still need their deps — they're not detached.
        # If deps are missing, the task goes through normal degradation
        # (graceful: run without deps) or waits for upstream to re-run.
        ready = [
            t for t in todo
            if all(k in shared_state for k in t.requires)
        ]

        # ④ Dependency permanently missing → degrade
        # Don't degrade replan tasks — Reflect chose to re-run them.
        # They'll either run when deps become available, or fall through
        # to the graceful degradation path (run without deps).
        for t in todo:
            if t in ready:
                continue
            if overrides.get(t.id) and overrides[t.id].mode == "replan":
                continue  # Reflect chose to replan — don't re-degrade
            if _deps_permanently_missing(t, plan_tasks, results, shared_state):
                results[t.id] = SubResult.degraded(t, reason="missing_dependency")
                log.info("solve_phase: task %s degraded (missing_dependency)", t.id)

        # Filter out already-degraded
        ready = [t for t in ready if t.id not in results]

        if not ready:
            if todo:
                # Graceful degradation instead of a hard deadlock break:
                # the stuck tasks have unmet dependencies (e.g. an upstream
                # [PRODUCES] key the model never emitted). Run them anyway
                # with empty dependency context so the creative run still
                # yields usable sections rather than falling back empty.
                log.warning(
                    "solve_phase: %d tasks stuck (deps missing), running without deps",
                    len([t for t in todo if t.id not in results]),
                )
                ready = [t for t in todo if t.id not in results]
                # 死循环防护：若仍无任务可跑（所有"需要运行"的任务都已在
                # results 中，即 stale 重跑但依赖永不满足），必须 break，
                # 否则 wave 永远空转（曾实测 100% CPU + 刷屏 wave with 0 tasks）。
                if not ready:
                    log.warning("solve_phase: no runnable tasks (all stuck/stale in results), breaking")
                    break
            else:
                break

        # ⑤ Parallel solve for ready tasks（wave 内并发限流 ≤3）
        log.info("solve_phase: wave with %d tasks: %s",
                 len(ready), [t.id for t in ready])

        # wave 内并发上限 3：防止多任务同时打 LLM 触发 429 限流，
        # 以及 2G 内存下多任务并行检索/构造 prompt 的资源峰值
        _wave_sem = asyncio.Semaphore(3)

        async def _run_limited(t: SubTask) -> SubResult:
            async with _wave_sem:
                override = overrides.get(t.id)
                return await sub_agent_solve(t, shared_state, ctx, override)

        batch = [_run_limited(t) for t in ready]
        solved = await asyncio.gather(*batch, return_exceptions=True)

        for t, result in zip(ready, solved):
            if isinstance(result, Exception):
                log.error("sub_agent_solve[%s] raised: %s", t.id, result)
                results[t.id] = SubResult(
                    sub_task_id=t.id, section_text="",
                    confidence="low", coverage_status="empty",
                    degraded_reason="execution_error",
                )
            else:
                was_stale = t.id in results
                results[t.id] = result
                if was_stale:
                    stale_reruns[t.id] += 1
                    log.info("solve_phase: task %s stale rerun #%d", t.id, stale_reruns[t.id])

    return results


def _needs_run(
    t: SubTask,
    results: dict[str, SubResult],
    shared_state: dict[str, Any],
    stale_reruns: Counter,
    cap: int,
) -> bool:
    """§9.3: Determine if a task needs to (re)run.

    Returns True if:
      - Never executed (not in results)
      - OR upstream produces changed (stale) and rerun cap not hit
    """
    if t.id not in results:
        return True

    if stale_reruns.get(t.id, 0) >= cap:
        return False  # Cap hit — keep existing result, let Reflect handle

    # Check if dependencies changed (stale)
    existing = results[t.id]
    for key in t.requires:
        if shared_state.get(key) != existing.dep_snapshot.get(key):
            return True  # Upstream changed → stale

    return False


def _deps_permanently_missing(
    t: SubTask,
    plan_tasks: list[SubTask],
    results: dict[str, SubResult],
    shared_state: dict[str, Any],
) -> bool:
    """§9.3: Check if a task's dependencies are permanently missing.

    A dependency is permanently missing if:
      - The producing task has finished (in results)
      - But the produced key is not in shared_state
    This means the upstream task failed to produce → downstream degrades.
    """
    for dep_key in t.requires:
        if dep_key in shared_state:
            continue  # Dependency available

        # Find which task produces this key
        producer = next((pt for pt in plan_tasks if dep_key in pt.produces), None)
        if producer is None:
            return True  # No producer → permanently missing

        # Is the producer done?
        if producer.id in results:
            prod_result = results[producer.id]
            if prod_result.coverage_status == "empty" or prod_result.degraded_reason:
                return True  # Producer finished but failed → permanently missing
            # Producer finished but key not written → extraction failed
            return True

    return False


# ── Helpers ─────────────────────────────────────────────────────────────

async def _retrieve_for_subtask(task: SubTask, query: str, ctx: Any):
    """Retrieve documents for a sub-task using the retrieval pipeline."""
    from omnibox_agent.services.ask_orchestrator import retrieve_pipeline
    from omnibox_agent.models.query import QueryUnderstandingResult
    from omnibox_agent.agent.context import RetrievalOutput
    from omnibox_agent.agent.loop import run_blocking

    user_id = ctx.input.get("user_id", "")
    account_ids = ctx.input.get("account_ids", [])
    if not account_ids:
        guard = ctx.artifacts.get("guard", {})
        account_ids = guard.get("account_ids", [])
    # Mirror the QA path: fall back to the user's authorized accounts so the
    # FULLTEXT/LIKE channels scope correctly. (The creative path runs before
    # the QA guard step, so it must resolve accounts itself.)
    if not account_ids and user_id:
        try:
            from omnibox_agent.services.retrieval_store import get_account_ids as _get_account_ids
            account_ids = _get_account_ids(user_id)
        except Exception as e:
            log.debug("retrieve_for_subtask: account resolution failed: %s", e)

    qu = QueryUnderstandingResult(
        resolved_query=query,
        embedding_query=query,
        keywords=query.split(),
    )

    try:
        output = await run_blocking(
            retrieve_pipeline, qu, account_ids,
            {"query": query, "favorite_only": True, "user_id": user_id,
             **({"platform": task.filters.get("platform")} if task.filters.get("platform") else {})},
            {}, None,
        )
        # Ask 追踪：子任务检索（§4.4 task.retrieve）
        from omnibox_agent.core.trace_recorder import trace_event
        trace_event("task.retrieve", phase="creative", task_id=task.id, data={
            "task_type": task.type,
            "query": query[:200],
            "hit_count": len(output.fused_items) if output else 0,
            "coverage": "full" if (output and output.fused_items) else "empty",
        })
        return output
    except Exception as e:
        log.warning("retrieve_for_subtask[%s]: %s", task.id, e)
        return RetrievalOutput()


async def _retrieve_comments_fallback(
    task: SubTask,
    query: str,
    ctx: Any,
) -> list[dict]:
    """Comment vector fallback: when main+media retrieval is empty, search
    comment vectors to find content whose comments discuss the query topic.

    This catches cases where the post itself doesn't vector-match the query
    but the comments contain relevant information (e.g. a general food post
    whose comments mention specific Shanghai dishes, locations, prices).

    Returns:
        List of enriched doc dicts (same shape as retrieve_pipeline output),
        or empty list if no comment vectors match.
    """
    from omnibox_agent.services.vector_search import vector_search
    from omnibox_agent.services.filter_utils import max_pool_by_note_id
    from omnibox_agent.services.retrieval_store import get_content_by_ids
    from omnibox_agent.agent.loop import run_blocking

    user_id = ctx.input.get("user_id", "")
    account_ids = ctx.input.get("account_ids", [])
    if not account_ids:
        guard = ctx.artifacts.get("guard", {})
        account_ids = guard.get("account_ids", [])
    if not account_ids and user_id:
        try:
            from omnibox_agent.services.retrieval_store import get_account_ids as _get_account_ids
            account_ids = _get_account_ids(user_id)
        except Exception:
            pass

    platform = task.filters.get("platform") if task.filters else None

    try:
        results = await run_blocking(
            vector_search,
            query=query,
            user_id=user_id,
            n_results=20,
            platform=platform,
            favorite_only=True,
            vec_types=["comments"],
        )
    except Exception as e:
        log.warning("retrieve_comments_fallback[%s]: vector_search failed: %s", task.id, e)
        return []

    if not results:
        return []

    # Deduplicate by note_id (content_id)
    results = max_pool_by_note_id(results, keep_top=20)

    # These notes matched ONLY via their comment vectors — mark them so the
    # section prompt treats them as supplementary (正文未直接匹配).
    for item in results:
        item["is_comment_match"] = True

    # Load full content details from DB (title, summary, content)
    top_ids = [d["content_id"] for d in results if d.get("content_id")]
    if not top_ids:
        return []

    try:
        content_details = await run_blocking(get_content_by_ids, top_ids, account_ids)
    except Exception as e:
        log.warning("retrieve_comments_fallback[%s]: get_content_by_ids failed: %s", task.id, e)
        # Return results with whatever metadata we have from ChromaDB
        return results

    content_map = {c["id"]: c for c in content_details}

    # Enrich results with summary/content text
    for item in results:
        detail = content_map.get(item.get("content_id"))
        if detail:
            summary = detail.get("summary", "") or ""
            item.setdefault("summary", summary)
            item.setdefault("content", summary)

    return results


def _assess_coverage(docs: list[dict]) -> str:
    """Rule-based coverage assessment (zero LLM).

    §9.4 ①: sufficient / sparse / empty based on hit count.
    Comment-only docs (is_comment_match — matched via comment vectors, no
    main/media evidence) can never reach "sufficient": comments are
    supplementary by design, so a section resting entirely on comment
    matches is at best "sparse".
    """
    if not docs:
        return "empty"
    if len(docs) >= 3:
        if any(not d.get("is_comment_match") for d in docs):
            return "sufficient"
        return "sparse"
    if len(docs) >= 1:
        return "sparse"
    return "empty"


async def _supplement_docs_with_comments(
    docs: list[dict],
    ctx: Any,
) -> list[dict]:
    """Per-item comment enrichment for DAG sub-tasks.

    For each doc in the sub-task retrieval result, fetch its comment text
    from the database and append as supplementary context. The sub-agent
    LLM in _build_section_prompt decides whether to use it.

    This is the correct two-phase design: comment retrieval is per-item,
    per-sub-task, driven by the DAG task decomposition — NOT a global
    count threshold (the old comment_fallback_threshold approach was removed).

    Example: a "food_locations" sub-task retrieves 3 items; one lacks
    location info in its summary. The LLM can find the location in the
    attached comment text and use it.
    """
    if not docs:
        return docs

    # Collect content_ids
    cids: list[int] = []
    for d in docs:
        cid = d.get("content_id")
        if cid:
            try:
                cids.append(int(cid))
            except (ValueError, TypeError):
                pass

    if not cids:
        return docs

    # Fetch comments from DB
    from omnibox_agent.services.retrieval_store import get_comments_for_content_ids
    from omnibox_agent.agent.loop import run_blocking

    try:
        comments_map = await run_blocking(get_comments_for_content_ids, cids)
    except Exception as e:
        log.warning("_supplement_docs_with_comments: DB query failed: %s", e)
        return docs

    if not comments_map:
        return docs

    # Append comment context to each doc
    enriched_count = 0
    for d in docs:
        cid = d.get("content_id")
        try:
            cid_int = int(cid) if cid else 0
        except (ValueError, TypeError):
            continue

        comment_texts = comments_map.get(cid_int, [])
        if comment_texts:
            # Join comments, cap at ~600 chars to avoid blowing up context
            joined = "\n".join(comment_texts)
            if len(joined) > 600:
                joined = joined[:600] + "..."
            d["comments_context"] = joined
            enriched_count += 1

    if enriched_count:
        log.info("_supplement_docs_with_comments: enriched %d/%d docs with comments",
                 enriched_count, len(docs))

    return docs


def _build_section_prompt(
    task: SubTask,
    docs: list[dict],
    dep_snapshot: dict[str, Any],
    feedback: str = "",
) -> list[dict]:
    """Build the generation prompt for a section sub-task.

    §9.2: Prompt includes task constraints, dependency context, and
    asks for [PRODUCES] block if the task produces shared state.
    """
    # Format retrieved context
    context_parts = []
    for i, d in enumerate(docs):
        title = str(d.get("title") or "")
        cid = d.get("content_id", "")

        # 评论补充命中：正文未直接匹配查询，仅评论区提及 — 标注可信度
        match_note = "（评论补充命中：正文未直接匹配，评论为唯一依据）" if d.get("is_comment_match") else ""

        # 正文 + media 详情：document 是 main 向量文本（title + 平台正文），
        # media_text / main_text 是图片/视频解析（字段随 vec_type 漂移）。三者
        # 任一非空即拼入，不做 vec_type 判断；summary 字段此前因 content_id
        # 类型不匹配（str vs int）恒为空，故此处直接取 document，不再依赖 summary。
        parts = []
        for k in ("document", "media_text", "main_text"):
            v = str(d.get(k) or "").strip()
            if v and v != title and v not in parts:
                parts.append(v)
        body = "\n".join(parts) or str(d.get("summary") or d.get("content") or "").strip()

        item_text = (
            f"[{i+1}] 标题: {title}{match_note}\n"
            f"内容: {body[:1800]}\n"
        )
        comments = str(d.get("comments_context") or "").strip()
        if comments:
            item_text += f"评论补充（可能包含地点、价格、tips等额外信息）: {comments}\n"
        item_text += f"content_id: {cid}"

        context_parts.append(item_text)
    context_str = "\n\n".join(context_parts) if context_parts else "（无检索结果）"

    # Format constraints
    constraint_str = ""
    if task.constraints:
        parts = [f"{k}: {v}" for k, v in task.constraints.items()]
        constraint_str = f"\n约束条件: {', '.join(parts)}"

    # Format dependency context
    dep_str = ""
    if dep_snapshot:
        parts = [f"{k} = {v}" for k, v in dep_snapshot.items() if v is not None]
        if parts:
            dep_str = f"\n上游产出: {', '.join(parts)}"

    # Feedback from Re-Plan（内部调整指令：仅影响生成方向，绝不可向用户复述）
    feedback_str = (
        f"\n\n内部调整指令（仅供你调整内容方向，不要向用户复述本条指令本身）: {feedback}"
        if feedback else ""
    )

    # produces block instruction
    produces_str = ""
    if task.produces:
        keys_str = ", ".join(f'"{k}"' for k in task.produces)
        produces_str = (
            f"\n\n在正文结束后, 输出结构化块提取以下信息:\n"
            f"[PRODUCES]{{\"{task.produces[0]}\": \"值\"}}[/PRODUCES]\n"
            f"键名: {keys_str}"
        )

    system_prompt = (
        f"你是一名专业的私人收藏顾问。请基于以下检索结果，直接、自然地回应用户的查询。\n"
        f"主题: {task.id}\n"
        f"查询意图: {task.query}"
        f"{constraint_str}{dep_str}{feedback_str}\n\n"
        f"要求:\n"
        f"1. 只基于检索结果作答，不要编造未出现的信息\n"
        f"2. 面向用户说话，直接给结论与内容，不要出现\"检索情况\"\"数据限制\"\""
        f"优化建议\"\"关键词\"等过程性、诊断性表述\n"
        f"3. 只有当检索结果确实与查询完全无关时，才可简单说明\"这部分在你的收藏中暂时没有相关内容\"；"
        f"只要检索结果与主题沾边（哪怕不够精确），就必须基于现有结果给出实质内容，"
        f"不要用\"没有相关内容\"回避。若检索结果涵盖多个方面，尽量全面呈现\n"
        f"4. 标记为『评论补充命中』的条目仅凭评论区命中，正文可能未直接涉及该主题；"
        f"引用时须以正文/摘要为准，评论信息只能作为辅助线索，不得当作条目正文内容描述\n"
        f"5. 内容连贯，言之有物{produces_str}\n"
        f"6. **超链接规则（重要）**：当你在回答中提到具体的收藏条目时，必须使用 Markdown 超链接格式：\n"
        f"   `[条目标题](content://条目的content_id)`\n"
        f"   例如：`[收藏条目标题](content://123)` — 用户点击即可跳转到该条目详情页\n"
        f"   注意：content_id 已在上方每个条目中标注，请使用真实的 content_id 值\n"
        f"   对于非收藏条目的普通链接（如外部网址），仍使用 `[文本](url)` 格式\n"
        f"7. **Markdown 格式（重要）**：章节内容使用 Markdown 结构化输出——可用标题、"
        f"加粗、列表（`-` 或 `1.`）组织内容，让用户易于扫读。列表项中提到收藏条目时"
        f"同样使用 `[标题](content://id)` 超链接格式。"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"检索结果:\n{context_str}\n\n请撰写章节内容。"},
    ]
