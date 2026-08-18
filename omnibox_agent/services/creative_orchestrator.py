"""Creative orchestrator entry point (migration unit two).

The hand-written 6-state machine that previously lived here has been migrated
to the LangGraph subgraph in `agent/graph_creative.py`. `handle_creative_query`
is now a thin shim that delegates to `run_creative_graph`, keeping the single
source of truth for the creative state machine in the graph (see
docs/omnibox-agent-langchain-langgraph-migration.md §4.4 / §6.2).

State machine (owned by graph_creative):
  PLAN → SOLVE → REFLECT → {SYNTHESIZE | REPLAN → SOLVE} → DONE
  PLAN → DONE (fallback to QA on plan failure)
  REPLAN → SYNTHESIZE (full-replan capped)

Four guarantees (§9.6) preserved in the graph:
  1. Round ceiling: max_rounds ≤ 10 (compute_max_rounds)
  2. Convergence stop: 2 consecutive rounds no improvement
  3. RERUN_CAP: single task max 2 stale reruns
  4. No time-based deadlines — the pipeline is not time-limited (user directive)
"""

from __future__ import annotations

from typing import Any

from omnibox_agent.agent.graph_creative import run_creative_graph


async def handle_creative_query(
    query: str,
    ctx: Any,
    clarify_cb: Any = None,
    clarify_count: int = 0,
    clarify_enabled: bool = True,
    progress_cb: Any = None,
    token_cb: Any = None,
    skill_manager: Any = None,
) -> dict[str, Any] | None:
    """§9.9: Full creative Plan-Solve-Reflect pipeline (delegates to LangGraph).

    This is the entry point called when route_task returns "creative".
    If anything fails, returns None (caller falls back to QA path).

    clarify_cb（可选）：DAG 澄清点回调。在 Plan/Reflect/Synthesize 三点若判定需要澄清，
    会调用 `await clarify_cb(decision, phase, context)`，由调用方 raise ClarifySignal
    暂停流。ClarifySignal 会从 run_creative_graph 向上传播，由 stream 层转成 clarify 事件。

    skill_manager（可选）：显式注入的 SkillManager，透传给 run_creative_graph（§5.4）。

    Returns:
        Dict with answer/confidence/sources/missing/partial, or None on fallback.
    """
    return await run_creative_graph(
        query, ctx,
        clarify_cb=clarify_cb,
        clarify_count=clarify_count,
        clarify_enabled=clarify_enabled,
        progress_cb=progress_cb,
        token_cb=token_cb,
        skill_manager=skill_manager,
    )
