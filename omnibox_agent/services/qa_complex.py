"""QA complex query support: shared retrieval across decomposed variants.

When ComplexityRouter judges a query as "complex" (multiple sub-intents),
variants share ONE retrieval pass (parallel recall → union RRF). Generation
is streaming-only (生产路径在 stream_creative_pipeline / graph 内)，本模块
只提供 shared_retrieval 供 creative_solver 复用。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from omnibox_agent.services.filter_utils import max_pool_by_note_id

log = logging.getLogger(__name__)


# ── Shared Retrieval ────────────────────────────────────────────────────

async def shared_retrieval(
    variants: list[str],
    filters: dict[str, Any],
    account_ids: list[str],
    ctx: Any = None,
    retrieval_cfg: Any = None,
) -> list[dict]:
    """§8: All variants share one retrieval pass.

    Each variant retrieves independently, then results are merged via
    union RRF and deduplicated by note_id.

    Cost: 0 LLM calls (retrieval is keyword + vector, no generation).

    Args:
        variants: List of query strings (variants[0] = main query)
        filters: Structured filters (platform, time, etc.) — same for all variants
        account_ids: Authorized account IDs
        ctx: AgentContext (for deadline check)
        retrieval_cfg: Retrieval config

    Returns:
        Merged + deduplicated list of retrieval result dicts.
    """
    from omnibox_agent.services.ask_orchestrator import retrieve_pipeline
    from omnibox_agent.models.query import QueryUnderstandingResult
    from omnibox_agent.agent.context import RetrievalOutput
    from omnibox_agent.agent.loop import run_blocking

    if not variants:
        return []

    # Parallel retrieval for each variant
    async def _retrieve_one(vq: str) -> list[dict]:
        qu = QueryUnderstandingResult(
            resolved_query=vq,
            embedding_query=vq,
            keywords=vq.split(),
        )
        try:
            result = await run_blocking(
                retrieve_pipeline,
                qu, account_ids,
                {"query": vq, "favorite_only": filters.get("favorite_only", True)},
                {}, retrieval_cfg,
            )
            return result.fused_items if result and result.fused_items else []
        except Exception as e:
            log.warning("Shared retrieval variant '%s' failed: %s", vq[:30], e)
            return []

    # Run all variants in parallel
    pools = await asyncio.gather(*[_retrieve_one(v) for v in variants])

    # Union RRF: merge all pools, aggregate scores by note_id
    merged = _union_rrf(pools)

    # Deduplicate by note_id (keep highest score)
    deduped = max_pool_by_note_id(merged)

    log.info("Shared retrieval: %d variants → %d raw hits → %d deduped",
             len(variants), len(merged), len(deduped))
    return deduped


def _union_rrf(pools: list[list[dict]], k: int = 60) -> list[dict]:
    """Union RRF fusion across multiple retrieval pools.

    Each pool is a ranked list. RRF score = sum(1 / (k + rank)) across pools.
    """
    from collections import defaultdict

    scores: dict[int, float] = defaultdict(float)
    best_item: dict[int, dict] = {}

    for pool in pools:
        for rank, item in enumerate(pool):
            cid = item.get("content_id")
            if cid is None:
                continue
            scores[cid] += 1.0 / (k + rank)
            if cid not in best_item or item.get("score", 0) > best_item[cid].get("score", 0):
                best_item[cid] = item

    # Sort by aggregated RRF score
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    result = []
    for cid in sorted_ids:
        item = dict(best_item[cid])
        item["score"] = scores[cid]
        item["rrf_fused"] = True
        result.append(item)

    return result
