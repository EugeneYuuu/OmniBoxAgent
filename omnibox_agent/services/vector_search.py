"""Vector search service: ChromaDB nearest neighbor with metadata filtering.

v4.1: Supports multi-vector max-pool aggregation (vec_main + vec_media).
v4.1 comment supplement: results carry `vec_type` so callers can split
main/media (primary) from comments (supplementary) after one shared pass.
"""

import logging
from datetime import datetime
from typing import Any

from omnibox_agent.core.config import get_config
from omnibox_agent.services.chroma_store import query_vectors
from omnibox_agent.services.embedding_service import embed_text
from omnibox_agent.services.filter_utils import to_chroma_where, max_pool_by_note_id

log = logging.getLogger(__name__)


def full_candidate_budget(
    user_id: str,
    vec_types: list[str] | None = None,
    fallback: int | None = None,
) -> int:
    """默认不限制 top-k：按用户库向量总数取全量召回预算。

    用户指令：默认取所有命中数据（除非用户显式限制条数）。ChromaDB 的
    n_results 必须传具体数值、无法真正"无限"——这里用 count 动态取该
    用户库的向量总数（min(总数, retrieval_max_candidates) 防御上限兜底），
    使召回覆盖全部命中而非固定 top-k。count 失败时回退 fallback
    （默认 unbounded_candidate_n）。
    """
    cfg = get_config()
    if fallback is None:
        fallback = cfg.retrieval.unbounded_candidate_n
    try:
        from omnibox_agent.services.chroma_store import count_vectors
        from omnibox_agent.services.filter_utils import to_chroma_where
        where = to_chroma_where({"user_id": user_id, "vec_types": vec_types or []})
        total = count_vectors(where=where)
        if total and total > 0:
            return max(min(int(total), cfg.retrieval.retrieval_max_candidates), 1)
    except Exception as e:
        log.warning("full_candidate_budget: count failed (user_id=%s), fallback=%d: %s",
                    user_id, fallback, e)
    return fallback


def vector_search(
    query: str,
    user_id: str,
    n_results: int = 20,
    time_start: datetime | None = None,
    time_end: datetime | None = None,
    platform: str | None = None,
    favorite_only: bool = True,
    filters: dict | None = None,
    max_pool: bool = True,
    vec_types: list[str] | None = None,
) -> list[dict]:
    """Vector-based search with metadata filtering.

    v4.1 §4: Supports multi-vector max-pool aggregation by default.
    When max_pool=True, queries vec_main + vec_media + vec_comments, pools by note_id.
    Use vec_types to restrict which vector types participate (e.g. ["main", "media"]
    to exclude comment vectors in Phase 1 of two-phase retrieval).

    Args:
        query: Clean query text for embedding.
        user_id: user_code (str) for tenant isolation.
        n_results: Number of results per vector type.
        time_start: Optional collected_at lower bound.
        time_end: Optional collected_at upper bound.
        platform: Optional platform filter.
        favorite_only: Only return favorite items.
        filters: Optional unified filters dict (overrides individual params).
        max_pool: If True, query selected vec_types and max-pool by note_id.
        vec_types: Explicit list of vec_type values to query (e.g. ["main", "media"]).
                   When None, all types are queried (main + media + comments).

    Returns:
        List of dicts: {content_id, score, metadata..., vec_type}
        Each hit carries its vec_type ("main" | "media" | "comments") so
        callers can separate primary (main/media) from supplementary
        (comments) matches after the shared pass.
    """
    # Generate query embedding
    query_emb = embed_text(query)
    if not query_emb:
        log.warning("Query embedding failed, vector search unavailable")
        return []

    # Build unified filters dict
    if filters is None:
        filters = {
            "user_id": user_id,
            "time_start": time_start,
            "time_end": time_end,
            "platform": platform,
            "favorite_only": favorite_only,
        }

    all_hits: list[dict] = []

    # Build filter: exclude vec_type from it, handle vec_types separately
    base_filters = {k: v for k, v in filters.items() if k not in ("vec_type", "vec_types")}
    if vec_types:
        # Explicit vec_types list — inject into ChromaDB where as $in filter
        base_filters["vec_types"] = vec_types
    where_filter = to_chroma_where(base_filters)

    if max_pool:
        log.debug("Vector search (max_pool): where=%s, vec_types=%s, n_results=%d",
                  where_filter, vec_types, n_results)

        try:
            results = query_vectors(
                query_embedding=query_emb,
                n_results=n_results * 2,  # Over-fetch since we'll dedup
                where=where_filter,
            )
            all_hits = _parse_results(results)
            all_hits = max_pool_by_note_id(all_hits, keep_top=n_results)
        except Exception as e:
            log.error("Vector search (max_pool) failed: %s", e)
            return []
    else:
        # Single-query mode: main 和 media 不去重，各自独立返回。
        # 同一 content_id 的两条向量各自分配 rank 进入 RRF，分数累加
        # ——这就是"两个向量放一起当一个内容计算分数"。
        # over-fetch 2x：main+media 混排占双倍位置，保证去重后仍有足够 content_id。
        # 但 n_results 不超过 ChromaDB 总量（时间列举查询时 candidate_n 可能已
        # 抬到全库，再 *2 无意义，ChromaDB 也只返回实际拥有的量）。
        over_fetch = min(n_results * 2, n_results + 200)
        log.debug("Vector search: where=%s, vec_types=%s, n_results=%d (over_fetch=%d)",
                  where_filter, vec_types, n_results, over_fetch)

        try:
            results = query_vectors(
                query_embedding=query_emb,
                n_results=over_fetch,
                where=where_filter,
            )
            all_hits = _parse_results(results)
        except Exception as e:
            log.error("Vector search failed: %s", e)
            return []

    return all_hits


def _parse_results(results: dict) -> list[dict]:
    """Parse ChromaDB query results into a list of dicts.

    ChromaDB returns:
    {
        "ids": [["id1", "id2", ...]],
        "distances": [[0.1, 0.2, ...]],
        "metadatas": [[{...}, {...}, ...]],
        "documents": [["text1", "text2", ...]]
    }
    """
    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    ids = results["ids"][0]
    distances = results.get("distances", [[0.0] * len(ids)])[0]
    metadatas = results.get("metadatas", [[{}] * len(ids)])[0]
    documents = results.get("documents", [[""] * len(ids)])[0]

    output = []
    for i, cid in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        # ChromaDB uses cosine distance; convert to similarity score (1 - distance)
        dist = distances[i] if i < len(distances) else 1.0
        score = 1.0 - dist

        output.append({
            "content_id": meta.get("content_id", _extract_content_id(cid)),
            "score": max(score, 0.0),
            "title": meta.get("title", ""),
            "platform": meta.get("platform", ""),
            "platform_name": meta.get("platform_name", ""),
            "author_name": meta.get("author_name", ""),
            "cover": meta.get("cover", ""),
            "original_url": meta.get("original_url", ""),
            "collected_at": meta.get("collected_at", ""),
            "tags": meta.get("tags", ""),
            "is_favorite": meta.get("is_favorite", True),
            "parsed": meta.get("parsed", False),
            "has_image": meta.get("has_image", False),
            "vec_type": meta.get("vec_type", ""),
            "channel": "vector",
            # 向量库的原始文本（main=正文, media=图片解析, comments=评论）
            # _parse_results 之前丢弃了 document，导致 prompt 渲染只能从
            # MySQL summary 取内容，media 向量的价目表/图片解析等丢失。
            # 保留 document 供 _build_system_prompt 渲染时补充。
            "document": documents[i] if i < len(documents) else "",
        })

    return output


def _extract_content_id(chroma_id: str) -> int:
    """Extract content ID from chroma ID (content_123 -> 123)."""
    try:
        return int(chroma_id.replace("content_", ""))
    except (ValueError, AttributeError):
        return 0
