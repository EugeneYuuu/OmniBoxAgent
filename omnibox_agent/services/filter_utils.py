"""v4.1 Unified filter utilities for consistent vector + BM25 filtering.

Design doc §4 (P1-3 fix): Structured filters must be applied consistently
on both the vector side (ChromaDB where) and the BM25 side (MySQL WHERE).
This module provides a single source of truth for filter construction.

Usage:
    filters = {
        "user_id": 123,
        "platform": "xhs",
        "time_start": datetime(...),
        "time_end": datetime(...),
        "favorite_only": True,
    }
    chroma_where = to_chroma_where(filters)
    # -> {"$and": [{"user_id": 123}, {"platform": "xhs"}, ...]}
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def to_chroma_where(filters: dict[str, Any]) -> dict | None:
    """Build ChromaDB metadata where clause from a unified filters dict.

    ChromaDB where supports:
      - equality: {"field": "value"}
      - $and: [{"field": ...}, ...]
      - $gte/$lte: {"field": {"$gte": value}}

    Args:
        filters: Dict with optional keys:
            - user_id: str (user_code, tenant isolation)
            - platform: str
            - time_start: datetime
            - time_end: datetime
            - favorite_only: bool
            - account_ids: list[str] (stored as comma-separated string in metadata)
            - vec_type: str ("main" | "media" | "comments") for single-type filtering
            - vec_types: list[str] (["main", "media"]) for multi-type filtering via $in

    Returns:
        ChromaDB where dict, or None if no filters.
    """
    conditions: list[dict] = []

    user_id = filters.get("user_id")
    if user_id is not None:
        conditions.append({"user_id": user_id})

    # favorite_only 已废弃：库内内容即收藏，向量库只存收藏内容，无需过滤。
    # 存量向量 metadata 中的 is_favorite 字段不再参与查询。

    platform = filters.get("platform")
    if platform:
        conditions.append({"platform": platform})

    # 时间窗过滤：ChromaDB 的 $gte/$lte 仅支持数值，collected_at 字符串过滤
    # 会抛 "Expected operand value to be an int or a float"（时间窗查询全空
    # 的根因）。改用数值时间戳 collected_ts（epoch 秒）比较；存量向量由
    # backfill_collected_ts 脚本回填，缺失该字段的向量不参与时间过滤。
    time_start = filters.get("time_start")
    if time_start:
        if isinstance(time_start, datetime):
            time_start = int(time_start.timestamp())
        conditions.append({"collected_ts": {"$gte": int(time_start)}})

    time_end = filters.get("time_end")
    if time_end:
        if isinstance(time_end, datetime):
            time_end = int(time_end.timestamp())
        conditions.append({"collected_ts": {"$lte": int(time_end)}})

    vec_type = filters.get("vec_type")
    if vec_type:
        conditions.append({"vec_type": vec_type})

    vec_types = filters.get("vec_types")
    if vec_types:
        conditions.append({"vec_type": {"$in": vec_types}})

    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def max_pool_by_note_id(
    vector_hits: list[dict],
    keep_top: int = 0,
) -> list[dict]:
    """Max-pool vector hits by note_id: one note's score = max(main, media).

    v4.1 §3.3: Retrieval granularity is note-level. A note with both vec_main
    and vec_media vectors should appear once, with its highest vector score.

    Args:
        vector_hits: List of result dicts, each with "content_id" and "score".
        keep_top: If > 0, return only top N notes by score.

    Returns:
        Deduplicated list, sorted by score descending.
    """
    best: dict[int, dict] = {}
    for hit in vector_hits:
        cid = hit.get("content_id", 0)
        if cid not in best:
            best[cid] = hit
        else:
            # 保留 score 更高的向量作为主命中，但合并另一条 vec_type 的 document。
            # 否则 media 向量的图片解析（价目表、菜品名等）在 max_pool 时丢失。
            if hit["score"] > best[cid]["score"]:
                # 新的 score 更高，把旧 record 的 document 补到新的上
                old_doc = (best[cid].get("document") or "").strip()
                old_vt = best[cid].get("vec_type", "")
                if old_doc and old_vt == "media":
                    hit.setdefault("media_text", old_doc)
                elif old_doc and old_vt == "main":
                    hit.setdefault("main_text", old_doc)
                best[cid] = hit
            else:
                # 旧的 score 更高，把新 hit 的 document 补到旧 record 上
                new_doc = (hit.get("document") or "").strip()
                new_vt = hit.get("vec_type", "")
                if new_doc and new_vt == "media":
                    best[cid].setdefault("media_text", new_doc)
                elif new_doc and new_vt == "main":
                    best[cid].setdefault("main_text", new_doc)

    result = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    if keep_top > 0:
        result = result[:keep_top]
    return result
