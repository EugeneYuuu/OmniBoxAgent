"""Pydantic models for Query Understanding."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    SEARCH_AND_SUMMARIZE = "search_and_summarize"
    EXIST_CHECK = "exist_check"
    COUNT = "count"
    GENERAL_LIST = "general_list"


class QueryUnderstandingResult(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    intent: Intent = Intent.SEARCH_AND_SUMMARIZE
    resolved_query: str = ""
    embedding_query: str = ""  # Clean query for embedding (no function words)
    recency: bool = False
    time_range_start: datetime | None = None
    time_range_end: datetime | None = None
    platform: str | None = None
    tags: list[str] = Field(default_factory=list)
    # True only when the user explicitly asked for a bounded result:
    #   - intent == COUNT ("多少/几个")  OR
    #   - an explicit time window was given ("最近一周"/"上个月")
    # When False, retrieval runs in "unbounded" mode (wide recall for
    # aggregation / summary queries) instead of the hard top_n cap.
    explicit_limit: bool = False

    # 用户显式指定的结果条数（如 "给我10条" / "推荐3个" / "前5篇"）。
    # None 表示用户未指定 → 检索不限制 topK（由门控 + token 预算收口）。
    limit_count: int | None = None

    # ── Classification preference ("将收藏分类" / "不按平台分类") ──
    # Captures the user's grouping intent so the answer prompt can honor it.
    # Without these, the answer prompt defaulted to platform grouping and
    # silently ignored "不要按平台分类" style constraints.
    want_classify: bool = False           # user asked to classify/group the collection
    classify_by: str | None = None       # "platform"|"theme"|"tag"|"type"|"topic"|"content"
    exclude_platform: bool = False        # user explicitly said NOT to classify by platform
