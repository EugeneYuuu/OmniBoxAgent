"""Pydantic models for Ask API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AskMessage(BaseModel):
    role: str
    content: str
    ts: int | None = None  # Unix ms timestamp
    meta: dict | None = None  # 结构化 meta（澄清气泡等），供前置过滤识别 meta.type=="clarify"


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    session_id: str | None = Field(None, alias="sessionId")
    history: list[dict[str, Any]] = Field(default_factory=list)
    user_id: str = Field(..., alias="userId")
    favorite_only: bool = Field(default=True, alias="favoriteOnly")
    scope: str = Field(default="favorite")

    # 追踪契约（docs/ask-trace-technical-design.md §3.1）：request_id 由后端
    # AiController 生成（req_ + 32hex）并经 body 透传；Agent 校验失败或缺省时自生成。
    request_id: str | None = Field(None, alias="requestId")

    # User AI config (passed through from OmniHub_server)
    ai_config: dict | None = Field(None, alias="aiConfig")

    # ── Ask 中间澄清（Clarify）契约（docs/clarify-mid-ask-design.md） ──
    # 用户回答澄清时，后端复用 /ask/stream 发起 resume，携带以下字段：
    clarify_session_id: str | None = Field(None, alias="clarifySessionId")
    # 用户回答类型：option（选了某个选项） / custom（自由输入）
    answer_type: str | None = Field(None, alias="answerType")
    answer_key: str | None = Field(None, alias="answerKey")
    answer_text: str | None = Field(None, alias="answerText")
    # 后端在澄清时缓存的上下文快照（top_items + content_map + plan），resume 时回传
    resume_context: dict | None = Field(None, alias="resumeContext")
    # 当前 ask 累计澄清次数（后端 askClarifyCounts 权威，用于前置过滤）
    clarify_count: int | None = Field(None, alias="clarifyCount")
    # 澄清是否启用（后端经 X-Clarify-Enabled 头透传；body 兜底）
    clarify_enabled: bool | None = Field(None, alias="clarifyEnabled")

    @field_validator("user_id", mode="before")
    @classmethod
    def _coerce_user_id(cls, v):
        # 防御：上游（OmniHub_server）可能以数字形式传 userId，
        # 这里统一转成字符串，避免 Pydantic 报 422 (Input should be a valid string)。
        return str(v) if v is not None else v

    @field_validator("history", mode="before")
    @classmethod
    def _coerce_history(cls, v):
        # 兼容：历史消息既可能是 AskMessage 对象，也可能是 dict；统一转 dict 并保留 meta。
        if v is None:
            return []
        out = []
        for m in v:
            if isinstance(m, AskMessage):
                d = {"role": m.role, "content": m.content}
                if m.ts is not None:
                    d["ts"] = m.ts
                if m.meta is not None:
                    d["meta"] = m.meta
                out.append(d)
            elif isinstance(m, dict):
                out.append(m)
            else:
                out.append(m)
        return out

    class Config:
        populate_by_name = True


class RefItem(BaseModel):
    id: int
    title: str | None = None
    cover: str | None = None
    platform_name: str | None = Field(None, alias="platformName")
    author_name: str | None = Field(None, alias="authorName")
    original_url: str | None = Field(None, alias="originalUrl")

    class Config:
        populate_by_name = True


class EmbedStatusResponse(BaseModel):
    ok: bool
    total_collections: int = 0
    synced_count: int = 0
    missing_count: int = 0
    stale_count: int = 0


class EmbedDeleteRequest(BaseModel):
    content_ids: list[int] = Field(alias="contentIds")

    class Config:
        populate_by_name = True
