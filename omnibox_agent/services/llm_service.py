"""Backward-compat shim — 所有函数已迁移到 llm_langchain.py。

新代码请直接 import llm_langchain；本模块仅保留旧 import 路径不断裂。
disable_thinking 迁到 core.llm_compat，此处 re-export。
"""

from __future__ import annotations

from omnibox_agent.core.llm_compat import disable_thinking, resolve_auth_token
from omnibox_agent.services.llm_langchain import (
    LLMReply,
    _call_llm,
    _parse_relevance,
    _parse_verdicts,
    batch_judge_sentences,
    call_with_tools,
    generate,
    generate_with_config,
    judge_batch,
    stream_chat,
    summarize_if_long,
)

__all__ = [
    "LLMReply",
    "_call_llm",
    "disable_thinking",
    "resolve_auth_token",
    "generate",
    "generate_with_config",
    "stream_chat",
    "call_with_tools",
    "judge_batch",
    "batch_judge_sentences",
    "summarize_if_long",
]