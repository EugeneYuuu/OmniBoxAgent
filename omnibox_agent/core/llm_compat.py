"""LLM 兼容工具：DeepSeek thinking 禁用、智谱 JWT 鉴权辅助。

从 llm_service.py 独立出来，供所有 LLM 调用方共享 import，
避免 llm_service 被重构为 shim 后相关 import 断裂。

disable_thinking 是纯函数（只改传入的 body dict），QA 调用禁用 thinking，
DAG 调用保持开启；智谱 JWT 由 auth.resolve_auth_token 负责。
"""

from __future__ import annotations

from typing import Any

from omnibox_agent.core.auth import resolve_auth_token  # noqa: F401  re-export


def disable_thinking(body: dict, model: str, base_url: str) -> dict:
    """Disable thinking mode for DeepSeek QA pipeline calls.

    For deepseek-reasoner: switch to deepseek-chat (reasoner can't disable thinking).
    For deepseek-chat (V3): add thinking={"type":"disabled"} parameter.
    For non-DeepSeek providers: no-op (parameter ignored or absent).

    Applied to ALL QA pipeline LLM calls (Parse, Gate, Reason, Act 流式等) for speed.
    NOT applied to DAG pipeline calls (generate) — those keep thinking enabled.
    """
    if "deepseek.com" not in (base_url or "").lower():
        return body

    # deepseek-reasoner can't disable thinking — switch to deepseek-chat
    if "reasoner" in (model or "").lower():
        body["model"] = "deepseek-chat"

    # Disable thinking for deepseek-chat (V3+ supports this parameter)
    body["thinking"] = {"type": "disabled"}
    return body