"""LLM-as-a-Judge complexity classifier.

Uses the user's own LLM config (apiKey / baseUrl / modelName) to classify
query complexity as "simple" or "complex" via Function Calling.

Three-layer fallback:
  1. Function Calling (tool_choice) -> parse tool_calls arguments
  2. Model rejects tools (400/422) -> retry without tools, parse JSON from content
  3. Timeout / failure -> default to "simple" (safe QA fallback)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from omnibox_agent.core.config import get_config

log = logging.getLogger(__name__)


@dataclass
class ComplexityResult:
    """Result of complexity classification."""
    type: str       # "simple" | "complex"
    reason: str     # classification reasoning


# ── Function Calling schema ──────────────────────────────────────────────

_CLASSIFY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "classify_complexity",
        "description": "Classify the complexity of a user query",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["simple", "complex"],
                    "description": "simple = single retrieval / fact lookup; complex = multi-step / planning / synthesis",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason for the classification",
                },
            },
            "required": ["type", "reason"],
        },
    },
}


# ── System prompt with Few-Shot ──────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一个查询复杂度分类器。判断用户查询是"simple"还是"complex"。

【simple 简单查询】
- 事实查询：询问特定信息，单次检索即可回答
- 存在性检查："有没有""是否存在"
- 计数："有多少""几个"
- 通用列表："有哪些""收藏了什么"
- 翻译、代码解释、通用闲聊
- 单一主题检索

【complex 复杂查询】
- 对比分析：需要对比多个对象的异同
- 跨文档综合：需要从多个来源整合信息成文
- 多步骤推理：需要分步骤思考、规划
- 任务规划：攻略、指南、报告、方案生成
- 深度分析：趋势分析、风格分析、特点总结

必须调用 classify_complexity 函数输出结果。

【示例】（仅示意查询结构，主题/平台/领域均为泛指占位）
用户: 我收藏了多少条某平台的内容
-> simple | 计数查询，单次检索

用户: 有没有关于某主题的收藏
-> simple | 存在性检查

用户: 帮我找一下关于某技术主题的收藏
-> simple | 单一主题检索

用户: 最近收藏了什么
-> simple | 通用列表

用户: 帮我翻译这段话
-> simple | 翻译任务

用户: 帮我做一份多日游攻略，包括住宿和路线
-> complex | 任务规划，多章节生成

用户: 对比我收藏的不同平台的内容差异
-> complex | 跨平台对比分析

用户: 总结我最近收藏的某领域趋势和热点
-> complex | 跨文档综合

用户: 分析这些某领域作者的内容风格并给建议
-> complex | 深度分析+综合

用户: 帮我整理某主题的学习资料做成知识体系
-> complex | 跨文档综合+规划

用户: 我有多少收藏，将我的收藏分类
-> simple | 计数+列表，单次检索即可回答

用户: 我收藏了多少条某平台的内容，按平台分类统计
-> simple | 计数+分组统计，单次检索即可回答

用户: 将我的收藏按主题归类
-> simple | 通用列表+分组，单次检索即可回答
"""


# ── Public API ───────────────────────────────────────────────────────────

async def classify_complexity(
    query: str,
    history: list[dict] | None = None,
    ai_config: dict | None = None,
) -> ComplexityResult:
    """Classify query complexity using the user's own LLM config.

    Args:
        query: User's raw query string
        history: Optional conversation history (not used in prompt yet, reserved)
        ai_config: User's AI config dict with apiKey/baseUrl/modelName

    Returns:
        ComplexityResult with type ("simple"|"complex") and reason.
        On any failure, returns ComplexityResult(type="simple") as safe default.
    """
    cfg = get_config()
    base_url = (ai_config or {}).get("baseUrl") or cfg.qu.base_url
    model = (ai_config or {}).get("modelName") or cfg.qu.model
    api_key = (ai_config or {}).get("apiKey")

    if not api_key:
        # Per the user directive, a missing user api-key must abort the task
        # rather than silently degrade — the route aborts earlier with the
        # "用户没有提供 API Key" message; this is the in-service guard.
        log.warning("Complexity classifier: no user API key, cannot classify")
        raise RuntimeError("用户没有提供 API Key，无法完成任务")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    # --- Layer 1: Function Calling with tool_choice ---
    # DeepSeek consistently returns 400 for tool_choice with specific function
    # name format, wasting ~1-2s per request. Skip directly to JSON mode.
    _skip_tool_choice = "deepseek.com" in base_url.lower()

    if not _skip_tool_choice:
        try:
            result = await _call_with_function_calling(
                messages, model, base_url, api_key, timeout=None,
            )
            if result is not None:
                log.info("Complexity (tool_choice): type=%s reason=%s", result.type, result.reason)
                return result
        except Exception as e:
            # 400/422（模型拒绝 tool_choice）已在函数内部降级返回 None → 落到 JSON 模式；
            # 其余异常 → 安全回退 simple。
            log.warning("Complexity classifier failed: %r, defaulting to simple", e)
            return ComplexityResult(type="simple", reason="error")
    else:
        log.debug("Skipping tool_choice for DeepSeek (known 400), using JSON mode directly")

    # --- Layer 2: Retry without tools, parse JSON from content ---
    try:
        result = await _call_without_tools(messages, model, base_url, api_key, timeout=None)
        if result is not None:
            log.info("Complexity (json fallback): type=%s reason=%s", result.type, result.reason)
            return result
    except Exception as e:
        log.warning("Complexity JSON fallback failed: %r, defaulting to simple", e)

    # --- Layer 3: Default to simple ---
    return ComplexityResult(type="simple", reason="fallback_default")


# ── Internal: Function Calling ───────────────────────────────────────────

async def _call_with_function_calling(
    messages: list[dict],
    model: str,
    base_url: str,
    api_key: str,
    *,
    timeout: float | None = None,  # None = not time-limited (user directive)
) -> ComplexityResult | None:
    """用 ChatOpenAI.bind_tools 强制函数调用，解析 tool_calls。

    400/422（模型拒绝 tool_choice）→ 返回 None，由调用方落到 JSON 模式；
    其余异常向上抛，由调用方安全回退 simple。
    """
    from omnibox_agent.services.llm_langchain import _build_model, _to_lc_messages
    from omnibox_agent.core.trace_recorder import incr_llm
    incr_llm()

    llm = _build_model(
        {"modelName": model, "baseUrl": base_url, "apiKey": api_key},
        temperature=0.0,
        max_tokens=1024,      # 复杂度判定：max_tokens=1024（用户定）
        no_thinking=True,     # 结构化工具调用：关闭思考（规范）
    )
    bound = llm.bind_tools(
        [_CLASSIFY_TOOL],
        tool_choice={"type": "function", "function": {"name": "classify_complexity"}},
    )

    try:
        resp = await bound.ainvoke(_to_lc_messages(messages))
    except Exception as e:
        status = getattr(e, "status_code", None)
        if status in (400, 422):
            log.warning("Model rejected tool_choice (status=%d), falling back to JSON", status)
            return None
        raise

    for tc in resp.tool_calls or []:
        args = tc.get("args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        ctype = str(args.get("type", "")).strip().lower()
        reason = str(args.get("reason", "")).strip()
        if ctype in ("simple", "complex"):
            return ComplexityResult(type=ctype, reason=reason)
    return None


# ── Internal: JSON fallback (no tools) ───────────────────────────────────

async def _call_without_tools(
    messages: list[dict],
    model: str,
    base_url: str,
    api_key: str,
    *,
    timeout: float | None = None,  # None = not time-limited (user directive)
) -> ComplexityResult | None:
    """Retry without tools, ask model to output JSON in content."""
    from omnibox_agent.services.llm_langchain import _build_model, _to_lc_messages
    from omnibox_agent.core.trace_recorder import incr_llm
    incr_llm()

    # Append instruction for JSON output
    messages_with_instruction = messages + [
        {"role": "system", "content": 'Please output JSON: {"type":"simple"|"complex","reason":"..."}'},
    ]

    llm = _build_model(
        {"modelName": model, "baseUrl": base_url, "apiKey": api_key},
        temperature=0.0,
        max_tokens=1024,      # 复杂度判定：max_tokens=1024（用户定）
        no_thinking=True,     # 结构化 JSON 输出：关闭思考
        response_format={"type": "json_object"},
    )
    resp = await llm.ainvoke(_to_lc_messages(messages_with_instruction))
    content = resp.content or ""
    return _parse_json_content(content)


def _parse_json_content(content: str) -> ComplexityResult | None:
    """Parse JSON from LLM text content."""
    text = content.strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            ctype = data.get("type", "").strip().lower()
            reason = data.get("reason", "").strip()
            if ctype in ("simple", "complex"):
                return ComplexityResult(type=ctype, reason=reason)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    import re
    json_match = re.search(r'\{[^}]*"type"[^}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            ctype = data.get("type", "").strip().lower()
            reason = data.get("reason", "").strip()
            if ctype in ("simple", "complex"):
                return ComplexityResult(type=ctype, reason=reason)
        except (json.JSONDecodeError, TypeError):
            pass

    return None
