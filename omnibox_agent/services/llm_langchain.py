"""LangChain-backed LLM layer (migration unit one).

Replaces the手写 httpx calls in llm_service.py with LangChain's ChatOpenAI,
while preserving every business constraint:

  - per-user API Key isolation  (new model per call, NO module-level singleton)
  - no read/write timeout        (connect 10s fast-fail only)
  - DeepSeek thinking disable    (QA calls via no_thinking)
  - Zhipu JWT auth               (ZhipuAuth httpx handler, per-request signing)
  - Ask tracing                  (incr_llm() before every real LLM call)

Functions mirror the llm_service.py signatures so callers can swap imports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from omnibox_agent.core.auth import resolve_auth_token
from omnibox_agent.core.config import get_config

log = logging.getLogger(__name__)


@dataclass
class LLMReply:
    """Structured LLM response with optional tool calls."""
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)


# ── Zhipu JWT auth handler ──────────────────────────────────────────────

class ZhipuAuth(httpx.Auth):
    """httpx auth handler: 对智谱请求实时签 JWT。

    ChatOpenAI 底层用 openai SDK，默认把 api_key 作为 `Authorization:
    Bearer <api_key>` 发送。智谱要求的是 `Bearer <jwt>`。通过注入自定义
    `httpx.AsyncClient` + `httpx.Auth`，在每次发请求前用现有
    resolve_auth_token 把 `id.secret` 现场签成 JWT 替换 header。

    非智谱（base_url 不含 bigmodel.cn）走标准 Bearer，零侵入。
    """

    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._is_zhipu = "bigmodel.cn" in (base_url or "") and "." in api_key

    def auth_flow(self, request: httpx.Request) -> httpx.Request:
        if self._is_zhipu:
            jwt = resolve_auth_token(self._api_key, "bigmodel.cn")
            request.headers["Authorization"] = f"Bearer {jwt}"
        yield request


# ── Model builder ───────────────────────────────────────────────────────

def _build_model(
    ai_config: dict | None,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    streaming: bool = False,
    no_thinking: bool = False,
    response_format: dict | None = None,
) -> ChatOpenAI:
    """按 per-user ai_config 构建 ChatOpenAI 实例（每次新实例，禁止单例）。"""
    cfg = get_config()
    base_url = (ai_config or {}).get("baseUrl") or cfg.qu.base_url
    model = (ai_config or {}).get("modelName") or cfg.qu.model
    api_key = (ai_config or {}).get("apiKey")
    if not api_key:
        raise RuntimeError("用户没有提供 API Key，无法完成任务")

    # DeepSeek: reasoner 无法禁用 thinking → 降级为 deepseek-chat
    if no_thinking and "reasoner" in (model or "").lower():
        model = "deepseek-chat"

    # 智谱 JWT：把 `id.secret` 现场签成 JWT 直接作为 api_key 传给 ChatOpenAI。
    # 每次调用都新建实例（resolve_auth_token 有 1h 缓存，JWT 每次都是新鲜的），
    # openai SDK 会把它作为 `Authorization: Bearer <jwt>` 发出。
    # 注：不宜用 ChatOpenAI(http_client=AsyncClient) 注入 httpx.Auth ——
    # langchain-openai 1.4.x 会同时创建同步+异步两个 openai client，
    # 只接受单个 `httpx.Client`，传 AsyncClient 会抛 TypeError。
    if "bigmodel.cn" in (base_url or ""):
        api_key = resolve_auth_token(api_key, base_url)

    extra_body: dict[str, Any] = {}
    if no_thinking and "deepseek.com" in (base_url or "").lower():
        extra_body["thinking"] = {"type": "disabled"}

    kwargs: dict[str, Any] = dict(
        model=model,
        base_url=base_url,
        api_key=api_key,          # 智谱场景已是 JWT；其余走标准 Bearer
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        timeout=None,             # ChatOpenAI 级别也无超时
        max_retries=0,            # 手动控制重试（保持现有语义）
    )
    if extra_body:
        kwargs["extra_body"] = extra_body
    if response_format:
        # langchain-openai 1.4.x 不把 response_format 作为顶层参数，
        # 显式放进 model_kwargs 可避免每次调用打出 "not default parameter" 警告。
        kwargs.setdefault("model_kwargs", {})["response_format"] = response_format

    return ChatOpenAI(**kwargs)


# ── Message conversion ──────────────────────────────────────────────────

def _lc_tool_calls(openai_tool_calls: list[dict]) -> list[dict]:
    """把 OpenAI 格式的 tool_calls 转成 LangChain AIMessage 需要的格式。

    OpenAI:  [{"id","type","function":{"name","arguments"(JSON 字符串)}}]
    LangChain: [{"name","args"(dict),"id","type":"tool_call"}]
    """
    out: list[dict] = []
    for tc in openai_tool_calls:
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        out.append({
            "name": fn.get("name", ""),
            "args": args,
            "id": tc.get("id", ""),
            "type": "tool_call",
        })
    return out


def _to_lc_messages(messages: list[dict]) -> list[Any]:
    """把 OpenAI 风格 messages 转为 LangChain 消息对象。

    逐个映射 role，保留 assistant 的 tool_calls 与 tool 消息的
    tool_call_id，否则多轮工具循环无法把工具结果回传给 LLM。
    """
    converted: list[Any] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            tool_calls = m.get("tool_calls")
            if tool_calls:
                converted.append(AIMessage(
                    content=content, tool_calls=_lc_tool_calls(tool_calls)))
            else:
                converted.append(AIMessage(content=content))
        elif role == "tool":
            converted.append(ToolMessage(
                content=content, tool_call_id=m.get("tool_call_id", "")))
        else:
            converted.append(HumanMessage(content=content))
    return converted


# ── Metrics ─────────────────────────────────────────────────────────────

def _incr_llm() -> None:
    """Ask 追踪：llm 只读计数器 +1（无活跃 recorder 时 no-op）。"""
    from omnibox_agent.core.trace_recorder import incr_llm
    incr_llm()


# ── Exposed functions (mirror llm_service.py signatures) ────────────────

async def generate(
    messages: list[dict],
    *,
    ai_config: dict | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: float | None = None,   # 与旧签名兼容；ChatOpenAI 层已无超时
    no_thinking: bool = False,      # QA 调用传 True；DAG 保持 False
    response_format: dict | None = None,
) -> str:
    """General-purpose LLM generation via ChatOpenAI.ainvoke."""
    model = _build_model(
        ai_config,
        temperature=temperature,
        max_tokens=max_tokens,
        no_thinking=no_thinking,
        response_format=response_format,
    )
    _incr_llm()
    resp = await model.ainvoke(_to_lc_messages(messages))
    return resp.content or ""


async def stream_chat(
    messages: list[dict],
    *,
    ai_config: dict | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    no_thinking: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream LLM tokens via ChatOpenAI.astream. Yields content deltas."""
    model = _build_model(
        ai_config,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        no_thinking=no_thinking,
    )
    _incr_llm()
    async for chunk in model.astream(_to_lc_messages(messages)):
        content = chunk.content
        if content:
            yield content


async def call_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    ai_config: dict | None = None,
) -> Any:
    """Call LLM with optional tool definitions (LangChain). Returns LLMReply.

    400/422 带 tools 时降级为纯生成重试（保持现有语义）。
    """
    try:
        model = _build_model(ai_config, max_tokens=4096)
    except RuntimeError as e:
        return LLMReply(text=str(e))

    lc_messages = _to_lc_messages(messages)

    if tools:
        bound = model.bind_tools(
            [{"type": "function", "function": t} for t in tools]
        )
    else:
        bound = model

    _incr_llm()
    try:
        resp = await bound.ainvoke(lc_messages)
    except Exception as e:
        status = getattr(e, "status_code", None)
        if status in (400, 422) and tools:
            log.warning("LLM rejected tools (status=%s), retrying without", status)
            _incr_llm()
            resp = await model.ainvoke(lc_messages)
        else:
            log.error("LLM call failed: %r", e)
            return LLMReply(text="")

    # 解析 tool_calls
    tool_calls: list[dict] = []
    for tc in resp.tool_calls or []:
        args = tc.get("args", {})
        tool_calls.append({
            "id": tc.get("id", ""),
            "name": tc["name"],
            "arguments": json.dumps(args) if isinstance(args, (dict, list)) else str(args),
        })
    return LLMReply(text=resp.content or "", tool_calls=tool_calls)


# ── Low-level call (mirror llm_service._call_llm) ───────────────────────

async def _call_llm(
    messages: list[dict],
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    ai_config: dict | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    response_format: dict | None = None,
    timeout: float | None = None,   # 与旧签名兼容；ChatOpenAI 层已无超时
    no_thinking: bool = False,      # QA 工具调用禁用；DAG 保持开启
) -> str:
    """Low-level LLM call to OpenAI-compatible /chat/completions endpoint.

    Per user directive, ONLY the embedding model runs under the system key;
    every other LLM task MUST run under the user's own api-key. If no user
    apiKey is present this raises (callers abort instead of using a system key).
    """
    cfg = get_config()
    if ai_config:
        _model = model or ai_config.get("modelName") or cfg.evaluator.model
        _base_url = base_url or ai_config.get("baseUrl") or cfg.evaluator.base_url
        _api_key = api_key or ai_config.get("apiKey")
    else:
        _model = model or cfg.evaluator.model
        _base_url = base_url or cfg.evaluator.base_url
        _api_key = api_key

    if not _api_key:
        raise RuntimeError("用户没有提供 API Key，无法完成任务")

    llm = _build_model(
        {"modelName": _model, "baseUrl": _base_url, "apiKey": _api_key},
        temperature=temperature,
        max_tokens=max_tokens,
        no_thinking=no_thinking,
        response_format=response_format,
    )
    _incr_llm()
    resp = await llm.ainvoke(_to_lc_messages(messages))
    content = resp.content or ""
    if not content.strip():
        reasoning = getattr(resp, "reasoning_content", "") or ""
        log.warning(
            "LLM returned empty content (model=%s, base_url=%s, "
            "reasoning_content_chars=%d) — treat as empty output",
            _model, _base_url, len(reasoning),
        )
    return content


# ── Generation with config (mirror llm_service.generate_with_config) ───

async def generate_with_config(
    messages: list[dict],
    ai_config: dict | None = None,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tools: list[dict] | None = None,
    timeout: float | None = None,   # 与旧签名兼容；ChatOpenAI 层已无超时
) -> str:
    """LLM generation with per-request AI config override."""
    cfg = get_config()
    base_url = (ai_config or {}).get("baseUrl") or cfg.qu.base_url
    model = (ai_config or {}).get("modelName") or cfg.qu.model
    api_key = (ai_config or {}).get("apiKey")
    if not api_key:
        raise RuntimeError("No user API key configured for generation")

    llm = _build_model(ai_config, temperature=temperature, max_tokens=max_tokens)
    _incr_llm()
    lc_messages = _to_lc_messages(messages)
    if tools:
        resp = await llm.bind_tools(
            [{"type": "function", "function": t} for t in tools]
        ).ainvoke(lc_messages)
    else:
        resp = await llm.ainvoke(lc_messages)
    return resp.content or ""


# ── Relevance parsing helpers (mirror llm_service) ──────────────────────

def _norm_label(value: Any) -> str:
    """Normalise a raw relevance value to one of the three tiers."""
    s = str(value).strip().lower()
    if "topic" in s:
        return "topic_relevant"
    if s in ("relevant", "true", "1", "yes", "high"):
        return "relevant"
    return "irrelevant"


def _parse_relevance(raw: str, expected: int) -> list[str] | None:
    """Parse judge_batch output into a list of relevance labels."""
    try:
        text = raw.strip()
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("results", "verdicts", "data", "items"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                arr = []
                for i in range(expected):
                    if str(i) in data:
                        arr.append(_norm_label(data[str(i)]))
                    else:
                        return None
                return arr

        if isinstance(data, list):
            labels = ["irrelevant"] * expected
            for item in data:
                if isinstance(item, dict):
                    idx = item.get("i", item.get("index", -1))
                    rel = item.get("relevance", item.get("relevant", "irrelevant"))
                    if 0 <= idx < expected:
                        labels[idx] = _norm_label(rel)
            return labels
    except (json.JSONDecodeError, TypeError, KeyError):
        log.debug("Parse relevance failed: %s", raw[:200])
    return None


def _parse_verdicts(raw: str, expected: int) -> list[bool] | None:
    """Parse LLM output into list of bools (used by sentence-level refinement)."""
    try:
        text = raw.strip()
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("results", "verdicts", "data", "items"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                arr = []
                for i in range(expected):
                    if str(i) in data:
                        arr.append(bool(data[str(i)]))
                    else:
                        return None
                return arr

        if isinstance(data, list):
            verdicts = [False] * expected
            for item in data:
                if isinstance(item, dict):
                    idx = item.get("i", item.get("index", -1))
                    rel = item.get("relevant", item.get("is_relevant", False))
                    if 0 <= idx < expected:
                        verdicts[idx] = bool(rel)
            if len(data) == expected and all(isinstance(x, bool) for x in data):
                return data
            return verdicts
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        log.debug("Parse verdicts failed: %s — raw: %s", e, raw[:200])
    return None


# ── Batch judge (quality gate) ──────────────────────────────────────────

async def judge_batch(
    docs: list[Any],
    query: str,
    ai_config: dict | None = None,
) -> list[str] | None:
    """Single LLM call to judge relevance of all candidate docs (three-tier).

    Returns list of labels aligned with docs, or None on parse failure.
    """
    if not docs:
        return []

    budget = max(512, len(docs) * 15)

    lines = []
    for i, d in enumerate(docs):
        if isinstance(d, dict):
            title = d.get("title", "") or ""
            content = d.get("summary", "") or d.get("content", "") or ""
        else:
            title = getattr(d, "title", "") or ""
            content = getattr(d, "content", "") or getattr(d, "summary", "") or ""
        content = content[:500]
        lines.append(f"[{i}] 标题: {title}\n内容: {content}")

    prompt = (
        f"用户查询: {query}\n\n"
        f"以下是候选文档列表, 请逐条判断每条文档与用户查询的相关性, 分三档:\n"
        f"  - \"relevant\": 直接回答或高度相关, 命中查询核心意图\n"
        f"  - \"topic_relevant\": 与查询同属一个主题/话题范畴, 虽不直接回答具体问题但属于相关知识"
        f"(聚合/总结/分析类查询中, 属于该主题范畴的收藏都应判为 topic_relevant)\n"
        f"  - \"irrelevant\": 完全不相关, 既不属于主题范畴也不能回答查询\n"
        f"输出 JSON 数组, 每个元素格式: {{\"i\": <序号>, \"relevance\": \"relevant\"|\"topic_relevant\"|\"irrelevant\"}}\n"
        f"数组长度必须等于 {len(docs)}。\n\n"
        + "\n".join(lines)
    )

    messages = [
        {"role": "system", "content": "你是一个相关性判定器。只输出 JSON 数组, 不要其他文字。"},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(2):
        try:
            out = await _call_llm(
                messages, ai_config=ai_config, temperature=0.0, max_tokens=budget * 2,
                response_format={"type": "json_object"},
                timeout=None, no_thinking=True,
            )
            labels = _parse_relevance(out, expected=len(docs))
            if labels is not None:
                return labels
        except Exception as e:
            log.warning("judge_batch attempt %d failed: %r", attempt + 1, e)

    return None  # Both attempts failed


# ── Sentence-level refinement (CRAG) ────────────────────────────────────

async def batch_judge_sentences(
    sentences: list[str],
    query: str,
    ai_config: dict | None = None,
) -> list[str]:
    """Single LLM call to judge which sentences are relevant to the query.

    Returns the list of sentences that passed (in original order).
    On parse failure, returns all sentences (fail-open — same as gate).
    """
    if not sentences:
        return []

    lines = [f"[{i}] {s}" for i, s in enumerate(sentences)]
    prompt = (
        f"用户查询: {query}\n\n"
        f"以下是文档中的句子, 请逐句判断每句是否与查询相关。\n"
        f"输出 JSON 数组, 每个元素: {{\"i\": <序号>, \"relevant\": true/false}}\n"
        f"数组长度必须等于 {len(sentences)}。\n\n"
        + "\n".join(lines)
    )

    messages = [
        {"role": "system", "content": "你是一个句子相关性判定器。只输出 JSON 数组, 不要其他文字。"},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(2):
        try:
            out = await _call_llm(
                messages, ai_config=ai_config, temperature=0.0, max_tokens=2048,
                response_format={"type": "json_object"},
                timeout=None, no_thinking=True,
            )
            verdicts = _parse_verdicts(out, expected=len(sentences))
            if verdicts is not None:
                return [s for s, ok in zip(sentences, verdicts) if ok]
        except Exception as e:
            log.warning("batch_judge_sentences attempt %d failed: %r", attempt + 1, e)

    # Fail-open: return all sentences
    return sentences


# ── Summary (for ingestion) ─────────────────────────────────────────────

async def summarize_if_long(
    text: str, anchor: str, ai_config: dict | None = None
) -> str:
    """Compress text for retrieval indexing if it exceeds SUMMARY_BUDGET.

    On failure, fall back to truncation — never block ingestion.
    """
    cfg = get_config()
    budget = cfg.ingestion.summary_budget

    if len(text) <= budget:
        return text

    try:
        messages = [
            {"role": "system", "content": (
                "你是一个检索索引压缩器。压缩以下内容用于检索索引。"
                "保留: 名称/品牌/数值/位置/时间等关键实体信息, "
                "关键事实与数字, "
                "与笔记主题相关的细节。去掉: 语气词/重复描述/无关闲聊。"
                f"笔记标题(语义锚点): {anchor}")},
            {"role": "user", "content": text},
        ]
        result = await _call_llm(
            messages, temperature=0.1, max_tokens=budget * 2, timeout=None,
            ai_config=ai_config, no_thinking=True,
        )
        return result.strip()
    except Exception as e:
        log.warning("Summary LLM failed, falling back to truncation: %r", e)
        return text[:budget * 2]  # Truncation fallback — never block ingestion