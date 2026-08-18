"""LLM 层真实回归：用真实智谱 key 跑通迁移后的 LangChain 实现。

覆盖 llm_langchain.py 的全部对外函数（除 embedding）：
  _call_llm / generate / generate_with_config / stream_chat /
  call_with_tools / judge_batch / batch_judge_sentences / summarize_if_long

会消费少量真实 token（max_tokens 已调小）。任一用例失败即退出码 1。

Usage (from OmniBoxAgent/):
    .venv/bin/python scripts/regression_llm_layer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omnibox_agent.core.config import get_config

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def ai_config() -> dict:
    cfg = get_config()
    key = cfg.qu.api_key or cfg.evaluator.api_key
    if not key:
        print("!! 无可用智谱 key（QU_API_KEY/EVALUATOR_API_KEY 为空）")
        sys.exit(2)
    return {"modelName": cfg.qu.model, "baseUrl": cfg.qu.base_url, "apiKey": key}


async def main() -> None:
    import omnibox_agent.services.llm_langchain as l
    ai = ai_config()

    # 1. generate / _call_llm：普通文本生成
    try:
        text = await l.generate(
            [{"role": "user", "content": "用一句话介绍你自己"}],
            ai_config=ai, max_tokens=32,
        )
        record("generate", bool(text.strip()), text.strip()[:40])
    except Exception as e:
        record("generate", False, repr(e))

    # 2. _call_llm：显式传参 + no_thinking（非 deepseek base_url 应无副作用）
    try:
        text = await l._call_llm(
            [{"role": "user", "content": "只回复：ok"}],
            model=ai["modelName"], base_url=ai["baseUrl"], api_key=ai["apiKey"],
            max_tokens=16, no_thinking=True,
        )
        record("_call_llm", text.strip().lower() == "ok", text.strip()[:20])
    except Exception as e:
        record("_call_llm", False, repr(e))

    # 3. generate_with_config
    try:
        text = await l.generate_with_config(
            [{"role": "user", "content": "只回复：配置ok"}],
            ai_config=ai, max_tokens=16,
        )
        record("generate_with_config", "配置ok" in text, text.strip()[:20])
    except Exception as e:
        record("generate_with_config", False, repr(e))

    # 4. stream_chat：逐 token 流式
    try:
        tokens: list[str] = []
        async for t in l.stream_chat(
            [{"role": "user", "content": "用三个词描述春天"}],
            ai_config=ai, max_tokens=32,
        ):
            tokens.append(t)
        joined = "".join(tokens)
        record("stream_chat", bool(joined.strip()), f"{len(tokens)} tokens, {joined.strip()[:30]}")
    except Exception as e:
        record("stream_chat", False, repr(e))

    # 5. call_with_tools：绑定一个简单工具，验证 bind_tools 不 400
    try:
        reply = await l.call_with_tools(
            [{"role": "user", "content": "今天天气如何？请用可用工具。"}],
            tools=[{
                "name": "get_weather",
                "description": "查询天气",
                "parameters": {"type": "object",
                               "properties": {"city": {"type": "string"}}},
            }],
            ai_config=ai,
        )
        # 不管模型是否真发起工具调用，只要返回了 text 或 tool_calls 即视为通过
        ok = bool(reply.text) or bool(reply.tool_calls)
        record("call_with_tools", ok,
               f"text={reply.text.strip()[:20]!r} tools={len(reply.tool_calls)}")
    except Exception as e:
        record("call_with_tools", False, repr(e))

    # 6. judge_batch：三档相关性判定 + JSON 契约解析
    try:
        docs = [
            {"title": "智谱AI大模型介绍", "summary": "智谱推出的GLM系列大模型与API开放平台。"},
            {"title": "周末爬山攻略", "summary": "推荐北京香山、百望山等爬山路线与装备清单。"},
        ]
        labels = await l.judge_batch(docs, "智谱AI大模型的能力", ai_config=ai)
        ok = labels is not None and len(labels) == 2 and "relevant" in labels
        record("judge_batch", ok, f"labels={labels}")
    except Exception as e:
        record("judge_batch", False, repr(e))

    # 7. batch_judge_sentences：句子相关性过滤
    try:
        sentences = [
            "智谱GLM-4支持多轮对话与工具调用。",
            "香山红叶在秋季最为壮观。",
        ]
        kept = await l.batch_judge_sentences(sentences, "智谱大模型功能", ai_config=ai)
        ok = len(kept) >= 1 and "智谱" in kept[0]
        record("batch_judge_sentences", ok, f"kept={kept}")
    except Exception as e:
        record("batch_judge_sentences", False, repr(e))

    # 8. summarize_if_long：长文本压缩
    try:
        long_text = "这是一段用于测试检索索引压缩功能的示例文本。" * 40  # >300 字
        summary = await l.summarize_if_long(long_text, anchor="测试笔记", ai_config=ai)
        ok = len(summary) < len(long_text)
        record("summarize_if_long", ok, f"{len(long_text)}->{len(summary)} chars")
    except Exception as e:
        record("summarize_if_long", False, repr(e))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS)-len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("FAILED:", ", ".join(r[0] for r in failed))
        sys.exit(1)
    print("REGRESSION PASS")