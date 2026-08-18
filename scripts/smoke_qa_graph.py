"""Smoke test: QA graph (LangGraph subgraph, streaming) with mock data.

Runs the full node pipeline: Parse→Guard→Retrieve→Gate→Skill→Reason→Act
(stream_chat 逐 token)→build，external dependencies (LLM, MySQL, ChromaDB)
mocked so the test is self-contained and fast.

Usage (from OmniBoxAgent/):
    .venv/bin/python scripts/smoke_qa_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# 让脚本可直接 import omnibox_agent
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omnibox_agent.agent.context import AgentContext
from omnibox_agent.agent.graph_qa import run_qa_graph
from omnibox_agent.models.query import QueryUnderstandingResult, Intent


# ── Mock service responses ──────────────────────────────────────────────

MOCK_QU_RESULT = QueryUnderstandingResult(
    keywords=["测试", "收藏"],
    intent=Intent.SEARCH_AND_SUMMARIZE,
    resolved_query="测试一下收藏的笔记",
    embedding_query="测试 收藏 笔记",
)

MOCK_ACCOUNT_IDS = ["mock_account_001"]

MOCK_ANSWER = "这是根据你的收藏生成的测试回答。"


def _make_mock_retrieval():
    """构造一个简单的 RetrievalOutput。"""
    from omnibox_agent.agent.context import RetrievalOutput
    out = RetrievalOutput(
        fused_items=[
            {
                "content_id": 1,
                "title": "测试笔记1",
                "summary": "这是一条测试收藏内容",
                "rrf_score": 0.95,
            },
            {
                "content_id": 2,
                "title": "测试笔记2",
                "summary": "另一条测试内容",
                "rrf_score": 0.80,
            },
        ],
        total_count=2,
        platform_dist={"小红书": 2},
        content_map={
            1: {"id": 1, "title": "测试笔记1", "platform_name": "小红书"},
            2: {"id": 2, "title": "测试笔记2", "platform_name": "小红书"},
        },
    )
    return out


# ── Mock patches ────────────────────────────────────────────────────────

def _make_patches():
    """返回一个上下文管理器列表，模拟所有外部依赖。"""

    # 1. 不需要真实 LLM 的 parse_query（parse_query 已改为 async）
    mock_parse = patch(
        "omnibox_agent.services.query_understanding.parse_query",
        new=AsyncMock(return_value=MOCK_QU_RESULT),
    )

    # 2. 不需要真实 MySQL 的 get_account_ids
    mock_accounts = patch(
        "omnibox_agent.services.retrieval_store.get_account_ids",
        return_value=MOCK_ACCOUNT_IDS,
    )

    # 3. 不需要真实 ChromaDB 的 retrieve_pipeline
    mock_retrieve = patch(
        "omnibox_agent.services.ask_orchestrator.retrieve_pipeline",
        return_value=_make_mock_retrieval(),
    )

    # 4. 不需要真实 LLM 的 quality_gate -> 直接返回 GATE_PROCEED
    mock_gate = patch(
        "omnibox_agent.services.quality_gate.quality_gate",
        return_value="proceed",
    )

    # 5. 不需要真实 LLM 的 stream_chat（流式逐 token 输出）
    async def _mock_stream_chat(messages, **kwargs):
        for ch in MOCK_ANSWER:
            yield ch

    mock_act = patch(
        "omnibox_agent.services.llm_service.stream_chat",
        new=_mock_stream_chat,
    )

    # 6. GateStep 在 gate=proceed 后仍会执行 CRAG refine_docs + fit_budget，
    #    所以这两个也要 mock 掉，避免真实调用。
    mock_refine = patch(
        "omnibox_agent.services.quality_gate.refine_docs",
        return_value=_make_mock_retrieval().fused_items,
    )
    mock_fit = patch(
        "omnibox_agent.services.quality_gate.fit_budget",
        return_value=_make_mock_retrieval().fused_items,
    )

    # 7. trace_event / incr_llm 无活跃 recorder 时是 no-op，无需 mock。

    return [mock_parse, mock_accounts, mock_retrieve, mock_gate,
            mock_refine, mock_fit, mock_act]


async def run_smoke() -> None:
    """构建 AgentContext → 运行 QA 图（流式）→ 验证输出。"""

    ctx = AgentContext(
        input={
            "query": "测试一下收藏的笔记",
            "user_id": "mock_user_001",
            "ai_config": {
                "modelName": "glm-4-flash",
                "baseUrl": "https://open.bigmodel.cn/api/paas/v4",
                "apiKey": "mock.api.key",
            },
            "history": [],
        },
        session_id="mock_session_001",
    )

    tokens: list[str] = []

    async def token_cb(tok: str) -> None:
        tokens.append(tok)

    managers = _make_patches()
    for m in managers:
        m.start()

    try:
        result = await run_qa_graph(ctx, token_cb=token_cb)
    finally:
        for m in managers:
            m.stop()

    # ── 验证 ────────────────────────────────────────────────────────
    full_text = "".join(tokens)
    assert full_text == MOCK_ANSWER, \
        f"stream mismatch: {full_text!r} != {MOCK_ANSWER!r}"

    # 验证中间产物
    assert result.artifacts.get("perception") is not None, "no perception artifact"
    assert result.artifacts.get("guard") is not None, "no guard artifact"
    assert result.artifacts.get("retrieval") is not None, "no retrieval artifact"
    assert result.artifacts.get("reasoning") is not None, "no reasoning artifact"

    # 验证 metrics（流式 Act 节点无 *_ms，只保留前置节点）
    assert "ParseStep_ms" in result.metrics, "missing ParseStep_ms"
    assert "GuardStep_ms" in result.metrics, "missing GuardStep_ms"
    assert "RetrieveStep_ms" in result.metrics, "missing RetrieveStep_ms"
    assert "GateStep_ms" in result.metrics, "missing GateStep_ms"
    assert "ReasonStep_ms" in result.metrics, "missing ReasonStep_ms"

    # 验证无错误
    assert not result.errors, f"unexpected errors: {result.errors}"
    assert "error" not in result.artifacts, f"graph aborted: {result.artifacts['error']}"

    print(f"[smoke] QA graph (streaming) ok: {len(full_text)} chars streamed")
    print(f"[smoke] parse/guard/retrieve/gate/reason passed (metrics present)")


async def run_abort_smoke() -> None:
    """验证 critical 中止路径：guard 无账号 → 图中止并映射为 error 产物。"""
    ctx = AgentContext(
        input={
            "query": "测试一下收藏的笔记",
            "user_id": "mock_user_001",
            "ai_config": {"modelName": "glm-4-flash"},
            "history": [],
        },
        session_id="mock_session_abort",
    )

    # 只 mock guard 依赖：get_account_ids 返回空 → GuardStep 抛 PipelineAborted
    mock_accounts = patch(
        "omnibox_agent.services.retrieval_store.get_account_ids",
        return_value=[],
    )
    mock_accounts.start()
    try:
        result = await run_qa_graph(ctx)
    finally:
        mock_accounts.stop()

    error = result.artifacts.get("error")
    assert error is not None, "expected an abort error artifact"
    assert error.get("code") == "guard", f"unexpected abort code: {error}"
    print(f"[smoke] guard abort ok: code={error['code']}")


def main() -> None:
    import asyncio
    asyncio.run(run_smoke())
    asyncio.run(run_abort_smoke())
    print("SMOKE PASS")


if __name__ == "__main__":
    main()