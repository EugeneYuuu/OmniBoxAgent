"""Smoke test: Creative graph (LangGraph subgraph) with mock data.

Runs the 5-node conditional-edge graph:
  PLAN -> SOLVE -> REFLECT -> {SYNTHESIZE | REPLAN -> SOLVE} -> DONE
with all external deps (LLM plan/solve/reflect/synthesize + clarify judge)
mocked so the test is self-contained and fast.

Covers:
  1. Happy path: PLAN valid -> SOLVE -> REFLECT(all_pass) -> SYNTHESIZE -> DONE
  2. Plan fallback: route_task returns "qa" -> graph returns None
  3. Replan path: REFLECT(has_fixable) -> REPLAN -> SOLVE -> REFLECT(all_pass) -> SYNTHESIZE
  4. Short circuit: all results empty -> build_short_circuit_response

Usage (from OmniBoxAgent/):
    .venv/bin/python scripts/smoke_creative_graph.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

# 让脚本可直接 import omnibox_agent
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omnibox_agent.agent.context import AgentContext
from omnibox_agent.agent.graph_creative import run_creative_graph
from omnibox_agent.models.note import (
    PlanOutput,
    ReflectResult,
    SubResult,
    SubTask,
)

NS = "omnibox_agent.agent.graph_creative"


# ── Mock objects ────────────────────────────────────────────────────────

def _make_plan(task_count: int = 2) -> PlanOutput:
    return PlanOutput(
        tasks=[SubTask(id=f"t{i + 1}", type="section", query=f"子任务{i + 1}")
               for i in range(task_count)],
        valid=True,
    )


def _make_results(plan: PlanOutput) -> dict[str, SubResult]:
    return {
        t.id: SubResult(sub_task_id=t.id, section_text=f"内容-{t.id}",
                        confidence="normal", coverage_status="sufficient")
        for t in plan.tasks
    }


def _make_ctx() -> AgentContext:
    return AgentContext(
        input={
            "query": "写一篇关于巴厘岛旅游的攻略",
            "user_id": "mock_user_001",
            "ai_config": {
                "modelName": "glm-4-flash",
                "baseUrl": "https://open.bigmodel.cn/api/paas/v4",
                "apiKey": "mock.api.key",
            },
            "history": [],
        },
        session_id="mock_creative_001",
    )


def _make_patches(**overrides):
    """构造默认的 mock patch 集合，overrides 可覆盖单项目标。"""
    plan = overrides.get("plan", _make_plan())
    route = overrides.get("route", "creative")
    max_rounds = overrides.get("max_rounds", 2)
    results = overrides.get("results")
    reflect_result = overrides.get("reflect_result", ReflectResult(all_pass=True))
    short_circuit = overrides.get("short_circuit", False)
    answer = overrides.get("answer", "这是生成的巴厘岛攻略回答。")

    patches = [
        patch(f"{NS}.plan", return_value=plan),
        patch(f"{NS}.route_task", return_value=route),
        patch(f"{NS}.compute_max_rounds", return_value=max_rounds),
        patch(f"{NS}.solve_phase", return_value=results),
        patch(f"{NS}.reflect", return_value=reflect_result),
        patch(f"{NS}.synthesize", return_value={
            "answer": answer,
            "confidence": "high",
            "partial": False,
            "missing": [],
            "sources": [{"id": "s1", "title": "来源1"}],
        }),
        patch(f"{NS}.should_short_circuit", return_value=short_circuit),
        patch(f"{NS}.build_short_circuit_response", return_value={
            "answer": "[短路] 无可用内容",
            "confidence": "empty",
            "partial": True,
            "missing": ["全部"],
            "sources": [],
        }),
    ]
    return patches


def _run(**overrides):
    """Start patches, run graph, stop patches, return (result, ctx)."""
    ctx = _make_ctx()
    managers = _make_patches(**overrides)
    for m in managers:
        m.start()
    try:
        result = asyncio.run(
            run_creative_graph(
                "写一篇关于巴厘岛旅游的攻略",
                ctx,
                clarify_count=0,
                clarify_enabled=True,
            )
        )
    finally:
        for m in managers:
            m.stop()
    return result, ctx


def smoke_happy_path() -> None:
    """PLAN valid -> SOLVE -> REFLECT(all_pass) -> SYNTHESIZE -> DONE. 返回 response。"""
    plan = _make_plan()
    result, ctx = _run(plan=plan, results=_make_results(plan))
    assert result is not None, "happy path should return a response"
    assert result["answer"] == "这是生成的巴厘岛攻略回答。", result
    assert ctx.metrics.get("creative_rounds") == 1, ctx.metrics
    assert ctx.metrics.get("creative_confidence") == "high", ctx.metrics
    print(f"[smoke] happy path ok: rounds={ctx.metrics.get('creative_rounds')}, "
          f"answer={len(result['answer'])} chars")


def smoke_plan_fallback() -> None:
    """route_task -> qa：图返回 None（回退 QA）。"""
    plan = _make_plan()
    result, ctx = _run(plan=plan, results=_make_results(plan), route="qa")
    assert result is None, "fallback should return None"
    print("[smoke] plan fallback ok: returned None")


def smoke_replan_path() -> None:
    """REFLECT(has_fixable) -> REPLAN -> SOLVE -> REFLECT(all_pass) -> SYNTHESIZE。

    验证 replan 路由被触发，且第二轮 SOLVE 后能正常合成。
    """
    plan = _make_plan()
    results = _make_results(plan)

    # 用 callable 模拟两次 reflect：第一轮 has_fixable，第二轮 all_pass
    reflect_calls = {"n": 0}

    def _reflect_stub(tasks, results_in, ctx, round_num, max_rounds):
        reflect_calls["n"] += 1
        if reflect_calls["n"] == 1:
            return ReflectResult(
                has_fixable=True,
                replan_actions={"t1": type("O", (), {"mode": "regenerate"})()},
            )
        return ReflectResult(all_pass=True)

    solve_calls = {"n": 0}

    def _solve_stub(tasks, shared_state, results_in, ctx, replan_overrides):
        solve_calls["n"] += 1
        return {t.id: SubResult(sub_task_id=t.id, section_text=f"内容-{t.id}",
                                confidence="normal", coverage_status="sufficient")
                for t in tasks}

    ctx = _make_ctx()
    managers = [
        patch(f"{NS}.plan", return_value=plan),
        patch(f"{NS}.route_task", return_value="creative"),
        patch(f"{NS}.compute_max_rounds", return_value=2),
        patch(f"{NS}.solve_phase", side_effect=_solve_stub),
        patch(f"{NS}.reflect", side_effect=_reflect_stub),
        patch(f"{NS}.synthesize", return_value={
            "answer": "重规划后的最终回答",
            "confidence": "high", "partial": False, "missing": [], "sources": [],
        }),
        patch(f"{NS}.should_short_circuit", return_value=False),
    ]
    for m in managers:
        m.start()
    try:
        result = asyncio.run(run_creative_graph(
            "写一篇关于巴厘岛旅游的攻略", ctx,
            clarify_count=0, clarify_enabled=True))
    finally:
        for m in managers:
            m.stop()

    assert result is not None, "replan path should return a response"
    assert result["answer"] == "重规划后的最终回答", result
    assert solve_calls["n"] == 2, f"solve should run twice, got {solve_calls['n']}"
    assert reflect_calls["n"] == 2, f"reflect should run twice, got {reflect_calls['n']}"
    assert ctx.metrics.get("creative_rounds") == 2, ctx.metrics
    print(f"[smoke] replan path ok: solve={solve_calls['n']}, "
          f"reflect={reflect_calls['n']}, rounds={ctx.metrics.get('creative_rounds')}")


def smoke_short_circuit() -> None:
    """REFLECT(all_empty) -> SYNTHESIZE(should_short_circuit=True) -> 短路响应。"""
    plan = _make_plan()
    result, ctx = _run(
        plan=plan,
        results={t.id: SubResult(sub_task_id=t.id, section_text="",
                                 coverage_status="empty") for t in plan.tasks},
        reflect_result=ReflectResult(all_empty=True),
        short_circuit=True,
    )
    assert result is not None, "short circuit should still return a response"
    assert result["confidence"] == "empty", result
    assert ctx.metrics.get("creative_short_circuit") is True, ctx.metrics
    print("[smoke] short circuit ok: confidence=empty")


def smoke_clarify_signal() -> None:
    """PLAN 阶段澄清判定 need=True 且 clarify_cb raise ClarifySignal → 图向上传播。

    验证 run_creative_graph 捕获 ClarifySignal 后 re-raise（与 handle_creative_query
    语义一致），由 stream 层转成 `clarify` 事件。
    """
    from types import SimpleNamespace
    from omnibox_agent.services.clarify import ClarifySignal

    plan = _make_plan()
    results = _make_results(plan)

    decision = SimpleNamespace(
        need=True,
        question="你更想要哪种风格？",
        options=[{"key": "a", "label": "美食"}],
        importance="high",
        recommended_key="a",
    )

    async def _clarify_cb(dec, phase, context):
        raise ClarifySignal(dec, phase, context)

    ctx = _make_ctx()
    managers = [
        patch(f"{NS}.plan", return_value=plan),
        patch(f"{NS}.route_task", return_value="creative"),
        patch(f"{NS}.compute_max_rounds", return_value=2),
        patch(f"{NS}.solve_phase", return_value=results),
        patch(f"{NS}.reflect", return_value=ReflectResult(all_pass=True)),
        patch("omnibox_agent.services.clarify.judge_dag_clarification",
              return_value=decision),
    ]
    for m in managers:
        m.start()
    try:
        raised = None
        try:
            asyncio.run(run_creative_graph(
                "写一篇关于巴厘岛旅游的攻略", ctx,
                clarify_cb=_clarify_cb,
                clarify_count=0, clarify_enabled=True))
        except ClarifySignal as e:
            raised = e
        assert raised is not None, "ClarifySignal should propagate out of the graph"
        assert raised.phase == "plan", raised.phase
    finally:
        for m in managers:
            m.stop()
    print(f"[smoke] clarify signal ok: phase={raised.phase}, "
          f"importance={raised.decision.importance}")


def main() -> None:
    smoke_happy_path()
    smoke_plan_fallback()
    smoke_replan_path()
    smoke_short_circuit()
    smoke_clarify_signal()
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
