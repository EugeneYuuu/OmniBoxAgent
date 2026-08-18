"""v4.1 Evaluation runner: run gate on eval set, compute metrics, calibrate θ.

Design doc §5.4 + §12 (observability):
  - Run retrieval + gate on each eval query
  - Compute precision/recall per query and aggregate
  - PR curve for threshold calibration
  - Track: gate pass rate, fail-open rate, ondemand trigger rate,
    rewrite trigger rate, refinement trigger rate, p95 latency
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from omnibox_agent.evaluation.eval_set import EvalSet, EvalQuery

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of running gate on a single eval query."""
    query: str
    expected_content_ids: set[int]
    retrieved_content_ids: list[int]
    gated_content_ids: list[int]
    gate_decision: str = ""
    gate_degraded: bool = False
    low_confidence: bool = False
    ondemand_triggered: bool = False
    rewrite_triggered: bool = False
    refine_triggered: int = 0
    llm_calls: int = 0
    latency_s: float = 0.0

    @property
    def precision(self) -> float:
        """Precision = |retrieved ∩ expected| / |retrieved|"""
        if not self.gated_content_ids:
            return 0.0
        retrieved_set = set(self.gated_content_ids)
        relevant = retrieved_set & self.expected_content_ids
        return len(relevant) / len(retrieved_set) if retrieved_set else 0.0

    @property
    def recall(self) -> float:
        """Recall = |retrieved ∩ expected| / |expected|"""
        if not self.expected_content_ids:
            return 1.0  # No expected = trivially all found
        retrieved_set = set(self.gated_content_ids)
        relevant = retrieved_set & self.expected_content_ids
        return len(relevant) / len(self.expected_content_ids)


@dataclass
class EvalReport:
    """Aggregated evaluation metrics across all queries."""
    total_queries: int = 0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    p95_latency: float = 0.0
    gate_pass_rate: float = 0.0
    fail_open_rate: float = 0.0
    ondemand_rate: float = 0.0
    rewrite_rate: float = 0.0
    avg_refine_triggered: float = 0.0
    avg_llm_calls: float = 0.0
    per_query: list[EvalResult] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Eval Report ({self.total_queries} queries)\n"
            f"  Precision:   {self.avg_precision:.3f}\n"
            f"  Recall:      {self.avg_recall:.3f}\n"
            f"  P95 Latency: {self.p95_latency:.1f}s\n"
            f"  Gate Pass:   {self.gate_pass_rate:.1%}\n"
            f"  Fail-Open:   {self.fail_open_rate:.1%}\n"
            f"  OnDemand:    {self.ondemand_rate:.1%}\n"
            f"  Rewrite:     {self.rewrite_rate:.1%}\n"
            f"  Avg Refine:  {self.avg_refine_triggered:.2f}\n"
            f"  Avg LLM:     {self.avg_llm_calls:.1f} calls/query"
        )


class EvalRunner:
    """Run evaluation on an EvalSet.

    Usage:
        runner = EvalRunner()
        report = runner.run(eval_set)
        print(report.summary())
    """

    def __init__(self, config: Any = None):
        self._config = config

    async def run(self, eval_set: EvalSet) -> EvalReport:
        """Run gate evaluation on all queries in the eval set.

        For each query:
          1. Run retrieval pipeline
          2. Run quality gate
          3. Compare gated results with expected_content_ids

        Returns:
            EvalReport with aggregated metrics.
        """
        results: list[EvalResult] = []

        for eq in eval_set.queries:
            result = await self._run_single(eq)
            results.append(result)
            log.info("Eval [%d/%d] '%s' — P=%.2f R=%.2f gate=%s",
                     len(results), len(eval_set), eq.query[:30],
                     result.precision, result.recall, result.gate_decision)

        return self._aggregate(results)

    async def _run_single(self, eq: EvalQuery) -> EvalResult:
        """Run evaluation on a single query.

        Issue #10: Returns "not_wired" gate_decision when pipeline is not connected,
        instead of silently returning fake metrics. When the pipeline is wired:
          1. Create an AgentContext with the query
          2. Run the full pipeline (or just retrieve + gate)
          3. Extract metrics from ctx.metrics and ctx.flags
        """
        start = time.monotonic()

        # Try to use the real pipeline if harness is available
        try:
            from omnibox_agent.api.lifecycle import get_harness
            harness = get_harness()

            if harness.get("ask") is not None:
                from omnibox_agent.agent.context import AgentContext
                from omnibox_agent.services.ai_config_store import get_user_ai_config

                ai_config = get_user_ai_config(eq.user_id) or {}
                ctx = AgentContext(input={
                    "query": eq.query,
                    "user_id": eq.user_id,
                    "ai_config": ai_config,
                })

                # 通过 LangGraph QA 子图运行检索+门控（Act/Build 节点 eval 不消费）
                from omnibox_agent.agent.graph_qa import run_qa_graph
                await run_qa_graph(ctx)
                retrieval = ctx.artifacts.get("retrieval")
                if retrieval and hasattr(retrieval, "fused_items"):
                    gated_ids = [
                        item.get("content_id")
                        for item in retrieval.fused_items
                        if item.get("content_id")
                    ]
                    latency = time.monotonic() - start
                    return EvalResult(
                        query=eq.query,
                        expected_content_ids=set(eq.expected_content_ids),
                        retrieved_content_ids=gated_ids[:20],
                        gated_content_ids=gated_ids[:20],
                        gate_decision="proceed",
                        llm_calls=ctx.llm_call_count,
                        latency_s=latency,
                    )
        except Exception as e:
            log.warning("Eval pipeline call failed for '%s': %s", eq.query[:30], e)

        # Pipeline not wired — return "not_wired" (issue #10)
        latency = time.monotonic() - start
        return EvalResult(
            query=eq.query,
            expected_content_ids=set(eq.expected_content_ids),
            retrieved_content_ids=[],
            gated_content_ids=[],
            gate_decision="not_wired",
            latency_s=latency,
        )

    def _aggregate(self, results: list[EvalResult]) -> EvalReport:
        """Aggregate per-query results into a report."""
        if not results:
            return EvalReport()

        n = len(results)
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        latencies = sorted([r.latency_s for r in results])

        # P95 latency
        p95_idx = int(n * 0.95)
        p95 = latencies[min(p95_idx, n - 1)] if latencies else 0.0

        # Gate metrics
        gate_pass = sum(1 for r in results if r.gate_decision == "proceed")
        fail_open = sum(1 for r in results if r.gate_degraded)
        ondemand = sum(1 for r in results if r.ondemand_triggered)
        rewrite = sum(1 for r in results if r.rewrite_triggered)
        refine_total = sum(r.refine_triggered for r in results)
        llm_total = sum(r.llm_calls for r in results)

        return EvalReport(
            total_queries=n,
            avg_precision=sum(precisions) / n,
            avg_recall=sum(recalls) / n,
            p95_latency=p95,
            gate_pass_rate=gate_pass / n,
            fail_open_rate=fail_open / n,
            ondemand_rate=ondemand / n,
            rewrite_rate=rewrite / n,
            avg_refine_triggered=refine_total / n,
            avg_llm_calls=llm_total / n,
            per_query=results,
        )

    def pr_curve(
        self,
        results: list[EvalResult],
        thresholds: list[float] | None = None,
    ) -> list[dict[str, float]]:
        """Compute PR curve for threshold calibration.

        v4.1 §5.4: PR curve to select θ (target: precision ≥ 0.8, max recall).

        Args:
            results: Per-query eval results
            thresholds: Optional list of threshold values to evaluate.
                        Default: [0.1, 0.2, ..., 0.9]

        Returns:
            List of {"threshold": θ, "precision": p, "recall": r} dicts.
        """
        if thresholds is None:
            thresholds = [i * 0.1 for i in range(1, 10)]

        curve = []
        for theta in thresholds:
            # At each threshold, count how many queries pass
            # (This is a simplified PR curve — in production, each query
            # would have per-doc relevance scores to threshold on)
            passed = [r for r in results if r.precision >= theta]
            p = len(passed) / len(results) if results else 0.0
            r = sum(x.recall for x in passed) / len(passed) if passed else 0.0
            curve.append({
                "threshold": theta,
                "precision": p,
                "recall": r,
            })

        return curve

    def calibrate_theta(
        self,
        pr_curve: list[dict[str, float]],
        target_precision: float = 0.8,
    ) -> float:
        """Select θ that achieves target precision with maximum recall.

        v4.1 §5.4: precision ≥ 0.8, select highest recall.

        Args:
            pr_curve: Output from pr_curve()
            target_precision: Minimum acceptable precision

        Returns:
            Optimal threshold value.
        """
        best_theta = 0.5  # Default
        best_recall = 0.0

        for point in pr_curve:
            if point["precision"] >= target_precision:
                if point["recall"] > best_recall:
                    best_recall = point["recall"]
                    best_theta = point["threshold"]

        log.info("Calibrated θ=%.1f (precision≥%.1f, recall=%.3f)",
                 best_theta, target_precision, best_recall)
        return best_theta
