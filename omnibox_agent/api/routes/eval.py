"""/v1/eval/* endpoints — evaluation framework (issue #8, #10)."""

import logging

from fastapi import APIRouter, Request

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/eval", tags=["eval"])


@router.get("/template")
async def eval_template(n: int = 50):
    """Generate an eval set template with n placeholder queries."""
    from omnibox_agent.evaluation.eval_set import EvalSet
    eval_set = EvalSet.create_template(n=n)
    return {
        "name": eval_set.name,
        "description": eval_set.description,
        "version": eval_set.version,
        "queries": [
            {"query": q.query, "expected_content_ids": q.expected_content_ids, "label": q.label}
            for q in eval_set.queries
        ],
    }


@router.post("/run")
async def eval_run(request: Request):
    """Run evaluation on an eval set.

    Issue #10: Returns clear error message when pipeline is not wired yet,
    instead of returning fake metrics.
    """
    try:
        body = await request.json()
        from omnibox_agent.evaluation.eval_set import EvalSet, EvalQuery
        from omnibox_agent.evaluation.eval_runner import EvalRunner

        queries_raw = body.get("queries", [])
        if not queries_raw:
            return {"error": "No queries provided in eval set"}

        queries = [
            EvalQuery(
                query=q.get("query", ""),
                expected_content_ids=q.get("expected_content_ids", []),
                label=q.get("label", ""),
                user_id=q.get("user_id", ""),
            )
            for q in queries_raw
        ]
        eval_set = EvalSet(name=body.get("name", "api"), queries=queries)

        runner = EvalRunner()
        report = await runner.run(eval_set)

        # Issue #10: If no real metrics computed (pipeline not wired), return clear message
        if not report.per_query or all(
            r.gate_decision in ("stub", "not_wired") for r in report.per_query
        ):
            return {
                "status": "not_available",
                "message": (
                    "Evaluation pipeline is not yet wired to the full ask pipeline. "
                    "The /v1/eval/run endpoint requires the retrieval + gate pipeline "
                    "to be connected. This is tracked as a planned feature."
                ),
            }

        pr = runner.pr_curve(report.per_query)
        theta = runner.calibrate_theta(pr, target_precision=0.8)

        return {
            "summary": report.summary(),
            "metrics": {
                "total_queries": report.total_queries,
                "avg_precision": round(report.avg_precision, 3),
                "avg_recall": round(report.avg_recall, 3),
                "p95_latency": round(report.p95_latency, 2),
                "gate_pass_rate": round(report.gate_pass_rate, 3),
                "fail_open_rate": round(report.fail_open_rate, 3),
                "ondemand_rate": round(report.ondemand_rate, 3),
                "rewrite_rate": round(report.rewrite_rate, 3),
                "avg_llm_calls": round(report.avg_llm_calls, 1),
            },
            "pr_curve": pr,
            "calibrated_theta": theta,
        }
    except Exception as e:
        log.exception("Eval run failed")
        return {"error": str(e)}
