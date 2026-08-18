"""ComplexityRouter: routes based on LLM complexity classification.

Reads the "complexity" artifact set by complexity_classifier.py.
No keyword matching — all routing decisions come from LLM-as-a-Judge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibox_agent.agent.context import AgentContext


class ComplexityRouter:
    """Route based on LLM complexity classification artifact.

    Usage:
        router = ComplexityRouter()
        route = router.route(ctx)
        if route == "ask":
            # 走 stream_qa_pipeline（LangGraph QA 子图）
        else:
            # 走 stream_creative_pipeline（LangGraph Creative 子图）
    """

    def route(self, ctx: AgentContext) -> str:
        """Determine routing: "ask" for simple, "dag" for complex.

        Reads ctx.artifacts["complexity"] set by classify_complexity().
        If artifact is missing (classifier failed/skipped), defaults to "ask".
        """
        complexity = ctx.artifacts.get("complexity")
        if complexity is not None:
            return "dag" if complexity.type == "complex" else "ask"

        # Fallback: classifier didn't run -> safe default to QA
        return "ask"
