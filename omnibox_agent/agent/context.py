"""Pipeline context: AgentContext + intermediate output dataclasses.

Uses artifacts: dict for extensibility -- new agents write convention keys
without changing the Context class.

v4.1 additions:
  - Counters dict (ondemand/rewrite attempts)
  - Flags dict (gate_degraded / low_confidence / partial)
  - llm_call_count for cost tracking

Note: no deadline clock — pipelines are not time-limited.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class RetrievalOutput:
    """Output from the retrieval step."""
    vector_results: list[dict] = field(default_factory=list)
    fulltext_results: list[dict] = field(default_factory=list)
    fused_items: list[dict] = field(default_factory=list)
    total_count: int = 0
    platform_dist: dict[str, int] = field(default_factory=dict)
    content_map: dict[int, dict] = field(default_factory=dict)


@dataclass
class ReasoningOutput:
    """Output from the reasoning/prompt-building step."""
    system_prompt: str = ""
    messages: list[dict] = field(default_factory=list)
    intent: str = ""


@dataclass
class AgentContext:
    # Input
    input: dict = field(default_factory=dict)
    session_id: str | None = None

    # Intermediate artifacts -- convention keys:
    #   "perception"  -> QueryUnderstandingResult
    #   "retrieval"   -> RetrievalOutput
    #   "reasoning"   -> ReasoningOutput
    #   "guard"       -> {"account_ids": [...]}
    #   "skills"      -> SkillResolution | None（SKILL 渐进式匹配结果，未命中为 None）
    artifacts: dict[str, Any] = field(default_factory=dict)

    # Meta
    trace_id: str = field(default_factory=lambda: uuid4().hex[:12])
    metrics: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    # Phase 2: Map-Reduce sub-results, indexed by label
    sub_results: dict[str, AgentContext] = field(default_factory=dict)

    # Pipeline start clock (for elapsed() metrics only — no deadline)
    _start_time: float = field(default_factory=time.monotonic)

    # v4.1: Counters for gate fallback chain (ondemand parse, rewrite)
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # v4.1: Flags for degraded/partial state propagation
    flags: dict[str, Any] = field(default_factory=dict)

    # v4.1: LLM call counter (cost tracking, §11)
    llm_call_count: int = 0

    def elapsed(self) -> float:
        """Elapsed time since pipeline start (seconds)."""
        return time.monotonic() - self._start_time

    def sub(self, label: str, **input_override: Any) -> AgentContext:
        """Fork a child context for parallel sub-tasks (Phase 2 foreach).

        Inherits session_id/trace_id, overrides input fields.
        Stores reference in sub_results for gather.
        """
        child = AgentContext(
            input={**self.input, **input_override},
            session_id=self.session_id,
            trace_id=self.trace_id,
            _start_time=self._start_time,
        )
        self.sub_results[label] = child
        return child
