"""Agent base class + AgentMeta capability declaration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibox_agent.agent.context import AgentContext
    from omnibox_agent.core.config import Config


@dataclass
class AgentMeta:
    """Agent capability declaration.

    Phase 2's AgentRegistry uses this for discovery and matching.
    """
    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)


@dataclass
class Agent:
    meta: AgentMeta
    config: Config

    async def initialize(self) -> None:
        """Called once during harness.start() before scheduler starts."""
        pass

    async def start(self) -> None:
        """Called on harness start; subclasses override for async init."""
        pass

    async def stop(self) -> None:
        """Called on harness stop; subclasses override for cleanup."""
        pass

    async def health_check(self) -> bool:
        """Base default: healthy. Subclasses override for real checks."""
        return True
