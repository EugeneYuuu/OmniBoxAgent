"""Agent package -- lazy exports to avoid circular imports.

Modules are loaded on first attribute access via __getattr__.
This avoids import-time cycles between agent submodules and
service/core modules they depend on.
"""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_MAP: dict[str, str] = {
    "Agent": "omnibox_agent.agent.base",
    "AgentMeta": "omnibox_agent.agent.base",
    "AgentContext": "omnibox_agent.agent.context",
    "RetrievalOutput": "omnibox_agent.agent.context",
    "ReasoningOutput": "omnibox_agent.agent.context",
    "PipelineStep": "omnibox_agent.agent.loop",
    "PipelineAborted": "omnibox_agent.agent.loop",
    "ExecutorBusyError": "omnibox_agent.agent.loop",
    "AgentHarness": "omnibox_agent.agent.harness",
    "create_ask_agent": "omnibox_agent.agent.ask_agent",
    "McpClient": "omnibox_agent.agent.mcp_client",
    "McpRegistry": "omnibox_agent.agent.mcp_client",
    "McpManager": "omnibox_agent.agent.mcp_client",
    "McpStore": "omnibox_agent.agent.mcp_client",
    # Orchestration
    "ComplexityRouter": "omnibox_agent.agent.orchestration.router",
}


def __getattr__(name: str) -> Any:
    if name in _MODULE_MAP:
        module = importlib.import_module(_MODULE_MAP[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
