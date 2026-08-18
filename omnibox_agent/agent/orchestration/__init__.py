"""Orchestration package -- complexity routing for query dispatch.

Lazy exports to avoid circular imports across orchestration modules.
"""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_MAP: dict[str, str] = {
    "ComplexityRouter": "omnibox_agent.agent.orchestration.router",
}


def __getattr__(name: str) -> Any:
    if name in _MODULE_MAP:
        module = importlib.import_module(_MODULE_MAP[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
