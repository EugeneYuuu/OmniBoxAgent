"""v4.1 MCP limited call wrapper with global semaphore + daily budget + retry.

All ingestion-time MCP calls (vision-mcp-server image analysis, ai-video-notes
task creation/query) go through this wrapper to ensure:
  - Global concurrency limit (semaphore, default 3)
  - Daily call budget with circuit-breaker fallback to queue
  - Exponential backoff retry (default max 2 retries)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import date
from typing import Any

from omnibox_agent.core.config import get_config

log = logging.getLogger(__name__)

# ── Global semaphore for MCP concurrency ──
_mcp_sem: asyncio.Semaphore | None = None

# ── Daily budget tracking ──
_daily_counts: dict[str, int] = {}          # date_str → count
_daily_budget_hit: dict[str, bool] = {}     # date_str → tripped

# ── MCP manager reference (set during harness startup) ──
_mcp_manager: Any = None


def init_mcp_wrapper(mcp_manager: Any) -> None:
    """Set the MCP manager reference and initialize semaphore."""
    global _mcp_manager, _mcp_sem
    _mcp_manager = mcp_manager
    cfg = get_config()
    if _mcp_sem is None:
        _mcp_sem = asyncio.Semaphore(cfg.ingestion.max_concurrent_mcp)
    log.info("MCP wrapper initialized (concurrency=%d, daily_budget=%d)",
             cfg.ingestion.max_concurrent_mcp, cfg.ingestion.mcp_daily_budget)


def _today_key() -> str:
    return date.today().isoformat()


def _check_budget() -> bool:
    """Returns True if budget allows another call."""
    cfg = get_config()
    today = _today_key()
    count = _daily_counts.get(today, 0)
    if count >= cfg.ingestion.mcp_daily_budget:
        if not _daily_budget_hit.get(today):
            log.warning("MCP daily budget hit: %d/%d calls today", count,
                        cfg.ingestion.mcp_daily_budget)
            _daily_budget_hit[today] = True
        return False
    return True


def _incr_count():
    today = _today_key()
    _daily_counts[today] = _daily_counts.get(today, 0) + 1


async def mcp_limited_call(tool: str, **kwargs) -> str:
    """Global rate-limited + budget-protected MCP call with exponential backoff retry.

    Args:
        tool: Prefixed tool name (server__tool)
        **kwargs: Tool arguments

    Returns:
        Tool result string

    Raises:
        RuntimeError: If MCP manager not initialized, budget exhausted, or all retries fail
    """
    if _mcp_manager is None:
        raise RuntimeError("MCP wrapper not initialized — call init_mcp_wrapper() first")
    global _mcp_sem
    if _mcp_sem is None:
        cfg = get_config()
        _mcp_sem = asyncio.Semaphore(cfg.ingestion.max_concurrent_mcp)

    if not _check_budget():
        raise RuntimeError(f"MCP daily budget exhausted ({_daily_counts.get(_today_key(), 0)} calls)")

    cfg = get_config()
    max_retries = cfg.ingestion.mcp_retry_max

    async with _mcp_sem:
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                _incr_count()
                result = await _mcp_manager.call(tool, **kwargs)
                return result
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    backoff = 2 ** attempt  # 1s, 2s, 4s...
                    log.warning("MCP call %s failed (attempt %d/%d): %s — retrying in %ds",
                                tool, attempt + 1, max_retries + 1, e, backoff)
                    await asyncio.sleep(backoff)
                else:
                    log.error("MCP call %s failed after %d attempts: %s",
                              tool, max_retries + 1, e)
        raise RuntimeError(f"MCP call {tool} failed after {max_retries + 1} attempts: {last_err}")


def get_mcp_daily_usage() -> dict[str, int]:
    """Get daily MCP call counts (for monitoring/dashboard)."""
    return dict(_daily_counts)


def is_mcp_budget_available() -> bool:
    """Check if MCP daily budget still allows calls."""
    return _check_budget()
