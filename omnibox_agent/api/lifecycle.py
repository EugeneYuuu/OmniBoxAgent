"""Harness lifecycle management and shared utilities (issue #7, #8).

Extracted from routes.py: startup/shutdown, harness singleton, and helpers.
"""

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

# ---- Harness singleton ----

_harness: Any = None

# ---- Ask 追踪事件批量 flush（docs/ask-trace-technical-design.md §3.3） ----
# 数据清理由后端 @Scheduled 统一负责（§6），Agent 启动不清理。

_trace_flush_task: asyncio.Task | None = None


async def _trace_flush_worker() -> None:
    """后台线程跑 trace_store.flush_loop_async（每 2s 批量写事件）。"""
    from omnibox_agent.core.trace_store import flush_loop_async

    await asyncio.to_thread(flush_loop_async)


def start_trace_flush() -> None:
    """启动事件批量 flush 任务（幂等）。"""
    global _trace_flush_task
    if _trace_flush_task is None or _trace_flush_task.done():
        _trace_flush_task = asyncio.create_task(_trace_flush_worker())
        log.info("Ask trace flush task started")


def stop_trace_flush() -> None:
    """停止 flush 任务（取消前排空队列）。"""
    global _trace_flush_task
    if _trace_flush_task is not None:
        from omnibox_agent.core.trace_store import flush_now
        try:
            flush_now()
        except Exception as e:
            log.warning("trace flush on stop failed: %s", e)
        _trace_flush_task.cancel()
        _trace_flush_task = None


def get_harness():
    """Get or create the harness singleton. Idempotent — only creates once."""
    global _harness
    if _harness is None:
        from omnibox_agent.agent.harness import AgentHarness
        from omnibox_agent.agent.ask_agent import create_ask_agent
        from omnibox_agent.agent.mcp_client import McpManager
        from omnibox_agent.core.config import get_config

        cfg = get_config()
        _harness = AgentHarness(cfg)

        # Set up MCP if enabled
        mcp_manager = None
        if cfg.mcp.enabled:
            mcp_manager = McpManager()

        # Set up SkillManager（SKILL 渐进式加载，docs/skill-support-design.md）
        skill_manager = None
        if cfg.skills.enabled:
            from omnibox_agent.skills.manager import SkillManager
            skill_manager = SkillManager(cfg.skills)

        # Set up MemoryManager（会话记忆 + 长期记忆生命周期托管，harness 统一管理，
        # MEMORY_HARNESS_INTEGRATION_DESIGN.md §4.4；构造条件 = 任一开关开启，默认两开）
        memory_manager = None
        if cfg.memory.enabled or cfg.memory.long_term_enabled:
            from omnibox_agent.services.memory_manager import MemoryManager
            memory_manager = MemoryManager(cfg.memory)

        ask_agent = create_ask_agent(cfg, mcp_registry=mcp_manager)
        _harness.register(ask_agent)
        _harness.mcp_manager = mcp_manager
        _harness.skill_manager = skill_manager
        _harness.memory_manager = memory_manager

    return _harness


def get_mcp_manager():
    """Get the McpManager from harness, or None."""
    h = get_harness()
    return h.mcp_manager


def get_skill_manager():
    """Get the SkillManager from harness, or None."""
    h = get_harness()
    return h.skill_manager


def get_memory_manager():
    """Get the MemoryManager from harness, or None（§4.4）."""
    return get_harness().memory_manager


async def start_harness() -> dict[str, Any]:
    """Start the harness. Returns health status dict.

    Issue #7: Fails fast if MySQL or ChromaDB is unavailable.
    """
    cfg = None
    from omnibox_agent.core.config import get_config
    cfg = get_config()

    errors = []

    # Verify MySQL connection (critical)
    try:
        from omnibox_agent.core.database import get_engine
        engine = get_engine()
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("MySQL connection verified")
    except Exception as e:
        msg = f"MySQL connection failed: {e}"
        log.error(msg)
        errors.append(msg)

    # Verify ChromaDB (critical)
    try:
        from omnibox_agent.services.chroma_store import get_collection
        coll = get_collection()
        log.info("ChromaDB collection '%s' ready, count=%s", coll.name, coll.count())
    except Exception as e:
        msg = f"ChromaDB init failed: {e}"
        log.error(msg)
        errors.append(msg)

    # Critical dependencies must be available
    if errors:
        return {"ok": False, "errors": errors}

    # Start harness (agents + scheduler)
    harness = get_harness()
    try:
        await harness.start()
        log.info("AgentHarness started with %d agents", len(harness.agents))
        # Ask 追踪：启动事件批量 flush 任务
        start_trace_flush()
        return {"ok": True, "agents": len(harness.agents)}
    except Exception as e:
        log.exception("AgentHarness start failed")
        return {"ok": False, "errors": [str(e)]}


async def stop_harness() -> None:
    """Stop harness gracefully."""
    global _harness
    # Ask 追踪：停止 flush 任务（先排空队列）
    stop_trace_flush()
    if _harness is None:
        return
    try:
        await _harness.stop()
        log.info("AgentHarness stopped")
    except Exception as e:
        log.warning("Harness stop error: %s", e)
    _harness = None
