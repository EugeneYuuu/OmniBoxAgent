"""AgentHarness: lifecycle management, registry.

Manages agent start/stop and provides
a single entry point for routes to access agent instances.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from omnibox_agent.agent.loop import shutdown_ask_executor, _get_ask_executor

if TYPE_CHECKING:
    from omnibox_agent.agent.base import Agent, AgentMeta
    from omnibox_agent.core.config import Config

log = logging.getLogger(__name__)


class AgentHarness:
    """Central harness: agent registry, lifecycle, scheduler.

    Usage:
        harness = AgentHarness(config)
        harness.register(ask_agent)
        await harness.start()
        # ... serve requests ...
        await harness.stop()
    """

    def __init__(self, config: Config):
        self.config = config
        self.agents: dict[str, Agent] = {}
        self.mcp_manager: Any = None  # McpManager | None
        self.skill_manager: Any = None  # SkillManager | None
        self.memory_manager: Any = None  # MemoryManager | None（MEMORY_HARNESS_INTEGRATION_DESIGN.md §4.3）
        self._running = False

    @property
    def mcp_registry(self) -> Any:
        """Backward-compatible alias for mcp_manager."""
        return self.mcp_manager

    @mcp_registry.setter
    def mcp_registry(self, value: Any) -> None:
        self.mcp_manager = value

    def register(self, agent: Agent) -> None:
        """Register an agent. Deduplicates by name."""
        name = agent.meta.name
        if name in self.agents:
            log.warning("Agent '%s' already registered, replacing", name)
        self.agents[name] = agent

    def get(self, name: str) -> Agent | None:
        """Get a registered agent by name."""
        return self.agents.get(name)

    async def start(self) -> None:
        """Start all agents.

        Idempotent: if already running, returns immediately.
        On failure, rolls back (shuts down executor).
        """
        if self._running:
            return

        try:
            # Start all agents
            for name, agent in self.agents.items():
                try:
                    await agent.start()
                except Exception as e:
                    log.exception("Agent '%s' start failed", name)
                    raise

            self._running = True

            # Start MCP if configured
            if self.mcp_manager and self.config.mcp.enabled:
                try:
                    await self.mcp_manager.startup(self.config.mcp.servers)
                    log.info("MCP manager started with %d servers", len(self.mcp_manager.clients))

                    # v4.1: Initialize MCP wrapper for ingestion pipeline
                    from omnibox_agent.services.mcp_wrapper import init_mcp_wrapper
                    init_mcp_wrapper(self.mcp_manager)
                except Exception as e:
                    log.warning("MCP manager start failed (non-fatal): %s", e)

            # v4.1: Start video enrichment worker
            try:
                from omnibox_agent.services.video_worker import get_video_worker
                from omnibox_agent.services.note_loader import load_note_by_id
                worker = get_video_worker()
                worker.set_note_loader(load_note_by_id)
                await worker.start()
            except Exception as e:
                log.warning("Video enrichment worker start failed (non-fatal): %s", e)

            # SKILL: start SkillManager if enabled
            if self.skill_manager and self.config.skills.enabled:
                try:
                    await self.skill_manager.startup()
                    log.info("Skill manager started with %d skills",
                             len(self.skill_manager.list_skills()))
                except Exception as e:
                    log.warning("Skill manager start failed (non-fatal): %s", e)

            # MEMORY: start MemoryManager（会话记忆 + 长期记忆生命周期托管，§4.3）
            # 门控 = 会话记忆 或 长期记忆 任一开启（默认两开）
            if self.memory_manager and (self.config.memory.enabled or self.config.memory.long_term_enabled):
                try:
                    await self.memory_manager.startup()
                    log.info("Memory manager started (cleanup_interval=%sh)",
                             self.config.memory.cleanup_interval_hours)
                except Exception as e:
                    log.warning("Memory manager start failed (non-fatal): %s", e)

            log.info("AgentHarness started with %d agents", len(self.agents))

        except Exception:
            # Rollback
            log.exception("Harness start failed, rolling back")
            shutdown_ask_executor()
            raise

    async def stop(self) -> None:
        """Stop agents and executors gracefully."""
        # v4.1: Stop video enrichment worker
        try:
            from omnibox_agent.services.video_worker import get_video_worker
            worker = get_video_worker()
            await worker.stop()
        except Exception as e:
            log.warning("Video enrichment worker stop error: %s", e)

        # Stop agents
        for name, agent in self.agents.items():
            try:
                await agent.stop()
            except Exception as e:
                log.warning("Agent '%s' stop error: %s", name, e)

        # Stop MEMORY manager（必须在 shutdown_ask_executor() 之前——若后台
        # 清理任务恰在执行中，先取消并 await 落地，避免 executor 置 None 后
        # 任务再调 run_blocking 重建线程池的风险，§4.3）
        if self.memory_manager:
            try:
                await self.memory_manager.shutdown()
            except Exception as e:
                log.warning("Memory manager stop error: %s", e)

        # Shutdown Ask executor
        shutdown_ask_executor()

        self._running = False

        # Stop MCP
        if self.mcp_manager:
            try:
                await self.mcp_manager.shutdown()
            except Exception as e:
                log.warning("MCP manager stop error: %s", e)

        # Stop SKILL manager
        if self.skill_manager:
            try:
                await self.skill_manager.shutdown()
            except Exception as e:
                log.warning("Skill manager stop error: %s", e)

        log.info("AgentHarness stopped")

    async def health_check(self) -> dict:
        """Aggregate health of all agents."""
        agents_status = {}
        all_healthy = True
        for name, agent in self.agents.items():
            try:
                healthy = await agent.health_check()
                agents_status[name] = healthy
                if not healthy:
                    all_healthy = False
            except Exception as e:
                agents_status[name] = False
                all_healthy = False
                log.warning("Health check for '%s' failed: %s", name, e)

        result: dict = {
            "status": "ok",
            "agents": agents_status,
        }

        # MySQL probe
        try:
            from omnibox_agent.core.database import get_engine
            from sqlalchemy import text
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            result["mysql"] = "connected"
        except Exception as e:
            result["mysql"] = "error"
            log.warning("MySQL health check failed: %s", e)

        # Chroma probe
        try:
            from omnibox_agent.services.chroma_store import get_collection
            coll = get_collection()
            result["chroma_count"] = coll.count()
        except Exception as e:
            result["chroma_count"] = "error"
            log.warning("Chroma health check failed: %s", e)

        # MCP probe
        if self.mcp_manager:
            try:
                result["mcp"] = await self.mcp_manager.health_check()
            except Exception as e:
                result["mcp"] = {"status": "error", "detail": str(e)}
                log.warning("MCP health check failed: %s", e)

        # MEMORY probe（门控同 start，§4.3）
        if self.memory_manager and (self.config.memory.enabled or self.config.memory.long_term_enabled):
            try:
                result["memory"] = await self.memory_manager.health_check()
            except Exception as e:
                result["memory"] = {"status": "error", "detail": str(e)}

        return result


# ---- Module-level singleton ----

_harness: AgentHarness | None = None


def get_harness(config: Config | None = None) -> AgentHarness:
    """Get or create the harness singleton."""
    global _harness
    if _harness is None:
        if config is None:
            from omnibox_agent.core.config import get_config
            config = get_config()
        _harness = AgentHarness(config)
    return _harness
