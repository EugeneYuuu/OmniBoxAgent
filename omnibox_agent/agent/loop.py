"""Pipeline primitives: PipelineStep, exceptions, thread-pool helpers.

AgentLoop（手写顺序驱动）已删除，QA 编排改由 LangGraph 子图
（agent.graph_qa.run_qa_graph）接管。本模块仅保留 PipelineStep 基类、
异常类型和 run_blocking 线程池辅助函数。
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from omnibox_agent.core.config import get_config

log = logging.getLogger(__name__)

# Ask executor workers (tunable via env)
_ASK_EXECUTOR_WORKERS = int(__import__("os").getenv("ASK_EXECUTOR_WORKERS", "8"))

# ---- Exceptions ----


class PipelineAborted(Exception):
    """Raised when a critical step fails.

    code values:
      "error"    -- critical step failure
      "busy"     -- executor saturated (no ticket)
      "guard"    -- guard step blocked (e.g. no accounts)
    """

    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code


class ExecutorBusyError(Exception):
    """Raised when the Ask executor has no available tickets."""
    pass


# ---- PipelineStep ----


class PipelineStep:
    """A single step in a pipeline.

    Convention: steps must be stateless. The same instance is shared
    across all concurrent requests. All request data flows through ctx.

    Note: no per-step timeouts — the pipeline is not time-limited. LLM calls
    carry their own (now removed) timeouts; actual network errors surface as
    exceptions handled per step.
    """

    name: str
    critical: bool

    def __init__(self, name: str = "", critical: bool = False):
        self.name = name or self.__class__.__name__
        self.critical = critical

    async def execute(self, ctx: Any) -> None:
        """Override in subclasses. ctx is AgentContext."""
        raise NotImplementedError


# ---- Ask thread-pool helpers ----

_ask_executor: ThreadPoolExecutor | None = None
_ask_tickets: threading.BoundedSemaphore | None = None


def _get_ask_executor() -> ThreadPoolExecutor:
    global _ask_executor, _ask_tickets
    if _ask_executor is None:
        _ask_executor = ThreadPoolExecutor(
            max_workers=_ASK_EXECUTOR_WORKERS,
            thread_name_prefix="ask-",
        )
        _ask_tickets = threading.BoundedSemaphore(_ASK_EXECUTOR_WORKERS)
    return _ask_executor


def shutdown_ask_executor() -> None:
    global _ask_executor, _ask_tickets
    if _ask_executor is not None:
        _ask_executor.shutdown(wait=False)
        _ask_executor = None
        _ask_tickets = None


async def run_blocking(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a sync function in the Ask thread pool with bounded admission.

    Acquires a non-blocking ticket; raises ExecutorBusyError if saturated.
    Ticket is returned exactly once: by the worker's finally on success/failure,
    or by the event loop catch if submission itself fails.
    Contextvars (e.g. trace_id) are propagated into the worker thread.
    """
    executor = _get_ask_executor()
    tickets = _ask_tickets
    assert tickets is not None

    acquired = tickets.acquire(blocking=False)
    if not acquired:
        raise ExecutorBusyError("Ask executor saturated, no available tickets")

    ctx_copy = contextvars.copy_context()

    def _wrapper() -> Any:
        try:
            return fn(*args, **kwargs)
        finally:
            tickets.release()

    try:
        # 提交与等待分离：只有 run_in_executor 提交本身失败（worker 未执行）
        # 才在此释放 ticket；worker 的 finally 已负责成功/异常路径的 release。
        # 若 await 抛 fn 的异常（worker 已 release），这里不能重复 release，
        # 否则 BoundedSemaphore 报 "Semaphore released too many times"。
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(executor, ctx_copy.run, _wrapper)
    except Exception:
        # Submission failed (unlikely); release ticket that worker won't release
        tickets.release()
        raise
    return await future
