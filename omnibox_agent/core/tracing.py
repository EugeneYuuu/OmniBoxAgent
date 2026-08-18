"""Trace ID propagation via contextvars and log filter injection."""

from __future__ import annotations

import contextvars
import logging
import uuid
from typing import Any


_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


def set_trace_id(trace_id: str | None = None) -> str:
    """Set the trace_id for the current context. Returns the trace_id."""
    tid = trace_id or uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    """Get the current trace_id, or empty string if not set."""
    return _trace_id.get()


class TraceIdFilter(logging.Filter):
    """Inject trace_id into log records.

    Sets record.trace_id so that %(trace_id)s in format strings
    renders correctly when this filter is attached to a logger.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id() or "-"
        return True


class SafeTraceFormatter(logging.Formatter):
    """Formatter that safely handles missing trace_id on records.

    Use this instead of standard Formatter when the format string
    includes %(trace_id)s but some loggers may not have TraceIdFilter.
    """

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "trace_id"):
            record.trace_id = "-"
        return super().format(record)


def install_trace_filter() -> None:
    """Install TraceIdFilter on root logger and uvicorn.error.

    Note: uvicorn.access uses its own formatter and does not get trace_id
    injected; access-log trace correlation is not supported in this design.
    """
    f = TraceIdFilter()

    root = logging.getLogger()
    root.addFilter(f)

    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_error.addFilter(f)


def create_trace_formatter(fmt: str | None = None) -> SafeTraceFormatter:
    """Create a SafeTraceFormatter with the given format string."""
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] [%(trace_id)s] %(name)s: %(message)s"
    return SafeTraceFormatter(fmt)
