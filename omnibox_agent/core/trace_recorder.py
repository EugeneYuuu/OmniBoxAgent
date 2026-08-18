"""Ask 请求追踪：TraceRecorder、requestId 契约与 llm 只读计数器。

对齐 docs/ask-trace-technical-design.md（§2.2 / §3.2 / §4）：

  - requestId 契约：`req_` + 32 位小写 hex；后端 AiController 生成并透传，
    Agent 优先用后端传的值，缺省（老调用方/直连调试）时自生成并回传。
  - trace_id 维持 12-hex 原格式用于日志上下文，与 requestId 分离，
    通过 ask_trace 表行关联。
  - 事件模型（§3.2）：
      {
        "request_id": "...", "trace_id": "...", "seq": 12,
        "ts": "2026-08-11T15:00:01.234Z",
        "phase": "ask|qa|creative", "event": "task.retrieve",
        "level": "info|warn|error", "task_id": "...",
        "duration_ms": 1234, "message": "...", "data": {}
      }
  - 事件量上限 MAX_EVENTS_PER_REQUEST=200，超限丢弃最旧并计数 dropped_events。
  - llm 只读计数器：每请求重置、仅累加（§1.3 A6），在 llm_service 等真实
    LLM 调用入口 incr_llm()，不回写 AgentContext.llm_call_count。

埋点零侵入：trace_event() 内部 try/except 全包，任何异常只打 warn。
并发穿透：contextvars 绑定，asyncio.gather/create_task 自动复制 context；
线程池（run_blocking 已 copy_context）同样穿透。
"""

from __future__ import annotations

import contextvars
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# 事件量上限（§3.2）
MAX_EVENTS_PER_REQUEST = 200

# request_id 契约：req_ + 32 位小写 hex
_REQUEST_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{8,64}$")


def normalize_request_id(raw: str | None) -> str | None:
    """校验外部传入的 request_id（契约 [a-zA-Z0-9_-]{8,64}）。

    满足契约则原样返回；否则返回 None（调用方生成新的）。
    """
    if not raw:
        return None
    raw = raw.strip()
    if _REQUEST_ID_RE.match(raw):
        return raw
    log.warning("request_id '%s' violates contract, ignored", raw[:32])
    return None


def new_request_id() -> str:
    """生成缺省 request_id：req_ + 32 位小写 hex（§3.1）。"""
    return "req_" + uuid.uuid4().hex


def _iso_ms_utc(ts_ms: int | None = None) -> str:
    """unix 毫秒 → ISO-8601 UTC（`2026-08-11T15:00:01.234Z`）。"""
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    sec = ts_ms // 1000
    ms = ts_ms % 1000
    dt = datetime.fromtimestamp(sec, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"


@dataclass
class TraceEvent:
    """单个追踪事件（文档 §3.2 事件通用字段）。"""

    event: str
    phase: str
    level: str
    seq: int
    ts: str
    task_id: str | None = None
    duration_ms: int | None = None
    message: str | None = None
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "event": self.event,
            "phase": self.phase,
            "level": self.level,
            "seq": self.seq,
            "ts": self.ts,
        }
        if self.task_id is not None:
            d["task_id"] = self.task_id
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        if self.message is not None:
            d["message"] = self.message
        if self.data is not None:
            d["data"] = self.data
        return d


class TraceRecorder:
    """单次 Ask 请求的追踪记录器（内存队列 + 事件上限兜底）。"""

    def __init__(
        self,
        request_id: str,
        trace_id: str,
        user_code: str,
        session_id: str | None,
        query: str,
    ):
        self.request_id = request_id
        self.trace_id = trace_id
        self.user_code = user_code or ""
        self.session_id = session_id
        self.query = query or ""
        self.started_at_ms = int(time.time() * 1000)
        self.events: list[TraceEvent] = []
        self.dropped_events = 0
        self.status = "running"  # running | done | error（§3.3 状态机）
        self.error_msg: str | None = None
        # 终态主记录字段（§3.3 ask_trace 列）
        self.route: str | None = None          # ask | dag
        self.complexity: str | None = None     # simple | complex
        self.rounds: int | None = None         # complex 状态机轮数
        self._seq = 0
        self._lock = threading.Lock()

    def add_event(
        self,
        event: str,
        phase: str = "ask",
        level: str = "info",
        *,
        task_id: str | None = None,
        duration_ms: int | None = None,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> TraceEvent | None:
        """记录一个事件（线程安全；超限丢弃最旧）。"""
        with self._lock:
            # 事件量上限：丢弃最旧（§3.2）
            if len(self.events) >= MAX_EVENTS_PER_REQUEST:
                self.events.pop(0)
                self.dropped_events += 1
            self._seq += 1
            seq = self._seq
            ev = TraceEvent(
                event=event,
                phase=phase,
                level=level,
                seq=seq,
                ts=_iso_ms_utc(),
                task_id=task_id,
                duration_ms=duration_ms,
                message=message,
                data=data,
            )
            self.events.append(ev)
        return ev

    def summary(self) -> dict[str, Any]:
        """汇总为落库/回传用的 dict（请求结束调用）。"""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "user_code": self.user_code,
            "session_id": self.session_id,
            "query": self.query,
            "status": self.status,
            "error_msg": self.error_msg,
            "llm_calls": get_llm_calls(),
            "duration_ms": int(time.time() * 1000) - self.started_at_ms,
            "started_at_ms": self.started_at_ms,
            "events": [e.to_dict() for e in self.events],
            "dropped_events": self.dropped_events,
        }


# ---- contextvars ----

_active_recorder: contextvars.ContextVar[TraceRecorder | None] = contextvars.ContextVar(
    "ask_trace_recorder", default=None
)
_llm_calls: contextvars.ContextVar[int] = contextvars.ContextVar("ask_trace_llm", default=0)


# ---- Trace 生命周期 ----

def begin_trace(
    request_id: str,
    trace_id: str,
    user_code: str,
    session_id: str | None,
    query: str,
) -> TraceRecorder:
    """开启一次 Ask 请求追踪，并挂到当前上下文。"""
    recorder = TraceRecorder(
        request_id=request_id,
        trace_id=trace_id,
        user_code=user_code,
        session_id=session_id,
        query=query,
    )
    _llm_calls.set(0)
    _active_recorder.set(recorder)
    return recorder


def get_recorder() -> TraceRecorder | None:
    """当前上下文中的 recorder；无活跃追踪时返回 None。"""
    return _active_recorder.get()


def end_trace() -> dict[str, Any] | None:
    """结束追踪，清理 contextvars，返回汇总 dict。"""
    recorder = _active_recorder.get()
    if recorder is None:
        return None
    _active_recorder.set(None)
    _llm_calls.set(0)
    return recorder.summary()


# ---- 埋点 API（§3.2 伪代码的正式实现） ----

def trace_event(
    event: str,
    phase: str = "ask",
    level: str = "info",
    *,
    task_id: str | None = None,
    duration_ms: int | None = None,
    message: str | None = None,
    data: dict[str, Any] | None = None,
) -> bool:
    """记录一个事件。无活跃追踪时静默 no-op。

    §3.2：内部 try/except 全包，任何异常只打 warn，绝不阻断主流程。
    """
    try:
        recorder = _active_recorder.get()
        if recorder is None:
            return False
        recorder.add_event(
            event, phase=phase, level=level,
            task_id=task_id, duration_ms=duration_ms,
            message=message, data=data,
        )
        return True
    except Exception as e:
        log.warning("trace_event(%s) failed: %s", event, e)
        return False


def incr_llm() -> None:
    """llm 只读计数器 +1（§1.3 A6：每请求重置、仅累加，不改变调用行为）。"""
    if _active_recorder.get() is not None:
        _llm_calls.set(_llm_calls.get() + 1)


def get_llm_calls() -> int:
    """读取当前上下文的 llm 只读计数。"""
    return _llm_calls.get()
