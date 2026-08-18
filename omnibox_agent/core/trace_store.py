"""Ask 追踪落库：MySQL 双表（ask_trace / ask_trace_event）+ 异步批量 + 文件兜底。

对齐 docs/ask-trace-technical-design.md §3.3：

  - 与 OmniHub_server 共用 `omnihub` 库两张表；表结构与文档 DDL 一致，
    时间字段以 UTC 存取（DATETIME(3)）。
  - 主记录：`ask.received` 时同步 INSERT（status='running'），终态用单独
    UPDATE 且带状态机守卫 `WHERE request_id=? AND status='running'`
    （只允许 running → done/error 单向流转）。
  - 事件：入内存队列，后台 asyncio 任务批量 INSERT（100 条/批 或 2s 一次）。
  - 幂等：uk_request_id 保证主记录重复写安全。
  - 文件兜底：写库失败时把整请求落盘 trace_fallback/（不阻塞主流程）。
  - 数据脱敏：query/message/task_id 入库前截断；apiKey/aiConfig 强制过滤。
  - 清理任务由后端 @Scheduled 负责（Agent 启动不清理）。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from omnibox_agent.core.database import get_engine

log = logging.getLogger(__name__)

_FALLBACK_DIR = Path("trace_fallback")

# 事件批量 flush 参数（§3.3）
_FLUSH_BATCH = 100
_FLUSH_INTERVAL_SECONDS = 2.0

# 字段长度（与 DDL 对齐，超长截尾加 ...）
_MAX_QUERY = 500
_MAX_MESSAGE = 500
_MAX_EVENT = 48
_MAX_TASK_ID = 64

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS ask_trace (
      id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
      request_id VARCHAR(64)  NOT NULL,
      trace_id   VARCHAR(16)  NULL,
      user_code  VARCHAR(64)  NOT NULL,
      query      VARCHAR(500) NOT NULL,
      route      VARCHAR(16)  NULL,
      complexity VARCHAR(16)  NULL,
      status     VARCHAR(16)  NOT NULL DEFAULT 'running',
      error_msg  VARCHAR(500) NULL,
      started_at DATETIME(3)  NOT NULL,
      ended_at   DATETIME(3)  NULL,
      duration_ms INT NULL,
      llm_calls  INT NULL,
      rounds     INT NULL,
      UNIQUE KEY uk_request_id (request_id),
      KEY idx_user_started (user_code, started_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS ask_trace_event (
      id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
      request_id VARCHAR(64)   NOT NULL,
      user_code  VARCHAR(64)   NOT NULL,
      seq        INT           NOT NULL,
      ts         DATETIME(3)   NOT NULL,
      phase      VARCHAR(16)   NOT NULL,
      event      VARCHAR(48)   NOT NULL,
      level      VARCHAR(8)    NOT NULL DEFAULT 'info',
      task_id    VARCHAR(64)   NULL,
      duration_ms INT          NULL,
      message    VARCHAR(500)  NULL,
      data       JSON          NULL,
      KEY idx_request_seq (request_id, seq),
      KEY idx_event (event, ts),
      KEY idx_user_event (user_code, event, ts)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]

# 敏感字段：绝不允许入库（§3.3 数据脱敏）
_SENSITIVE_KEYS = {"apiKey", "api_key", "aiConfig", "ai_config", "authorization", "token"}


def _ensure_tables() -> None:
    """建表（幂等）。失败仅 warn——追踪绝不阻断主流程。"""
    try:
        engine = get_engine()
        with engine.begin() as conn:
            for ddl in _DDL:
                conn.execute(text(ddl))
    except Exception as e:
        log.warning("trace_store: ensure tables failed: %s", e)


# ---- 脱敏 ----

def _truncate(s: str | None, limit: int) -> str | None:
    if not s:
        return s
    if len(s) <= limit:
        return s
    return s[:limit] + "..."


def _sanitize_data(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """递归过滤敏感键并截断字符串值（§3.3 数据脱敏）。"""
    if not data:
        return None
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k in _SENSITIVE_KEYS:
            continue
        if isinstance(v, dict):
            out[k] = _sanitize_data(v)
        elif isinstance(v, list):
            out[k] = [
                _sanitize_data(x) if isinstance(x, dict) else x for x in v
            ]
        elif isinstance(v, str):
            out[k] = _truncate(v, 500)
        else:
            out[k] = v
    return out or None


def _iso_to_datetime(ts_iso: str) -> datetime:
    """ISO-8601 Z → naive datetime（按 UTC 约定存 DATETIME）。"""
    try:
        s = ts_iso
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return datetime.utcnow()


# ---- 主记录：同步写（ask.received） ----

def persist_trace_start(recorder: Any) -> bool:
    """同步 INSERT 主记录（status='running'）。幂等（uk_request_id）。"""
    try:
        _ensure_tables()
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT IGNORE INTO ask_trace
                  (request_id, trace_id, user_code, query, status, started_at)
                VALUES
                  (:request_id, :trace_id, :user_code, :query, 'running', :started_at)
            """), {
                "request_id": recorder.request_id,
                "trace_id": recorder.trace_id,
                "user_code": recorder.user_code,
                "query": _truncate(recorder.query, _MAX_QUERY) or "",
                "started_at": datetime.fromtimestamp(
                    recorder.started_at_ms / 1000, tz=timezone.utc
                ).replace(tzinfo=None),
            })
        return True
    except Exception as e:
        log.warning("trace_store: persist_trace_start failed: %s", e)
        return False


# ---- 事件：异步批量 flush ----

_events: "queue.Queue[tuple[str, dict]]" = queue.Queue()


def enqueue_events(recorder: Any) -> None:
    """把 recorder 的事件入队（后台批量 flush；失败不阻塞）。"""
    try:
        for ev in recorder.events:
            row = {
                "request_id": recorder.request_id,
                "user_code": recorder.user_code,
                "seq": ev.seq,
                "ts": _iso_to_datetime(ev.ts),
                "phase": (ev.phase or "ask")[:16],
                "event": _truncate(ev.event, _MAX_EVENT),
                "level": (ev.level or "info")[:8],
                "task_id": _truncate(ev.task_id, _MAX_TASK_ID),
                "duration_ms": ev.duration_ms,
                "message": _truncate(ev.message, _MAX_MESSAGE),
                "data": _sanitize_data(ev.data),
            }
            _events.put(("event", row))
    except Exception as e:
        log.warning("trace_store: enqueue_events failed: %s", e)


def _flush_batch(items: list[tuple[str, dict]]) -> None:
    """写入一批（主记录终态 UPDATE + 事件批量 INSERT）。"""
    try:
        _ensure_tables()
        engine = get_engine()
        events = [row for kind, row in items if kind == "event"]
        finals = [row for kind, row in items if kind == "final"]

        with engine.begin() as conn:
            # 终态 UPDATE（状态机守卫：仅 running → done/error）
            for row in finals:
                conn.execute(text("""
                    UPDATE ask_trace
                    SET status = :status,
                        ended_at = :ended_at,
                        duration_ms = :duration_ms,
                        llm_calls = :llm_calls,
                        route = :route,
                        complexity = :complexity,
                        rounds = :rounds,
                        error_msg = :error_msg
                    WHERE request_id = :request_id AND status = 'running'
                """), row)

            # 事件批量 INSERT
            if events:
                # data 列是 JSON 类型：SQLAlchemy text() 不会自动把 dict 序列化，
                # 直接传 dict 会让 MySQL 驱动报 "dict can not be used as parameter"。
                # 这里显式 json.dumps；用副本避免污染文件兜底的原始行。
                prepared = [
                    {**r, "data": json.dumps(r["data"], ensure_ascii=False, default=str)}
                    if r.get("data") is not None else r
                    for r in events
                ]
                conn.execute(
                    text("""
                        INSERT INTO ask_trace_event
                          (request_id, user_code, seq, ts, phase, event, level,
                           task_id, duration_ms, message, data)
                        VALUES
                          (:request_id, :user_code, :seq, :ts, :phase, :event,
                           :level, :task_id, :duration_ms, :message, :data)
                    """),
                    prepared,
                )
    except Exception as e:
        log.warning("trace_store: batch flush failed (%d rows): %s", len(items), e)
        # 文件兜底：整批落盘
        for _, row in items:
            _write_fallback(row)


def flush_now() -> None:
    """同步排空事件队列（请求结束时兜底调用，保证不丢）。"""
    items: list[tuple[str, dict]] = []
    while not _events.empty():
        try:
            items.append(_events.get_nowait())
        except queue.Empty:
            break
    if items:
        _flush_batch(items)


def flush_loop_async() -> None:
    """后台 flush 任务主体（每 2s 批量写一次，100 条/批）。"""
    while True:
        time.sleep(_FLUSH_INTERVAL_SECONDS)
        items: list[tuple[str, dict]] = []
        try:
            for _ in range(_FLUSH_BATCH):
                try:
                    items.append(_events.get_nowait())
                except queue.Empty:
                    break
        except Exception:
            pass
        if items:
            try:
                _flush_batch(items)
            except Exception as e:
                log.warning("trace_store: flush loop error: %s", e)


# ---- 终态（请求结束时调用） ----

def persist_trace_end(recorder: Any, *, status: str = "done", error_msg: str | None = None) -> None:
    """终态落库：事件入队 + 终态 UPDATE 入队；随后同步 flush 保证落库。

    Args:
        status: done | error（§3.3 状态机 running → done/error）
        error_msg: 错误信息（截断）
    """
    recorder.status = status
    if error_msg:
        recorder.error_msg = _truncate(error_msg, _MAX_MESSAGE)

    summary = recorder.summary()
    duration_ms = int(time.time() * 1000) - recorder.started_at_ms

    final_row = {
        "request_id": recorder.request_id,
        "status": status if status in ("done", "error") else "done",
        "ended_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "duration_ms": duration_ms,
        "llm_calls": summary["llm_calls"],
        "route": getattr(recorder, "route", None),
        "complexity": getattr(recorder, "complexity", None),
        "rounds": getattr(recorder, "rounds", None),
        "error_msg": _truncate(error_msg, _MAX_MESSAGE),
    }
    _events.put(("final", final_row))

    # 事件入队
    enqueue_events(recorder)

    # 同步排空，确保请求结束时数据已落库（幂等：uk_request_id + 状态守卫）
    flush_now()


def set_route(recorder: Any, route: str | None) -> None:
    """记录路由（ask/dag），随终态 UPDATE 落库。"""
    try:
        recorder.route = route
    except AttributeError:
        pass


def set_complexity(recorder: Any, complexity: str | None) -> None:
    """记录复杂度判定（simple/complex）。"""
    try:
        recorder.complexity = complexity
    except AttributeError:
        pass


def set_rounds(recorder: Any, rounds: int | None) -> None:
    """记录 complex 状态机轮数。"""
    try:
        recorder.rounds = rounds
    except AttributeError:
        pass


def persist_trace_error(recorder: Any, error_msg: str) -> None:
    """以 error 状态收尾（异常路径用）。"""
    persist_trace_end(recorder, status="error", error_msg=error_msg)


# ---- 文件兜底（§3.3 / §6） ----

def _write_fallback(row: dict[str, Any]) -> None:
    """写库失败时兜底落盘（append-only，按天轮转；不抛异常）。"""
    try:
        _FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = _FALLBACK_DIR / f"trace-{day}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        log.warning("trace_store: fallback write failed: %s", e)
