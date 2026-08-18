"""v4.1 Video task store — SQLite persistence for async video parsing tasks.

ai-video-notes MCP is async: create_task returns immediately, results are
polled later. This store tracks pending video tasks so the background worker
can pick them up.

Schema:
  - video_tasks: (task_id, note_id, video_id, note_title, status, created_at, updated_at)
    status: pending | done | failed | giveup

SQLite is used (not MySQL) because:
  1. OmniBoxAgent doesn't write to MySQL (OmniHub_server owns it)
  2. Video task state is ephemeral agent-level state
  3. SQLite is zero-config, embedded, and sufficient for this volume
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omnibox_agent.core.config import get_config

log = logging.getLogger(__name__)

# Default DB path: project root / data / video_tasks.db
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "video_tasks.db"


@dataclass
class VideoTask:
    """A pending or completed video parsing task."""
    task_id: str = ""
    note_id: str = ""
    video_id: str = ""
    note_title: str = ""
    status: str = "pending"       # pending | done | failed | giveup
    created_at: float = 0.0
    updated_at: float = 0.0


class VideoTaskStore:
    """SQLite-backed video task store with thread-safe access."""

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = _DEFAULT_DB_PATH
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS video_tasks (
                        task_id     TEXT PRIMARY KEY,
                        note_id     TEXT NOT NULL,
                        video_id    TEXT NOT NULL,
                        note_title  TEXT DEFAULT '',
                        status      TEXT DEFAULT 'pending',
                        created_at  REAL DEFAULT 0,
                        updated_at  REAL DEFAULT 0
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON video_tasks(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_note_id ON video_tasks(note_id)")
                conn.commit()
            finally:
                conn.close()
        log.info("VideoTaskStore initialized at %s", self._db_path)

    def save_task(
        self,
        note_id: str,
        video_id: str,
        task_id: str,
        note_title: str = "",
    ) -> None:
        """Save a new video task as pending."""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO video_tasks
                        (task_id, note_id, video_id, note_title, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'pending', ?, ?)
                """, (task_id, note_id, video_id, note_title, now, now))
                conn.commit()
            finally:
                conn.close()
        log.debug("Saved video task: task_id=%s, note_id=%s", task_id, note_id)

    def save_many(self, tasks: list[dict]) -> None:
        """Save multiple video tasks at once."""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            try:
                for t in tasks:
                    conn.execute("""
                        INSERT OR REPLACE INTO video_tasks
                            (task_id, note_id, video_id, note_title, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, 'pending', ?, ?)
                    """, (
                        t["task_id"], t["note_id"], t["video_id"],
                        t.get("note_title", ""), now, now,
                    ))
                conn.commit()
            finally:
                conn.close()

    def get(self, task_id: str) -> VideoTask | None:
        """Get a single task by task_id (any status)."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM video_tasks WHERE task_id = ?", (task_id,)
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return None
        return VideoTask(
            task_id=row["task_id"],
            note_id=row["note_id"],
            video_id=row["video_id"],
            note_title=row["note_title"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def pending(self) -> list[VideoTask]:
        """Get all pending tasks (no TTL — poll until done or failed)."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT * FROM video_tasks
                    WHERE status = 'pending'
                    ORDER BY created_at ASC
                """).fetchall()
            finally:
                conn.close()

        tasks = []
        for row in rows:
            tasks.append(VideoTask(
                task_id=row["task_id"],
                note_id=row["note_id"],
                video_id=row["video_id"],
                note_title=row["note_title"],
                status=row["status"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            ))

        return tasks

    def mark_done(self, task_id: str) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE video_tasks SET status = 'done', updated_at = ?
                    WHERE task_id = ?
                """, (time.time(), task_id))
                conn.commit()
            finally:
                conn.close()

    def mark_giveup(self, task_id: str) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE video_tasks SET status = 'giveup', updated_at = ?
                    WHERE task_id = ?
                """, (time.time(), task_id))
                conn.commit()
            finally:
                conn.close()

    def mark_failed(self, task_id: str) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    UPDATE video_tasks SET status = 'failed', updated_at = ?
                    WHERE task_id = ?
                """, (time.time(), task_id))
                conn.commit()
            finally:
                conn.close()

    def get_status(self) -> dict[str, int]:
        """Get task counts by status (for monitoring)."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT status, COUNT(*) as cnt FROM video_tasks GROUP BY status
                """).fetchall()
            finally:
                conn.close()
        return {row["status"]: row["cnt"] for row in rows}


# ── Singleton ──
_store: VideoTaskStore | None = None


def get_video_task_store() -> VideoTaskStore:
    global _store
    if _store is None:
        _store = VideoTaskStore()
    return _store
