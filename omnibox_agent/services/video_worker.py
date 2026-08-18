"""v4.1 Background video enrichment worker — producer/consumer queue.

Independent async consumer that processes video tasks ONE AT A TIME from an
asyncio.Queue (producer/consumer pattern).  Each task is fully processed
(query → summarize → reindex → persist) before the next task is dequeued,
so we never fire a burst of MCP calls (which triggered server-side 429
rate limiting).

Task status mapping (Baidu ai_note API, wrapped by MCP in ``raw``):
  - ``errno``/``status`` 10000 or show_msg containing "running" → pending
  - ``errno``/``status`` 10002 or show_msg containing "done/complete/success"
    → done; notes live under ``notes`` / ``notes_text`` / ``result`` / ``content``
  - anything else → failed
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from omnibox_agent.core.config import get_config
from omnibox_agent.services.mcp_wrapper import mcp_limited_call
from omnibox_agent.services.llm_service import summarize_if_long
from omnibox_agent.services.ai_config_store import get_user_ai_config
from omnibox_agent.services.ingestion import reindex_note
from omnibox_agent.services.video_task_store import get_video_task_store
from omnibox_agent.models.note import NoteRecord

log = logging.getLogger(__name__)


class VideoEnrichmentWorker:
    """Background worker: serial consumer of a video-task queue.

    Producer(s) enqueue pending tasks via ``enqueue()`` / ``enqueue_all()``.
    The single consumer loop dequeues ONE task, processes it end-to-end, and
    only then pulls the next one — strictly serial, no concurrent MCP bursts.
    """

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running = False
        self._note_loader: Any = None  # Injected during init
        self._queue: asyncio.Queue = asyncio.Queue()
        # task_ids that are enqueued or currently being processed — guards
        # against duplicate enqueue (producer + store seeding + retry).
        self._queued: set[str] = set()
        self._queued_lock = threading.Lock()
        # Consecutive error count per task_id — prevents a permanently-failing
        # task (e.g. MCP tool gone) from occupying the queue forever.
        self._error_counts: dict[str, int] = {}
        self._error_counts_lock = threading.Lock()

    def set_note_loader(self, loader: Any) -> None:
        """Set the function used to load NoteRecord by note_id.

        Must be set before starting the worker.
        """
        self._note_loader = loader

    async def start(self) -> None:
        """Start the background consumer loop."""
        if self._running:
            log.warning("VideoEnrichmentWorker already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._consume_loop())
        log.info("VideoEnrichmentWorker started (serial consumer)")

    async def stop(self) -> None:
        """Stop the background consumer loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("VideoEnrichmentWorker stopped")

    # ── Producer interface ──────────────────────────────────────────────

    def enqueue(self, task_id: str) -> None:
        """Enqueue a single task id (deduped).

        Thread-safe: schedules onto the loop if called from a non-main thread
        (e.g. after ingestion persists a new pending task).
        """
        with self._queued_lock:
            if task_id in self._queued:
                return
            self._queued.add(task_id)
            do_put = lambda: self._queue.put_nowait(task_id)  # noqa: E731
        loop = asyncio.get_event_loop()
        try:
            if loop.is_running() and threading.current_thread() is not threading.main_thread():
                loop.call_soon_threadsafe(do_put)
            else:
                do_put()
        except Exception:
            # Queue closed / loop not running — drop silently.
            with self._queued_lock:
                self._queued.discard(task_id)

    def enqueue_all(self, task_ids: list[str]) -> None:
        """Enqueue multiple task ids at once (deduped, thread-safe)."""
        for tid in task_ids:
            self.enqueue(tid)

    # ── Consumer loop ───────────────────────────────────────────────────

    async def _consume_loop(self) -> None:
        """Main consumer loop: dequeue one task → process fully → next."""
        # Bootstrap: pull any already-persisted pending tasks into the queue
        # so tasks created before startup (or by other workers) are picked up.
        try:
            self._seed_from_store()
        except Exception as e:
            log.warning("VideoEnrichmentWorker seed-from-store failed: %s", e)

        while self._running:
            task_id = await self._next_task()
            if task_id is None:
                continue

            try:
                await self._process_one(task_id)
            except Exception as e:
                log.warning("VideoEnrichmentWorker: error processing task %s: %s",
                            task_id, e)
                # Transient error — allow retry with a small backoff, but give
                # up after MAX_CONSECUTIVE_ERRORS so a permanently-broken task
                # can't occupy the serial queue forever.
                if self._bump_error(task_id):
                    log.error("Video task %s exceeded consecutive error limit; giving up",
                              task_id)
                    try:
                        get_video_task_store().mark_giveup(task_id)
                    except Exception as ex:
                        log.warning("video_worker: mark_giveup failed for %s: %s",
                                    task_id, ex)
                    continue
                await asyncio.sleep(get_config().ingestion.video_poll_interval)
                self.enqueue(task_id)

    def _bump_error(self, task_id: str) -> bool:
        """Increment consecutive-error count. Returns True when limit reached."""
        limit = get_config().ingestion.video_max_consecutive_errors
        with self._error_counts_lock:
            count = self._error_counts.get(task_id, 0) + 1
            self._error_counts[task_id] = count
            return count >= limit

    async def _next_task(self) -> str | None:
        """Get the next task id, waiting for work."""
        while self._running:
            try:
                return await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Queue empty → seed from the store (picks up externally-created
                # tasks), then sleep briefly to avoid a busy spin.
                try:
                    self._seed_from_store()
                except Exception:
                    pass
                await asyncio.sleep(1.0)

    def _seed_from_store(self) -> None:
        """Pull all pending task ids from the store into the queue (dedup)."""
        store = get_video_task_store()
        pending = store.pending()
        if not pending:
            return
        for task in pending:
            self.enqueue(task.task_id)

    # ── Single-task processing ──────────────────────────────────────────

    async def _process_one(self, task_id: str) -> None:
        """Process exactly ONE video task end-to-end."""
        store = get_video_task_store()

        try:
            task = store.get(task_id)
            if task is None or task.status != "pending":
                log.debug("VideoEnrichmentWorker: task %s not pending (skipped)", task_id)
                return

            log.info("VideoEnrichmentWorker: processing task %s (note %s)",
                     task.task_id, task.note_id)

            result = await mcp_limited_call(
                "ai-video-notes__ai_notes_query_task",
                task_id=task.task_id,
            )

            status, notes_text = self._parse_task_result(result)

            if status == "done" and notes_text:
                # Resolve the note owner's key so the summary runs under
                # the user's own model (only embedding uses Zhipu).
                owner_cfg: dict | None = None
                try:
                    if self._note_loader:
                        note = self._note_loader(task.note_id)
                        if note and note.user_id:
                            owner_cfg = get_user_ai_config(note.user_id)
                except Exception as e:
                    log.warning(
                        "video_worker: failed to resolve owner ai_config for %s: %s",
                        task.note_id, e,
                    )
                # Summarize OUTSIDE the lock (expensive LLM call)
                summary = await summarize_if_long(
                    notes_text, task.note_title, ai_config=owner_cfg
                )

                # Critical section: load → mark → collect → store
                await reindex_note(
                    note_id=task.note_id,
                    new_media=[summary],
                    mark_done=self._make_mark_done(task.video_id),
                    note_loader=self._note_loader,
                )
                # Persist completion (thread-safe SQLite write) — cross-thread
                # notification that this task finished and is persisted.
                store.mark_done(task.task_id)
                log.info("Video task %s done, note %s enriched", task.task_id, task.note_id)

            elif status == "done":
                # Task reached a definitive "done" state (errno=0 / 10002) but
                # the response carried no notes content. This is a terminal
                # state — mark it done instead of retrying forever.
                store.mark_done(task.task_id)
                log.info("Video task %s done (no notes content), marked done (note %s)",
                         task.task_id, task.note_id)

            elif status == "failed":
                store.mark_giveup(task.task_id)
                log.warning("Video task %s failed, giving up (note %s)",
                            task.task_id, task.note_id)

            else:
                # status == "pending" → task still running on MCP side; retry
                # later with a backoff (don't hammer the query endpoint).
                # Note: we do NOT re-enqueue here — the finally block releases
                # the dedup slot, and _next_task re-seeds pending tasks from the
                # store, so the task is naturally picked up again with backoff.
                backoff = get_config().ingestion.video_poll_interval
                log.debug("Video task %s still pending (note %s) — retry in %ss",
                          task.task_id, task.note_id, backoff)
                await asyncio.sleep(backoff)
        finally:
            # Always pair the get() with task_done(), and free the dedup slot
            # so the task can be retried (pending) or is no longer scheduled
            # (done/failed/giveup).
            self._queue.task_done()
            with self._queued_lock:
                self._queued.discard(task_id)

    # ── Parsing ─────────────────────────────────────────────────────────

    def _parse_task_result(self, result: str) -> tuple[str, str]:
        """Parse ai-video-notes query_task result.

        Compatible with the MCP-wrapped Baidu response:
          running: {"success": true, "raw": {"errno": 10000,
                    "show_msg": "task running", ...}}
          done:    {"success": true, "raw": {"status": 10002, "notes": [...]}}
                   or notes_text / notes / result / content at top level or raw.

        Returns (status, notes_text): status ∈ {"done", "pending", "failed"}.
        """
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            data = None

        if isinstance(data, dict):
            # success: false → definitive failure
            if data.get("success") is False:
                return "failed", ""

            raw = data.get("raw")
            raw = raw if isinstance(raw, dict) else {}

            # Status number: Baidu ai_note codes (10000 running, 10002 done).
            # Prefer explicit status, then nested raw, then top-level errno.
            status_num = data.get("status", raw.get("status", raw.get("errno")))
            if status_num is None:
                status_num = data.get("errno")
            show_msg = str(raw.get("show_msg") or data.get("show_msg") or "")
            msg_lower = show_msg.lower()

            done_indicators = ("done", "complete", "success", "finished")
            run_indicators = ("running", "processing", "progress", "pending", "wait")

            # If notes are already present, the task is effectively done.
            extracted = self._extract_notes(data, raw)
            if extracted:
                return "done", extracted

            if status_num is not None and status_num != "":
                if str(status_num) in ("10002", "0"):
                    return "done", ""
                if str(status_num) in ("10000", "1"):
                    return "pending", ""
                # Other numeric codes → failed (e.g. errno 2 "params error",
                # errno 100010 "no audio")
                return "failed", ""

            if any(k in msg_lower for k in done_indicators) and not any(
                k in msg_lower for k in ("fail", "error")
            ):
                return "done", ""
            if any(k in msg_lower for k in run_indicators):
                return "pending", ""
            if any(k in msg_lower for k in ("fail", "error")):
                return "failed", ""

            # Fallback: top-level status string
            top_status = str(data.get("status", "")).lower()
            if top_status in ("done", "complete", "success", "completed"):
                return "done", self._extract_notes(data, raw)
            if top_status in ("pending", "running", "processing"):
                return "pending", ""

            # Unknown JSON → treat as pending (retry later), never drop.
            return "pending", ""

        # Not JSON: look for plain text indicators
        if result:
            result_lower = result.lower().strip()
            if any(k in result_lower for k in ("done", "complete", "success")):
                return "done", result
            if any(k in result_lower for k in ("fail", "error")):
                return "failed", ""

        return "pending", ""

    def _extract_notes(self, data: dict, raw: dict) -> str:
        """Extract note text from MCP response (top-level or raw)."""
        for scope in (data, raw):
            text = self._notes_from_scope(scope)
            if text:
                return text
        # Also try data.data (extra nesting)
        inner = data.get("data")
        if isinstance(inner, dict):
            return self._notes_from_scope(inner) or ""
        return ""

    @staticmethod
    def _notes_from_scope(scope: dict) -> str:
        """Extract notes text from a single dict scope."""
        for key in ("notes_text", "notes", "note_text", "text", "result", "content", "output"):
            val = scope.get(key)
            if val is None:
                continue
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, list) and val:
                parts = []
                for item in val:
                    if isinstance(item, dict):
                        for k in ("contents", "content", "text", "note"):
                            v = item.get(k)
                            if isinstance(v, list):
                                parts.extend(str(x) for x in v if str(x).strip())
                            elif isinstance(v, str) and v.strip():
                                parts.append(v.strip())
                            elif v is not None:
                                parts.append(str(v))
                    elif isinstance(item, str) and item.strip():
                        parts.append(item.strip())
                if parts:
                    return "\n".join(parts)
            if isinstance(val, dict):
                text = VideoEnrichmentWorker._notes_from_scope(val)
                if text:
                    return text
        return ""

    def _make_mark_done(self, video_id: str):
        """Create a mark_done callback for a specific video."""
        def mark_done(note: NoteRecord) -> None:
            for vid in note.videos:
                if vid.id == video_id:
                    vid.mark_parsed(vid.parsed_text)  # Status flip
                    break
        return mark_done


# ── Singleton ──
_worker: VideoEnrichmentWorker | None = None


def get_video_worker() -> VideoEnrichmentWorker:
    global _worker
    if _worker is None:
        _worker = VideoEnrichmentWorker()
    return _worker
