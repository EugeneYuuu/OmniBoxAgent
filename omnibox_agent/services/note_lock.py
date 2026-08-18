"""v4.1 per-note lock manager.

All write paths to the same note's index (ingestion, video worker, query-time
on-demand re-parse) share the same per-note asyncio.Lock. This prevents
read-modify-write races where two paths concurrently load stale parsed state.

The lock is not reentrant — store_vectors must NEVER be called while already
holding the lock (all callers go through reindex_note or explicit get_note_lock).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

# LRU-style dict to prevent unbounded growth
_MAX_LOCKS = 5000
_PRUNE_INTERVAL_S = 3600  # prune every hour

_note_locks: OrderedDict[str, asyncio.Lock] = {}
_last_prune = time.monotonic()


def get_note_lock(note_id: str) -> asyncio.Lock:
    """Get or create the per-note lock.

    All three write paths (ingestion sync, video worker, query-time on-demand)
    call this to get the same lock instance for a given note_id.
    """
    global _last_prune

    if note_id not in _note_locks:
        _maybe_prune()
        _note_locks[note_id] = asyncio.Lock()

    # Move to end (LRU touch) — only if not currently locked
    lock = _note_locks.pop(note_id)
    _note_locks[note_id] = lock
    return lock


def _maybe_prune():
    """Periodically prune unlocked entries to prevent dict growth."""
    global _last_prune
    now = time.monotonic()
    if now - _last_prune < _PRUNE_INTERVAL_S:
        return
    _last_prune = now

    pruned = 0
    # Only remove entries that are not currently locked
    to_remove = []
    for nid, lock in _note_locks.items():
        if not lock.locked():
            to_remove.append(nid)
    for nid in to_remove:
        del _note_locks[nid]
        pruned += 1

    # Hard cap: if still too many, remove oldest unlocked
    while len(_note_locks) > _MAX_LOCKS:
        for nid, lock in list(_note_locks.items()):
            if not lock.locked():
                del _note_locks[nid]
                pruned += 1
                break
        else:
            break  # all locked, can't prune more

    if pruned > 0:
        logger.debug("Pruned %d note locks (%d remaining)", pruned, len(_note_locks))
