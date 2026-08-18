"""Vector sync service: keeps ChromaDB in sync with MySQL content_items.

v4.1: single-item ingest and backfill moved to /v1/ingest (multi-modal
pipeline). This module now only provides:
  - get_sync_status(): vector coverage check (/v1/embed/status)
  - delete_vectors_for_content_ids(): vector deletion (/v1/embed/delete)
  - _cleanup_deleted_for_user(): prune stale vectors during ingest-backfill
  - get_all_user_ids(): user_code enumeration for ingest-backfill
"""

import logging

from omnibox_agent.services.chroma_store import (
    delete_vectors,
    get_fingerprints,
    get_collection,
)
from omnibox_agent.services.embedding_service import compute_content_fingerprint
from omnibox_agent.services.retrieval_store import (
    get_all_content_for_user,
    get_account_ids,
)

log = logging.getLogger(__name__)


def get_sync_status(user_id: str) -> dict:
    """Get vector embedding coverage status for a user."""
    account_ids = get_account_ids(user_id)
    all_content = get_all_content_for_user(account_ids)
    total = len(all_content)

    content_ids = [c["id"] for c in all_content]
    chroma_fps = get_fingerprints(content_ids)

    synced = 0
    stale = 0
    for c in all_content:
        cid = c["id"]
        mysql_fp = compute_content_fingerprint(
            title=c.get("title"),
            summary=c.get("summary"),
            content_text=None,
            tags=None,
            collected_at=str(c.get("collected_at")) if c.get("collected_at") else None,
            updated_at=str(c.get("updated_at")) if c.get("updated_at") else None,
            ai_tag=c.get("ai_tag"),
        )
        chroma_fp = chroma_fps.get(cid, "")
        if chroma_fp == mysql_fp:
            synced += 1
        elif chroma_fp:
            stale += 1

    return {
        "total": total,
        "synced": synced,
        "missing": total - synced,
        "stale": stale,
    }


def delete_vectors_for_content_ids(content_ids: list[int]) -> None:
    """Delete vectors for specific content IDs.

    v4.1: supports both legacy (content_{id}) and multi-vector ({id}#main /
    {id}#media) ID formats so deletion works regardless of which pipeline
    wrote the vectors.
    """
    ids: list[str] = []
    for cid in content_ids:
        ids.append(f"content_{cid}")
        ids.append(f"{cid}#main")
        ids.append(f"{cid}#media")
    delete_vectors(ids)


def _cleanup_deleted_for_user(user_id: str, existing_content_ids: list[int]) -> None:
    """Remove ChromaDB vectors for a user that no longer exist in MySQL.

    Matches Chroma entries by account_ids using $in filter, then deletes
    vectors whose content_ids are not in the existing set from MySQL.
    """
    account_ids = get_account_ids(user_id)
    if not account_ids:
        return

    try:
        coll = get_collection()

        # Get all Chroma entries for this user
        # v4.1 note: ChromaDB metadata content_id is stored as str; normalize
        # both sides to str for comparison.
        existing_set = {str(cid) for cid in existing_content_ids}

        # Query all vectors for this user (stored user_id == real user_id)
        # FIX A: previously matched by account_id, now by the REAL user_id.
        where_clause: dict = {"user_id": user_id}

        # Get all Chroma IDs for this user
        fetched = coll.get(where=where_clause, include=["metadatas"])
        if not fetched or not fetched.get("ids"):
            return

        to_delete = []
        for chroma_id, meta in zip(fetched["ids"], fetched.get("metadatas", [])):
            if meta and "content_id" in meta:
                cid = str(meta["content_id"])
                if cid not in existing_set:
                    to_delete.append(chroma_id)

        if to_delete:
            coll.delete(ids=to_delete)
            log.info("Cleaned up %d deleted vectors for user %s", len(to_delete), user_id)

    except Exception as e:
        log.warning("Per-user vector cleanup failed for user %s: %s", user_id, e)


def get_all_user_ids() -> list[str]:
    """Get all user_codes from MySQL (for backfill/scheduler).

    Returns public user_codes (not internal ids) so that downstream backfill
    stores ChromaDB user_id = user_code, matching what vector_search filters on.
    """
    from omnibox_agent.core.database import get_session
    from sqlalchemy import text

    session = get_session()
    try:
        result = session.execute(text("SELECT user_code FROM users"))
        return [row[0] for row in result.fetchall()]
    finally:
        session.close()
