"""/v1/embed/* endpoints — vector management.

v4.1: only /status (coverage check) and /delete are kept. Single-item ingest and
backfill now live under /v1/ingest (multi-modal pipeline: image parse + async
video enrichment). The legacy /sync-item and /backfill routes were removed.
"""

import logging

from fastapi import APIRouter

from omnibox_agent.models.ask import (
    EmbedStatusResponse,
    EmbedDeleteRequest,
)
from omnibox_agent.services.vector_sync import (
    get_sync_status,
    delete_vectors_for_content_ids,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/embed", tags=["embed"])


@router.get("/status", response_model=EmbedStatusResponse)
async def embed_status(user_id: str):
    """Get vector embedding coverage status for a user."""
    try:
        stats = get_sync_status(user_id)
        return EmbedStatusResponse(
            ok=True,
            total_collections=stats["total"],
            synced_count=stats["synced"],
            missing_count=stats["missing"],
            stale_count=stats["stale"],
        )
    except Exception as e:
        log.exception("Embed status failed for user %s", user_id)
        return EmbedStatusResponse(ok=False)


@router.post("/delete", response_model=dict)
async def embed_delete(request: EmbedDeleteRequest):
    """Delete vectors for specific content IDs."""
    try:
        delete_vectors_for_content_ids(request.content_ids)
        return {"ok": True, "deleted": len(request.content_ids)}
    except Exception as e:
        log.exception("Embed delete failed")
        return {"ok": False, "reason": str(e)}
