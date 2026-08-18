"""/v1/ingest/* endpoints — multi-modal ingestion pipeline (issue #8)."""

import logging

from fastapi import APIRouter, HTTPException, Request

from omnibox_agent.api.lifecycle import get_harness

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/ingest", tags=["ingest"])


@router.post("")
async def ingest_content(request: Request):
    """v4.1 Ingestion endpoint: multi-modal parsing + multi-vector indexing."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    content_id = body.get("content_id")
    user_id = body.get("user_id")

    if not content_id:
        raise HTTPException(status_code=400, detail="content_id is required")

    from omnibox_agent.services.note_loader import load_note_by_id
    from omnibox_agent.services.ingestion import ingest_note

    note = load_note_by_id(content_id)
    if note is None:
        raise HTTPException(status_code=404, detail=f"Content {content_id} not found")

    note.user_id = str(user_id) if user_id else note.user_id

    try:
        result = await ingest_note(note)
        log.info("Ingestion complete for content %s: %s", content_id, result)
        return {"ok": True, **result}
    except Exception as e:
        log.exception("Ingestion failed for content %s", content_id)
        return {"ok": False, "reason": str(e)}


@router.post("/backfill")
async def ingest_backfill(user_id: str | None = None):
    """v4.1 Full backfill with multi-modal ingestion."""
    from omnibox_agent.services.note_loader import load_notes_for_user
    from omnibox_agent.services.ingestion import ingest_note
    from omnibox_agent.services.retrieval_store import get_account_ids
    # get_all_user_ids lives in vector_sync (NOT retrieval_store) — importing it
    # from the wrong module here previously caused an ImportError on this route.
    from omnibox_agent.services.vector_sync import get_all_user_ids

    try:
        if user_id is not None:
            user_ids = [user_id]
        else:
            user_ids = get_all_user_ids()

        total_processed = 0
        total_skipped = 0
        total_failed = 0

        for uid in user_ids:
            account_ids = get_account_ids(uid)
            if not account_ids:
                continue

            notes = load_notes_for_user(uid, account_ids)
            for note in notes:
                try:
                    result = await ingest_note(note)
                    if result.get("images_parsed", 0) > 0 or result.get("videos_queued", 0) > 0:
                        total_processed += 1
                    else:
                        total_skipped += 1
                except Exception as e:
                    log.warning("Backfill ingest failed for note %s: %s", note.id, e)
                    total_failed += 1

        return {
            "ok": True,
            "processed": total_processed,
            "skipped": total_skipped,
            "failed": total_failed,
        }
    except Exception as e:
        log.exception("v4.1 backfill failed")
        return {"ok": False, "reason": str(e)}


@router.get("/video-tasks")
async def video_task_status():
    """Get video task processing status (v4.1 monitoring)."""
    from omnibox_agent.services.video_task_store import get_video_task_store
    from omnibox_agent.services.mcp_wrapper import get_mcp_daily_usage

    store = get_video_task_store()
    task_status = store.get_status()
    mcp_usage = get_mcp_daily_usage()

    return {
        "ok": True,
        "video_tasks": task_status,
        "mcp_daily_usage": mcp_usage,
    }
