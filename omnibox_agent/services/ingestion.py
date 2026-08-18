"""v4.1 Layer 0 Ingestion Pipeline.

Core functions:
  - store_vectors(): Multi-vector upsert (vec_main + vec_media) with fingerprint
  - reindex_note(): Read-modify-write critical section with per-note lock
  - parse_one_image(): Single image MCP call + summary
  - ingest_note(): Main entry point (concurrent image parse + video fire-and-forget)

Concurrency safety (v4.0 P1-4 fix):
  All three write paths share the same per-note lock:
    1. Ingestion sync path (ingest_note)
    2. Background video worker (video_enrichment_worker)
    3. Query-time on-demand re-parse (quality_gate fallback)
  The lock wraps the full load→mark→collect→store cycle in reindex_note.
  Expensive operations (summary LLM) happen OUTSIDE the lock.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Callable

import httpx

from omnibox_agent.core.config import get_config
from omnibox_agent.models.note import (
    ImageRef,
    VideoRef,
    NoteRecord,
    MediaStatus,
    compute_index_fingerprint,
    dedupe_by_cdn_hash,
)
from omnibox_agent.services.note_lock import get_note_lock
from omnibox_agent.services.mcp_wrapper import mcp_limited_call
from omnibox_agent.services.llm_service import summarize_if_long
from omnibox_agent.services.ai_config_store import get_user_ai_config
from omnibox_agent.services import chroma_store
from omnibox_agent.services.embedding_service import embed_text, embed_texts

log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────

def _note_main_id(note_id: str) -> str:
    """ChromaDB vector ID for the main vector."""
    return f"{note_id}#main"


def _note_media_id(note_id: str) -> str:
    """ChromaDB vector ID for the media vector."""
    return f"{note_id}#media"


def _note_comments_id(note_id: str) -> str:
    """ChromaDB vector ID for the comments vector."""
    return f"{note_id}#comments"


def _all_media_parsed(note: NoteRecord) -> bool:
    """Check if all media items are parsed (not pending)."""
    return note.all_media_parsed()


def _collect_parsed_media(note: NoteRecord) -> list[str]:
    """Collect all successfully parsed media texts."""
    return note.collect_parsed_media()


def _build_metadata(note: NoteRecord, vec_type: str) -> dict:
    """Build ChromaDB metadata for a vector.

    Metadata is shared between main and media vectors, differing only in vec_type.
    All fields needed for filtering at query time are included.
    """
    return {
        "note_id": note.id,
        "content_id": note.id,       # Backward compat
        # user_id is the user_code (str) for tenant isolation; stored as-is so
        # vector_search(user_id=<user_code>) tenant filtering matches ChromaDB.
        "user_id": note.user_id,
        "account_ids": ",".join(note.account_ids),
        "platform": note.platform,
        "platform_name": note.platform_name,
        "title": note.title[:200],
        "author_name": note.author_name,
        "cover_url": note.cover_url,
        "original_url": note.original_url,
        "collected_at": note.collected_at,
        # 数值时间戳（epoch 秒）：ChromaDB 的 $gte/$lte 仅支持数值比较，
        # collected_at 字符串过滤会直接抛错（时间窗查询全空的根因）。
        "collected_ts": _parse_collected_ts(note.collected_at),
        "tags": note.tags,
        "vec_type": vec_type,
        "has_image": bool(note.images),
        "has_video": bool(note.videos),
        "parsed": _all_media_parsed(note),
        "index_version": get_config().ingestion.index_version,
    }


def _parse_collected_ts(collected_at: str) -> int:
    """Parse collected_at string (CST naive datetime) to epoch seconds.

    Formats handled: "2026-08-17 12:00:16.653531" / "2026-08-17 12:00:16" /
    ISO variants. Returns 0 on failure (never raises — metadata write must
    not break ingestion).
    """
    if not collected_at:
        return 0
    from datetime import timezone, timedelta
    cst = timezone(timedelta(hours=8))
    s = str(collected_at).replace("T", " ").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            from datetime import datetime as _dt
            dt = _dt.strptime(s[:26 if ".%f" in fmt else len(s)], fmt)
            return int(dt.replace(tzinfo=cst).timestamp())
        except ValueError:
            continue
    return 0


# ── Multi-vector store ──────────────────────────────────────────────────

async def store_vectors(note: NoteRecord, media_text: str) -> None:
    """Multi-vector upsert: main text, media text, and comments are embedded separately.

    v4.1 Section 3.1 & 3.3:
      - vec_main = embed(title + content) — topic vector, not diluted by media/comments
      - vec_media = embed(media summaries) — multimedia entry point (optional)
      - vec_comments = embed(comments) — social context vector (optional)
      - Fingerprint covers main_text + media_text + comments_text + parsed state + INDEX_VERSION

    MUST be called within get_note_lock(note.id) — see reindex_note.
    """
    cfg = get_config()
    main_text = note.main_text
    comments_text = note.comments_text
    all_parsed = _all_media_parsed(note)

    # Build vectors to upsert (skip empty texts — prevents embedding API 400
    # on empty strings and Chroma upsert length mismatch).
    vectors_to_embed = []
    vector_specs = []

    if main_text.strip():
        vectors_to_embed.append(main_text)
        vector_specs.append({
            "id": _note_main_id(note.id),
            "text": main_text,
            "meta": _build_metadata(note, "main"),
        })

    if media_text:
        vectors_to_embed.append(media_text)
        vector_specs.append({
            "id": _note_media_id(note.id),
            "text": media_text,
            "meta": _build_metadata(note, "media"),
        })

    if comments_text:
        vectors_to_embed.append(comments_text)
        vector_specs.append({
            "id": _note_comments_id(note.id),
            "text": comments_text,
            "meta": _build_metadata(note, "comments"),
        })

    if not vectors_to_embed:
        log.warning("store_vectors: note=%s has no content to embed, skipping", note.id)
        return

    # Batch embed (more efficient than sequential)
    embeddings = embed_texts(vectors_to_embed)

    # Compute fingerprint BEFORE upsert so it goes into the single batch call.
    fp = compute_index_fingerprint(
        main_text=main_text,
        media_text=media_text,
        comments_text=comments_text,
        all_parsed=all_parsed,
        index_version=cfg.ingestion.index_version,
        salt=cfg.fingerprint_salt,
    )

    # Inject fingerprint into the main vector's metadata (single upsert).
    # Modify vector_specs[0]["meta"] in place so the upsert below writes
    # everything (vectors + fingerprint) once.
    vector_specs[0]["meta"] = dict(vector_specs[0]["meta"], fingerprint=fp)

    # Upsert each vector — handle partial embedding failures explicitly
    # so no vector spec is silently skipped and lengths stay aligned
    # (the old zip() truncated to the shortest list; a bare filter on
    # embeddings desynced ids vs embeddings → Chroma "Unequal lengths").
    if len(embeddings) == len(vector_specs):
        ids, embs, metas, docs = [], [], [], []
        for spec, emb in zip(vector_specs, embeddings):
            if emb:
                ids.append(spec["id"])
                embs.append(emb)
                metas.append(spec["meta"])
                docs.append(spec["text"])
            else:
                log.warning(
                    "store_vectors: embedding failed for %s, vector not upserted",
                    spec["id"],
                )
        if ids:
            chroma_store.upsert_vectors(ids, embs, metas, docs)
    else:
        # Partial batch failure — upsert individually, warn on failed embeddings
        failed_count = 0
        for i, spec in enumerate(vector_specs):
            emb = embeddings[i] if i < len(embeddings) else None
            if emb:
                chroma_store.upsert_vectors(
                    [spec["id"]], [emb], [spec["meta"]], [spec["text"]]
                )
            else:
                failed_count += 1
                log.warning(
                    "store_vectors: embedding failed for %s (index %d/%d), "
                    "vector not upserted",
                    spec["id"], i, len(vector_specs),
                )
        if failed_count:
            log.warning(
                "store_vectors: note=%s, %d/%d embeddings failed",
                note.id, failed_count, len(vector_specs),
            )

    log.debug("store_vectors: note=%s, vectors=%d, fp=%s, parsed=%s",
              note.id, len(vector_specs), fp, all_parsed)

    # Also upsert BM25 text in MySQL (if we have write access)
    # Note: Currently MySQL is read-only from OmniBoxAgent's perspective.
    # BM25 search uses the existing FULLTEXT index on content_items table,
    # which is maintained by OmniHub_server. The media_text augmentation
    # would require either a new table or an API to OmniHub_server.
    # For now, BM25 operates on original content only — media enrichment
    # benefits vector search primarily.


# ── Reindex critical section ────────────────────────────────────────────

async def reindex_note(
    note_id: str,
    new_media: list[str],
    mark_done: Callable[[NoteRecord], None] | None = None,
    note_loader: Callable[[str], NoteRecord | None] | None = None,
) -> None:
    """Read-modify-write critical section: load → mark → collect → store.

    v4.1 Section 3.1:
      - Lock acquired INSIDE this function (per-note lock, shared by all write paths)
      - Expensive operations (summary LLM) must be done OUTSIDE — pass results in new_media
      - mark_done callback flips parsed state BEFORE collect (fingerprint depends on it)

    Args:
        note_id: The note's ID
        new_media: List of new media summary texts to add (from outside the lock)
        mark_done: Callback to mark media as parsed (e.g., mark_video_parsed)
        note_loader: Function to load the NoteRecord from storage
    """
    async with get_note_lock(note_id):
        # Load latest note state inside the lock
        if note_loader:
            note = note_loader(note_id)
        else:
            note = _default_note_loader(note_id)

        if note is None:
            log.warning("reindex_note: note %s not found", note_id)
            return

        # Mark parsed state first (fingerprint depends on it)
        if mark_done:
            mark_done(note)

        # Collect all parsed media + new media
        existing_media = _collect_parsed_media(note)
        all_media = existing_media + new_media
        media_text = "\n".join(all_media) if all_media else ""

        await store_vectors(note, media_text)


def _default_note_loader(note_id: str) -> NoteRecord | None:
    """Default note loader — loads from MySQL via note_loader.

    Called inside the async lock in reindex_note. Should be fast (just a DB
    read). Uses the shared note_loader module that queries content_items +
    content_images + content_comments.
    """
    from omnibox_agent.services.note_loader import load_note_by_id

    try:
        return load_note_by_id(note_id)
    except Exception as e:
        log.warning("Default note loader failed for %s: %s", note_id, e)
        return None


# ── Image parsing ───────────────────────────────────────────────────────

async def parse_one_image(
    img: ImageRef, anchor: str, ai_config: dict | None = None
) -> str | None:
    """Parse a single image via vision-mcp-server + summarize if long.

    v4.1 Section 3.1:
      - MCP call to vision-mcp-server__analyze_image (param name: `image`)
      - On success: mark parsed, summarize if >300 chars, return text
      - On failure: mark failed, return None (don't block the batch)

    Download-403 fallback: 小红书等 CDN 对 vision MCP 的直接下载常返回 403
    (反爬/签名)。先尝试直传 URL（快路径）；若 MCP 返回下载失败/图片错误，
    则用带 Referer/UA 的 httpx 下载图片并转为 base64 data URI 重试——让
    403 图片也能被视觉模型解析。

    Args:
        img: ImageRef with URL and note context
        anchor: Note title (semantic anchor for summary)
        ai_config: Owner's AI config (modelName/baseUrl/apiKey) so the summary
            runs under the note owner's key, not the evaluator/Zhipu default.

    Returns:
        Summary text (or raw text if short), or None on failure
    """
    try:
        raw = await mcp_limited_call(
            "vision-mcp-server__analyze_image",
            image=img.url,   # vision-mcp-server 参数名是 image（不是 url）
            focus=anchor,
        )
        if raw and not raw.startswith("["):
            img.mark_parsed(raw)
            summary = await summarize_if_long(raw, anchor, ai_config=ai_config)
            return summary

        # MCP returned an error (likely IMAGE_DOWNLOAD_FAILED 403)
        log.warning("parse_one_image: direct URL failed for %s: %s", img.url, str(raw)[:100])
        data_uri = await _download_image_as_data_uri(img.url)
        if not data_uri:
            log.warning("parse_one_image: image download fallback failed for %s", img.url)
            img.mark_failed()
            return None

        raw2 = await mcp_limited_call(
            "vision-mcp-server__analyze_image",
            image=data_uri,
            focus=anchor,
        )
        if not raw2 or raw2.startswith("["):
            log.warning("parse_one_image: base64 retry failed for %s: %s", img.url, str(raw2)[:100])
            img.mark_failed()
            return None

        img.mark_parsed(raw2)
        summary = await summarize_if_long(raw2, anchor, ai_config=ai_config)
        return summary
    except Exception as e:
        log.warning("parse_one_image failed for %s: %s", img.url, e)
        img.mark_failed()
        return None


def _download_image_as_data_uri(url: str, max_bytes: int = 5 * 1024 * 1024) -> str | None:
    """Download an image (with browser-like headers) and return a base64 data URI.

    Some CDNs (e.g. xiaohongshu sns-webpic) block plain downloads (HTTP 403).
    Sending Referer + a mobile UA often bypasses the anti-scraping check.
    Returns None on failure or if the response is not an image.
    """
    import base64
    import mimetypes

    try:
        resp = httpx.get(
            url,
            timeout=20.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
                ),
                "Referer": "https://www.xiaohongshu.com/",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            },
        )
        if resp.status_code != 200 or not resp.content:
            return None

        content_type = resp.headers.get("Content-Type", "")
        if content_type and not content_type.startswith("image/"):
            return None
        if len(resp.content) > max_bytes:
            log.warning("parse_one_image: image too large (%d bytes), skip", len(resp.content))
            return None

        ext = mimetypes.guess_extension(content_type.split(";")[0]) or ".jpg"
        mime = content_type.split(";")[0] or "image/jpeg"
        b64 = base64.b64encode(resp.content).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        log.warning("parse_one_image: download failed for %s: %s", url, e)
        return None


# ── Main ingestion entry point ──────────────────────────────────────────

async def ingest_note(note: NoteRecord) -> dict[str, Any]:
    """Main ingestion entry point.

    v4.1 Section 3.1 flow:
      1. Parse images concurrently (gather + semaphore=3, single failure doesn't block)
      2. Store vectors synchronously (main + image summaries) — note immediately searchable
      3. For videos: create async tasks (fire-and-forget), don't block ingestion
      4. Video results picked up by background worker later

    Args:
        note: NoteRecord with title, content, images, videos

    Returns:
        Dict with status info: {"note_id": ..., "images_parsed": N, "videos_queued": M}
    """
    log.info("Ingesting note %s: %d images, %d videos", note.id, len(note.images), len(note.videos))

    # Resolve the note owner's AI config ONCE so image/video summarization
    # runs under the user's own key (only embedding uses Zhipu).
    owner_cfg = get_user_ai_config(note.user_id) if note.user_id else None

    # 1. Parse images concurrently (dedup by CDN hash first)
    deduped_images = dedupe_by_cdn_hash(note.images)
    if deduped_images:
        results = await asyncio.gather(*[
            parse_one_image(img, note.title, ai_config=owner_cfg) for img in deduped_images
        ])
        image_media = [r for r in results if r]
    else:
        image_media = []

    # Propagate parsed status back to original images (dedup may have merged)
    # Note: dedupe_by_cdn_hash already propagates status to duplicates

    # 2. Synchronous path: original text + image summaries → immediate indexing
    media_text = "\n".join(image_media) if image_media else ""
    async with get_note_lock(note.id):
        await store_vectors(note, media_text)

    log.info("Note %s indexed: %d/%d images parsed, media_text=%d chars",
             note.id, len(image_media), len(note.images), len(media_text))

    # 3. Video: async task, fire-and-forget
    video_tasks_created = 0
    if note.videos:
        from omnibox_agent.services.video_task_store import get_video_task_store
        store = get_video_task_store()

        for vid in note.videos:
            try:
                result = await mcp_limited_call(
                    "ai-video-notes__ai_notes_create_task",
                    video_url=vid.url,
                )
                # Parse task_id from result
                task_id = _parse_task_id(result)
                if task_id:
                    vid.task_id = task_id
                    store.save_task(
                        note_id=note.id,
                        video_id=vid.id,
                        task_id=task_id,
                        note_title=note.title,
                    )
                    video_tasks_created += 1
                    # 生产者 → 消费者：新任务入队，串行消费
                    try:
                        from omnibox_agent.services.video_worker import get_video_worker
                        get_video_worker().enqueue(task_id)
                    except Exception as e:
                        log.debug("video_worker enqueue failed (non-fatal): %s", e)
            except Exception as e:
                log.warning("Failed to create video task for note %s video %s: %s",
                           note.id, vid.id, e)
                vid.mark_failed()

    return {
        "note_id": note.id,
        "images_parsed": len(image_media),
        "images_total": len(note.images),
        "videos_queued": video_tasks_created,
        "videos_total": len(note.videos),
    }


def _parse_task_id(result: str) -> str | None:
    """Extract task_id from MCP tool result string."""
    import json
    try:
        data = json.loads(result)
        if isinstance(data, dict):
            # Direct match at top level
            tid = data.get("task_id") or data.get("taskId") or data.get("id")
            if tid:
                return tid
            # Nested in data object (e.g. {"success": true, "data": {"task_id": "..."}})
            inner = data.get("data")
            if isinstance(inner, dict):
                return inner.get("task_id") or inner.get("taskId") or inner.get("id")
    except (json.JSONDecodeError, TypeError):
        # Maybe it's just a plain string ID
        result = result.strip()
        if result and not result.startswith("[") and len(result) < 100:
            return result
    return None


# ── Query-time on-demand image re-parse ─────────────────────────────────

async def parse_and_reindex(
    notes: list[NoteRecord],
    max_images: int = 6,
) -> int:
    """Query-time fallback: parse unparsed images and reindex.

    v4.1 Section 5.2:
      - Only images (video is async, can't wait in query path)
      - Shared per-note lock via reindex_note (concurrent-safe with worker)
      - Image count cap (default 6, semaphore=3 → ≤2 batches)
      - Results written back to index (closed loop, not just in-context)

    Args:
        notes: Notes with unparsed images
        max_images: Maximum images to parse

    Returns:
        Number of images successfully parsed
    """
    # Collect unparsed images, capped
    all_images: list[ImageRef] = []
    for note in notes:
        all_images.extend(note.unparsed_images())
    all_images = all_images[:max_images]

    if not all_images:
        return 0

    # Resolve each note owner's AI config so image summarization uses their
    # own key (only embedding uses Zhipu). Bounded by max_images (~6).
    notes_by_id = {n.id: n for n in notes}
    owner_cfgs: dict[str, dict | None] = {}
    for img in all_images:
        note = notes_by_id.get(img.note_id)
        if note and note.user_id and img.note_id not in owner_cfgs:
            owner_cfgs[img.note_id] = get_user_ai_config(note.user_id)

    # Parse concurrently
    results = await asyncio.gather(*[
        parse_one_image(img, img.note_id, ai_config=owner_cfgs.get(img.note_id))
        for img in all_images
    ])

    # Group by note_id and reindex
    from collections import defaultdict
    by_note: dict[str, list[str]] = defaultdict(list)
    for img, result in zip(all_images, results):
        if result:
            by_note[img.note_id].append(result)

    for note_id, summaries in by_note.items():
        await reindex_note(note_id, summaries)

    return sum(1 for r in results if r)
