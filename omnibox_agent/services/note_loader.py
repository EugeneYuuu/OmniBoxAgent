"""v4.1 NoteRecord loader — bridges MySQL content_items/content_images to NoteRecord.

Reads from the existing OmniHub_server MySQL database (read-only):
  - content_items: id, title, summary, platform, ..., video_url, ai_tag
  - content_images: id, content_item_id, image_url, sort_order

Maps to NoteRecord for the v4.1 ingestion pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from sqlalchemy import text

from omnibox_agent.core.database import get_session
from omnibox_agent.models.note import NoteRecord, ImageRef, VideoRef, CommentRef, MediaStatus

log = logging.getLogger(__name__)


def load_note_by_id(note_id: str | int) -> NoteRecord | None:
    """Load a single note from MySQL by content_items.id.

    Returns a NoteRecord with images from content_images and video from video_url.
    Returns None if the content item doesn't exist.
    """
    session = get_session()
    try:
        # Load content item
        row = session.execute(
            text("""
                SELECT c.id, c.title, c.summary, c.platform, c.platform_name,
                       c.author_name, c.cover, c.original_url, c.video_url,
                       c.collected_at, c.updated_at, c.ai_tag,
                       c.account_id, a.user_id, u.user_code
                FROM content_items c
                JOIN platform_accounts a ON c.account_id = a.id
                JOIN users u ON a.user_id = u.id
                WHERE c.id = :cid
            """),
            {"cid": int(note_id)},
        ).fetchone()

        if row is None:
            return None

        data = dict(row._mapping)

        # Load images
        img_rows = session.execute(
            text("""
                SELECT id, image_url, sort_order
                FROM content_images
                WHERE content_item_id = :cid
                ORDER BY sort_order ASC
            """),
            {"cid": int(note_id)},
        ).fetchall()

        images = []
        for ir in img_rows:
            img_data = dict(ir._mapping)
            cdn_hash = _compute_cdn_hash(img_data["image_url"])
            images.append(ImageRef(
                id=str(img_data["id"]),
                url=img_data["image_url"],
                note_id=str(note_id),
                cdn_hash=cdn_hash,
            ))

        # 封面图兜底：多数同步内容没有 content_images（feed 卡片不返回图片列表），
        # 但 cover 封面通常含关键视觉信息（如菜品/店名/文字）。有 content_images
        # 时不重复加 cover（第一张图通常就是封面）；无图时把 cover 作为唯一图片源。
        cover_url = (data.get("cover") or "").strip()
        if cover_url and not images:
            images.append(ImageRef(
                id=f"cover_{note_id}",
                url=cover_url,
                note_id=str(note_id),
                cdn_hash=_compute_cdn_hash(cover_url),
            ))

        # Build video ref if video_url exists
        videos = []
        if data.get("video_url"):
            videos.append(VideoRef(
                id=f"video_{note_id}",
                url=data["video_url"],
                note_id=str(note_id),
            ))

        # Load comments from content_comments (ParseVideoLink 解析的评论)
        comments = _load_comments(int(note_id), session)

        # Parse ai_tag (JSON array string → comma-separated)
        tags = data.get("ai_tag") or ""
        if tags:
            try:
                tag_list = json.loads(tags)
                if isinstance(tag_list, list):
                    tags = ",".join(tag_list)
            except (json.JSONDecodeError, TypeError):
                pass

        return NoteRecord(
            id=str(data["id"]),
            # Public user_code (string) for tenant isolation in ChromaDB,
            # matching what vector_search filters on. Fall back to the internal
            # id if the JOIN to users somehow yields no user_code.
            user_id=data.get("user_code") or str(data.get("user_id", "")),
            account_ids=[str(data.get("account_id", ""))],
            title=data.get("title") or "",
            content=data.get("summary") or "",      # summary is the main content text
            summary=data.get("summary") or "",
            platform=data.get("platform") or "",
            platform_name=data.get("platform_name") or "",
            author_name=data.get("author_name") or "",
            cover_url=data.get("cover") or "",
            original_url=data.get("original_url") or "",
            collected_at=str(data.get("collected_at")) if data.get("collected_at") else "",
            tags=tags,
            images=images,
            videos=videos,
            comments=comments,
        )
    except Exception as e:
        log.error("load_note_by_id failed for %s: %s", note_id, e)
        return None
    finally:
        session.close()


def load_notes_for_user(user_id: str, account_ids: list[str]) -> list[NoteRecord]:
    """Load all notes for a user (used by backfill).

    Returns list of NoteRecord objects.
    """
    if not account_ids:
        return []

    session = get_session()
    try:
        rows = session.execute(
            text("""
                SELECT c.id, c.title, c.summary, c.platform, c.platform_name,
                       c.author_name, c.cover, c.original_url, c.video_url,
                       c.collected_at, c.updated_at, c.ai_tag,
                       c.account_id
                FROM content_items c
                WHERE c.account_id IN :aids
                ORDER BY c.id ASC
            """),
            {"aids": tuple(account_ids)},
        ).fetchall()

        # Batch-load all comments for these notes (avoids N+1 queries)
        all_note_ids = [int(row._mapping["id"]) for row in rows]
        comments_map = _load_comments_batch(all_note_ids, session)

        notes = []
        for row in rows:
            data = dict(row._mapping)
            note_id = str(data["id"])

            # Load images for this note
            img_rows = session.execute(
                text("""
                    SELECT id, image_url FROM content_images
                    WHERE content_item_id = :cid ORDER BY sort_order ASC
                """),
                {"cid": data["id"]},
            ).fetchall()

            images = []
            for ir in img_rows:
                img_data = dict(ir._mapping)
                cdn_hash = _compute_cdn_hash(img_data["image_url"])
                images.append(ImageRef(
                    id=str(img_data["id"]),
                    url=img_data["image_url"],
                    note_id=note_id,
                    cdn_hash=cdn_hash,
                ))

            # 封面图兜底（同 load_note_by_id）：无 content_images 时用 cover 作图片源
            cover_url = (data.get("cover") or "").strip()
            if cover_url and not images:
                images.append(ImageRef(
                    id=f"cover_{note_id}",
                    url=cover_url,
                    note_id=note_id,
                    cdn_hash=_compute_cdn_hash(cover_url),
                ))

            videos = []
            if data.get("video_url"):
                videos.append(VideoRef(
                    id=f"video_{note_id}",
                    url=data["video_url"],
                    note_id=note_id,
                ))

            tags = data.get("ai_tag") or ""
            if tags:
                try:
                    tag_list = json.loads(tags)
                    if isinstance(tag_list, list):
                        tags = ",".join(tag_list)
                except (json.JSONDecodeError, TypeError):
                    pass

            notes.append(NoteRecord(
                id=note_id,
                user_id=str(user_id),
                account_ids=[str(data.get("account_id", ""))],
                title=data.get("title") or "",
                content=data.get("summary") or "",
                summary=data.get("summary") or "",
                platform=data.get("platform") or "",
                platform_name=data.get("platform_name") or "",
                author_name=data.get("author_name") or "",
                cover_url=data.get("cover") or "",
                original_url=data.get("original_url") or "",
                collected_at=str(data.get("collected_at")) if data.get("collected_at") else "",
                tags=tags,
                images=images,
                videos=videos,
                comments=comments_map.get(data["id"], []),
            ))

        return notes
    except Exception as e:
        log.error("load_notes_for_user failed: %s", e)
        return []
    finally:
        session.close()


def _load_comments(note_id: int, session) -> list[CommentRef]:
    """Load comments for a single note from content_comments table.

    Only loads top-level comments (parent_comment_id IS NULL) with their basic
    info. Sub-comments are not loaded individually — they will be part of the
    content text if the platform returns them nested.
    """
    try:
        rows = session.execute(
            text("""
                SELECT comment_id, content, author_name, like_count, create_time
                FROM content_comments
                WHERE content_item_id = :cid AND parent_comment_id IS NULL
                ORDER BY sort_order ASC
                LIMIT 50
            """),
            {"cid": note_id},
        ).fetchall()

        comments = []
        for row in rows:
            d = dict(row._mapping)
            comments.append(CommentRef(
                comment_id=str(d.get("comment_id", "")),
                content=d.get("content") or "",
                author_name=d.get("author_name") or "",
                like_count=d.get("like_count") or 0,
                create_time=d.get("create_time") or 0,
            ))
        return comments
    except Exception as e:
        log.debug("_load_comments failed for note %s (table may not exist yet): %s", note_id, e)
        return []


def _load_comments_batch(note_ids: list[int], session) -> dict[int, list[CommentRef]]:
    """Batch-load comments for multiple notes.

    Returns dict of note_id -> list of CommentRef (only top-level comments).
    """
    if not note_ids:
        return {}
    try:
        rows = session.execute(
            text("""
                SELECT content_item_id, comment_id, content, author_name, like_count, create_time
                FROM content_comments
                WHERE content_item_id IN :cids AND parent_comment_id IS NULL
                ORDER BY content_item_id, sort_order ASC
                LIMIT 1000
            """),
            {"cids": tuple(note_ids)},
        ).fetchall()

        result: dict[int, list[CommentRef]] = {}
        for row in rows:
            d = dict(row._mapping)
            cid = d["content_item_id"]
            if cid not in result:
                result[cid] = []
            if len(result[cid]) < 50:
                result[cid].append(CommentRef(
                    comment_id=str(d.get("comment_id", "")),
                    content=d.get("content") or "",
                    author_name=d.get("author_name") or "",
                    like_count=d.get("like_count") or 0,
                    create_time=d.get("create_time") or 0,
                ))
        return result
    except Exception as e:
        log.debug("_load_comments_batch failed (table may not exist yet): %s", e)
        return {}


def _compute_cdn_hash(url: str) -> str:
    """Compute a hash from image URL for cross-note dedup.

    Uses the URL path (ignoring query params that may be CDN signatures)
    to identify the same underlying image across different notes.
    """
    # Strip query parameters (CDN signatures, tokens)
    path = url.split("?")[0].split("#")[0]
    return hashlib.md5(path.encode("utf-8")).hexdigest()
