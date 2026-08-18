"""SQL data layer: tenant-scoped content lookups + plain filtered counts.

v4.1 flow fix: retrieval is vector-driven (ChromaDB embedding). The legacy
FULLTEXT / LIKE / recent search channels and keyword-gated statistics were
removed from the flow — they produced counts inconsistent with the vector
recall (e.g. "共找到 0 条" while 58 items were shown). Only the functions
still used by the pipeline remain here.
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import text

from omnibox_agent.core.database import get_session

log = logging.getLogger(__name__)


def resolve_user_id(user_code: str) -> int | None:
    """Map a public user_code (string, e.g. 'AbXyZwQr1785932997') to the
    internal bigint users.id that platform_accounts / user_ai_config reference.

    The frontend identifies users by user_code (8-char random + timestamp);
    MySQL stores the internal integer id. This resolver bridges the two so the
    rest of the pipeline can keep treating user_id as the public user_code.
    Returns None if the code is unknown.
    """
    if not user_code:
        return None
    session = get_session()
    try:
        row = session.execute(
            text("SELECT id FROM users WHERE user_code = :code"),
            {"code": user_code},
        ).fetchone()
        return row[0] if row else None
    except Exception as e:
        log.warning("resolve_user_id failed for %r: %r", user_code, e)
        return None
    finally:
        session.close()


def get_account_ids(user_id: str) -> list[str]:
    """Return the platform_accounts.id values (as strings) belonging to the
    user identified by user_code. These are what content_items.account_id
    references for tenant-scoped MySQL retrieval.

    user_id is the public user_code; it is resolved to the internal bigint id
    before querying platform_accounts.user_id (which is a bigint column).
    """
    internal = resolve_user_id(user_id)
    if internal is None:
        return []
    session = get_session()
    try:
        result = session.execute(
            text("SELECT id FROM platform_accounts WHERE user_id = :uid"),
            {"uid": internal},
        )
        return [str(row[0]) for row in result.fetchall()]
    finally:
        session.close()


def get_content_by_ids(content_ids: list[int], account_ids: list[str]) -> list[dict]:
    if not content_ids:
        return []
    session = get_session()
    try:
        result = session.execute(
            text("""
                SELECT c.id, c.title, c.summary, c.platform, c.platform_name,
                       c.author_name, c.cover, c.original_url, c.collected_at,
                       c.updated_at, c.ai_tag
                FROM content_items c
                WHERE c.id IN :ids AND c.account_id IN :aids
            """),
            {"ids": tuple(content_ids), "aids": tuple(account_ids)},
        )
        return [dict(row._mapping) for row in result.fetchall()]
    finally:
        session.close()


def count_with_filters(
    account_ids: list[str],
    time_start: datetime | None = None,
    time_end: datetime | None = None,
    platform: str | None = None,
    favorite_only: bool = True,
) -> int:
    """Plain filtered COUNT of the user's content — NO keywords, NO FULLTEXT,
    NO LIKE. Used for COUNT-intent queries so "how many" stays accurate and
    consistent with the vector recall instead of keyword-gated at 0.

    favorite_only 参数已废弃：库内内容即收藏（is_favorite 字段已从表删除）。
    """
    session = get_session()
    try:
        conditions = ["c.account_id IN :aids"]
        params: dict[str, Any] = {"aids": tuple(account_ids)}

        if time_start:
            conditions.append("c.collected_at >= :ts")
            params["ts"] = time_start
        if time_end:
            conditions.append("c.collected_at <= :te")
            params["te"] = time_end
        if platform:
            conditions.append("c.platform = :plat")
            params["plat"] = platform

        where_clause = " AND ".join(conditions)
        sql = f"SELECT COUNT(DISTINCT c.id) FROM content_items c WHERE {where_clause}"
        result = session.execute(text(sql), params)
        return result.scalar() or 0
    except Exception as e:
        log.warning("Count query failed: %r", e)
        return 0
    finally:
        session.close()


def get_comments_for_content_ids(content_ids: list[int]) -> dict[int, list[str]]:
    """Fetch flattened comment text for multiple content items.

    Used by the DAG creative solver to enrich per-item context so the
    sub-agent LLM can pull supplementary information (location, price,
    tips) from comments when the main summary lacks it.

    Returns:
        Dict of content_item_id -> list of comment text strings (each
        truncated to 300 chars, sorted by create_time ascending).
    """
    if not content_ids:
        return {}
    session = get_session()
    try:
        result = session.execute(
            text("""
                SELECT content_item_id, content
                FROM content_comments
                WHERE content_item_id IN :cids
                ORDER BY content_item_id, create_time ASC
            """),
            {"cids": tuple(content_ids)},
        )
        comments: dict[int, list[str]] = {}
        for row in result.fetchall():
            cid = row[0]
            # 用 comment_text 而非 text：避免遮蔽上方 sqlalchemy.text()，
            # 否则 text("""SQL""") 在此函数内被当作局部变量 → UnboundLocalError
            comment_text = (row[1] or "").strip()
            if comment_text:
                comments.setdefault(cid, []).append(comment_text[:300])
        return comments
    finally:
        session.close()


def get_all_content_for_user(account_ids: list[str], favorite_only: bool = True) -> list[dict]:
    """库内内容即收藏（favorite_only 参数已废弃，is_favorite 列已删除）。"""
    session = get_session()
    try:
        conditions = ["c.account_id IN :aids"]
        params = {"aids": tuple(account_ids)}
        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT c.id, c.title, c.summary, c.platform, c.platform_name,
                   c.author_name, c.cover, c.original_url, c.collected_at,
                   c.updated_at, c.ai_tag, c.account_id
            FROM content_items c WHERE {where_clause}
        """
        result = session.execute(text(sql), params)
        return [dict(row._mapping) for row in result.fetchall()]
    finally:
        session.close()
