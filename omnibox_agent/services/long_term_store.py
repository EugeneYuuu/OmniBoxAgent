"""长期记忆存储层（MEMORY_HARNESS_INTEGRATION_DESIGN.md 第二部分 §10 / §13.1）。

三层存储：
  L1 用户画像    agent_user_profile（每用户一行：profile_json LLM 版 + stats_json SQL 版）
  L2 跨会话偏好  agent_user_memory 行式 + supersede 链（MySQL 权威源）
  L3 向量记忆    Chroma omnihub_user_memories（仅召回索引，与 L2 行双写同步）

设计要点：
  - 键约定（§10）：所有表与 Chroma metadata 的 user_id 一律存 user_code（与
    agent_session 同标准）；聚合收藏库必须走 resolve_user_id 桥接
    （user_code → users.id → platform_accounts → content_items.account_id）。
  - 用户级并发（§13.1）：per-user 进程内锁（仿 session_store）+ 关键计数原子 SQL
    ——去重防重 / supersede / hit_count 三处读-改-写不可复用 session 级锁。
  - MySQL 是权威源，Chroma 仅索引；删除/降权双写同步（§16）。
  - best-effort：任何失败只记日志返回 None/[]，不触碰会话树读写路径。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import bindparam, text

from omnibox_agent.core.database import get_session as db_session

log = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))

# L3 Chroma 集合名（§10.3）
USER_MEMORIES_COLLECTION = "omnihub_user_memories"

# 写入去重阈值（§11.4）：embedding 相似度 ≥ 0.85 视为重复
DEDUP_SIMILARITY = 0.85
# 召回过滤阈值（§12.1）：score ≥ 0.55
RECALL_MIN_SCORE = 0.55
# 衰减参数（§15）：情景记忆 30 天半衰期；低分且 90 天未命中软删；superseded 180 天物理清理
EPISODIC_HALF_LIFE_DAYS = 30.0
DECAY_UNHIT_DAYS = 90
SUPERSEDED_PURGE_DAYS = 180
DECAY_MIN_IMPORTANCE = 0.15

# 基础重要度（§15 importance = base(type) × recency_decay × log(1+hit_count)）
_BASE_IMPORTANCE = {"preference": 0.8, "fact": 0.7, "episodic": 0.5}

# 敏感信息黑名单（§11.5：规则正则二次校验，零成本兜底）
_SENSITIVE_PATTERNS = [
    r"\b\d{17}[\dXx]\b",                                # 身份证
    r"\b1[3-9]\d{9}\b",                                  # 手机号
    r"(住址|地址|门牌)\s*[:：]?\s*[\u4e00-\u9fff]{4,}",   # 住址
    r"(银行卡|卡号)\s*[:：]?\s*\d{8,}",                   # 银行卡
    r"(存款|余额|工资|负债)\s*[:：]?\s*\d+",               # 财务
    r"(患病|病历|诊断|高血压|糖尿病|抑郁症)",               # 健康状况
]
import re as _re
_SENSITIVE_RES = [_re.compile(p) for p in _SENSITIVE_PATTERNS]


def contains_sensitive_info(text: str) -> bool:
    """敏感信息命中检测（§11.5）。"""
    if not text:
        return False
    return any(r.search(text) for r in _SENSITIVE_RES)


# =====================================================================
# per-user 进程内锁（§13.1；仿 session_store.per_session_lock，含空闲清理）
# =====================================================================

_per_user_locks: dict[str, tuple[asyncio.Lock, float]] = {}
_USER_LOCK_IDLE_PRUNE_SECONDS = 600.0
_user_lock_op_counter = 0


def per_user_lock(user_id: str) -> asyncio.Lock:
    global _user_lock_op_counter
    _user_lock_op_counter += 1
    if _user_lock_op_counter % 64 == 0:
        _prune_user_locks()
    now = time.monotonic()
    entry = _per_user_locks.get(user_id)
    if entry is None:
        lock = asyncio.Lock()
        _per_user_locks[user_id] = (lock, now)
        return lock
    lock, _ts = entry
    _per_user_locks[user_id] = (lock, now)
    return lock


def _prune_user_locks() -> None:
    now = time.monotonic()
    stale = [uid for uid, (lock, ts) in _per_user_locks.items()
             if (now - ts) > _USER_LOCK_IDLE_PRUNE_SECONDS and not lock.locked()]
    for uid in stale:
        _per_user_locks.pop(uid, None)


# =====================================================================
# 同步核心（纯 DB / Chroma 操作；asyncio.to_thread 或 run_blocking 执行）
# =====================================================================

# ---- L1 画像（agent_user_profile） ----

def ensure_profile_row_sync(user_id: str) -> bool:
    """确保画像行存在（INSERT IGNORE 幂等；首次触达 / 轮次记账前调用）。"""
    s = db_session()
    try:
        s.execute(
            text("INSERT IGNORE INTO agent_user_profile (user_id, updated_at) VALUES (:uid, :now)"),
            {"uid": user_id, "now": datetime.now(CST)},
        )
        s.commit()
        return True
    except Exception as e:
        log.warning("ensure_profile_row failed (best-effort): %s", e)
        s.rollback()
        return False
    finally:
        s.close()


def get_profile_sync(user_id: str) -> dict | None:
    """读 L1 画像行。返回 {profile, stats, lt_round_count, last_lt_extract_at}；失败 None。"""
    s = db_session()
    try:
        row = s.execute(
            text("SELECT profile_json, stats_json, lt_round_count, last_lt_extract_at "
                 "FROM agent_user_profile WHERE user_id = :uid"),
            {"uid": user_id},
        ).fetchone()
        if row is None:
            return None
        def _parse(v):
            if not v:
                return None
            try:
                return json.loads(v) if isinstance(v, str) else v
            except (ValueError, TypeError):
                return None
        return {
            "profile": _parse(row[0]),
            "stats": _parse(row[1]),
            "lt_round_count": row[2] or 0,
            "last_lt_extract_at": str(row[3]) if row[3] else None,
        }
    except Exception as e:
        log.warning("get_profile failed (best-effort): %s", e)
        return None
    finally:
        s.close()


def upsert_profile_sync(user_id: str, profile: dict | None = None,
                        stats: dict | None = None) -> bool:
    """更新画像字段（profile_json / stats_json 至少一项）。"""
    if profile is None and stats is None:
        return False
    sets, params = ["updated_at = :now"], {"uid": user_id, "now": datetime.now(CST)}
    if profile is not None:
        sets.append("profile_json = :profile")
        params["profile"] = json.dumps(profile, ensure_ascii=False)
    if stats is not None:
        sets.append("stats_json = :stats")
        params["stats"] = json.dumps(stats, ensure_ascii=False)
    s = db_session()
    try:
        s.execute(
            text(f"UPDATE agent_user_profile SET {', '.join(sets)} WHERE user_id = :uid"),
            params,
        )
        s.commit()
        return True
    except Exception as e:
        log.warning("upsert_profile failed (best-effort): %s", e)
        s.rollback()
        return False
    finally:
        s.close()


def incr_lt_round_sync(user_id: str) -> int | None:
    """轻量提取轮次原子自增（§11.2 轮次记账）。返回自增后的值；失败 None。"""
    s = db_session()
    try:
        row = s.execute(
            text("UPDATE agent_user_profile SET lt_round_count = lt_round_count + 1, "
                 "updated_at = :now WHERE user_id = :uid"),
            {"uid": user_id, "now": datetime.now(CST)},
        )
        if row.rowcount == 0:
            s.rollback()
            return None
        s.commit()
        cur = s.execute(
            text("SELECT lt_round_count FROM agent_user_profile WHERE user_id = :uid"),
            {"uid": user_id},
        ).fetchone()
        return int(cur[0]) if cur else None
    except Exception as e:
        log.warning("incr_lt_round failed (best-effort): %s", e)
        s.rollback()
        return None
    finally:
        s.close()


def reset_lt_round_sync(user_id: str) -> bool:
    """触发轻量提取后清零 + 记录时间（§11.2）。"""
    s = db_session()
    try:
        s.execute(
            text("UPDATE agent_user_profile SET lt_round_count = 0, "
                 "last_lt_extract_at = :now, updated_at = :now WHERE user_id = :uid"),
            {"uid": user_id, "now": datetime.now(CST)},
        )
        s.commit()
        return True
    except Exception as e:
        log.warning("reset_lt_round failed (best-effort): %s", e)
        s.rollback()
        return False
    finally:
        s.close()


# ---- L2 记忆行（agent_user_memory） ----

def _row_to_memory(row: Any) -> dict:
    meta = None
    if row.meta:
        try:
            meta = json.loads(row.meta) if isinstance(row.meta, str) else row.meta
        except (ValueError, TypeError):
            meta = None
    return {
        "memory_id": row.memory_id,
        "user_id": row.user_id,
        "mem_type": row.mem_type,
        "content": row.content,
        "meta": meta or {},
        "status": row.status,
        "hit_count": row.hit_count or 0,
        "last_accessed_at": str(row.last_accessed_at) if row.last_accessed_at else None,
        "created_at": str(row.created_at) if row.created_at else None,
    }


def list_active_memories_sync(user_id: str, mem_type: str | None = None,
                              limit: int = 50) -> list[dict]:
    """列出用户 active 记忆（召回只取 active，§10.2）。"""
    s = db_session()
    try:
        sql = ("SELECT memory_id, user_id, mem_type, content, meta, status, hit_count, "
               "last_accessed_at, created_at FROM agent_user_memory "
               "WHERE user_id = :uid AND status = 'active'")
        params: dict = {"uid": user_id}
        if mem_type:
            sql += " AND mem_type = :mtype"
            params["mtype"] = mem_type
        sql += " ORDER BY id DESC LIMIT :lim"
        params["lim"] = limit
        rows = s.execute(text(sql), params).fetchall()
        return [_row_to_memory(r) for r in rows]
    except Exception as e:
        log.warning("list_active_memories failed (best-effort): %s", e)
        return []
    finally:
        s.close()


def list_memories_by_status_sync(user_id: str, statuses: tuple[str, ...] = ("active",),
                                 limit_per_status: int = 50) -> dict[str, list[dict]]:
    """按 status 分组列出用户记忆（admin 画像页：active/superseded/deleted，§5.2）。"""
    out: dict[str, list[dict]] = {st: [] for st in statuses}
    s = db_session()
    try:
        for st in statuses:
            rows = s.execute(
                text("SELECT memory_id, user_id, mem_type, content, meta, status, hit_count, "
                     "last_accessed_at, created_at FROM agent_user_memory "
                     "WHERE user_id = :uid AND status = :st "
                     "ORDER BY id DESC LIMIT :lim"),
                {"uid": user_id, "st": st, "lim": limit_per_status},
            ).fetchall()
            out[st] = [_row_to_memory(r) for r in rows]
        return out
    except Exception as e:
        log.warning("list_memories_by_status failed (best-effort): %s", e)
        return out
    finally:
        s.close()


def search_profiles_sync(query: str = "", limit: int = 20) -> list[dict]:
    """画像列表检索（admin 平台用户列表页）：user_id 前缀匹配 + updated_at 倒序。"""
    s = db_session()
    try:
        sql = ("SELECT user_id, profile_json, stats_json, lt_round_count, updated_at "
               "FROM agent_user_profile")
        params: dict = {}
        if query:
            sql += " WHERE user_id LIKE :q"
            params["q"] = f"{query}%"
        sql += " ORDER BY updated_at DESC LIMIT :lim"
        params["lim"] = max(1, min(int(limit), 20))
        rows = s.execute(text(sql), params).fetchall()
        out: list[dict] = []
        for r in rows:
            out.append({"user_id": r[0], "lt_round_count": r[3] or 0,
                        "updated_at": str(r[4]) if r[4] else None,
                        "has_profile": bool(r[1]), "has_stats": bool(r[2])})
        return out
    except Exception as e:
        log.warning("search_profiles failed (best-effort): %s", e)
        return []
    finally:
        s.close()


def get_memory_sync(user_id: str, memory_id: str) -> dict | None:
    """读单条记忆（归属校验：user_id 必须匹配，§14.3）。"""
    s = db_session()
    try:
        row = s.execute(
            text("SELECT memory_id, user_id, mem_type, content, meta, status, hit_count, "
                 "last_accessed_at, created_at FROM agent_user_memory "
                 "WHERE user_id = :uid AND memory_id = :mid"),
            {"uid": user_id, "mid": memory_id},
        ).fetchone()
        return _row_to_memory(row) if row else None
    except Exception as e:
        log.warning("get_memory failed (best-effort): %s", e)
        return None
    finally:
        s.close()


def insert_memory_sync(user_id: str, mem_type: str, content: str,
                       meta: dict | None = None,
                       memory_id: str | None = None) -> str | None:
    """INSERT 一条记忆行。返回 memory_id；失败 None。"""
    mid = memory_id or uuid.uuid4().hex
    s = db_session()
    try:
        s.execute(
            text(
                "INSERT INTO agent_user_memory "
                "(memory_id, user_id, mem_type, content, meta, status, created_at) "
                "VALUES (:mid, :uid, :mtype, :content, :meta, 'active', :now)"
            ),
            {"mid": mid, "uid": user_id, "mtype": mem_type, "content": content,
             "meta": json.dumps(meta or {}, ensure_ascii=False), "now": datetime.now(CST)},
        )
        s.commit()
        return mid
    except Exception as e:
        log.warning("insert_memory failed (best-effort): %s", e)
        s.rollback()
        return None
    finally:
        s.close()


def supersede_by_key_sync(user_id: str, key: str, new_memory_id: str) -> int:
    """同 key 旧偏好 → superseded（meta 记 superseded_by + superseded_at，不物理删，§10.2）。

    superseded_at 供 180 天物理清理判龄（按**被替代时间**而非创建时间——
    创建早于 180 天的老偏好若刚被替代即删，审计链会立即丢失）。
    返回受影响行数；无 key 偏好体系外调用（如 episodic）返回 0。
    """
    s = db_session()
    try:
        res = s.execute(
            text(
                "UPDATE agent_user_memory SET status = 'superseded', "
                "meta = JSON_SET(COALESCE(meta, JSON_OBJECT()), "
                "'$.superseded_by', :new_id, '$.superseded_at', :sup_at) "
                "WHERE user_id = :uid AND mem_type = 'preference' AND status = 'active' "
                "AND JSON_UNQUOTE(JSON_EXTRACT(meta, '$.key')) = :key"
            ),
            {"uid": user_id, "key": key, "new_id": new_memory_id,
             "sup_at": datetime.now(CST).isoformat()},
        )
        s.commit()
        return res.rowcount or 0
    except Exception as e:
        log.warning("supersede_by_key failed (best-effort): %s", e)
        s.rollback()
        return 0
    finally:
        s.close()


def find_active_preference_sync(user_id: str, key: str) -> dict | None:
    """按 key 找当前 active 偏好（冲突/升信判定用）。"""
    s = db_session()
    try:
        row = s.execute(
            text("SELECT memory_id, user_id, mem_type, content, meta, status, hit_count, "
                 "last_accessed_at, created_at FROM agent_user_memory "
                 "WHERE user_id = :uid AND mem_type = 'preference' AND status = 'active' "
                 "AND JSON_UNQUOTE(JSON_EXTRACT(meta, '$.key')) = :key LIMIT 1"),
            {"uid": user_id, "key": key},
        ).fetchone()
        return _row_to_memory(row) if row else None
    except Exception as e:
        log.warning("find_active_preference failed (best-effort): %s", e)
        return None
    finally:
        s.close()


def bump_confidence_sync(user_id: str, memory_id: str,
                         new_confidence: float) -> bool:
    """重复出现升信（0.7 → 0.85 → 0.95 封顶，§11.4）+ hit_count 原子 +1。"""
    s = db_session()
    try:
        s.execute(
            text(
                "UPDATE agent_user_memory SET "
                "meta = JSON_SET(COALESCE(meta, JSON_OBJECT()), '$.confidence', :conf), "
                "hit_count = hit_count + 1, last_accessed_at = :now "
                "WHERE user_id = :uid AND memory_id = :mid AND status = 'active'"
            ),
            {"uid": user_id, "mid": memory_id, "conf": new_confidence,
             "now": datetime.now(CST)},
        )
        s.commit()
        return True
    except Exception as e:
        log.warning("bump_confidence failed (best-effort): %s", e)
        s.rollback()
        return False
    finally:
        s.close()


def bump_hit_sync(user_id: str, memory_id: str) -> None:
    """召回命中强化：hit_count+1 与 last_accessed_at 刷新（原子 SQL，§12.1）。

    TODO(QPS 增长后)：改为批量/采样回写。
    """
    s = db_session()
    try:
        s.execute(
            text("UPDATE agent_user_memory SET hit_count = hit_count + 1, "
                 "last_accessed_at = :now WHERE user_id = :uid AND memory_id = :mid"),
            {"uid": user_id, "mid": memory_id, "now": datetime.now(CST)},
        )
        s.commit()
    except Exception as e:
        log.warning("bump_hit failed (best-effort): %s", e)
        s.rollback()
    finally:
        s.close()


def soft_delete_memory_sync(user_id: str, memory_id: str) -> bool:
    """用户自管理软删（§14.3）：MySQL status + Chroma 同步。"""
    s = db_session()
    try:
        res = s.execute(
            text("UPDATE agent_user_memory SET status = 'deleted' "
                 "WHERE user_id = :uid AND memory_id = :mid"),
            {"uid": user_id, "mid": memory_id},
        )
        s.commit()
        ok = (res.rowcount or 0) > 0
        if ok:
            try:
                from omnibox_agent.services.chroma_store import (
                    get_named_collection, USER_MEMORIES_COLLECTION)
                get_named_collection(USER_MEMORIES_COLLECTION).delete(ids=[memory_id])
            except Exception as e:
                log.warning("chroma delete on soft-delete failed (will re-sync): %s", e)
        return ok
    except Exception as e:
        log.warning("soft_delete_memory failed (best-effort): %s", e)
        s.rollback()
        return False
    finally:
        s.close()


def downgrade_by_session_sync(session_id: str, user_id: str) -> int:
    """会话树删除级联：源自该会话的情景记忆**降权**而非删除（§15）。

    按 meta->>'$.source_session_id' 匹配（JSON 路径查询）；不改写
    session_store.delete_session_tree 本体（存储层边界）。
    """
    s = db_session()
    try:
        res = s.execute(
            text(
                "UPDATE agent_user_memory SET "
                "meta = JSON_SET(COALESCE(meta, JSON_OBJECT()), '$.importance', 0.25) "
                "WHERE user_id = :uid AND mem_type = 'episodic' AND status = 'active' "
                "AND JSON_UNQUOTE(JSON_EXTRACT(meta, '$.source_session_id')) = :sid"
            ),
            {"uid": user_id, "sid": session_id},
        )
        s.commit()
        return res.rowcount or 0
    except Exception as e:
        log.warning("downgrade_by_session failed (best-effort): %s", e)
        s.rollback()
        return 0
    finally:
        s.close()


# ---- 衰减打分与周期清理（§15） ----

def compute_importance(mem_type: str, meta: dict | None, created_at: str | None,
                       last_accessed_at: str | None, hit_count: int) -> float:
    """importance = base(type) × recency_decay × log(1 + hit_count)（§15）。"""
    base = _BASE_IMPORTANCE.get(mem_type, 0.5)
    if meta and isinstance(meta.get("importance"), (int, float)):
        base = float(meta["importance"])
    ref = last_accessed_at or created_at
    age_days = 0.0
    if ref:
        try:
            dt = datetime.fromisoformat(str(ref))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=CST)
            age_days = max(0.0, (datetime.now(CST) - dt).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            pass
    decay = 0.5 ** (age_days / EPISODIC_HALF_LIFE_DAYS)
    import math
    return base * decay * math.log1p(max(0, hit_count))


def cleanup_decay_sync(limit: int = 200) -> dict:
    """周期衰减清理（§15，与 cleanup 同循环不同事务）。

    - importance < 阈值 且 90 天未命中 → 软删（status='deleted' + Chroma 同步删）
    - superseded 180 天 → 物理清理
    """
    now = datetime.now(CST)
    soft_deleted = 0
    s = db_session()
    try:
        rows = s.execute(
            text("SELECT memory_id, user_id, mem_type, content, meta, hit_count, "
                 "last_accessed_at, created_at FROM agent_user_memory "
                 "WHERE status = 'active' AND mem_type = 'episodic' "
                 "AND COALESCE(last_accessed_at, created_at) < :cutoff LIMIT :lim"),
            {"cutoff": now - timedelta(days=DECAY_UNHIT_DAYS), "lim": limit},
        ).fetchall()
        to_soft_delete: list[tuple[str, str]] = []
        for r in rows:
            mem = _row_to_memory(r)
            imp = compute_importance(mem["mem_type"], mem["meta"], mem["created_at"],
                                     mem["last_accessed_at"], mem["hit_count"])
            if imp < DECAY_MIN_IMPORTANCE:
                to_soft_delete.append((mem["user_id"], mem["memory_id"]))
        for uid, mid in to_soft_delete:
            s.execute(
                text("UPDATE agent_user_memory SET status = 'deleted' "
                     "WHERE user_id = :uid AND memory_id = :mid"),
                {"uid": uid, "mid": mid},
            )
            soft_deleted += 1
        # superseded 物理清理：按被替代时间判龄（$.superseded_at；旧数据缺失回退 created_at）
        s.execute(
            text("DELETE FROM agent_user_memory WHERE status = 'superseded' "
                  "AND COALESCE(JSON_UNQUOTE(JSON_EXTRACT(meta, '$.superseded_at')), "
                  "created_at) < :cutoff"),
            {"cutoff": now - timedelta(days=SUPERSEDED_PURGE_DAYS)},
        )
        s.commit()
        # Chroma 同步删（软删向量；superseded 行本就不在 active 召回 where 内，
        # 但仍删除避免索引膨胀——按 status 过滤需重查，简化为直接删 ids）
        if to_soft_delete:
            try:
                from omnibox_agent.services.chroma_store import (
                    get_named_collection, USER_MEMORIES_COLLECTION)
                get_named_collection(USER_MEMORIES_COLLECTION).delete(
                    ids=[mid for _uid, mid in to_soft_delete])
            except Exception as e:
                log.warning("chroma decay delete failed (will re-sync): %s", e)
        return {"soft_deleted": soft_deleted}
    except Exception as e:
        log.warning("cleanup_decay failed (best-effort): %s", e)
        s.rollback()
        return {"soft_deleted": -1}
    finally:
        s.close()


# ---- L3 Chroma 双写与查询（omnihub_user_memories） ----

def upsert_memory_vector_sync(memory_id: str, user_id: str, content: str,
                              mem_type: str, importance: float = 0.5) -> bool:
    """L3 向量写入（embedding 服务端通道，§12.1；失败不阻断 MySQL 权威源）。"""
    try:
        from omnibox_agent.services.chroma_store import (
            get_named_collection, USER_MEMORIES_COLLECTION)
        from omnibox_agent.services.embedding_service import embed_text
        emb = embed_text(content)
        if not emb:
            return False
        get_named_collection(USER_MEMORIES_COLLECTION).upsert(
            ids=[memory_id],
            embeddings=[emb],
            documents=[content],
            metadatas=[{
                "user_id": user_id,
                "mem_type": mem_type,
                "importance": importance,
                "status": "active",
                "created_at": datetime.now(CST).isoformat(),
            }],
        )
        return True
    except Exception as e:
        log.warning("upsert_memory_vector failed (best-effort): %s", e)
        return False


def query_similar_memories_sync(query_embedding: list[float], user_id: str,
                                top_k: int = 5) -> list[dict]:
    """L3 相似召回：显式 $and 过滤（user_id + status=active，§10.3）。

    返回 [{memory_id, score, document, meta}]，score = 1 − cosine distance。
    """
    try:
        from omnibox_agent.services.chroma_store import (
            get_named_collection, USER_MEMORIES_COLLECTION)
        coll = get_named_collection(USER_MEMORIES_COLLECTION)
        res = coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"$and": [{"user_id": user_id}, {"status": "active"}]},
        )
        out: list[dict] = []
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        for mid, dist, doc, meta in zip(ids, dists, docs, metas):
            out.append({
                "memory_id": mid,
                "score": 1.0 - float(dist or 1.0),
                "document": doc or "",
                "meta": meta or {},
            })
        return out
    except Exception as e:
        log.warning("query_similar_memories failed (best-effort): %s", e)
        return []


def find_duplicate_memory_sync(user_id: str, content: str) -> dict | None:
    """写入前去重检查（§11.4）：同 user active 记忆 embedding 相似 ≥ 0.85 → 返回既有记忆。"""
    try:
        from omnibox_agent.services.embedding_service import embed_text
        emb = embed_text(content)
        if not emb:
            return None
        hits = query_similar_memories_sync(emb, user_id, top_k=5)
        for h in hits:
            if h["score"] >= DEDUP_SIMILARITY:
                return h
        return None
    except Exception as e:
        log.warning("find_duplicate_memory failed (best-effort): %s", e)
        return None


# ---- 周期统计画像（§11.3，SQL 聚合，零 LLM） ----

def refresh_stats_profiles_sync(batch_users: int = 100) -> int:
    """批量刷新 stats_json（近期活跃优先：updated_at 倒序取 N 个）。

    ⚠️ 租户桥接（§10 键约定）：content_items 无 user_id 列，必须经
    user_code → resolve_user_id → users.id → platform_accounts 桥接；
    未解析成功的用户跳过（绝不对共享 MySQL 全表 GROUP BY）。
    返回刷新的用户数。
    """
    s = db_session()
    try:
        user_rows = s.execute(
            text("SELECT user_id FROM agent_user_profile ORDER BY updated_at DESC LIMIT :lim"),
            {"lim": batch_users},
        ).fetchall()
        user_codes = [r[0] for r in user_rows]
        if not user_codes:
            return 0
        # 批量解析 user_code → users.id（桥接，§11.3）
        # IN :codes 用 expanding bindparam（tuple 直绑是 pymysql 驱动宽容行为，换驱动会炸）
        id_rows = s.execute(
            text("SELECT user_code, id FROM users WHERE user_code IN :codes")
            .bindparams(bindparam("codes", expanding=True)),
            {"codes": list(user_codes)},
        ).fetchall()
        code_to_internal = {r[0]: r[1] for r in id_rows}
        if not code_to_internal:
            return 0
        internal_ids = list(code_to_internal.values())
        # 平台分布（JOIN 键：content_items.account_id / platform_accounts.user_id，§11.3）
        _ids_param = bindparam("ids", expanding=True)
        platform_rows = s.execute(
            text(
                "SELECT a.user_id, c.platform, COUNT(*) AS n "
                "FROM content_items c "
                "JOIN platform_accounts a ON c.account_id = a.id "
                "WHERE a.user_id IN :ids "
                "GROUP BY a.user_id, c.platform"
            ).bindparams(_ids_param),
            {"ids": internal_ids},
        ).fetchall()
        # 收藏总量
        total_rows = s.execute(
            text("SELECT a.user_id, COUNT(*) AS n FROM content_items c "
                 "JOIN platform_accounts a ON c.account_id = a.id "
                 "WHERE a.user_id IN :ids GROUP BY a.user_id")
            .bindparams(bindparam("ids", expanding=True)),
            {"ids": internal_ids},
        ).fetchall()
        # 月度收藏趋势（近 6 个月）
        trend_rows = s.execute(
            text(
                "SELECT a.user_id, DATE_FORMAT(c.collected_at, '%Y-%m') AS ym, COUNT(*) AS n "
                "FROM content_items c "
                "JOIN platform_accounts a ON c.account_id = a.id "
                "WHERE a.user_id IN :ids "
                "AND c.collected_at >= :cutoff GROUP BY a.user_id, ym"
            ).bindparams(bindparam("ids", expanding=True)),
            {"ids": internal_ids,
             "cutoff": datetime.now(CST) - timedelta(days=180)},
        ).fetchall()

        plat_map: dict[int, dict[str, int]] = {}
        for uid_i, platform, n in platform_rows:
            plat_map.setdefault(uid_i, {})[platform or "unknown"] = int(n)
        total_map = {uid_i: int(n) for uid_i, n in total_rows}
        trend_map: dict[int, dict[str, int]] = {}
        for uid_i, ym, n in trend_rows:
            trend_map.setdefault(uid_i, {})[str(ym)] = int(n)

        now = datetime.now(CST)
        updated = 0
        for code, uid_i in code_to_internal.items():
            platforms = plat_map.get(uid_i, {})
            stats = {
                "total_favorites": total_map.get(uid_i, 0),
                "platform_dist": dict(sorted(platforms.items(), key=lambda x: -x[1])),
                "top_platforms": sorted(platforms, key=platforms.get, reverse=True)[:3],
                "monthly_trend": dict(sorted(trend_map.get(uid_i, {}).items())),
                "refreshed_at": now.isoformat(),
            }
            s.execute(
                text("UPDATE agent_user_profile SET stats_json = :stats, updated_at = :now "
                     "WHERE user_id = :uid"),
                {"stats": json.dumps(stats, ensure_ascii=False), "now": now, "uid": code},
            )
            updated += 1
        s.commit()
        return updated
    except Exception as e:
        log.warning("refresh_stats_profiles failed (best-effort): %s", e)
        s.rollback()
        return 0
    finally:
        s.close()


# ---- clarify 节点读取助手（§11.1：按 entry_id 读 content/meta） ----

def read_entry_meta_sync(session_id: str, user_id: str, entry_id: str) -> dict | None:
    """读树节点 content + meta（长期记忆提取需要 clarify 节点维度信息）。"""
    s = db_session()
    try:
        row = s.execute(
            text("SELECT content, meta FROM agent_session_entry "
                 "WHERE session_id = :sid AND user_id = :uid AND entry_id = :eid"),
            {"sid": session_id, "uid": user_id, "eid": entry_id},
        ).fetchone()
        if row is None:
            return None
        try:
            meta = json.loads(row[1]) if row[1] else None
        except (ValueError, TypeError):
            meta = None
        return {"content": row[0], "meta": meta}
    except Exception as e:
        log.warning("read_entry_meta failed (best-effort): %s", e)
        return None
    finally:
        s.close()


__all__ = [
    "USER_MEMORIES_COLLECTION",
    "contains_sensitive_info",
    "per_user_lock",
    "ensure_profile_row_sync",
    "get_profile_sync",
    "upsert_profile_sync",
    "incr_lt_round_sync",
    "reset_lt_round_sync",
    "list_active_memories_sync",
    "list_memories_by_status_sync",
    "search_profiles_sync",
    "get_memory_sync",
    "insert_memory_sync",
    "supersede_by_key_sync",
    "find_active_preference_sync",
    "bump_confidence_sync",
    "bump_hit_sync",
    "soft_delete_memory_sync",
    "downgrade_by_session_sync",
    "compute_importance",
    "cleanup_decay_sync",
    "upsert_memory_vector_sync",
    "query_similar_memories_sync",
    "find_duplicate_memory_sync",
    "refresh_stats_profiles_sync",
    "read_entry_meta_sync",
]
