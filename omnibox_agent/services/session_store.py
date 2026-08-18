"""Session Tree 存储层（MEMORY_SYSTEM_DESIGN.md v1.1 §4）。

基于现有 MySQL 基础设施（core/database.py）实现会话树双表：
  agent_session          会话元数据 + leaf 指针
  agent_session_entry    树节点（内容只增不改；parent_id / status 允许结构变更）

设计要点（见设计文档）：
  - 归属校验：所有读写必须携带 user_id（sessionId 为客户端生成，可猜测/复用），
    禁止只按 session_id 查询 —— 防跨用户读写（§3.1）。
  - per-session 进程内锁：串行化同会话的 append / compaction / 中断标记（§4.7）。
  - 幂等：append_user / append_assistant 以 (session_id, request_id, role) 去重
    （uk_request_role 唯一键兜底；一轮请求产生一问一答两个节点，按 role 区分，§4.1）。
  - best-effort：任何 DB 故障 / 执行器繁忙只记日志并返回 None，绝不抛给问答主流程
    （记忆故障不得影响现有问答行为，§4.8）。
  - 结构变更白名单：mark_user_complete / mark_interrupted / reparent
    （仅 parent_id 与 status 可 UPDATE，content / role / entry_type 不可变，§3.2）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text, bindparam
from sqlalchemy.exc import IntegrityError

from omnibox_agent.core.database import get_session as db_session
from omnibox_agent.agent.loop import ExecutorBusyError, run_blocking

log = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))

# 路径遍历深度上限（防循环；正常会话远达不到）
_PATH_MAX_DEPTH = 10000


class SessionOwnershipError(Exception):
    """sessionId 已存在但属于其他用户（与后端 AskService.saveSession 的防越权一致）。"""


@dataclass
class SessionEntry:
    """agent_session_entry 的一行（只读视图）。"""
    entry_id: str
    entry_type: str
    role: str | None
    status: str
    content: str | None
    meta: dict | None
    token_est: int
    parent_id: str | None
    created_at: datetime | None


# ---- per-session 进程内锁（单 uvicorn worker 成立；多 worker/多实例需换 Redis 锁，§4.7） ----

# 锁 + 最后使用时间戳（monotonic）。长期运行下会话数可能增长，空闲锁定期清理，
# 避免字典无限膨胀（修复：2026-08-15 审计）。
_per_session_locks: dict[str, tuple[asyncio.Lock, float]] = {}
_LOCK_IDLE_PRUNE_SECONDS = 600.0   # 空闲 10 分钟即清理
_LOCK_PRUNE_INTERVAL = 64          # 每 N 次获取做一次裁剪检查
_lock_op_counter = 0


def per_session_lock(session_id: str) -> asyncio.Lock:
    """按 session_id 获取（或创建）进程内锁（含空闲清理）。"""
    now = time.monotonic()
    entry = _per_session_locks.get(session_id)
    if entry is None:
        lock = asyncio.Lock()
        _per_session_locks[session_id] = (lock, now)
        return lock
    lock, _ts = entry
    _per_session_locks[session_id] = (lock, now)  # 刷新最后使用
    return lock


def _prune_session_locks() -> None:
    """移除空闲超过阈值的锁（正在持有/有等待者的不删）。"""
    now = time.monotonic()
    stale = [
        sid for sid, (lock, ts) in _per_session_locks.items()
        if (now - ts) > _LOCK_IDLE_PRUNE_SECONDS and not lock.locked()
    ]
    for sid in stale:
        _per_session_locks.pop(sid, None)


# =====================================================================
# 同步核心（纯 DB 操作；可单测、可被 run_blocking 包装）
# =====================================================================

def _row_to_entry(row: Any) -> SessionEntry:
    meta = None
    if row.meta:
        try:
            meta = json.loads(row.meta)
        except (ValueError, TypeError):
            meta = None
    return SessionEntry(
        entry_id=row.entry_id,
        entry_type=row.entry_type,
        role=row.role,
        status=row.status,
        content=row.content,
        meta=meta,
        token_est=row.token_est or 0,
        parent_id=row.parent_id,
        created_at=row.created_at,
    )


def load_session(session_id: str, user_id: str) -> dict | None:
    """归属校验后的会话读取（WHERE session_id AND user_id）。失败返回 None。"""
    if not session_id or not user_id:
        return None
    s = db_session()
    try:
        row = s.execute(
            text(
                "SELECT session_id, user_id, leaf_entry_id, title, status, last_active_at "
                "FROM agent_session WHERE session_id = :sid AND user_id = :uid"
            ),
            {"sid": session_id, "uid": user_id},
        ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row.session_id,
            "user_id": row.user_id,
            "leaf_entry_id": row.leaf_entry_id,
            "title": row.title,
            "status": row.status,
            "last_active_at": row.last_active_at,
        }
    except Exception as e:
        log.warning("load_session failed (best-effort): %s", e)
        return None
    finally:
        s.close()


def create_session(session_id: str, user_id: str, title: str | None = None) -> dict | None:
    """新建会话。sessionId 已被他人占用 → 抛 SessionOwnershipError（由调用方 best-effort 兜底）。"""
    now = datetime.now(CST)
    s = db_session()
    try:
        s.execute(
            text(
                "INSERT INTO agent_session "
                "(session_id, user_id, leaf_entry_id, title, status, last_active_at, created_at) "
                "VALUES (:sid, :uid, NULL, :title, 'active', :now, :now)"
            ),
            {"sid": session_id, "uid": user_id, "title": title, "now": now},
        )
        s.commit()
    except IntegrityError:
        s.rollback()
        existing = s.execute(
            text("SELECT user_id FROM agent_session WHERE session_id = :sid"),
            {"sid": session_id},
        ).fetchone()
        if existing is None:
            raise
        if existing[0] != user_id:
            raise SessionOwnershipError(f"sessionId {session_id} 已被其他用户占用")
        # 已存在且属本人 → 幂等复用
        return load_session(session_id, user_id)
    except Exception as e:
        log.warning("create_session failed (best-effort): %s", e)
        return None
    finally:
        s.close()
    return load_session(session_id, user_id)


def _insert_entry(
    *,
    session_id: str,
    user_id: str,
    entry_id: str,
    parent_id: str | None,
    entry_type: str,
    role: str | None,
    status: str,
    content: str | None,
    meta: dict | None,
    request_id: str | None = None,
    token_est: int = 0,
) -> str:
    """INSERT 一个节点。uk_request 冲突（同 request_id）→ 返回已存在 entry_id（幂等）。"""
    from omnibox_agent.services.compaction import estimate_tokens  # 延迟导入防循环依赖

    if token_est <= 0:
        token_est = estimate_tokens(content or "")
    meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
    s = db_session()
    try:
        s.execute(
            text(
                "INSERT INTO agent_session_entry "
                "(entry_id, session_id, user_id, parent_id, entry_type, role, status, "
                " content, meta, request_id, token_est, created_at) "
                "VALUES (:eid, :sid, :uid, :pid, :etype, :role, :status, "
                "        :content, :meta, :rid, :tokens, :now)"
            ),
            {
                "eid": entry_id,
                "sid": session_id,
                "uid": user_id,
                "pid": parent_id,
                "etype": entry_type,
                "role": role,
                "status": status,
                "content": content,
                "meta": meta_json,
                "rid": request_id,
                "tokens": token_est,
                "now": datetime.now(CST),
            },
        )
        s.commit()
        return entry_id
    except IntegrityError:
        s.rollback()
        # 幂等兜底：同 (session_id, request_id, role) 已存在 → 复用既有 entry
        # （一轮请求产生 user + assistant 两个节点，按 role 区分，见 uk_request_role）
        if request_id:
            existing = _find_by_request(session_id, user_id, request_id, role or "")
            if existing:
                return existing
        raise
    except Exception as e:
        log.warning("_insert_entry failed (best-effort): %s", e)
        s.rollback()
        raise
    finally:
        s.close()


def _update_leaf(session_id: str, user_id: str, entry_id: str, expected_leaf: str | None) -> bool:
    """乐观锁移动 leaf 指针（expected_leaf 不匹配则不动，返回 False）。"""
    s = db_session()
    try:
        res = s.execute(
            text(
                "UPDATE agent_session SET leaf_entry_id = :eid, last_active_at = :now "
                "WHERE session_id = :sid AND user_id = :uid "
                "AND ((leaf_entry_id = :exp) OR (:exp IS NULL AND leaf_entry_id IS NULL))"
            ),
            {
                "eid": entry_id,
                "now": datetime.now(CST),
                "sid": session_id,
                "uid": user_id,
                "exp": expected_leaf,
            },
        )
        s.commit()
        return res.rowcount > 0
    except Exception as e:
        log.warning("_update_leaf failed (best-effort): %s", e)
        s.rollback()
        return False
    finally:
        s.close()


def _find_by_request(session_id: str, user_id: str, request_id: str, role: str) -> str | None:
    s = db_session()
    try:
        row = s.execute(
            text(
                "SELECT entry_id FROM agent_session_entry "
                "WHERE session_id = :sid AND user_id = :uid AND request_id = :rid AND role = :role LIMIT 1"
            ),
            {"sid": session_id, "uid": user_id, "rid": request_id, "role": role},
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None
    finally:
        s.close()


def _mark_user_complete(entry_id: str, session_id: str | None = None,
                        user_id: str | None = None) -> None:
    """User Entry 收到对应 Assistant 后标记 complete（结构变更白名单内）。

    session_id/user_id 可选：传入时收紧到本会话本用户（防御性，防 entry_id 越权）。
    """
    s = db_session()
    try:
        sql = "UPDATE agent_session_entry SET status = 'complete' WHERE entry_id = :eid AND role = 'user'"
        params: dict = {"eid": entry_id}
        if session_id and user_id:
            sql += " AND session_id = :sid AND user_id = :uid"
            params["sid"] = session_id
            params["uid"] = user_id
        s.execute(text(sql), params)
        s.commit()
    except Exception as e:
        log.warning("_mark_user_complete failed (best-effort): %s", e)
        s.rollback()
    finally:
        s.close()


# ---- 写操作（同步核心） ----

def append_user_sync(
    session_id: str,
    user_id: str,
    content: str,
    meta: dict | None = None,
    request_id: str | None = None,
    parent_id: str | None = None,
) -> str | None:
    """请求开始时追加 User Entry（status=pending）。失败/归属冲突返回 None。

    parent_id 为 None 时挂在当前 leaf 下；显式传入时（澄清 resume 的回答轮）
    挂在指定节点下（设计 §4.5 步骤 3）。
    """
    try:
        session = load_session(session_id, user_id)
        if session is None:
            try:
                session = create_session(session_id, user_id)
            except SessionOwnershipError:
                log.warning("append_user ownership conflict: %s", session_id)
                return None
            if session is None:
                return None
        # 幂等：同 (session_id, request_id, role='user') 已存在 → 复用
        # （防后端/客户端重试重复节点；assistant 节点同 rid 但 role 不同，不受影响）
        if request_id:
            existing = _find_by_request(session_id, user_id, request_id, "user")
            if existing:
                return existing
        entry_id = f"e_{uuid.uuid4().hex}"
        effective_parent = parent_id if parent_id is not None else session["leaf_entry_id"]
        _insert_entry(
            session_id=session_id, user_id=user_id, entry_id=entry_id,
            parent_id=effective_parent, entry_type="message", role="user",
            status="pending", content=content, meta=meta, request_id=request_id,
        )
        _update_leaf(session_id, user_id, entry_id, expected_leaf=effective_parent)
        return entry_id
    except Exception as e:
        log.warning("append_user_sync failed (best-effort): %s", e)
        return None


def append_assistant_sync(
    session_id: str,
    user_id: str,
    content: str,
    parent_entry_id: str,
    meta: dict | None = None,
    request_id: str | None = None,
) -> str | None:
    """响应完成后追加 Assistant Entry 并完结对应 User Entry。失败返回 None。"""
    try:
        entry_id = f"e_{uuid.uuid4().hex}"
        _insert_entry(
            session_id=session_id, user_id=user_id, entry_id=entry_id,
            parent_id=parent_entry_id, entry_type="message", role="assistant",
            status="complete", content=content, meta=meta, request_id=request_id,
        )
        _mark_user_complete(parent_entry_id, session_id=session_id, user_id=user_id)
        leaf_ok = _update_leaf(session_id, user_id, entry_id, expected_leaf=parent_entry_id)
        if not leaf_ok:
            # ⚠️ 生产实证（2026-08-15）：长耗时生成（如 DAG ~90s）期间用户抢先发了下一条
            # → 本 final 晚到时 leaf 已被新问题占用，乐观锁失败 → final 成旁支，主链永远缺它。
            # 修复：把 parent 的第一个主链子节点（最早 append 的）改挂到本 final 下，
            # 使主链变为 ... → parent → final → 后续轮次（认父不认子，线性对话下安全）。
            _splice_first_child(session_id, user_id, parent_entry_id, entry_id)
        return entry_id
    except Exception as e:
        log.warning("append_assistant_sync failed (best-effort): %s", e)
        return None


def _splice_first_child(session_id: str, user_id: str, parent_id: str, new_parent_id: str) -> None:
    """把 parent 的第一个子节点（按 id 最早 = 主链延续）改挂到 new_parent_id 下。"""
    s = db_session()
    try:
        s.execute(
            text(
                "UPDATE agent_session_entry SET parent_id = :np "
                "WHERE session_id = :sid AND user_id = :uid AND parent_id = :p "
                "ORDER BY id LIMIT 1"
            ),
            {"np": new_parent_id, "sid": session_id, "uid": user_id, "p": parent_id},
        )
        s.commit()
    except Exception as e:
        log.warning("_splice_first_child failed (best-effort): %s", e)
        s.rollback()
    finally:
        s.close()


def mark_interrupted_sync(session_id: str, user_id: str, entry_id: str) -> bool:
    """客户端中断：把 pending 的 User Entry 标记 interrupted（不追加 assistant，§4.6）。"""
    s = db_session()
    try:
        res = s.execute(
            text(
                "UPDATE agent_session_entry SET status = 'interrupted' "
                "WHERE entry_id = :eid AND session_id = :sid AND user_id = :uid "
                "AND role = 'user' AND status = 'pending'"
            ),
            {"eid": entry_id, "sid": session_id, "uid": user_id},
        )
        s.commit()
        return res.rowcount > 0
    except Exception as e:
        log.warning("mark_interrupted_sync failed (best-effort): %s", e)
        s.rollback()
        return False
    finally:
        s.close()


def reparent_sync(session_id: str, user_id: str, entry_id: str, new_parent_id: str | None) -> bool:
    """结构重定向（Compaction 上树用；仅 parent_id 可变更，§4.2）。"""
    s = db_session()
    try:
        res = s.execute(
            text(
                "UPDATE agent_session_entry SET parent_id = :pid "
                "WHERE entry_id = :eid AND session_id = :sid AND user_id = :uid"
            ),
            {"pid": new_parent_id, "eid": entry_id, "sid": session_id, "uid": user_id},
        )
        s.commit()
        return res.rowcount > 0
    except Exception as e:
        log.warning("reparent_sync failed (best-effort): %s", e)
        s.rollback()
        return False
    finally:
        s.close()


# ---- 读操作（同步核心；无需 per-session 锁，事务原子性保证读到一致状态） ----

def get_path_entries_sync(
    session_id: str, user_id: str, leaf_entry_id: str | None = None
) -> list[SessionEntry]:
    """从 leaf 沿 parent_id 上溯至根后反转（归属校验 + 循环保护）。失败返回 []。"""
    try:
        if leaf_entry_id is None:
            session = load_session(session_id, user_id)
            if session is None:
                return []
            leaf_entry_id = session["leaf_entry_id"]
        if not leaf_entry_id:
            return []
        chain: list[SessionEntry] = []
        seen: set[str] = set()
        current = leaf_entry_id
        s = db_session()
        try:
            while current and len(chain) < _PATH_MAX_DEPTH:
                if current in seen:
                    log.warning("session tree cycle detected at %s (session=%s)", current, session_id)
                    break
                seen.add(current)
                row = s.execute(
                    text(
                        "SELECT entry_id, entry_type, role, status, content, meta, token_est, "
                        "parent_id, created_at FROM agent_session_entry "
                        "WHERE session_id = :sid AND user_id = :uid AND entry_id = :eid"
                    ),
                    {"sid": session_id, "uid": user_id, "eid": current},
                ).fetchone()
                if row is None:
                    break
                chain.append(_row_to_entry(row))
                current = row.parent_id
        finally:
            s.close()
        chain.reverse()
        return chain
    except Exception as e:
        log.warning("get_path_entries_sync failed (best-effort): %s", e)
        return []


def _trim_recent(recent: list[dict], summary: str | None, budget: int) -> list[dict]:
    """溢出兜底：summary + recent 总 token 超预算时从头部裁剪（设计 §4.3）。"""
    from omnibox_agent.services.compaction import estimate_tokens

    total = estimate_tokens(summary or "")
    kept: list[dict] = []
    for msg in reversed(recent):
        t = estimate_tokens(msg.get("content") or "")
        if total + t > budget:
            break
        kept.insert(0, msg)
        total += t
    return kept


def load_backend_history_sync(user_code: str, session_id: str) -> list[dict]:
    """从后端 ask_session（同一 MySQL 实例，服务端权威对话记录）读取历史。

    设计原则（生产实证修正）：**记忆来源必须是服务端**——前端注入的 history
    是客户端镜像（本地存储、单设备、仅最近 N 轮、可能被过滤），不可作记忆来源。
    后端 ask_session.messages_json 由服务端保存，含记忆启用前的完整轮次。

    ask_session.user_id 是内部 bigint 主键 → 先按 user_code 解析再查。
    消息字段：user/assistant 均取 `text`（澄清气泡的 text 即最终回答）。
    返回 [{role, content}, ...]（存储顺序）；失败/无记录返回 []。
    """
    if not user_code or not session_id:
        return []
    s = db_session()
    try:
        uid = s.execute(
            text("SELECT id FROM users WHERE user_code = :code"),
            {"code": user_code},
        ).fetchone()
        if uid is None:
            return []
        row = s.execute(
            text("SELECT messages_json FROM ask_session WHERE id = :sid AND user_id = :uid"),
            {"sid": session_id, "uid": uid[0]},
        ).fetchone()
        if row is None or not row[0]:
            return []
        try:
            msgs = json.loads(row[0])
        except (ValueError, TypeError):
            return []
        out: list[dict] = []
        for m in msgs or []:
            if not isinstance(m, dict):
                continue
            role = m.get("role") if m.get("role") in ("user", "assistant") else None
            content = m.get("text") or m.get("content") or ""
            if role and content.strip():
                out.append({"role": role, "content": content})
        return out
    except Exception as e:
        log.warning("load_backend_history failed (best-effort): %s", e)
        return []
    finally:
        s.close()


def merge_history(tree_history: list | None, injected_history: list | None) -> list[dict]:
    """树历史 + 前端注入历史合并去重（树优先，注入补缺）。

    场景：记忆启用前产生的轮次不在树里，但前端每次请求都带最近 N 轮注入
    history——合并后补全"启用前的旧轮"，避免 Agent 失忆（生产实证，§10）。
    返回 [{role, content, ts?}]；注入条目无 ts（QU 过滤时会丢弃，不影响作答 LLM）。
    """
    out: list[dict] = [dict(m) for m in (tree_history or [])]
    seen = {(m.get("role"), m.get("content")) for m in out}
    for m in injected_history or []:
        content = m.get("content") or ""
        role = m.get("role") if m.get("role") in ("user", "assistant") else "user"
        if content and (role, content) not in seen:
            out.append({"role": role, "content": content})
            seen.add((role, content))
    return out


def build_session_context_sync(
    session_id: str, user_id: str, budget: int | None = None,
    fallback_history: list | None = None,
) -> dict | None:
    """重放构建 {"summary": str|None, "recent": [{role, content}, ...]}。

    过滤规则（§4.3，v1.1 生产实证修正）：只过滤 **interrupted** 的 user 节点
    （用户中断、问题未回答）；**pending 注入**——长耗时生成（如 DAG 可达 90s）期间
    用户抢先发下一条时，该轮是 pending，若过滤会导致"上面说了什么"看不到进行中的
    轮次而误判失忆。失败/无内容返回 None（调用方回退注入 history）。

    fallback_history：前端注入的 history，用于补全树里缺失的旧轮
    （记忆启用前的轮次、跨端恢复等），与树 recent 合并去重后统一裁剪预算。
    """
    try:
        entries = get_path_entries_sync(session_id, user_id)
        summary: str | None = None
        recent: list[dict] = []
        if entries:
            for entry in entries:
                if entry.entry_type == "message" and entry.role == "user" and entry.status in ("interrupted",):
                    continue
                if entry.entry_type == "compaction":
                    summary = entry.content
                    recent = []  # 摘要代表之前的一切
                elif entry.entry_type == "message":
                    recent.append({"role": entry.role, "content": entry.content})
                elif entry.entry_type == "branch_summary":
                    recent.append({"role": "user", "content": f"[历史分支信息] {entry.content}"})
        # 合并前端注入 history（补全树缺失的旧轮；去重，树优先）
        if fallback_history:
            recent = merge_history(recent, fallback_history)
        if budget:
            recent = _trim_recent(recent, summary, budget)
        return {"summary": summary, "recent": recent}
    except Exception as e:
        log.warning("build_session_context_sync failed (best-effort): %s", e)
        # 降级：树读取失败时不返回 None（否则回答侧静默失忆），改用
        # fallback_history（ask_session 权威记录）构建 recent。summary 只能置空
        # （树读取失败拿不到 compaction 摘要），但至少保留完整近期轮次。
        if fallback_history:
            recent = merge_history([], fallback_history)
            if budget:
                recent = _trim_recent(recent, None, budget)
            return {"summary": None, "recent": recent}
        return None


def bootstrap_session_from_history_sync(
    session_id: str, user_id: str, history: list[dict],
) -> bool:
    """会话首次触达记忆时，用前端注入 history 种子化会话树（§4.1 补充）。

    场景：会话创建于记忆启用之前（或树从未建立），前端注入的最近 N 轮
    是唯一完整来源——把它们按序写进树（complete 对），后续轮次才能
    累积记忆、参与 QU/压缩。幂等：会话已存在则跳过。返回是否执行了种子化。
    """
    if not history or not session_id or not user_id:
        return False
    if load_session(session_id, user_id) is not None:
        return False  # 会话已存在（树已建），由 merge_history 补缺即可
    s = db_session()
    try:
        now = datetime.now(CST)
        from omnibox_agent.services.compaction import estimate_tokens

        parent: str | None = None
        last_id: str | None = None
        for m in history:
            role = m.get("role") if m.get("role") in ("user", "assistant") else "user"
            content = m.get("content") or ""
            if not content:
                continue
            entry_id = f"e_{uuid.uuid4().hex}"
            s.execute(
                text(
                    "INSERT INTO agent_session_entry "
                    "(entry_id, session_id, user_id, parent_id, entry_type, role, status, "
                    " content, meta, request_id, token_est, created_at) "
                    "VALUES (:eid, :sid, :uid, :pid, 'message', :role, 'complete', "
                    "        :content, NULL, NULL, :tokens, :now)"
                ),
                {"eid": entry_id, "sid": session_id, "uid": user_id, "pid": parent,
                 "role": role, "content": content, "tokens": estimate_tokens(content),
                 "now": now},
            )
            parent = entry_id
            last_id = entry_id
        if last_id is None:
            return False
        # 会话行 + leaf 指针
        s.execute(
            text(
                "INSERT INTO agent_session "
                "(session_id, user_id, leaf_entry_id, title, status, last_active_at, created_at) "
                "VALUES (:sid, :uid, :leaf, NULL, 'active', :now, :now) "
                "ON DUPLICATE KEY UPDATE leaf_entry_id = :leaf2, last_active_at = :now2"
            ),
            {"sid": session_id, "uid": user_id, "leaf": last_id, "now": now,
             "leaf2": last_id, "now2": now},
        )
        s.commit()
        log.info("session tree bootstrapped from injected history: session=%s turns=%d",
                 session_id, len(history))
        return True
    except Exception as e:
        log.warning("bootstrap_session_from_history_sync failed (best-effort): %s", e)
        s.rollback()
        return False
    finally:
        s.close()


def get_qu_history_sync(session_id: str, user_id: str, hours: int = 12) -> list[dict]:
    """QU 指代消解用历史（近 N 小时，含 ts 毫秒；只排除 interrupted 的 user；pending 注入）。

    返回 [{role, content, ts}]；失败/空返回 []（调用方回退注入 history）。
    """
    def _aware(dt: datetime | None) -> datetime | None:
        # pymysql 读回 DATETIME 为 naive（服务器本地时间）；统一补 CST 时区再比较/换算
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=CST)
        return dt

    try:
        entries = get_path_entries_sync(session_id, user_id)
        cutoff = datetime.now(CST) - timedelta(hours=hours)
        out: list[dict] = []
        for e in entries:
            if e.entry_type != "message":
                continue
            if e.role == "user" and e.status in ("interrupted",):
                continue
            created = _aware(e.created_at)
            if created is not None and created < cutoff:
                continue
            out.append({
                "role": e.role,
                "content": e.content or "",
                "ts": int(created.timestamp() * 1000) if created else None,
            })
        return out
    except Exception as e:
        log.warning("get_qu_history_sync failed (best-effort): %s", e)
        return []


def find_last_clarify_entry_sync(session_id: str, user_id: str) -> str | None:
    """路径上最近一次澄清提问（assistant + meta.type=='clarify'）的 entry_id。

    澄清 resume 用：回答轮要挂到该节点下（设计 §4.5 步骤 3）；无则返回 None。
    """
    try:
        entries = get_path_entries_sync(session_id, user_id)
        for e in reversed(entries):
            if (e.entry_type == "message" and e.role == "assistant"
                    and e.meta and e.meta.get("type") == "clarify"):
                return e.entry_id
        return None
    except Exception as e:
        log.warning("find_last_clarify_entry_sync failed (best-effort): %s", e)
        return None


def session_memory_suffix(session_context: dict | None) -> str:
    """返回 <session_memory> 注入片段（无摘要时为空串）。

    供三条管线（ReasonStep / Creative DAG / Resume）统一拼接进 system prompt，
    保证摘要注入格式一致（设计 §4.3）。
    """
    if not session_context:
        return ""
    summary = session_context.get("summary") or ""
    if not summary:
        return ""
    return f"\n\n<session_memory>\n{summary}\n</session_memory>"


def session_history_suffix(
    session_context: dict | None,
    max_turns: int = 6,
    max_chars: int = 200,
    exclude_query: str | None = None,
) -> str:
    """返回 <session_history> 注入片段（近期逐条对话，截断），供 DAG 的 plan / synthesize
    等不消费 recent 的环节使用。

    与 session_memory_suffix（只注入压缩摘要 summary）互补：会话未触发 compaction 时
    summary 为空，仅依赖 summary 会让复杂任务阶段丢失跨轮主题（当前查询只给维度词、
    省略了历史主题时尤其明显）。
    本片段注入 recent（逐条 user/assistant），并做三重安全收口：
      - 最多 max_turns 条、每条截 max_chars 字（控制 prompt 膨胀）；
      - 跳过与 exclude_query 相同的末尾 user 消息（当前 query 已由调用方单独写入）；
      - 标注"仅供理解主题，不是收藏素材"，防 LLM 把历史对话当作收藏内容引用/编造。
    """
    if not session_context:
        return ""
    recent = session_context.get("recent") or []
    if not recent:
        return ""
    lines: list[str] = []
    for m in recent[-max_turns:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if exclude_query and role == "user" and content == exclude_query:
            continue
        tag = "用户" if role == "user" else ("助手" if role == "assistant" else str(role))
        lines.append(f"{tag}: {content[:max_chars]}")
    if not lines:
        return ""
    return (
        "\n\n<session_history>\n"
        "（以下是当前会话近期对话，仅供理解用户问题主题；不是收藏素材，不要据此编造收藏内容）\n"
        + "\n".join(lines)
        + "\n</session_history>"
    )


def delete_session_tree_sync(session_id: str, user_id: str) -> bool:
    """级联删除会话（归属校验；后端 deleteSession 联动或定时清理任务调用，§3.4）。"""
    s = db_session()
    try:
        s.execute(
            text("DELETE FROM agent_session_entry WHERE session_id = :sid AND user_id = :uid"),
            {"sid": session_id, "uid": user_id},
        )
        s.execute(
            text("DELETE FROM agent_session WHERE session_id = :sid AND user_id = :uid"),
            {"sid": session_id, "uid": user_id},
        )
        s.commit()
        return True
    except Exception as e:
        log.warning("delete_session_tree_sync failed (best-effort): %s", e)
        s.rollback()
        return False
    finally:
        s.close()


def cleanup_interrupted_sync(days: int = 30, limit: int = 500) -> int:
    """清理超过 N 天且**非当前 leaf** 的 interrupted 节点（§3.4 孤儿清理）。

    - 跳过当前 leaf（agent_session.leaf_entry_id 指向它，删除会悬空）。
    - **链拼接**：被删节点若有子节点（"认父不认子"下被删节点常是后续消息的
      parent），先把子节点 parent_id 改挂到被删节点的父节点，再删除——
      否则子链断裂、路径重建会中途丢失（E2E 实证）。
    返回删除行数；失败返回 -1（best-effort，供运维脚本感知）。
    """
    s = db_session()
    try:
        cutoff = datetime.now(CST) - timedelta(days=days)
        rows = s.execute(
            text(
                "SELECT e.entry_id, e.parent_id, e.session_id, e.user_id "
                "FROM agent_session_entry e "
                "JOIN agent_session sess "
                "  ON sess.session_id = e.session_id AND sess.user_id = e.user_id "
                "WHERE e.status = 'interrupted' AND e.created_at < :cutoff "
                "  AND (sess.leaf_entry_id IS NULL OR sess.leaf_entry_id <> e.entry_id) "
                "LIMIT :lim"
            ),
            {"cutoff": cutoff, "lim": limit},
        ).fetchall()
        if not rows:
            return 0
        deleted = 0
        for entry_id, _snapshot_parent, session_id, user_id in rows:
            # 链拼接：把该节点的子节点改挂到其父节点（内容不动，仅结构指针）。
            # ⚠️ 父节点必须**实时重读**：同批内前面的删除可能已把它改挂/删除
            #（E2E 实证：用快照 parent 会把子链指向已删节点）。
            cur = s.execute(
                text("SELECT parent_id FROM agent_session_entry "
                     "WHERE entry_id = :eid AND session_id = :sid AND user_id = :uid"),
                {"eid": entry_id, "sid": session_id, "uid": user_id},
            ).fetchone()
            if cur is None:  # 已被同批前序删除
                continue
            gp = cur[0]
            s.execute(
                text(
                    "UPDATE agent_session_entry SET parent_id = :gp "
                    "WHERE parent_id = :eid AND session_id = :sid AND user_id = :uid"
                ),
                {"gp": gp, "eid": entry_id, "sid": session_id, "uid": user_id},
            )
            s.execute(
                text(
                    "DELETE FROM agent_session_entry "
                    "WHERE entry_id = :eid AND session_id = :sid AND user_id = :uid"
                ),
                {"eid": entry_id, "sid": session_id, "uid": user_id},
            )
            deleted += 1
        s.commit()
        return deleted
    except Exception as e:
        log.warning("cleanup_interrupted_sync failed: %s", e)
        s.rollback()
        return -1
    finally:
        s.close()


def cleanup_stale_sessions_sync(days: int = 90, limit: int = 50) -> int:
    """删除超过 N 天未活跃的整棵会话树（§3.4）。

    MySQL 多表 DELETE 不支持 LIMIT，故先 SELECT 候选再逐棵级联删除。
    返回删除的会话数；失败返回 -1。
    """
    s = db_session()
    try:
        cutoff = datetime.now(CST) - timedelta(days=days)
        rows = s.execute(
            text(
                "SELECT session_id, user_id FROM agent_session "
                "WHERE last_active_at < :cutoff LIMIT :lim"
            ),
            {"cutoff": cutoff, "lim": limit},
        ).fetchall()
        n = 0
        for session_id, user_id in rows:
            s.execute(
                text("DELETE FROM agent_session_entry WHERE session_id = :sid AND user_id = :uid"),
                {"sid": session_id, "uid": user_id},
            )
            s.execute(
                text("DELETE FROM agent_session WHERE session_id = :sid AND user_id = :uid"),
                {"sid": session_id, "uid": user_id},
            )
            n += 1
        s.commit()
        return n
    except Exception as e:
        log.warning("cleanup_stale_sessions_sync failed: %s", e)
        s.rollback()
        return -1
    finally:
        s.close()


# =====================================================================
# 异步包装（ask.py / compaction.py 使用）
#   - 写操作：per-session 锁 + run_blocking（线程池，best-effort）
#   - 读操作：仅 run_blocking（无需锁；事务原子性保证一致视图）
# =====================================================================

async def _write_locked(session_id: str, fn, *args, **kwargs):
    global _lock_op_counter
    _lock_op_counter += 1
    if _lock_op_counter % _LOCK_PRUNE_INTERVAL == 0:
        _prune_session_locks()
    async with per_session_lock(session_id):
        try:
            return await run_blocking(fn, *args, **kwargs)
        except ExecutorBusyError:
            log.warning("memory write skipped: executor busy (session=%s)", session_id)
            return None
        except Exception as e:
            log.warning("memory write failed (best-effort, session=%s): %s", session_id, e)
            return None


async def _read(fn, *args, **kwargs):
    # ExecutorBusyError 是瞬时饱和（8-ticket 线程池被并发请求占用），短退避重试
    # 可大幅降低静默丢失；重试仍失败才降级返回 None（best-effort）。
    for attempt in range(3):
        try:
            return await run_blocking(fn, *args, **kwargs)
        except ExecutorBusyError:
            if attempt < 2:
                await asyncio.sleep(0.05 * (attempt + 1))
                continue
            log.warning("memory read skipped: executor busy (after retries)")
            return None
        except Exception as e:
            log.warning("memory read failed (best-effort): %s", e)
            return None
    return None


async def append_user(session_id, user_id, content, meta=None, request_id=None, parent_id=None) -> str | None:
    return await _write_locked(session_id, append_user_sync, session_id, user_id, content,
                               meta=meta, request_id=request_id, parent_id=parent_id)


async def append_assistant(session_id, user_id, content, parent_entry_id, meta=None, request_id=None) -> str | None:
    return await _write_locked(session_id, append_assistant_sync, session_id, user_id, content,
                               parent_entry_id, meta=meta, request_id=request_id)


async def mark_interrupted(session_id, user_id, entry_id) -> bool | None:
    return await _write_locked(session_id, mark_interrupted_sync, session_id, user_id, entry_id)


async def find_last_clarify_entry(session_id, user_id) -> str | None:
    return await _read(find_last_clarify_entry_sync, session_id, user_id)


async def build_session_context(session_id, user_id, budget=None, fallback_history=None) -> dict | None:
    return await _read(build_session_context_sync, session_id, user_id, budget=budget,
                       fallback_history=fallback_history)


async def bootstrap_session_from_history(session_id, user_id, history) -> bool | None:
    return await _write_locked(session_id, bootstrap_session_from_history_sync, session_id, user_id, history)


async def load_backend_history(user_code, session_id) -> list[dict]:
    result = await _read(load_backend_history_sync, user_code, session_id)
    return result if isinstance(result, list) else []


async def get_qu_history(session_id, user_id, hours: int = 12) -> list[dict]:
    """返回 [{role, content, ts}]；失败/禁用返回 []（调用方回退注入 history）。"""
    result = await _read(get_qu_history_sync, session_id, user_id, hours=hours)
    return result if isinstance(result, list) else []


async def delete_session_tree(session_id, user_id) -> bool | None:
    return await _write_locked(session_id, delete_session_tree_sync, session_id, user_id)


__all__ = [
    "SessionEntry",
    "SessionOwnershipError",
    "per_session_lock",
    "load_session",
    "create_session",
    "append_user",
    "append_assistant",
    "mark_interrupted",
    "find_last_clarify_entry",
    "session_memory_suffix",
    "session_history_suffix",
    "merge_history",
    "bootstrap_session_from_history",
    "load_backend_history",
    "load_backend_history_sync",
    "build_session_context",
    "get_qu_history",
    "delete_session_tree",
    "get_path_entries_sync",
    "reparent_sync",
    "append_user_sync",
    "append_assistant_sync",
    "mark_interrupted_sync",
    "find_last_clarify_entry_sync",
    "session_memory_suffix",
    "merge_history",
    "bootstrap_session_from_history_sync",
    "build_session_context_sync",
    "get_qu_history_sync",
    "delete_session_tree_sync",
    "cleanup_interrupted_sync",
    "cleanup_stale_sessions_sync",
]
