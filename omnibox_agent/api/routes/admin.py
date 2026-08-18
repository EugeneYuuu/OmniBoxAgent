"""开关平台 Admin API（USER_FLAG_PLATFORM_DESIGN.md §5，F2）。

  GET    /admin/flags?user_code=xxx              用户开关面板（覆盖/默认 + 审计字段）
  PUT    /admin/flags/{user_code}/{flag_key}     写覆盖行（body: enabled/reason）
  DELETE /admin/flags/{user_code}/{flag_key}     删覆盖行（回到默认）
  GET    /admin/flags/overview                   全局视图（override/disabled 计数）
  GET    /admin/profiles/{user_code}             用户画像（L1 + L2/L3 记忆 + flag + 会话）
  GET    /admin/profiles?query=xxx&limit=20      画像列表检索（前缀 + updated_at 倒序）

鉴权：**无 token 鉴权**（产品决定，部署于受控网络/网关层；app 级 admin 组限流
仍生效，见 app.py rate_limit_middleware / RATE_LIMIT_ADMIN_RPM）。
审计：flag 表自带 reason/updated_by/updated_at；每次 PUT 追加 trace_event（无活跃
追踪时 no-op，同时落日志）。
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Query, Request
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ---- 开关管理（§5.1） ----

@router.get("/flags")
async def get_user_flags(user_code: str = Query(..., min_length=1)):
    """用户开关面板：FLAG_REGISTRY 全量 + 用户覆盖行合并。"""
    from omnibox_agent.services import flag_service

    rows = await asyncio.to_thread(flag_service.list_flag_rows_sync, user_code)
    by_flag = {r["flag"]: r for r in rows}
    flags: list[dict[str, Any]] = []
    for name, (env_name, default) in flag_service.FLAG_REGISTRY.items():
        wired = name in flag_service.WIRED_FLAGS
        env_v = flag_service.env_override(name)  # None/True/False
        row = by_flag.get(name)
        if row:
            flags.append({
                "key": name, "enabled": row["enabled"], "source": "user_override",
                "reason": row.get("reason"), "updated_by": row.get("updated_by"),
                "updated_at": row.get("updated_at"), "default": default, "wired": wired,
                "env_override": env_v,
            })
        else:
            flags.append({"key": name, "enabled": default,
                          "source": "registry_default", "default": default, "wired": wired,
                          "env_override": env_v})
    return {"ok": True, "user_code": user_code, "flags": flags}


@router.put("/flags/{user_code}/{flag_key}")
async def put_user_flag(user_code: str, flag_key: str, request: Request):
    """写/更新用户 flag 覆盖行。body: {"enabled": bool|null, "reason": "..."}。

    enabled=null（或缺省）= 删除覆盖行（回到默认，等价 DELETE）。
    写库成功即失效进程内缓存（同 worker 即时生效，§4.4）。
    """
    from omnibox_agent.services import flag_service

    if flag_key not in flag_service.FLAG_REGISTRY:
        return JSONResponse(status_code=404, content={"ok": False, "reason": "unknown flag"})
    try:
        body = await request.json()
    except Exception:
        body = {}
    enabled = body.get("enabled")
    # 类型校验：enabled 必须是布尔或 null，避免字符串 "false" 被 bool() 判为 True
    # 翻转语义（"false" → True 会错误地开启开关）
    if enabled is not None and not isinstance(enabled, bool):
        return JSONResponse(status_code=400,
                            content={"ok": False, "reason": "enabled must be boolean or null"})
    reason = (body.get("reason") or None) and str(body["reason"])[:255]
    # updated_by 服务端绑定（§5.3）：统一记录为 "admin"；不信任 body 自报（可伪造）
    updated_by = "admin"

    if enabled is None:
        ok = await asyncio.to_thread(
            flag_service.delete_flag_row_sync, user_code, flag_key)
        action = "reset_default"
    else:
        ok = await asyncio.to_thread(
            flag_service.set_flag_row_sync, user_code, flag_key, bool(enabled),
            reason, updated_by)
        action = "override"
    if not ok:
        return JSONResponse(status_code=500, content={"ok": False, "reason": "write failed"})

    # 审计（§5.3）：trace 体系（无活跃追踪时 no-op）+ 日志双落
    from omnibox_agent.core.trace_recorder import trace_event
    trace_event("admin.flag_update", phase="admin", data={
        "user_code": user_code, "flag": flag_key, "enabled": enabled,
        "action": action, "reason": reason, "updated_by": updated_by,
    })
    log.info("admin.flag_update user=%s flag=%s enabled=%s action=%s reason=%s by=%s",
             user_code, flag_key, enabled, action, reason, updated_by)
    return {"ok": True}


@router.delete("/flags/{user_code}/{flag_key}")
async def delete_user_flag(user_code: str, flag_key: str):
    """删除用户 flag 覆盖行（回到 FLAG_REGISTRY 默认值）。"""
    from omnibox_agent.services import flag_service

    if flag_key not in flag_service.FLAG_REGISTRY:
        return JSONResponse(status_code=404, content={"ok": False, "reason": "unknown flag"})
    ok = await asyncio.to_thread(flag_service.delete_flag_row_sync, user_code, flag_key)
    if not ok:
        return JSONResponse(status_code=500, content={"ok": False, "reason": "delete failed"})
    from omnibox_agent.core.trace_recorder import trace_event
    trace_event("admin.flag_update", phase="admin", data={
        "user_code": user_code, "flag": flag_key, "action": "reset_default",
    })
    log.info("admin.flag_update user=%s flag=%s action=reset_default", user_code, flag_key)
    return {"ok": True}


@router.get("/flags/overview")
async def flags_overview():
    """全局视图（平台首页）：每个 flag 的默认值 + 覆盖/关闭计数。"""
    from omnibox_agent.services import flag_service

    stats = await asyncio.to_thread(flag_service.overview_stats_sync)
    overview = []
    for name, (_env_name, default) in flag_service.FLAG_REGISTRY.items():
        st = stats.get(name) or {}
        overview.append({
            "key": name, "default": default, "wired": name in flag_service.WIRED_FLAGS,
            "env_override": flag_service.env_override(name),
            "override_count": st.get("override_count", 0),
            "disabled_count": st.get("disabled_count", 0),
        })
    return {"ok": True, "flags": overview}


# ---- 画像查看（§5.2） ----

@router.get("/profiles/{user_code}")
async def get_user_profile(user_code: str):
    """用户画像：L1 profile/stats + L2/L3 记忆分组 + flag + 会话统计。"""
    from omnibox_agent.services import flag_service
    from omnibox_agent.services.long_term_store import (
        list_memories_by_status_sync, get_profile_sync)
    from omnibox_agent.core.database import get_session as db_session
    from sqlalchemy import text

    profile = await asyncio.to_thread(get_profile_sync, user_code)
    if profile is None:
        return {"ok": True, "user_code": user_code, "profile": None,
                "hint": "尚未生成（首个周期任务/压缩后产生）"}

    memories, flags = await asyncio.gather(
        asyncio.to_thread(list_memories_by_status_sync, user_code,
                          ("active", "superseded", "deleted")),
        asyncio.to_thread(flag_service.list_flag_rows_sync, user_code),
    )

    # 会话统计（admin 只读聚合，agent_session 表）
    sessions: dict[str, Any] = {"count": 0, "last_active_at": None}
    try:
        def _q():
            s = db_session()
            try:
                row = s.execute(text(
                    "SELECT COUNT(*), MAX(last_active_at) FROM agent_session "
                    "WHERE user_id = :uid"), {"uid": user_code}).fetchone()
                return (int(row[0] or 0), str(row[1]) if row[1] else None)
            finally:
                s.close()
        sessions["count"], sessions["last_active_at"] = await asyncio.to_thread(_q)
    except Exception as e:
        log.debug("admin sessions stat failed (best-effort): %s", e)

    return {
        "ok": True,
        "user_code": user_code,
        "profile": profile.get("profile"),
        "stats": profile.get("stats"),
        "lt_round_count": profile.get("lt_round_count", 0),
        "memories": {
            "active": [_brief(m) for m in memories.get("active", [])],
            "superseded": [_brief(m) for m in memories.get("superseded", [])],
            "deleted": [_brief(m) for m in memories.get("deleted", [])],
        },
        "flags": flags,
        "sessions": sessions,
    }


def _brief(m: dict) -> dict:
    meta = m.get("meta") or {}
    return {
        "memoryId": m.get("memory_id"),
        "memType": m.get("mem_type"),
        "content": m.get("content"),
        "status": m.get("status"),
        "confidence": meta.get("confidence"),
        "importance": meta.get("importance"),
        "hitCount": m.get("hit_count", 0),
        "createdAt": m.get("created_at"),
    }


@router.get("/profiles")
async def search_profiles(
    query: str = Query("", description="user_code 前缀"),
    limit: int = Query(20, ge=1, le=20),
):
    """画像列表检索（平台用户列表页）：user_code 前缀 / updated_at 倒序。"""
    from omnibox_agent.services.long_term_store import search_profiles_sync

    items = await asyncio.to_thread(search_profiles_sync, query, limit)
    return {"ok": True, "query": query, "items": items}
