"""用户级开关平台 F1：FLAG_REGISTRY + 判定链 + 进程内缓存（USER_FLAG_PLATFORM_DESIGN.md）。

判定链（MEMORY_HARNESS_INTEGRATION_DESIGN.md §17.2）：

    env 显式 false（部署层总闸，恒关）
      > env 显式 true（部署层强制开，极少用）
      > 用户 flag 覆盖（agent_user_feature_flag 表，平台 toggle，即时生效）
      > FLAG_REGISTRY 默认值（缺席即默认）

要点：
  - 缓存：进程内 dict[(uid, flag)] -> bool | None，**写时失效**（admin PUT 后同步
    失效），DB 读仅发生在缓存 miss（每用户每 flag 进程生命周期内至多一次）；
  - 缓存 miss 的 DB 读走 asyncio.to_thread（同步 pymysql 不得进事件循环），
    失败回退默认值（best-effort）；
  - env kill 语义：MEMORY_ENABLED=false / MEMORY_LONG_TERM_ENABLED=false 为全量
    熔断，用户级"开"覆盖失效；env=true 为部署层强制开。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# flag 名 -> (env 变量名, 注册表默认值)
FLAG_REGISTRY: dict[str, tuple[str, bool]] = {
    "memory_session": ("MEMORY_ENABLED", True),
    "memory_long_term": ("MEMORY_LONG_TERM_ENABLED", True),
    "mcp_tools": ("MCP_USER_ENABLED", True),
    "skill_injection": ("SKILL_USER_ENABLED", True),
}

# 已接线（存在消费方）的 flag（USER_FLAG_PLATFORM_DESIGN.md §4.1）——
# 未接线者在监控平台标注「未接线」并禁用 toggle，防止运营误判"已生效"。
# mcp_tools 待 MCP 工具进入 ask 流程后再接线（ask 链路当前无 MCP 调用点）。
WIRED_FLAGS: frozenset[str] = frozenset({
    "memory_session",      # MemoryConfig.is_enabled_for
    "memory_long_term",    # MemoryConfig.is_enabled_for_lt
    "skill_injection",     # graph_skill.skill_node（F4）
})

# ---- 进程内缓存：写时失效 ----
# (user_id, flag_name) -> True/False（已判定）；显式存 bool，miss 即不在 dict
_flag_cache: dict[tuple[str, str], bool] = {}


def invalidate_cache(user_id: str | None = None, flag_name: str | None = None) -> None:
    """写时失效：admin PUT / DELETE 后同步调用（可按用户/flag 精确失效）。"""
    if user_id is None and flag_name is None:
        _flag_cache.clear()
        return
    for key in list(_flag_cache.keys()):
        uid, flag = key
        if user_id is not None and uid != user_id:
            continue
        if flag_name is not None and flag != flag_name:
            continue
        _flag_cache.pop(key, None)


# ---- 同步 DB 读（缓存 miss 时经 asyncio.to_thread 执行） ----

def _load_flag_row_sync(user_id: str, flag_name: str) -> bool | None:
    """读 agent_user_feature_flag 行。返回 None = 无行（用默认值）。

    表不存在（迁移未执行）等任何异常 → None（回退默认，best-effort）。
    """
    from sqlalchemy import text

    from omnibox_agent.core.database import get_session as db_session

    s = db_session()
    try:
        row = s.execute(
            text(
                "SELECT enabled FROM agent_user_feature_flag "
                "WHERE user_id = :uid AND flag_name = :flag LIMIT 1"
            ),
            {"uid": user_id, "flag": flag_name},
        ).fetchone()
        return bool(row[0]) if row is not None else None
    except Exception as e:
        log.debug("flag row load failed (fallback to default): %s", e)
        return None
    finally:
        s.close()


def _env_verdict(env_name: str) -> bool | None:
    """env 判定：显式 false → False；显式 true → True；未设置 → None（继续往下）。"""
    raw = os.getenv(env_name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_override(flag_name: str) -> bool | None:
    """当前 flag 的 env 覆盖值：True=部署层强制开 / False=部署层熔断(kill) / None=未设。

    admin 面板展示用——若返回非 None，说明用户级 toggle 不生效（被 env 覆盖）。
    """
    entry = FLAG_REGISTRY.get(flag_name)
    if entry is None:
        return None
    return _env_verdict(entry[0])


async def is_enabled(user_id: str | None, flag_name: str) -> bool:
    """判定入口（§17.2 判定链）。任何失败回退默认值，绝不抛出。"""
    entry = FLAG_REGISTRY.get(flag_name)
    if entry is None:
        log.warning("unknown flag %r, defaulting False", flag_name)
        return False
    env_name, default = entry

    # 1) env 显式（部署层总闸/强制开）
    env_v = _env_verdict(env_name)
    if env_v is not None:
        return env_v

    # 2) 用户 flag 覆盖（缓存 → DB → 默认）
    uid = user_id or ""
    key = (uid, flag_name)
    cached = _flag_cache.get(key)
    if cached is not None:
        return cached
    try:
        row_enabled: bool | None = await asyncio.wait_for(
            asyncio.to_thread(_load_flag_row_sync, uid, flag_name), timeout=3.0)
    except Exception as e:
        log.debug("flag DB read failed (fallback default): %s", e)
        row_enabled = None
    verdict = default if row_enabled is None else row_enabled
    _flag_cache[key] = verdict
    return verdict


# ---- admin 读写通道（F2 admin.py 消费） ----

def set_flag_row_sync(user_id: str, flag_name: str, enabled: bool,
                      reason: str | None = None, updated_by: str | None = None) -> bool:
    """写入/更新用户 flag 行（admin PUT），成功后调用方负责失效缓存。

    reason / updated_by 为审计字段（USER_FLAG_PLATFORM_DESIGN.md §3.1 / §5.3）。
    """
    from sqlalchemy import text

    from omnibox_agent.core.database import get_session as db_session
    from datetime import datetime, timedelta, timezone

    CST = timezone(timedelta(hours=8))
    if flag_name not in FLAG_REGISTRY:
        return False
    s = db_session()
    try:
        s.execute(
            text(
                "INSERT INTO agent_user_feature_flag "
                "(user_id, flag_name, enabled, reason, updated_by, updated_at) "
                "VALUES (:uid, :flag, :enabled, :reason, :by, :now) "
                "ON DUPLICATE KEY UPDATE enabled = :enabled2, reason = :reason2, "
                "updated_by = :by2, updated_at = :now2"
            ),
            {"uid": user_id, "flag": flag_name, "enabled": 1 if enabled else 0,
             "reason": reason, "by": updated_by,
             "now": datetime.now(CST),
             "enabled2": 1 if enabled else 0, "reason2": reason, "by2": updated_by,
             "now2": datetime.now(CST)},
        )
        s.commit()
        invalidate_cache(user_id, flag_name)  # 写时失效（同步，同进程立即生效）
        return True
    except Exception as e:
        log.warning("set_flag_row failed: %s", e)
        s.rollback()
        return False
    finally:
        s.close()


def delete_flag_row_sync(user_id: str, flag_name: str) -> bool:
    """删除用户 flag 行（回到默认值），成功后调用方缓存已失效。"""
    from sqlalchemy import text

    from omnibox_agent.core.database import get_session as db_session

    s = db_session()
    try:
        s.execute(
            text("DELETE FROM agent_user_feature_flag WHERE user_id = :uid AND flag_name = :flag"),
            {"uid": user_id, "flag": flag_name},
        )
        s.commit()
        invalidate_cache(user_id, flag_name)
        return True
    except Exception as e:
        log.warning("delete_flag_row failed: %s", e)
        s.rollback()
        return False
    finally:
        s.close()


def list_flag_rows_sync(user_id: str) -> list[dict[str, Any]]:
    """列出用户全部 flag 行（admin/画像页；含审计字段）。"""
    from sqlalchemy import text

    from omnibox_agent.core.database import get_session as db_session

    s = db_session()
    try:
        rows = s.execute(
            text("SELECT flag_name, enabled, reason, updated_by, updated_at "
                 "FROM agent_user_feature_flag WHERE user_id = :uid"),
            {"uid": user_id},
        ).fetchall()
        return [{"flag": r[0], "enabled": bool(r[1]),
                 "reason": r[2], "updated_by": r[3], "updated_at": str(r[4])}
                for r in rows]
    except Exception as e:
        log.warning("list_flag_rows failed: %s", e)
        return []
    finally:
        s.close()


def overview_stats_sync() -> dict[str, dict[str, int]]:
    """全局视图（admin overview）：每个 flag 的 override_count / disabled_count。"""
    from sqlalchemy import text

    from omnibox_agent.core.database import get_session as db_session

    s = db_session()
    try:
        rows = s.execute(text(
            "SELECT flag_name, COUNT(*), SUM(enabled = 0) "
            "FROM agent_user_feature_flag GROUP BY flag_name"
        )).fetchall()
        return {r[0]: {"override_count": int(r[1] or 0), "disabled_count": int(r[2] or 0)}
                for r in rows}
    except Exception as e:
        log.warning("overview_stats failed: %s", e)
        return {}
    finally:
        s.close()


__all__ = [
    "FLAG_REGISTRY",
    "WIRED_FLAGS",
    "is_enabled",
    "env_override",
    "invalidate_cache",
    "set_flag_row_sync",
    "delete_flag_row_sync",
    "list_flag_rows_sync",
    "overview_stats_sync",
]
