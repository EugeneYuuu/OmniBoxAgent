#!/usr/bin/env python3
"""记忆系统数据清理脚本（MEMORY_SYSTEM_DESIGN.md v1.1 §3.4；V2.0 扩展长期记忆）。

按项目约定"数据清理由运维/后端统一负责（Agent 启动不清理）"，本脚本供
cron / 后端 @Scheduled / 手动运维调用，安全可重复执行（幂等）：

  - interrupted 清理：删除超过 N 天且**非当前 leaf** 的 interrupted 节点（孤儿残留）
  - 会话清理：删除超过 N 天未活跃的整棵会话树（含全部 entry）
  - 长期记忆维护（V2.0，§13.2/§15）：统计画像批量刷新 + 衰减软删
    （importance < 阈值且 90 天未命中 → deleted；superseded 链 180 天物理清理）

进程内清理（MemoryManager._cleanup_loop）与本脚本职责相同，二选一：
MEMORY_CLEANUP_ENABLED=false 时 Agent 不启动后台任务，交本脚本托管。

用法：
  .venv/bin/python scripts/cleanup_memory.py                 # 默认参数执行
  .venv/bin/python scripts/cleanup_memory.py --dry-run        # 只统计不删除
  .venv/bin/python scripts/cleanup_memory.py --interrupted-days 30 --stale-days 90 --limit 500
  .venv/bin/python scripts/cleanup_memory.py --skip-lt        # 跳过长期记忆维护

生产 ECS 建议 cron（每天一次）：
  0 3 * * * cd /opt/omniboxagent && venv/bin/python scripts/cleanup_memory.py >> /var/log/omnibox_memory_cleanup.log 2>&1
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cleanup_memory")


def _counts() -> dict:
    """统计待清理量（dry-run 与真实执行共用）。"""
    from omnibox_agent.services import session_store

    s = session_store
    from omnibox_agent.core.database import get_session as db
    from sqlalchemy import text
    from datetime import datetime, timedelta, timezone

    CST = timezone(timedelta(hours=8))
    out = {}
    dbc = db()
    try:
        out["interrupted"] = dbc.execute(
            text(
                "SELECT COUNT(*) FROM agent_session_entry e "
                "JOIN agent_session sess ON sess.session_id = e.session_id AND sess.user_id = e.user_id "
                "WHERE e.status = 'interrupted' AND e.created_at < :cutoff "
                "AND (sess.leaf_entry_id IS NULL OR sess.leaf_entry_id <> e.entry_id)"
            ),
            {"cutoff": datetime.now(CST) - timedelta(days=args.interrupted_days)},
        ).fetchone()[0]
        out["stale_sessions"] = dbc.execute(
            text("SELECT COUNT(*) FROM agent_session WHERE last_active_at < :cutoff"),
            {"cutoff": datetime.now(CST) - timedelta(days=args.stale_days)},
        ).fetchone()[0]
    finally:
        dbc.close()
    return out


def main() -> int:
    global args
    parser = argparse.ArgumentParser(description="OmniBoxAgent 记忆系统数据清理")
    parser.add_argument("--interrupted-days", type=int, default=30,
                        help="interrupted 节点保留天数（默认 30）")
    parser.add_argument("--stale-days", type=int, default=90,
                        help="无活跃会话保留天数（默认 90）")
    parser.add_argument("--limit", type=int, default=500,
                        help="单次最多处理量（默认 500；会话树 50）")
    parser.add_argument("--skip-lt", action="store_true",
                        help="跳过长期记忆维护（画像刷新 + 衰减清理）")
    parser.add_argument("--lt-batch-users", type=int, default=100,
                        help="长期记忆画像刷新单批用户数（默认 100）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计并打印，不删除")
    args = parser.parse_args()

    from omnibox_agent.services import session_store

    counts = _counts()
    log.info("待清理预估: interrupted=%d 条, stale_sessions=%d 棵",
             counts["interrupted"], counts["stale_sessions"])
    if args.dry_run:
        log.info("[dry-run] 未执行删除")
        return 0

    n_int = session_store.cleanup_interrupted_sync(days=args.interrupted_days, limit=args.limit)
    n_stale = session_store.cleanup_stale_sessions_sync(days=args.stale_days, limit=min(args.limit, 50))
    log.info("已清理: interrupted=%d 条, stale_sessions=%d 棵", n_int, n_stale)
    if n_int < 0 or n_stale < 0:
        log.error("清理部分失败（-1），请检查 MySQL 权限/连接")
        return 1

    # ---- 长期记忆维护（V2.0 §13.2：统计画像刷新 + 衰减清理；独立 try 不影响上面） ----
    if not args.skip_lt:
        try:
            from omnibox_agent.services import long_term_store
            refreshed = long_term_store.refresh_stats_profiles_sync(args.lt_batch_users)
            decay = long_term_store.cleanup_decay_sync()
            log.info("长期记忆维护: stats_refreshed=%s decay=%s", refreshed, decay)
        except Exception as e:
            log.warning("长期记忆维护失败（best-effort，下轮重试）: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
