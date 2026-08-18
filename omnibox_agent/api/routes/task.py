"""/v1/task 端点 —— 澄清机制所需的 Agent 侧任务控制（docs/clarify-mid-ask-design.md §5.5.3）。

Agent 的 ask 流是请求级（由后端 forwardAskStream 驱动，后端断开即取消），
这里的任务控制以 best-effort 为主：
  - POST /v1/task/cancel: 后端 cancel 澄清/中断时调用，停止推理任务（内存登记清理）

上下文实体（/v1/task/context）按 v4.0 简化：simple QA / DAG 的 resume 上下文
随 clarify 事件内联传递，Agent 侧无需额外 Store，故不实现 /v1/task/context。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/task", tags=["task"])

# 内存登记：active_task_id -> 注册时间（cancel 时用于审计 / 清理）
_active_tasks: dict[str, float] = {}


@router.post("/cancel")
async def cancel_task(body: dict):
    """停止 Agent 推理任务（best-effort）。

    body: { taskId, reason }
    Agent 的流式任务是请求级的，后端通过关闭底层 HTTP 连接即已级联取消；
    此处做登记清理与审计，幂等。
    """
    task_id = (body or {}).get("taskId")
    reason = (body or {}).get("reason")
    if not task_id:
        return {"ok": False, "reason": "taskId 不能为空"}

    existed = _active_tasks.pop(task_id, None)
    log.info("Task cancel (best-effort): taskId=%s reason=%s registered=%s",
             task_id, reason, existed is not None)

    from omnibox_agent.core.trace_recorder import trace_event
    trace_event("clarify.task_cancel", phase="clarify",
                data={"task_id": task_id, "reason": reason,
                      "registered": existed is not None})

    return {"ok": True, "cancelled": existed is not None}


def register_task(task_id: str) -> None:
    """登记一个活跃任务 id（可选，供 cancel 审计；线程安全地忽略重复）。"""
    if task_id:
        _active_tasks[task_id] = datetime.now(timezone.utc).timestamp()


def unregister_task(task_id: str | None) -> None:
    """流结束时清理登记（防 _active_tasks 字典累积泄漏）。

    cancel API 是 pop，这里是显式 remove；两者幂等。
    流的 finally / _finalize 必须调一次，否则 cancel API 在 stream 结束后调用
    仍会返回 registered=True（误判为活跃任务），且字典会无限增长。
    """
    if task_id:
        _active_tasks.pop(task_id, None)
