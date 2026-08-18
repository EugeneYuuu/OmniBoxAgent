"""用户记忆自管理端点（MEMORY_HARNESS_INTEGRATION_DESIGN.md §14.3，M4a）。

  GET    /v1/memory?userId=xxx          列出我的 active 记忆（L2 偏好 + L3 情景）
  DELETE /v1/memory/{memory_id}?userId=xxx  软删单条（MySQL status + Chroma 同步）

鉴权与现有 DELETE /v1/session 同级：请求须携带 user_id（user_code）且**归属校验**
（memory 行 user_id 必须等于调用者 user_code，存储层 WHERE 条件保证）；后续若引入
统一鉴权中间件再升级。小程序后续可做"记忆管理"页消费本组端点。
"""

import asyncio
import logging

from fastapi import APIRouter, Query
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["memory"])


@router.get("/memory")
async def list_my_memories(user_id: str = Query(..., alias="userId")):
    """列出我的 active 记忆（按 mem_type 分组返回，前端"记忆管理"页消费）。"""
    from omnibox_agent.services import long_term_store

    if not user_id:
        return JSONResponse(status_code=400, content={"ok": False, "reason": "missing userId"})
    try:
        prefs = await asyncio.to_thread(
            long_term_store.list_active_memories_sync, user_id, "preference", 100)
        others = await asyncio.to_thread(
            long_term_store.list_active_memories_sync, user_id, None, 100)
        facts = [m for m in others if m.get("mem_type") == "fact"]
        episodic = [m for m in others if m.get("mem_type") == "episodic"]

        def _brief(m: dict) -> dict:
            meta = m.get("meta") or {}
            return {
                "memoryId": m.get("memory_id"),
                "memType": m.get("mem_type"),
                "content": m.get("content"),
                "confidence": meta.get("confidence"),
                "importance": meta.get("importance"),
                "hitCount": m.get("hit_count", 0),
                "createdAt": m.get("created_at"),
            }

        return {
            "ok": True,
            "userId": user_id,
            "preferences": [_brief(m) for m in prefs],
            "facts": [_brief(m) for m in facts],
            "episodic": [_brief(m) for m in episodic],
        }
    except Exception as e:
        log.warning("list_my_memories failed (best-effort): %s", e)
        return JSONResponse(status_code=500, content={"ok": False, "reason": "list failed"})


@router.delete("/memory/{memory_id}")
async def delete_my_memory(memory_id: str, user_id: str = Query(..., alias="userId")):
    """软删单条记忆（§14.3）：MySQL status='deleted' + Chroma 向量同步删除。

    归属校验由存储层 WHERE user_id 保证（跨用户 memory_id 不可见 → rowcount=0 → 404）。
    """
    from omnibox_agent.services import long_term_store

    if not memory_id or not user_id:
        return JSONResponse(status_code=400, content={"ok": False, "reason": "missing memoryId/userId"})
    try:
        ok = await asyncio.to_thread(
            long_term_store.soft_delete_memory_sync, user_id, memory_id)
        if not ok:
            return JSONResponse(status_code=404,
                                content={"ok": False, "reason": "memory not found"})
        log.info("LT memory deleted by user: uid=%s mid=%s", user_id, memory_id)
        return {"ok": True}
    except Exception as e:
        log.warning("delete_my_memory failed (best-effort): %s", e)
        return JSONResponse(status_code=500, content={"ok": False, "reason": "delete failed"})
