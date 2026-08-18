"""/v1/skills/* endpoints — SKILL 渐进式加载管理（docs/skill-support-design.md §6）。

写操作（POST/PATCH/DELETE/reload/rescan）需要管理侧鉴权 `X-Skill-Admin-Key`
（与技能配置 `SKILL_ADMIN_KEY` 常量时间比对）；未配置密钥则写操作默认拒绝（fail-closed）。
只读（列表/详情/health/match）匿名。
"""

import hmac
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from omnibox_agent.api.lifecycle import get_skill_manager
from omnibox_agent.core.config import get_config

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/skills", tags=["skills"])

_ADMIN_HEADER = "X-Skill-Admin-Key"


def _require_admin(request: Request) -> None:
    cfg = get_config().skills
    expected = cfg.admin_key
    if not expected:
        # fail-closed：未配置密钥则写操作默认拒绝
        raise HTTPException(status_code=401, detail="SKILL_ADMIN_KEY 未配置，写操作已禁用")
    provided = request.headers.get(_ADMIN_HEADER) or ""
    if not hmac.compare_digest(provided.strip(), expected):
        raise HTTPException(status_code=401, detail="无效的管理密钥")


def _mgr():
    mgr = get_skill_manager()
    if mgr is None:
        raise HTTPException(status_code=503, detail="Skill manager 未启用（SKILL_ENABLED=false）")
    return mgr


# ── 只读 ──

@router.get("")
async def skill_list():
    return {"ok": True, "skills": _mgr().list_skills()}


@router.get("/health")
async def skill_health():
    mgr = get_skill_manager()
    if mgr is None:
        return {"ok": False, "enabled": False}
    return {
        "ok": True,
        "enabled": True,
        "skill_count": len(mgr.list_skills()),
        "level1_ready": mgr._level1_ready,
    }


@router.get("/{name}")
async def skill_detail(name: str):
    full = await _mgr().get_full(name)
    if full is None:
        raise HTTPException(status_code=404, detail=f"skill '{name}' not found")
    return {"ok": True, "skill": full}


@router.post("/match")
async def skill_match(request: Request):
    """调试：预览 query 匹配哪些技能（只读，匿名）。"""
    mgr = get_skill_manager()
    if mgr is None:
        raise HTTPException(status_code=503, detail="Skill manager 未启用")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    query = body.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Field 'query' is required")
    res = await mgr.resolve(query)
    if res is None:
        return {"ok": True, "matched_by": None, "candidates": [],
                "selected": [], "match_score": None}
    return {
        "ok": True,
        "matched_by": res.matched_by,
        "candidates": res.candidates,
        "selected": [s.name for s in res.selected],
        "match_score": res.match_score,
        "degraded": res.degraded,
        "resources_injected": res.resources_injected,
    }


# ── 写操作（需鉴权） ──

@router.post("")
async def skill_add(request: Request):
    _require_admin(request)
    mgr = _mgr()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Field 'name' is required")

    try:
        result = await mgr.add_skill(
            name,
            source=body.get("source"),
            description=body.get("description") or "",
            tags=body.get("tags") or [],
            instructions=body.get("instructions") or "",
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/{name}")
async def skill_update(name: str, request: Request):
    _require_admin(request)
    mgr = _mgr()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    try:
        result = await mgr.update_skill(name, **body)
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{name}")
async def skill_remove(name: str, request: Request, delete_files: bool = False):
    _require_admin(request)
    mgr = _mgr()
    try:
        return await mgr.remove_skill(name, delete_files=delete_files)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{name}/reload")
async def skill_reload(name: str, request: Request):
    _require_admin(request)
    mgr = _mgr()
    try:
        return await mgr.reload_skill(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/rescan")
async def skill_rescan(request: Request):
    _require_admin(request)
    mgr = _mgr()
    results = await mgr.reload_all()
    return {"ok": True, "rescanned": len(results)}