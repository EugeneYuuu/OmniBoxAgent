"""API routes for OmniBoxAgent — assembled from route modules (issue #8).

Endpoints:
  POST /v1/ask/stream       - Streaming Ask endpoint
  GET  /v1/embed/status     - Vector embedding coverage status
  POST /v1/embed/backfill   - Trigger backfill
  POST /v1/embed/delete     - Delete vectors
  POST /v1/embed/sync-item  - Event-driven single-item sync
  POST /v1/ingest           - Multi-modal ingestion
  POST /v1/ingest/backfill  - Full backfill
  GET  /v1/ingest/video-tasks - Video task status
  GET/POST/DELETE /v1/mcp/* - MCP server management
  GET  /v1/eval/template    - Eval set template
  POST /v1/eval/run         - Run evaluation
  GET  /health              - Health check

v4.1: Rate limiting (issue #15), CORS from config (issue #11),
       startup dependency checks (issue #7).
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from omnibox_agent.core.config import get_config

log = logging.getLogger(__name__)

app = FastAPI(
    title="OmniBoxAgent",
    description="OmniHub Ask Agent - RAG Smart Q&A Service",
    version="0.1.0",
)

# ---- CORS from config (issue #11) ----
cfg = get_config()
cors = cfg.cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors.allow_origins,
    allow_credentials=cors.allow_credentials,
    allow_methods=cors.allow_methods,
    allow_headers=cors.allow_headers,
)

# ---- Rate limiting middleware (issue #15) ----

if cfg.rate_limit.enabled:
    import time
    from collections import defaultdict

    _rate_limits: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        """Simple sliding-window rate limiter per IP per endpoint.

        Issue #15: Prevents LLM cost explosion from unlimited /v1/ask/stream calls.
        """
        rl_cfg = cfg.rate_limit
        path = request.url.path

        # Determine rate limit for this endpoint
        if path == "/v1/ask/stream":
            max_rpm = rl_cfg.requests_per_minute
        elif path.startswith("/v1/ingest"):
            max_rpm = rl_cfg.ingest_per_minute
        elif path.startswith("/admin"):
            # admin 组独立配额（USER_FLAG_PLATFORM_DESIGN.md §8）：无 token 鉴权，靠限流兜底
            # ⚠️ /admin 是平台页面(FileResponse)，/admin/flags 等为 API，统一限流（页面低频）
            max_rpm = rl_cfg.admin_per_minute
        else:
            # No rate limit for other endpoints
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        now = time.time()
        window = 60.0  # 1 minute sliding window

        # Prune old entries
        key = f"{client_ip}:{path}"
        entries = _rate_limits[key]
        entries = [t for t in entries if now - t < window]
        _rate_limits[key] = entries

        if len(entries) >= max_rpm:
            return JSONResponse(
                status_code=429,
                content={
                    "ok": False,
                    "reason": "请求过于频繁，请稍后重试",
                    "retry_after": int(window - (now - entries[0])),
                },
            )

        entries.append(now)
        return await call_next(request)


# ---- Register all route modules (issue #8) ----

from omnibox_agent.api.routes.ask import router as ask_router
from omnibox_agent.api.routes.embed import router as embed_router
from omnibox_agent.api.routes.ingest import router as ingest_router
from omnibox_agent.api.routes.mcp import router as mcp_router
from omnibox_agent.api.routes.eval import router as eval_router
from omnibox_agent.api.routes.task import router as task_router
from omnibox_agent.api.routes.skills import router as skills_router
from omnibox_agent.api.routes.memory import router as memory_router
from omnibox_agent.api.routes.admin import router as admin_router

app.include_router(ask_router)
app.include_router(embed_router)
app.include_router(ingest_router)
app.include_router(mcp_router)
app.include_router(eval_router)
app.include_router(task_router)
app.include_router(skills_router)
app.include_router(memory_router)
app.include_router(admin_router)


# ---- Startup / Shutdown (issue #7) ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify dependencies, start harness. Fails fast on critical deps."""
    from omnibox_agent.api.lifecycle import start_harness, stop_harness

    log.info("OmniBoxAgent starting on %s:%s", cfg.agent.host, cfg.agent.port)

    # Start harness with dependency verification (issue #7)
    status = await start_harness()
    if not status.get("ok"):
        errors = status.get("errors", ["Unknown startup error"])
        log.critical("OmniBoxAgent startup failed: %s", "; ".join(errors))
        # FastAPI will return 503 for any request when lifespan fails
        raise RuntimeError(f"Startup failed: {'; '.join(errors)}")

    log.info("OmniBoxAgent started successfully with %d agents", status.get("agents", 0))

    yield

    # Shutdown
    await stop_harness()
    log.info("OmniBoxAgent shutting down")


app.router.lifespan_context = lifespan


# ---- Health ----

@app.get("/health")
async def health():
    """Health check endpoint — aggregates harness + dependency health."""
    try:
        from omnibox_agent.api.lifecycle import get_harness
        harness = get_harness()
        health_data = await harness.health_check()
        return health_data
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(e)},
        )


# ---- 开关平台 F3：/admin 平台前端（§19.4 托管前提）----
# 单文件托管（admin.html 内联 CSS/JS，无静态目录挂载）。无 token 鉴权
# （产品决定；admin 组限流由 app 级中间件保护，见 RATE_LIMIT_ADMIN_RPM）。
# CORS 由 app 级中间件统一放行，平台若独立部署 admin 页，通过页面内 "API 地址"
# 指向本服务并依赖 CORS_ALLOW_ORIGINS。

_ADMIN_HTML = Path(__file__).resolve().parents[2] / "website" / "admin.html"


@app.get("/admin", include_in_schema=False)
async def admin_page():
    if not _ADMIN_HTML.is_file():
        return JSONResponse(status_code=404, content={"ok": False, "detail": "admin.html missing"})
    return FileResponse(_ADMIN_HTML)
