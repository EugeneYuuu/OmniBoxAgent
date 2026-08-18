"""API route modules — split from monolithic routes.py (issue #8)."""

from omnibox_agent.api.routes.ask import router as ask_router
from omnibox_agent.api.routes.embed import router as embed_router
from omnibox_agent.api.routes.ingest import router as ingest_router
from omnibox_agent.api.routes.mcp import router as mcp_router
from omnibox_agent.api.routes.eval import router as eval_router

__all__ = ["ask_router", "embed_router", "ingest_router", "mcp_router", "eval_router"]
