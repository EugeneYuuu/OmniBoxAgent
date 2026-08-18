"""/v1/mcp/* endpoints — MCP server management (issue #8)."""

import logging

from fastapi import APIRouter, HTTPException, Request

from omnibox_agent.api.lifecycle import get_mcp_manager

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/mcp", tags=["mcp"])


@router.get("/servers")
async def mcp_list_servers():
    """List all MCP servers with connection status and tool count."""
    mgr = get_mcp_manager()
    if mgr is None:
        return {"ok": False, "reason": "MCP not configured"}
    servers = mgr.list_servers()
    total_tools = sum(s.get("tool_count", 0) for s in servers)
    return {
        "ok": True,
        "server_count": len(servers),
        "total_tools": total_tools,
        "servers": servers,
    }


@router.get("/servers/{name}")
async def mcp_get_server(name: str):
    """Get status of a single MCP server."""
    mgr = get_mcp_manager()
    if mgr is None:
        return {"ok": False, "reason": "MCP not configured"}
    status = mgr.get_server(name)
    if status is None:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not found")
    return {"ok": True, "server": status}


@router.post("/servers")
async def mcp_add_server(request: Request):
    """Add a new MCP server at runtime."""
    mgr = get_mcp_manager()
    if mgr is None:
        return {"ok": False, "reason": "MCP not configured"}

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Field 'name' is required")

    from omnibox_agent.core.config import McpServerConfig

    transport = body.get("transport", "streamable_http")
    config = McpServerConfig(
        name=name,
        transport=transport,
        command=body.get("command", ""),
        args=body.get("args", []),
        url=body.get("url", ""),
        env={str(k): str(v) for k, v in body.get("env", {}).items()},
    )

    try:
        result = await mgr.add_server(config)
        return {"ok": True, "server": result}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/servers/{name}")
async def mcp_remove_server(name: str):
    """Remove an MCP server."""
    mgr = get_mcp_manager()
    if mgr is None:
        return {"ok": False, "reason": "MCP not configured"}
    try:
        result = await mgr.remove_server(name)
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/servers/{name}/reload")
async def mcp_reload_server(name: str):
    """Reload (reconnect) an MCP server."""
    mgr = get_mcp_manager()
    if mgr is None:
        return {"ok": False, "reason": "MCP not configured"}
    try:
        result = await mgr.reload_server(name)
        return {"ok": True, "server": result}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/servers/{name}/tools")
async def mcp_server_tools(name: str, refresh: bool = False):
    """List tools from a specific MCP server."""
    mgr = get_mcp_manager()
    if mgr is None:
        return {"ok": False, "reason": "MCP not configured"}
    tools = await mgr.get_server_tools(name, refresh=refresh)
    if tools is None:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not found")
    return {"ok": True, "server": name, "tool_count": len(tools), "tools": tools}


@router.get("/tools")
async def mcp_all_tools(refresh: bool = False):
    """List all tools from all MCP servers."""
    mgr = get_mcp_manager()
    if mgr is None:
        return {"ok": False, "reason": "MCP not configured"}
    tools = await mgr.list_all_tools_async(refresh=refresh)
    return {"ok": True, "total_tools": len(tools), "tools": tools}


@router.get("/health")
async def mcp_health():
    """MCP connection health check."""
    mgr = get_mcp_manager()
    if mgr is None:
        return {"ok": False, "reason": "MCP not configured"}
    return await mgr.health_check()


@router.get("/status")
async def mcp_status_legacy():
    """Legacy alias for GET /v1/mcp/servers."""
    return await mcp_list_servers()
