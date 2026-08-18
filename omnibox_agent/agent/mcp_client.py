"""MCP Client: connect to external MCP servers and expose tools to agents.

Redesigned architecture:
  - McpClient: wraps a single MCP server connection (stdio / sse / streamable_http).
  - McpManager: central manager replacing the old McpRegistry.
    Supports runtime add / remove / reload of MCP servers with JSON persistence.
  - McpStore: JSON-file persistence layer for server configs.

Tool naming convention: {server_name}__{tool_name}  (double underscore separator).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omnibox_agent.core.config import McpServerConfig, MCPConfig, get_config

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

TOOL_SEP = "__"
# Default store path: project root (where .env lives), not inside the package.
DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "mcp_servers.json"
MCP_PROTOCOL_VERSION = "2024-11-05"


# ── McpClient ──────────────────────────────────────────────────────────────


class McpClient:
    """Wraps an MCP session to a single server.

    Supports three transports:
      - stdio:           local subprocess via mcp SDK
      - sse:             Server-Sent Events via mcp SDK
      - streamable_http: raw JSON-RPC 2.0 over httpx

    Exposes public methods only — no private attribute access from callers.
    """

    def __init__(self, server_config: McpServerConfig) -> None:
        self._config: McpServerConfig = server_config
        self._session: Any = None
        self._tools_cache: list[dict] | None = None
        self._stale: bool = False
        self._tool_timeout_s: float = 10.0
        self._transport_cm: Any = None   # stdio/sse context manager (entered)
        self._session_cm: Any = None      # ClientSession context manager (entered)
        self._session_id: str | None = None
        self._http_client: Any = None  # httpx.AsyncClient
        self._request_id: int = 0
        self._connect_lock: asyncio.Lock = asyncio.Lock()

    # ── Properties ──

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def transport(self) -> str:
        return self._config.transport

    @property
    def is_connected(self) -> bool:
        """True if the session appears healthy (not stale)."""
        if self._config.transport in ("streamable_http", "http"):
            return not self._stale and self._session_id is not None
        return not self._stale and self._session is not None

    # ── Connection lifecycle ──

    async def connect(self) -> None:
        """Establish MCP session based on transport type."""
        if TOOL_SEP in self._config.name:
            raise ValueError(
                f"MCP server name '{self._config.name}' contains '{TOOL_SEP}', "
                f"which conflicts with the tool-name separator"
            )

        cfg = get_config().mcp
        self._tool_timeout_s = cfg.tool_timeout_s

        async with self._connect_lock:
            try:
                if self._config.transport == "stdio":
                    await self._connect_stdio()
                elif self._config.transport == "sse":
                    await self._connect_sse()
                elif self._config.transport in ("streamable_http", "http"):
                    await self._connect_streamable_http()
                else:
                    raise ValueError(f"Unknown MCP transport: {self._config.transport}")

                self._stale = False
                self._tools_cache = None  # Invalidate cache on (re)connect
                log.info("MCP client connected to '%s' [%s]", self._config.name, self._config.transport)
            except Exception as e:
                log.error("MCP connect to '%s' failed: %s", self._config.name, e)
                self._stale = True
                raise

    async def _connect_stdio(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args,
            env={**os.environ, **self._config.env} if self._config.env else None,
        )
        # Manually enter the stdio_client context manager and keep both the
        # transport CM and the ClientSession CM so we can exit them explicitly
        # in close(). This avoids the AsyncExitStack + anyio "cancel scope in a
        # different task" noise when the stack is torn down from another task.
        self._transport_cm = stdio_client(params)
        read, write = await self._transport_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

    async def _connect_sse(self) -> None:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        self._transport_cm = sse_client(self._config.url)
        read, write = await self._transport_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

    async def _connect_streamable_http(self) -> None:
        import httpx

        self._http_client = httpx.AsyncClient(timeout=30.0)
        self._session_id = None
        self._request_id = 0

        resp = await self._http_client.post(
            self._config.url,
            headers={"Accept": "application/json"},
            json={
                "jsonrpc": "2.0", "id": "init-1", "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "omnibox-agent", "version": "0.1.0"},
                },
            },
        )
        resp.raise_for_status()
        self._session_id = resp.headers.get("mcp-session-id", "")
        data = resp.json()
        server_info = data.get("result", {}).get("serverInfo", {})
        log.info(
            "MCP Streamable HTTP connected to '%s' [%s v%s], session=%s",
            self._config.name,
            server_info.get("name", "unknown"),
            server_info.get("version", ""),
            self._session_id[:8] if self._session_id else "",
        )

        # Send initialized notification
        await self._http_client.post(
            self._config.url,
            headers={
                "Accept": "application/json",
                "mcp-session-id": self._session_id,
            },
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

    async def close(self) -> None:
        """Close the MCP session and release all resources."""
        async with self._connect_lock:
            try:
                if self._config.transport in ("stdio", "sse"):
                    # Exit the ClientSession CM, then the transport CM.
                    # Swallow anyio "cancel scope in a different task" noise:
                    # the stdio/sse transport spawns background pump tasks whose
                    # cancel scope was entered in the connect task; closing from
                    # another task (e.g. lifespan shutdown) can raise here. The
                    # connection is already dead, so this is safe to ignore.
                    if self._session_cm is not None:
                        try:
                            await self._session_cm.__aexit__(None, None, None)
                        except Exception as e:  # noqa: BLE001
                            log.debug("stdio/sse session exit noise: %s", e)
                        self._session_cm = None
                    if self._transport_cm is not None:
                        try:
                            await self._transport_cm.__aexit__(None, None, None)
                        except Exception as e:  # noqa: BLE001
                            log.debug("stdio/sse transport exit noise: %s", e)
                        self._transport_cm = None
                elif self._config.transport in ("streamable_http", "http") and self._http_client:
                    await self._http_client.aclose()
                    self._http_client = None
            except Exception as e:
                log.warning("Error closing MCP session '%s': %s", self._config.name, e)
            self._session = None
            self._session_id = None
            self._tools_cache = None
            self._stale = True

    async def _ensure_session(self) -> None:
        """Reconnect if session is stale or missing."""
        if self._config.transport in ("streamable_http", "http"):
            if self._stale or self._session_id is None:
                await self.close()
                await self.connect()
        else:
            if self._stale or self._session is None:
                await self.close()
                await self.connect()

    # ── Tool operations ──

    async def list_tools(self, *, refresh: bool = False) -> list[dict]:
        """List tools from this server, with caching.

        Args:
            refresh: If True, bypass cache and re-fetch from server.
        """
        if self._tools_cache is not None and not refresh:
            return self._tools_cache

        await self._ensure_session()
        if not self.is_connected:
            return []

        try:
            if self._config.transport in ("streamable_http", "http"):
                tools = await self._list_tools_http()
            else:
                tools = await self._list_tools_mcp_session()
            self._tools_cache = tools
            return tools
        except Exception as e:
            log.warning("Failed to list tools from '%s': %s", self._config.name, e)
            self._stale = True
            return []

    async def _list_tools_http(self) -> list[dict]:
        self._request_id += 1
        resp = await self._http_client.post(
            self._config.url,
            headers={
                "Accept": "application/json",
                "mcp-session-id": self._session_id,
            },
            json={
                "jsonrpc": "2.0",
                "id": f"list-{self._request_id}",
                "method": "tools/list",
                "params": {},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        raw_tools = data.get("result", {}).get("tools", [])
        return [
            {
                "name": f"{self._config.name}{TOOL_SEP}{tool['name']}",
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {}),
            }
            for tool in raw_tools
        ]

    async def _list_tools_mcp_session(self) -> list[dict]:
        result = await self._session.list_tools()
        return [
            {
                "name": f"{self._config.name}{TOOL_SEP}{tool.name}",
                "description": tool.description or "",
                "parameters": tool.inputSchema if hasattr(tool, "inputSchema") else {},
            }
            for tool in result.tools
        ]

    async def call_tool(self, tool_name: str, **kwargs: Any) -> str:
        """Call a tool on this server.

        The *tool_name* must already be stripped of the server prefix.
        Implements stale-session detection and automatic reconnection.
        """
        await self._ensure_session()

        try:
            if self._config.transport in ("streamable_http", "http"):
                return await self._call_tool_http(tool_name, **kwargs)
            else:
                return await self._call_tool_mcp(tool_name, **kwargs)
        except asyncio.TimeoutError:
            log.warning("Tool '%s' call timed out after %.1fs", tool_name, self._tool_timeout_s)
            self._stale = True
            return "[Tool call timed out]"
        except Exception as e:
            log.warning("Tool '%s' call failed: %s", tool_name, e)
            self._stale = True
            return f"[Tool call failed: {type(e).__name__}]"

    async def _call_tool_http(self, tool_name: str, **kwargs: Any) -> str:
        async with asyncio.timeout(self._tool_timeout_s):
            self._request_id += 1
            resp = await self._http_client.post(
                self._config.url,
                headers={
                    "Accept": "application/json",
                    "mcp-session-id": self._session_id,
                },
                json={
                    "jsonrpc": "2.0",
                    "id": f"call-{self._request_id}",
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": kwargs},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            result = data.get("result", {})
            content_list = result.get("content", [])
            texts = [
                item["text"]
                for item in content_list
                if isinstance(item, dict) and "text" in item
            ]
            return "\n".join(texts) if texts else json.dumps(result)

    async def _call_tool_mcp(self, tool_name: str, **kwargs: Any) -> str:
        if self._session is None:
            return "[MCP session unavailable]"
        async with asyncio.timeout(self._tool_timeout_s):
            result = await self._session.call_tool(tool_name, arguments=kwargs)
            if hasattr(result, "content") and result.content:
                texts = [
                    item.text for item in result.content if hasattr(item, "text")
                ]
                return "\n".join(texts)
            return str(result)

    # ── Public status (no private attribute access from callers) ──

    def get_status(self) -> dict:
        """Return a serializable status dict for this client."""
        tools = self._tools_cache or []
        endpoint = (
            self._config.url
            if self._config.transport not in ("stdio",)
            else self._config.command
        )
        return {
            "name": self._config.name,
            "transport": self._config.transport,
            "endpoint": endpoint,
            "connected": self.is_connected,
            "tool_count": len(tools),
            "tools": [
                {"name": t["name"], "description": t.get("description", "")[:200]}
                for t in tools
            ],
        }


# ── McpStore (JSON persistence) ────────────────────────────────────────────


class McpStore:
    """JSON-file persistence for MCP server configurations.

    On startup, servers from the JSON file are merged with those from the
    MCP_SERVERS env var (env takes precedence for duplicates).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else DEFAULT_STORE_PATH

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[McpServerConfig]:
        """Load server configs from the JSON file."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return []
            servers_data = data.get("servers", [])
            if not isinstance(servers_data, list):
                return []
            return [self._dict_to_config(item) for item in servers_data if isinstance(item, dict)]
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load MCP store from %s: %s", self._path, e)
            return []

    def save(self, servers: list[McpServerConfig]) -> None:
        """Persist server configs to the JSON file."""
        data = {
            "servers": [self._config_to_dict(s) for s in servers],
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            log.error("Failed to save MCP store to %s: %s", self._path, e)

    def _dict_to_config(self, item: dict) -> McpServerConfig:
        env = item.get("env", {})
        if not isinstance(env, dict):
            env = {}
        return McpServerConfig(
            name=item.get("name", ""),
            transport=item.get("transport", "stdio"),
            command=item.get("command", ""),
            args=item.get("args", []),
            url=item.get("url", ""),
            env={str(k): str(v) for k, v in env.items()},
        )

    def _config_to_dict(self, config: McpServerConfig) -> dict:
        return {
            "name": config.name,
            "transport": config.transport,
            "command": config.command,
            "args": config.args,
            "url": config.url,
            "env": config.env,
        }


# ── McpManager (replaces McpRegistry) ──────────────────────────────────────


class McpManager:
    """Central manager for MCP server connections.

    Replaces the old McpRegistry with:
      - Runtime add / remove / reload of MCP servers
      - JSON-file persistence (survives restarts)
      - Thread-safe async operations
      - Public API only (no private attribute access from callers)

    Usage:
        manager = McpManager()
        await manager.startup()           # load from env + store, connect all
        await manager.add_server(config)  # add at runtime
        await manager.remove_server(name) # remove at runtime
        await manager.reload_server(name) # reconnect a server
        tools = manager.list_all_tools()  # sync, cached
        result = await manager.call("server__tool", **kwargs)
        await manager.shutdown()          # disconnect all
    """

    def __init__(self, store_path: Path | str | None = None) -> None:
        self._clients: dict[str, McpClient] = {}
        self._store = McpStore(store_path)
        self._lock = asyncio.Lock()
        self._started = False

    # ── Lifecycle ──

    async def startup(self, env_servers: list[McpServerConfig] | None = None) -> None:
        """Initialize from env + persisted store, then connect to all servers."""
        async with self._lock:
            if self._started:
                log.warning("McpManager already started")
                return

            # Merge env servers with persisted servers (env takes precedence)
            env_servers = env_servers or []
            env_names = {s.name for s in env_servers}
            stored_servers = self._store.load()
            merged = list(env_servers)
            for s in stored_servers:
                if s.name not in env_names:
                    merged.append(s)

            # Persist the merged set so the store stays in sync
            self._store.save(merged)

            for srv in merged:
                if srv.name in self._clients:
                    continue
                client = McpClient(srv)
                self._clients[srv.name] = client
                try:
                    await client.connect()
                except Exception as e:
                    log.warning("MCP server '%s' connect failed (non-fatal): %s", srv.name, e)

            self._started = True
            log.info("McpManager started with %d servers", len(self._clients))

    async def shutdown(self) -> None:
        """Disconnect all MCP clients."""
        async with self._lock:
            for name, client in self._clients.items():
                try:
                    await client.close()
                except Exception as e:
                    log.warning("Error stopping MCP client '%s': %s", name, e)
            self._clients.clear()
            self._started = False
            log.info("McpManager shut down")

    # ── Runtime CRUD ──

    async def add_server(self, config: McpServerConfig) -> dict:
        """Add a new MCP server, connect to it, and persist.

        Returns a status dict.
        Raises ValueError if the name is already in use.
        """
        if TOOL_SEP in config.name:
            raise ValueError(
                f"Server name '{config.name}' contains '{TOOL_SEP}', "
                f"which conflicts with the tool-name separator"
            )

        async with self._lock:
            if config.name in self._clients:
                raise ValueError(f"MCP server '{config.name}' already exists")

            client = McpClient(config)
            self._clients[config.name] = client
            connect_error = None
            try:
                await client.connect()
            except Exception as e:
                connect_error = str(e)
                log.warning("MCP server '%s' connect failed (added but not connected): %s", config.name, e)

            # Persist updated server list
            self._persist()

            result = client.get_status()
            if connect_error:
                result["warning"] = f"Server added but connection failed: {connect_error}"
            return result

    async def remove_server(self, name: str) -> dict:
        """Remove an MCP server, disconnect, and persist.

        Returns a confirmation dict.
        Raises KeyError if the server doesn't exist.
        """
        async with self._lock:
            if name not in self._clients:
                raise KeyError(f"MCP server '{name}' not found")

            client = self._clients.pop(name)
            try:
                await client.close()
            except Exception as e:
                log.warning("Error closing MCP client '%s': %s", name, e)

            self._persist()
            return {"ok": True, "removed": name}

    async def reload_server(self, name: str) -> dict:
        """Reconnect to an existing MCP server.

        Raises KeyError if the server doesn't exist.
        """
        async with self._lock:
            if name not in self._clients:
                raise KeyError(f"MCP server '{name}' not found")

            client = self._clients[name]
            # Close and reconnect with the same config
            try:
                await client.close()
            except Exception as e:
                log.warning("Error closing MCP client '%s' during reload: %s", name, e)

            try:
                await client.connect()
            except Exception as e:
                log.warning("MCP server '%s' reload connect failed: %s", name, e)

            return client.get_status()

    async def reload_all(self) -> list[dict]:
        """Reconnect all MCP servers."""
        async with self._lock:
            results = []
            for name, client in list(self._clients.items()):
                try:
                    await client.close()
                    await client.connect()
                except Exception as e:
                    log.warning("MCP server '%s' reload failed: %s", name, e)
                results.append(client.get_status())
            return results

    # ── Query ──

    def list_servers(self) -> list[dict]:
        """Return status of all registered servers (sync, no network calls)."""
        return [client.get_status() for client in self._clients.values()]

    def get_server(self, name: str) -> dict | None:
        """Return status of a single server, or None if not found."""
        client = self._clients.get(name)
        return client.get_status() if client else None

    async def get_server_tools(self, name: str, *, refresh: bool = False) -> list[dict] | None:
        """Return tools from a specific server.

        If refresh=True, bypasses cache and re-fetches from the server.
        Returns None if the server doesn't exist.
        """
        client = self._clients.get(name)
        if client is None:
            return None
        return await client.list_tools(refresh=refresh)

    def list_all_tools(self) -> list[dict]:
        """Return all tool schemas from all connected servers (sync, cached).

        Tools are fetched lazily on first access. Returns cached results only.
        """
        all_tools: list[dict] = []
        for client in self._clients.values():
            tools = client._tools_cache
            if tools:
                all_tools.extend(tools)
        return all_tools

    async def list_all_tools_async(self, *, refresh: bool = False) -> list[dict]:
        """Async version: fetches tools from all servers, populating caches."""
        all_tools: list[dict] = []
        for client in self._clients.values():
            try:
                tools = await client.list_tools(refresh=refresh)
                all_tools.extend(tools)
            except Exception as e:
                log.warning("Failed to list tools from '%s': %s", client.name, e)
        return all_tools

    async def call(self, prefixed_name: str, **kwargs: Any) -> str:
        """Call a tool by its prefixed name ({server}__{tool}).

        Routes to the correct McpClient based on the prefix.
        If the call fails, refreshes tool lists and retries once.
        """
        if TOOL_SEP not in prefixed_name:
            log.warning("Tool name '%s' lacks server prefix, cannot route", prefixed_name)
            return "[Tool routing failed: missing server prefix]"

        server_name, tool_name = prefixed_name.split(TOOL_SEP, 1)
        client = self._clients.get(server_name)
        if client is None:
            return f"[Unknown server: {server_name}]"

        try:
            return await client.call_tool(tool_name, **kwargs)
        except Exception:
            # Unknown tool or stale: refresh and retry once
            try:
                await client.list_tools(refresh=True)
            except Exception:
                pass
            try:
                return await client.call_tool(tool_name, **kwargs)
            except Exception as e:
                return f"[Tool call failed: {type(e).__name__}]"

    # ── Health ──

    async def health_check(self) -> dict:
        """Check health of all MCP connections."""
        servers = {}
        all_healthy = True
        for name, client in self._clients.items():
            status = client.get_status()
            if not status["connected"]:
                all_healthy = False
            servers[name] = status["connected"]
        return {
            "status": "ok" if all_healthy else "degraded",
            "server_count": len(self._clients),
            "servers": servers,
        }

    # ── Internal ──

    def _persist(self) -> None:
        """Save current server configs to the store."""
        configs = [
            McpServerConfig(
                name=c.name,
                transport=c.transport,
                command=c._config.command,
                args=c._config.args,
                url=c._config.url,
                env=c._config.env,
            )
            for c in self._clients.values()
        ]
        self._store.save(configs)

    # ── Backward compat with old McpRegistry ──

    @property
    def clients(self) -> dict[str, McpClient]:
        """Backward-compatible access to the client dict."""
        return self._clients

    async def start(self, servers: list[McpServerConfig]) -> None:
        """Backward-compatible start() — delegates to startup()."""
        await self.startup(env_servers=servers)

    async def stop(self) -> None:
        """Backward-compatible stop() — delegates to shutdown()."""
        await self.shutdown()

    async def _refresh_tools(self) -> None:
        """Refresh tool schemas from all servers."""
        for client in self._clients.values():
            try:
                await client.list_tools(refresh=True)
            except Exception as e:
                log.warning("Tool refresh failed for '%s': %s", client.name, e)


# ── Backward-compatible alias ──────────────────────────────────────────────

# McpRegistry is kept as an alias so existing imports continue to work.
# New code should use McpManager directly.
McpRegistry = McpManager
