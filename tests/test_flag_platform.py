"""开关平台测试矩阵（USER_FLAG_PLATFORM_DESIGN.md §9，F5）。

全部 mock DB 层（不依赖 MySQL/Chroma）：判定链 / 缓存失效 / admin 免鉴权 CRUD /
skill_node 用户级接线 / 默认开等价单测。e2e 冒烟（真实 DB + 下一请求生效）见
MEMORY_HARNESS_INTEGRATION_DESIGN.md §19.3，不在本文件范围。

⚠️ admin 端点【无 token 鉴权】（产品决定，USER_FLAG_PLATFORM_DESIGN.md §5）；用例
只验证"无任何请求头即放行 + 限流由中间件覆盖"。

运行：.venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from omnibox_agent.services import flag_service
from omnibox_agent.services.flag_service import (
    FLAG_REGISTRY,
    WIRED_FLAGS,
    invalidate_cache,
    is_enabled,
)

ENV_KEYS = [v[0] for v in FLAG_REGISTRY.values()]


def _clear_flag_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def _mock_flag_rows(monkeypatch, table: dict):
    """_load_flag_row_sync → 内存表 {(uid, flag): bool}；返回 (load, calls)。"""
    invalidate_cache()
    calls = []

    def fake_load(uid, flag):
        calls.append((uid, flag))
        v = table.get((uid, flag))
        if v is None and (uid, flag) not in table:
            return None
        return v

    monkeypatch.setattr(flag_service, "_load_flag_row_sync", fake_load)
    return fake_load, calls


# ── 判定链（§4.2） ────────────────────────────────────────────────────────

def test_flag_default_on(monkeypatch):
    """无 flag 行 + env 未设 → memory_long_term / memory_session 均默认 True。"""
    _clear_flag_env(monkeypatch)
    _mock_flag_rows(monkeypatch, {})
    assert asyncio.run(is_enabled("u_new", "memory_long_term")) is True
    assert asyncio.run(is_enabled("u_new", "memory_session")) is True


def test_flag_env_kill(monkeypatch):
    """env=false → 用户行 enabled=1 也判 False；env=true → 用户行 enabled=0 也判 True。"""
    _clear_flag_env(monkeypatch)
    _mock_flag_rows(monkeypatch, {("u1", "memory_long_term"): True,
                                  ("u2", "memory_long_term"): False})
    monkeypatch.setenv("MEMORY_LONG_TERM_ENABLED", "false")
    assert asyncio.run(is_enabled("u1", "memory_long_term")) is False  # 开覆盖失效
    monkeypatch.setenv("MEMORY_LONG_TERM_ENABLED", "true")
    assert asyncio.run(is_enabled("u2", "memory_long_term")) is True   # 关覆盖失效


def test_flag_user_override(monkeypatch):
    """写行 enabled=0 → 判 False；删行 → 回默认 True。"""
    _clear_flag_env(monkeypatch)
    table = {("u1", "memory_long_term"): False}
    _mock_flag_rows(monkeypatch, table)
    assert asyncio.run(is_enabled("u1", "memory_long_term")) is False
    # 模拟平台 DELETE 覆盖行（写时失效 → 回默认）
    del table[("u1", "memory_long_term")]
    invalidate_cache("u1", "memory_long_term")
    assert asyncio.run(is_enabled("u1", "memory_long_term")) is True


def test_flag_cache_invalidation(monkeypatch):
    """PUT 后（写时失效）同进程立即读新值；DB 读失败 → 回退默认值。"""
    _clear_flag_env(monkeypatch)
    table = {("u1", "memory_long_term"): False}
    _mock_flag_rows(monkeypatch, table)
    assert asyncio.run(is_enabled("u1", "memory_long_term")) is False

    # 模拟平台 PUT 翻开：改行 + 写时失效 → 下一请求（同进程）立即生效
    table[("u1", "memory_long_term")] = True
    invalidate_cache("u1", "memory_long_term")
    assert asyncio.run(is_enabled("u1", "memory_long_term")) is True

    # DB 读失败（新用户缓存 miss）→ best-effort 回退默认值 True
    def boom(uid, flag):
        raise RuntimeError("db down")
    invalidate_cache()
    monkeypatch.setattr(flag_service, "_load_flag_row_sync", boom)
    assert asyncio.run(is_enabled("u_no_row", "memory_long_term")) is True


def test_flag_cache_hit_no_repeat_db(monkeypatch):
    """缓存命中：同 (uid, flag) 第二次判定不再触发 DB 读。"""
    _clear_flag_env(monkeypatch)
    _, calls = _mock_flag_rows(monkeypatch, {})
    asyncio.run(is_enabled("u1", "memory_session"))
    asyncio.run(is_enabled("u1", "memory_session"))
    assert calls.count(("u1", "memory_session")) == 1


# ── Admin API（§5） ──────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """TestClient 不用 with —— 不触发 lifespan(避免启动 harness 连 MySQL/Chroma)。"""
    from fastapi.testclient import TestClient
    from omnibox_agent.api.app import app

    invalidate_cache()
    return TestClient(app)


def test_admin_no_auth(client, monkeypatch):
    """admin 端点免鉴权：无请求头即放行（限流由中间件单独覆盖）。"""
    monkeypatch.setattr(flag_service, "overview_stats_sync", lambda: {})
    assert client.get("/admin/flags/overview").status_code == 200          # 无 header 即放行
    assert client.get("/admin/flags/overview",
                      headers={"X-Auth": "whatever"}).status_code == 200   # 任意头也放行


def test_admin_flag_crud(client, monkeypatch):
    """GET/PUT/DELETE 全链路 + reason 落参 + updated_by 服务端绑定 + wired 标注。"""
    rows = {"u1": [{"flag": "memory_long_term", "enabled": False, "reason": "投诉 #123",
                    "updated_by": "admin", "updated_at": "2026-08-16 10:00:00"}]}
    monkeypatch.setattr(flag_service, "list_flag_rows_sync", lambda uid: rows.get(uid, []))

    set_calls, del_calls = [], []
    monkeypatch.setattr(flag_service, "set_flag_row_sync",
                        lambda uid, f, e, r=None, by=None: (set_calls.append((uid, f, e, r, by)), True)[1])
    monkeypatch.setattr(flag_service, "delete_flag_row_sync",
                        lambda uid, f: (del_calls.append((uid, f)), True)[1])

    # GET：覆盖行合并 registry + wired 字段
    data = client.get("/admin/flags", params={"user_code": "u1"}).json()
    assert data["ok"] is True
    by_key = {f["key"]: f for f in data["flags"]}
    assert set(by_key) == set(FLAG_REGISTRY)
    assert by_key["memory_long_term"]["source"] == "user_override"
    assert by_key["memory_long_term"]["enabled"] is False
    assert by_key["memory_long_term"]["reason"] == "投诉 #123"
    assert by_key["memory_session"]["source"] == "registry_default"
    # wired：三个已接线；mcp_tools 未接线（ask 链路无调用点）
    assert by_key["mcp_tools"]["wired"] is False
    assert all(by_key[k]["wired"] for k in ("memory_session", "memory_long_term", "skill_injection"))

    # PUT：enabled=false + reason；updated_by 服务端绑定（body 自报被忽略）
    resp = client.put("/admin/flags/u1/memory_long_term",
                      json={"enabled": False, "reason": "投诉 #123", "updated_by": "hacker"})
    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert set_calls == [("u1", "memory_long_term", False, "投诉 #123", "admin")]

    # PUT enabled=null = 删除覆盖行（回默认）
    resp = client.put("/admin/flags/u1/memory_long_term",
                      json={"enabled": None})
    assert resp.status_code == 200
    assert del_calls == [("u1", "memory_long_term")]

    # PUT 未知 flag → 404
    assert client.put("/admin/flags/u1/nope",
                      json={"enabled": True}).status_code == 404

    # PUT enabled 类型校验：字符串 "false" 不得被 bool() 翻转为 True（400）
    resp = client.put("/admin/flags/u1/memory_long_term",
                      json={"enabled": "false"})
    assert resp.status_code == 400
    assert set_calls == [("u1", "memory_long_term", False, "投诉 #123", "admin")]  # 未新增写


def test_admin_env_override_field(client, monkeypatch):
    """GET /flags 与 overview 返回 env_override(None/True/False)，运营可见 env 覆盖。"""
    monkeypatch.setattr(flag_service, "list_flag_rows_sync", lambda uid: [])

    # env 未设 → env_override None
    for k in ("MEMORY_ENABLED", "MEMORY_LONG_TERM_ENABLED",
              "MCP_USER_ENABLED", "SKILL_USER_ENABLED"):
        monkeypatch.delenv(k, raising=False)
    data = client.get("/admin/flags", params={"user_code": "u1"}).json()
    by_key = {f["key"]: f for f in data["flags"]}
    assert by_key["memory_long_term"]["env_override"] is None

    # env=false（部署层熔断）→ env_override False
    monkeypatch.setenv("MEMORY_LONG_TERM_ENABLED", "false")
    data = client.get("/admin/flags", params={"user_code": "u1"}).json()
    by_key = {f["key"]: f for f in data["flags"]}
    assert by_key["memory_long_term"]["env_override"] is False

    # env=true（部署层强制开）→ env_override True
    monkeypatch.setenv("MEMORY_LONG_TERM_ENABLED", "true")
    data = client.get("/admin/flags", params={"user_code": "u1"}).json()
    by_key = {f["key"]: f for f in data["flags"]}
    assert by_key["memory_long_term"]["env_override"] is True

    # overview 同样带 env_override
    monkeypatch.setattr(flag_service, "overview_stats_sync", lambda: {})
    data = client.get("/admin/flags/overview").json()
    assert all("env_override" in f for f in data["flags"])


def test_admin_profile(client, monkeypatch):
    """无画像 → hint；有画像 → 完整结构 + user_code 透传（跨用户隔离由 SQL WHERE 保证）。"""
    from omnibox_agent.services import long_term_store as lts

    seen = {}

    def fake_get_profile(uid):
        seen["get_profile"] = uid
        return None
    monkeypatch.setattr(lts, "get_profile_sync", fake_get_profile)

    r = client.get("/admin/profiles/u_nope").json()
    assert r["profile"] is None and "hint" in r
    assert seen["get_profile"] == "u_nope"

    def fake_get_profile2(uid):
        return {"profile": {"library": {"size": 10}}, "stats": {"platforms": []},
                "lt_round_count": 2}
    monkeypatch.setattr(lts, "get_profile_sync", fake_get_profile2)
    monkeypatch.setattr(lts, "list_memories_by_status_sync",
                        lambda uid, statuses: {"active": [{"memory_id": "m1", "mem_type": "preference",
                                                           "content": "平台偏好:优先 bilibili",
                                                           "status": "active", "meta": {"confidence": 0.9},
                                                           "hit_count": 3, "created_at": "2026-08-16"}],
                                               "superseded": [], "deleted": []})
    monkeypatch.setattr(flag_service, "list_flag_rows_sync", lambda uid: [])

    r = client.get("/admin/profiles/u1").json()
    assert r["ok"] is True
    assert r["profile"] == {"library": {"size": 10}}
    assert r["memories"]["active"][0]["hitCount"] == 3
    assert r["memories"]["active"][0]["confidence"] == 0.9


# ── skill_node 用户级接线（§4.5，F4） ───────────────────────────────────

class _FakeSkillManager:
    def __init__(self):
        self.config = SimpleNamespace(enabled=True)
        self.resolved = []

    async def resolve(self, query):
        self.resolved.append(query)
        return {"matched": True, "query": query}


def _fake_ctx(uid: str):
    return SimpleNamespace(input={"user_id": uid, "query": "q"}, artifacts={})


def test_mcp_skill_user_flag(monkeypatch):
    """flag 关 → 该用户跳过 SKILL 注入（resolve 不被调用）；其他用户不受影响。"""
    from omnibox_agent.agent.graph_skill import skill_node

    _clear_flag_env(monkeypatch)
    verdicts = {("u_off", "skill_injection"): False}
    invalidate_cache()

    async def fake_is_enabled(uid, flag):
        return verdicts.get((uid, flag), True)
    monkeypatch.setattr(flag_service, "is_enabled", fake_is_enabled)

    # 关闭用户：跳过注入
    ctx_off, sm = _fake_ctx("u_off"), _FakeSkillManager()
    asyncio.run(skill_node(ctx_off, "q", sm))
    assert ctx_off.artifacts["skills"] is None and sm.resolved == []

    # 其他用户不受影响（默认开 → 正常匹配）
    ctx_on, sm2 = _fake_ctx("u_on"), _FakeSkillManager()
    asyncio.run(skill_node(ctx_on, "q", sm2))
    assert ctx_on.artifacts["skills"] == {"matched": True, "query": "q"}
    assert sm2.resolved == ["q"]

    # mcp_tools 未接线断言（§4.1：平台禁用其 toggle）
    assert "mcp_tools" not in WIRED_FLAGS


# ── 默认开等价单测（§9 test_default_on_e2e 的单测部分） ─────────────────

def test_default_on_e2e(monkeypatch):
    """新用户首问（无行 + env 未设）→ 长期记忆判定开；平台关闭（行=0 + 写时失效）→ 即 False。"""
    from omnibox_agent.core.config import MemoryConfig

    _clear_flag_env(monkeypatch)
    table: dict = {}
    _mock_flag_rows(monkeypatch, table)
    cfg = MemoryConfig()

    # 新用户：默认开（画像空但判定不报错）
    assert asyncio.run(cfg.is_enabled_for_lt("u_new")) is True

    # 平台 toggle 关闭（写覆盖行 + 写时失效，等价 admin PUT 路径）
    table[("u_new", "memory_long_term")] = False
    invalidate_cache("u_new", "memory_long_term")
    assert asyncio.run(cfg.is_enabled_for_lt("u_new")) is False
