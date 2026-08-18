"""Smoke test: Zhipu JWT auth through ChatOpenAI.

Two parts:
  1. Offline: verify ZhipuAuth (httpx.Auth) injects a structurally valid JWT
     on the Authorization header (no network).
  2. Online: real call to bigmodel.cn via ChatOpenAI built by
     omnibox_agent.services.llm_langchain._build_model — confirms the JWT is
     accepted and a 200 answer returns.

Usage (from OmniBoxAgent/):
    .venv/bin/python scripts/smoke_zhipu_jwt.py [--offline-only]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

# 让脚本可直接 import omnibox_agent（从仓库根目录运行）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omnibox_agent.core.config import get_config


def _decode_b64url(s: str) -> dict:
    padded = s + "=" * (-len(s) % 4)
    return json.loads(base64.urlsafe_b64decode(padded.encode()))


def offline_check(api_key: str, base_url: str) -> None:
    """不发网络请求，校验 ZhipuAuth 会注入一个结构合法的 JWT。"""
    import httpx

    from omnibox_agent.services.llm_langchain import ZhipuAuth

    auth = ZhipuAuth(api_key, base_url)
    assert auth._is_zhipu, "expected is_zhipu=True for bigmodel.cn + dotted key"

    # 驱动 auth_flow，模拟 httpx 发请求前回调，捕获注入的 Authorization header
    real = httpx.Request("POST", f"{base_url}/chat/completions")
    for _ in auth.auth_flow(real):  # noqa: B007
        pass
    token = real.headers.get("Authorization", "").replace("Bearer ", "")
    assert token, "no Authorization header injected"

    header_b64, payload_b64, _sig = token.split(".")
    header = _decode_b64url(header_b64)
    payload = _decode_b64url(payload_b64)
    assert header.get("alg") == "HS256", f"unexpected alg: {header}"
    assert header.get("sign_type") == "SIGN", f"unexpected sign_type: {header}"
    assert payload.get("api_key", "").startswith(api_key.split(".")[0]), \
        f"payload api_key mismatch: {payload.get('api_key')}"
    assert payload.get("exp", 0) > payload.get("timestamp", 0), "exp not after timestamp"
    print("[offline] JWT ok: alg=HS256 sign_type=SIGN")
    print(f"[offline] Authorization=Bearer {token[:40]}...")


async def online_check(ai_config: dict) -> None:
    """真实调用 bigmodel.cn，验证 JWT 被智谱网关接受并返回 200 内容。"""
    from omnibox_agent.services.llm_langchain import _build_model

    model = _build_model(ai_config, temperature=0.0, max_tokens=16)
    resp = await model.ainvoke(
        [{"role": "user", "content": "只回复两个字：你好"}]
    )
    text = (resp.content or "").strip()
    assert text, "empty response from bigmodel.cn (JWT may be rejected)"
    print(f"[online] 200 ok, model replied: {text!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-only", action="store_true",
                        help="只做离线 JWT 校验，不发真实网络请求")
    args = parser.parse_args()

    cfg = get_config()
    api_key = cfg.qu.api_key or cfg.evaluator.api_key
    base_url = cfg.qu.base_url
    if not api_key:
        print("!! no zhipu api key found in .env (QU_API_KEY/EVALUATOR_API_KEY)")
        sys.exit(1)

    print(f"base_url={base_url}")
    ai_config = {
        "modelName": cfg.qu.model,
        "baseUrl": base_url,
        "apiKey": api_key,
    }

    offline_check(api_key, base_url)

    if args.offline_only:
        print("[offline-only] skipped real network call")
        return

    import asyncio
    asyncio.run(online_check(ai_config))
    print("SMOKE PASS")


if __name__ == "__main__":
    main()