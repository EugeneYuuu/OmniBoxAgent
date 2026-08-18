"""Zhipu JWT token generation for API authentication."""

import base64
import hashlib
import hmac
import json
import logging
import time
from urllib.parse import urlparse

log = logging.getLogger(__name__)


def resolve_auth_token(api_key: str, base_url: str) -> str:
    """Resolve auth token: JWT for zhipu (bigmodel.cn), raw key otherwise."""
    if not api_key:
        return ""
    if "bigmodel.cn" in base_url and "." in api_key:
        try:
            return _build_zhipu_jwt(api_key)
        except Exception as e:
            log.warning("Failed to build zhipu JWT, using raw key: %s", e)
    return api_key


def _build_zhipu_jwt(raw_key: str) -> str:
    """Build JWT token per Zhipu official spec.
    Header: {"alg":"HS256","sign_type":"SIGN"}
    Payload: {"api_key":id,"exp":now+3600000,"timestamp":now}
    """
    parts = raw_key.split(".", 1)
    if len(parts) != 2:
        raise ValueError("Invalid zhipu key format (expected id.secret)")

    key_id, secret = parts
    now_ms = int(time.time() * 1000)
    exp_ms = now_ms + 3_600_000  # 1 hour

    header = {"alg": "HS256", "sign_type": "SIGN"}
    payload = {"api_key": key_id, "exp": exp_ms, "timestamp": now_ms}

    header_b64 = _b64url(json.dumps(header, separators=(",", ":")))
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")))
    signing_input = f"{header_b64}.{payload_b64}"

    mac = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256)
    sig = _b64url_bytes(mac.digest())
    return f"{signing_input}.{sig}"


def _b64url(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).rstrip(b"=").decode()


def _b64url_bytes(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
