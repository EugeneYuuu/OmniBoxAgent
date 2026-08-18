"""Read user AI config directly from MySQL user_ai_config table.

Decrypts api_key using the shared AES-256-GCM key.
Falls back to request ai_config if DB lookup fails.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from omnibox_agent.core.database import get_session
from omnibox_agent.core.aes_util import decrypt_api_key

log = logging.getLogger(__name__)

# Module-level cache for the decryption key
_encryption_key: str | None = None


def _get_encryption_key() -> str:
    """Lazy-load AES key from env or config."""
    global _encryption_key
    if _encryption_key is None:
        import os
        _encryption_key = os.getenv("AI_CONFIG_ENCRYPTION_KEY", "")
    return _encryption_key


def get_user_ai_config(user_id: str) -> dict[str, Any] | None:
    """Read user AI config from MySQL and decrypt api_key.

    user_id is the public user_code (string); it is resolved to the internal
    bigint users.id that user_ai_config.user_id references.

    Returns dict with keys matching aiConfig format:
      {"baseUrl": ..., "modelName": ..., "apiKey": ..., "provider": ...}

    Returns None if not found or decryption fails.
    """
    from omnibox_agent.services.retrieval_store import resolve_user_id
    internal = resolve_user_id(user_id)
    if internal is None:
        return None

    encryption_key = _get_encryption_key()
    if not encryption_key:
        log.debug("AI_CONFIG_ENCRYPTION_KEY not set, skipping DB ai_config lookup")
        return None

    session = get_session()
    try:
        result = session.execute(
            text(
                "SELECT provider, api_key, base_url, model_name, enabled "
                "FROM user_ai_config WHERE user_id = :uid ORDER BY updated_at DESC LIMIT 1"
            ),
            {"uid": internal},
        )
        row = result.fetchone()
        if not row:
            return None

        provider, encrypted_key, base_url, model_name, enabled = row._mapping.values()

        if not enabled:
            return None

        # Decrypt the API key
        try:
            api_key = decrypt_api_key(encrypted_key or "", encryption_key)
        except Exception as e:
            log.warning("Failed to decrypt ai_config api_key for user %s: %s", user_id, e)
            return None

        config = {
            "provider": provider or "",
            "apiKey": api_key,
        }
        if base_url:
            config["baseUrl"] = base_url
        if model_name:
            config["modelName"] = model_name

        log.debug("Loaded ai_config from DB for user %s: provider=%s model=%s",
                  user_id, provider, model_name)
        return config

    except Exception as e:
        log.warning("Failed to query user_ai_config for user %s: %s", user_id, e)
        return None
    finally:
        session.close()
