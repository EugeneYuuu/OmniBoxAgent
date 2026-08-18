"""Embedding service: OpenAI-compatible embeddings with content fingerprinting.

Each content item -> single vector (dimension defined by global EmbeddingConfig,
currently Zhipu embedding-3) via multi-segment sampling of
title+summary+content_text+tags.

Embedding 始终使用全局配置(智谱 embedding-3),不接受任何 per-user provider 覆盖。

Text construction:
  title (full) + summary (first 200 chars) + content_text (first 200 chars)
  + tags (comma-separated)

Embedding is computed as the average of up to 3 segments (beginning, middle, end)
if the total text exceeds 600 chars.
"""

import hashlib
import json
import logging
import time
from typing import Any

import httpx

from omnibox_agent.core.config import get_config
from omnibox_agent.core.auth import resolve_auth_token

log = logging.getLogger(__name__)


def compute_content_fingerprint(
    title: str | None,
    summary: str | None,
    content_text: str | None,
    tags: list[str] | None,
    collected_at: str | None,
    updated_at: str | None,
    ai_tag: str | None,
) -> str:
    """Compute a content fingerprint for change detection.

    Covers: title, summary, content_text, tags, collected_at, updated_at, ai_tag.
    """
    cfg = get_config()
    parts = [
        title or "",
        summary or "",
        content_text or "",
        ",".join(sorted(tags)) if tags else "",
        collected_at or "",
        updated_at or "",
        ai_tag or "",
        cfg.fingerprint_salt,
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_embedding_text(
    title: str | None,
    summary: str | None,
    content_text: str | None,
    tags: list[str] | None,
) -> str:
    """Build the text to be embedded for a content item.

    Order: title -> summary -> content_text -> tags.
    Each field is trimmed to reasonable length.
    """
    parts = []
    if title:
        parts.append(title.strip())
    if summary:
        parts.append(summary.strip()[:300])
    if content_text:
        parts.append(content_text.strip()[:300])
    if tags:
        parts.append(", ".join(tags))

    return " | ".join(parts)


def embed_text(text: str, timeout: float = 30.0) -> list[float] | None:
    """Generate embedding for a single text via OpenAI-compatible API.

    Uses multi-segment sampling for long texts (average of up to 3 segments).
    始终使用全局 embedding 配置(智谱 embedding-3,.env EMBEDDING_* /
    EmbeddingConfig),不使用任何 per-user provider 配置。

    timeout（设计 §12.1 热路径延迟控制）：默认 30.0 不变（摄取管线等既有
    调用方零改动）；长期记忆 recall 热路径传 timeout=3.0，配合外层
    asyncio.wait_for 快速降级。
    """
    cfg = get_config().embedding
    if not cfg.api_key:
        log.error("Embedding API key not configured")
        return None

    segments = _sample_segments(text, max_total_chars=600)
    if not segments:
        return None

    embeddings = []
    for seg in segments:
        emb = _call_embedding_api(seg, timeout=timeout)
        if emb:
            embeddings.append(emb)
        else:
            log.warning("Failed to embed segment of length %d", len(seg))

    if not embeddings:
        return None

    # Average multiple segment embeddings
    dim = len(embeddings[0])
    avg = [0.0] * dim
    for emb in embeddings:
        for i in range(dim):
            avg[i] += emb[i]
    for i in range(dim):
        avg[i] /= len(embeddings)

    return avg


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple texts. Falls back to sequential on failure.

    始终使用全局 embedding 配置(智谱 embedding-3)。
    """
    cfg = get_config().embedding
    if not cfg.api_key:
        log.error("Embedding API key not configured")
        return [[] for _ in texts]

    # Try batch API call
    try:
        result = _call_embedding_api_batch(texts)
        if result and len(result) == len(texts):
            return result
    except Exception as e:
        log.warning("Batch embedding failed, falling back to sequential: %s", e)

    # Sequential fallback
    result = []
    for text in texts:
        emb = embed_text(text)
        result.append(emb or [])
    return result


def _sample_segments(text: str, max_total_chars: int = 600) -> list[str]:
    """Sample up to 3 segments from text for embedding.

    If text <= max_total_chars, return it as a single segment.
    Otherwise return beginning, middle, and end segments.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_total_chars:
        return [text]

    seg_len = max_total_chars // 3
    segments = []
    # Beginning
    segments.append(text[:seg_len])
    # Middle
    mid_start = (len(text) - seg_len) // 2
    segments.append(text[mid_start:mid_start + seg_len])
    # End
    segments.append(text[-seg_len:])
    return segments


def _call_embedding_api(text: str, timeout: float = 30.0) -> list[float] | None:
    """Call OpenAI-compatible embedding endpoint for a single text.

    始终使用全局配置的智谱 embedding-3(.env EMBEDDING_* /
    EmbeddingConfig),不使用任何 per-user provider 配置。
    timeout 默认 30.0（现状不变）；recall 热路径由 embed_text 透传 3.0（§12.1）。
    """
    cfg = get_config().embedding
    url = _build_url(cfg.base_url, "/embeddings")
    auth_token = resolve_auth_token(cfg.api_key, cfg.base_url)

    body = {
        "model": cfg.model,
        "input": text,
    }
    # embedding-3 支持 dimensions 参数(512/1024/2048),
    # 显式请求全局配置的维度,与 chroma 索引保持一致。
    if cfg.model == "embedding-3":
        body["dimensions"] = cfg.dimension

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code != 200:
                log.error("Embedding API error %s: %s", resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            return data["data"][0]["embedding"]
    except Exception as e:
        log.error("Embedding API call failed: %s", e)
        return None


def _call_embedding_api_batch(texts: list[str]) -> list[list[float]] | None:
    """Call OpenAI-compatible embedding endpoint with batch input.

    始终使用全局配置的智谱 embedding-3。
    """
    cfg = get_config().embedding
    url = _build_url(cfg.base_url, "/embeddings")
    auth_token = resolve_auth_token(cfg.api_key, cfg.base_url)

    body = {
        "model": cfg.model,
        "input": texts,
    }
    # embedding-3 支持 dimensions 参数(512/1024/2048),
    # 显式请求全局配置的维度,与 chroma 索引保持一致。
    if cfg.model == "embedding-3":
        body["dimensions"] = cfg.dimension

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code != 200:
                log.error("Batch embedding API error %s: %s", resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            return [item["embedding"] for item in data["data"]]
    except Exception as e:
        log.error("Batch embedding API call failed: %s", e)
        return None


def _build_url(base_url: str, path: str) -> str:
    """Build full URL from base_url and path."""
    base = base_url.rstrip("/")
    return f"{base}{path}"


def validate_dimension(embedding: list[float]) -> bool:
    """Validate that embedding dimension matches config."""
    cfg = get_config().embedding
    return len(embedding) == cfg.dimension
