"""ChromaDB vector store for OmniHub content items."""

import logging
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from omnibox_agent.core.config import get_config

log = logging.getLogger(__name__)

COLLECTION_NAME = "omnihub_items"

# 长期记忆 L3 向量集合（MEMORY_HARNESS_INTEGRATION_DESIGN.md §10.3）
USER_MEMORIES_COLLECTION = "omnihub_user_memories"

_client: Any = None
_collection: Any = None

# 多集合缓存：name -> collection（get_or_create 语义，§10.3）
_named_collections: dict[str, Any] = {}


def _get_client():
    global _client
    if _client is not None:
        return _client

    cfg = get_config().chroma
    if cfg.mode == "http":
        _client = chromadb.HttpClient(
            host=cfg.host,
            port=cfg.port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        log.info("ChromaDB HTTP client connected to %s:%s", cfg.host, cfg.port)
    else:
        _client = chromadb.PersistentClient(
            path=cfg.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        log.info("ChromaDB PersistentClient initialized at %s", cfg.persist_dir)

    return _client


def get_collection() -> Any:
    """Get or create the omnihub_items collection."""
    global _collection
    if _collection is not None:
        return _collection

    client = _get_client()
    cfg = get_config().chroma
    name = cfg.collection_name

    try:
        _collection = client.get_collection(name)
        log.info("ChromaDB collection '%s' found, count=%s", name, _collection.count())
    except Exception:
        _collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("ChromaDB collection '%s' created", name)

    return _collection


def get_named_collection(name: str) -> Any:
    """按名字获取（或创建）任意集合——多集合支持（设计 §10.3）。

    现有 get_collection() 是单集合模块单例（硬编码 cfg.collection_name），
    长期记忆 L3 集合 omnihub_user_memories 走本函数；模块级 dict 缓存，
    get_or_create 语义，复用现有 _get_client()。探测失败向上抛（调用方
    按非 fatal 降级，遵循 §6.1）。
    """
    if name in _named_collections:
        return _named_collections[name]

    client = _get_client()
    try:
        coll = client.get_collection(name)
        log.info("ChromaDB collection '%s' found, count=%s", name, coll.count())
    except Exception:
        coll = client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("ChromaDB collection '%s' created", name)
    _named_collections[name] = coll
    return coll


def reset_collection() -> None:
    """Reset the collection (for testing/re-index)."""
    global _collection
    client = _get_client()
    cfg = get_config().chroma
    name = cfg.collection_name
    try:
        client.delete_collection(name)
        log.warning("ChromaDB collection '%s' deleted", name)
    except Exception:
        pass
    _collection = None


def upsert_vectors(
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    documents: list[str],
) -> None:
    coll = get_collection()
    coll.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    log.debug("Upserted %d vectors", len(ids))


def delete_vectors(ids: list[str]) -> None:
    coll = get_collection()
    coll.delete(ids=ids)
    log.debug("Deleted %d vectors", len(ids))


def query_vectors(
    query_embedding: list[float],
    n_results: int = 20,
    where: Optional[dict] = None,
    where_document: Optional[dict] = None,
) -> dict:
    coll = get_collection()
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }
    if where:
        kwargs["where"] = where
    if where_document:
        kwargs["where_document"] = where_document
    return coll.query(**kwargs)


def count_vectors(where: Optional[dict] = None) -> int:
    """按 where 过滤统计向量数。

    用于"默认不限制 top-k"（用户指令）的全量召回预算：ChromaDB 的
    n_results 必须传具体数值，用 count 动态取用户库向量总数作为召回
    预算，使召回覆盖全部命中而非固定 top-k。失败返回 0。
    """
    coll = get_collection()
    try:
        return int(coll.count(where=where))
    except Exception:
        return 0


def get_comment_docs(content_ids: list[int]) -> dict[int, str]:
    """按 content_id 取回各内容的评论区全文（{note_id}#comments 向量的 document）。

    评论区兜底检索用（用户原则：内容正文找不到问题的语义时，再去该内容的
    评论向量找）。评论向量按内容聚合存储（每条内容一个 #comments 向量，
    document=评论文本），按 content_id 直接 get 即拿到评论全文——不做
    场景关键词预判，"在评论里找答案"由生成 LLM 阅读完成。

    Returns:
        {content_id: 评论文本}，仅含有评论向量的内容。
    """
    if not content_ids:
        return {}
    coll = get_collection()
    result: dict[int, str] = {}
    try:
        fetched = coll.get(
            where={"$and": [
                {"content_id": {"$in": list(content_ids)}},
                {"vec_type": "comments"},
            ]},
            include=["metadatas", "documents"],
        )
        metas = fetched.get("metadatas") or []
        docs = fetched.get("documents") or []
        for i in range(len(metas)):
            meta = metas[i] or {}
            cid = meta.get("content_id")
            text = (docs[i] if i < len(docs) else "") or ""
            if cid is None or not text.strip():
                continue
            cid = int(cid)
            # 同一 content_id 理论上只有一条 #comments 向量；防御性保留最长文本
            if cid not in result or len(text) > len(result[cid]):
                result[cid] = text
    except Exception as e:
        log.warning("get_comment_docs failed: %s", e)
    return result


def get_fingerprints(content_ids: list[int]) -> dict[int, str]:
    """Get fingerprints for content IDs (both legacy and v4.1 vector formats).

    v4.1: vectors are stored as {id}#main / {id}#media (multi-vector), not the
    legacy content_{id}. Query by metadata (user_id not known here, so by
    content_id) and return the main vector's fingerprint.
    """
    coll = get_collection()
    result: dict[int, str] = {}

    # 1. Try legacy ids first (fast path)
    chrome_ids = [f"content_{cid}" for cid in content_ids]
    try:
        fetched = coll.get(ids=chrome_ids)
        if fetched and fetched["ids"]:
            for cid_str, meta in zip(fetched["ids"], fetched["metadatas"]):
                if meta and "content_id" in meta:
                    result[meta["content_id"]] = meta.get("fingerprint", "")
    except Exception:
        pass

    # 2. v4.1 multi-vector format: query by content_id metadata for missing ids
    missing = [cid for cid in content_ids if cid not in result]
    if missing:
        try:
            # ChromaDB where with $in on metadata content_id
            fetched = coll.get(
                where={"content_id": {"$in": missing}},
                include=["metadatas"],
            )
            if fetched and fetched["metadatas"]:
                for meta in fetched["metadatas"]:
                    if meta and "content_id" in meta:
                        cid = meta["content_id"]
                        if cid in missing:
                            # Prefer main vector fingerprint
                            if meta.get("vec_type") == "main" or cid not in result:
                                result[cid] = meta.get("fingerprint", "")
        except Exception:
            pass

    return result


def get_all_content_ids() -> set[int]:
    coll = get_collection()
    all_ids: set[int] = set()
    try:
        fetched = coll.get(include=["metadatas"])
        if fetched and fetched["metadatas"]:
            for meta in fetched["metadatas"]:
                if meta and "content_id" in meta:
                    all_ids.add(int(meta["content_id"]))
    except Exception:
        pass
    return all_ids
