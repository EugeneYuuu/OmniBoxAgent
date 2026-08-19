"""Rerank service: SiliconFlow bge-reranker-v2-m3 精召回（RAG 两阶段检索第二阶段）。

设计（RAG_TWO_STAGE_RETRIEVAL_DESIGN.md §4）：
  - 云端 HTTP rerank，粗召回（ChromaDB 向量）之后、eff_top_n 截取之前。
  - 每条候选写回 item["rerank_score"]（relevance_score，0~1），不覆盖 rrf_score
    （方案 B：下游 fit_budget/_drill_comments 按 rerank_score 定序、降级回退 rrf_score）。
  - 缓存按 (rerank_model, query_hash) → {document 内容哈希: score} 对齐——relevance
    是 (model, query, document) 的确定性函数，与候选池顺序/批次无关，收藏库更新后
    index 映射会把分数安到错误条目上，内容哈希不会。
  - 降级策略：异常/超时/限流 → raise，由调用方回退 RRF（log.warning + trace），
    本模块不吞异常（禁止静默 except: pass）。

线程模型：retrieve_pipeline 的全部调用方（ask_agent / qa_complex / creative_solver）
已用 run_blocking 把它放进 8-ticket 有界线程池执行，rerank 作为其内部同步 HTTP
调用天然跑在线程池中，无需再包一层。因此：
  - HTTP 必须设显式 rerank_timeout_s——无超时的慢调用会占住线程位；
  - 重试严格限死 rerank_max_retries（默认 1）——重试会让单个请求占住 ticket
    远超 rerank_timeout_s，并发 ≥8 时连 vector_search 等 DB 任务都会
    ExecutorBusyError（设计 §4.9 线程池竞争约束）。
"""

import hashlib
import logging
import threading
import time

import httpx

from omnibox_agent.core.config import get_config

log = logging.getLogger(__name__)

# ── TTL 内存缓存（线程安全：8-ticket 线程池并发读写，§4.8）──
# 结构：{(model, query_hash): (expire_ts, {doc_hash: score})}
_CACHE_TTL_S = 3600.0
_CACHE_MAX_QUERIES = 256
_cache_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, dict[str, float]]] = {}

# 云端 QPS 限流信号量：rerank 请求体大（十万级 token），限并发防 TPM 打满
_MAX_CONCURRENT = 4
_qps_sem = threading.Semaphore(_MAX_CONCURRENT)


def rerank(query: str, candidates: list[dict], *, top_n: int) -> list[dict]:
    """调用 SiliconFlow rerank 复排候选池，返回按相关度降序、截断到 top_n。

    Args:
        query: 精排 query（与向量检索同一 search_query，保证语义口径一致）。
        candidates: 候选 item 列表（fused 的全部非评论子集，不预设条数上限；
            超出 rerank_max_candidates 时内部按批分片请求，全量返回）。
        top_n: 返回条数（调用方传 len(candidates)，截断交给检索层 eff_top_n）。

    Returns:
        按 relevance_score 降序的候选子集；每条已写回 item["rerank_score"]。

    Raises:
        ValueError: api_key 未配置（调用方应短路，不发起调用）或候选为空。
        httpx.HTTPError / RuntimeError: 网络/超时/限流且重试耗尽——由调用方回退 RRF。
    """
    cfg = get_config().retrieval

    # fail-fast 短路：key 为空直接视为 disabled（调用方 retrieve_pipeline 已先判，
    # 此处兜底防直接调用 rerank_service 的未来路径白付一次 401 往返）
    if not cfg.rerank_api_key:
        raise ValueError("rerank disabled: rerank_api_key is empty")

    if not candidates:
        # SiliconFlow documents 要求 ≥1，空列表必然 4xx——调用方已短路，兜底
        raise ValueError("rerank candidates is empty")

    # 请求文本：优先向量库原始 document（main=正文 / media=图片解析），
    # 回退 title。summary 在精排时刻尚不可用（enrich 在 eff_top_n 截取之后）。
    documents = [
        ((it.get("document") or "").strip() or (it.get("title") or "").strip())[: cfg.rerank_doc_max_chars]
        for it in candidates
    ]

    # 缓存命中部分先取回；未命中的按 batch_size 分成多批，逐批请求后并入缓存
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    cached = _cache_get(cfg.rerank_model, query_hash)
    doc_hashes = [hashlib.sha256(doc.encode("utf-8")).hexdigest() for doc in documents]
    need_idx = [i for i, dh in enumerate(doc_hashes) if dh not in cached]

    if need_idx:
        # 批次上限 = rerank_max_candidates（自设防线，§6：真实约束是 TPM/单请求
        # token 规模）。粗排召回的全量候选（未指定条数时可达数百）在此被切成
        # 多批串行请求——relevance 是 (model, query, document) 的确定性函数，
        # 分批不改变各文档得分，合并后全量统一按分数排序，无混合序。
        batch_size = max(1, cfg.rerank_max_candidates)
        for start in range(0, len(need_idx), batch_size):
            batch_idx = need_idx[start:start + batch_size]
            scores = _call_rerank_api(
                query=query,
                documents=[documents[i] for i in batch_idx],
                top_n=len(batch_idx),
            )
            for i, score in zip(batch_idx, scores):
                cached[doc_hashes[i]] = score
        _cache_put(cfg.rerank_model, query_hash, cached)

    # 写回 rerank_score（不覆盖 rrf_score，方案 B），按分数降序截断
    for it, dh in zip(candidates, doc_hashes):
        it["rerank_score"] = float(cached.get(dh, 0.0))

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]


def _call_rerank_api(query: str, documents: list[str], top_n: int) -> list[float]:
    """POST /v1/rerank，返回按请求顺序对齐的 relevance_score 列表。

    重试：指数退避，严格限死 rerank_max_retries（§4.9 线程池竞争约束）。
    限流：429/5xx 退避重试；非重试类 4xx（鉴权/参数）立即 raise。
    """
    cfg = get_config().retrieval
    url = f"{cfg.rerank_api_url.rstrip('/')}/v1/rerank"
    body = {
        "model": cfg.rerank_model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
    }
    headers = {
        "Authorization": f"Bearer {cfg.rerank_api_key}",
        "Content-Type": "application/json",
    }

    last_err: Exception | None = None
    attempts = cfg.rerank_max_retries + 1
    for attempt in range(attempts):
        if attempt > 0:
            # 指数退避：0.5s / 1s / 2s ...（重试上限已限死，退避总时长可控）
            time.sleep(0.5 * (2 ** (attempt - 1)))
        try:
            with _qps_sem:
                with httpx.Client(timeout=cfg.rerank_timeout_s) as client:
                    resp = client.post(url, json=body, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results") or []
                # resp.results 按相关度降序，每项含 index + relevance_score
                # → 重映射回请求 documents 顺序，保证与候选池一一对应
                scores = [0.0] * len(documents)
                for r in results:
                    idx = r.get("index")
                    if isinstance(idx, int) and 0 <= idx < len(documents):
                        scores[idx] = float(r.get("relevance_score", 0.0))
                return scores
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(
                    f"rerank API retryable error {resp.status_code}: {resp.text[:200]}")
                log.warning("rerank attempt %d/%d failed (retryable): %s",
                            attempt + 1, attempts, last_err)
                continue
            # 400/401/403 等非重试类错误：重试无意义，立即失败
            raise RuntimeError(
                f"rerank API error {resp.status_code}: {resp.text[:200]}")
        except (httpx.TimeoutException, httpx.TransportError) as e:
            # 超时/网络错误可重试
            last_err = e
            log.warning("rerank attempt %d/%d failed (transport): %s",
                        attempt + 1, attempts, e)
            continue

    raise RuntimeError(f"rerank failed after {attempts} attempts: {last_err}")


def _cache_get(model: str, query_hash: str) -> dict[str, float]:
    """取缓存（线程安全）；过期/未命中返回空 dict（可安全 merge）。"""
    with _cache_lock:
        entry = _cache.get((model, query_hash))
        if entry is None:
            return {}
        expire_ts, scores = entry
        if time.monotonic() > expire_ts:
            _cache.pop((model, query_hash), None)
            return {}
        return dict(scores)


def _cache_put(model: str, query_hash: str, scores: dict[str, float]) -> None:
    """写缓存（线程安全）；简单 LRU：超容量丢最旧条目。"""
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX_QUERIES:
            # dict 保序：弹出最早写入的条目（近似 LRU 够用，避免引入 OrderedDict）
            oldest = next(iter(_cache))
            _cache.pop(oldest, None)
        _cache[(model, query_hash)] = (time.monotonic() + _CACHE_TTL_S, dict(scores))
