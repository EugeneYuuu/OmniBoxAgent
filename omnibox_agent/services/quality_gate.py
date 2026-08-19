"""v4.1 Layer 2: Quality Gate — batch relevance judgment + fallback chain.

Design doc §5 (§5.1-5.5):
  1. Batch judge all candidates (single LLM call, JSON contract)
  2. If any relevant → proceed (filter to relevant only)
  3. If all irrelevant → fallback chain:
     a. On-demand image re-parse (≤1, dual budget: image count + time)
     b. Query rewrite + re-retrieve (≤1, filters unchanged)
     c. Degrade: low_confidence, proceed with original docs

  Fail-open: If judge_batch fails twice, keep all docs + set gate_degraded flag.

Integration:
  Called as a pipeline step between RetrieveStep and ReasonStep.
  Returns gate decision that drives the pipeline loop:
    "proceed"    → filtered docs, go to generation
    "re_retrieve" → on-demand parse done, need re-retrieval
    "rewrite"    → query rewritten, need re-retrieval with new query
"""

from __future__ import annotations

import logging
from typing import Any

from omnibox_agent.core.config import get_config
from omnibox_agent.models.query import Intent
from omnibox_agent.services.llm_service import judge_batch, generate

log = logging.getLogger(__name__)

# Gate decision constants
GATE_PROCEED = "proceed"
GATE_RE_RETRIEVE = "re_retrieve"
GATE_REWRITE = "rewrite"


async def quality_gate(ctx: Any) -> str:
    """Execute the quality gate on retrieved results.

    v4.1 §5.1: Batch relevance judgment with fallback chain.
    Reads ctx.artifacts["retrieval"] and modifies it in-place.
    Sets ctx.flags["gate_degraded"] / ctx.flags["low_confidence"] as needed.

    Args:
        ctx: AgentContext with:
            - artifacts["retrieval"] -> RetrievalOutput
            - input["query"] -> user query
            - counters (defaultdict)

    Returns:
        Gate decision: "proceed" | "re_retrieve" | "rewrite"
    """
    retrieval = ctx.artifacts.get("retrieval")
    if retrieval is None or not retrieval.fused_items:
        # No results to gate — proceed to generation (will produce "not found")
        ctx.metrics["gate_decision"] = "proceed_empty"
        return GATE_PROCEED

    query = ctx.input.get("query", "")
    docs = retrieval.fused_items
    cfg = get_config()

    # ① Skip judging when the candidate set is very large (e.g. unbounded
    # aggregation queries). Judging 100+ docs is expensive and useless — we
    # keep everything and let refinement + budget fit run as normal.
    if len(docs) > cfg.gate.max_judge_docs:
        log.info("Gate: %d docs > max_judge_docs(%d), skipping judge (keep all)",
                 len(docs), cfg.gate.max_judge_docs)
        ctx.metrics["gate_total"] = len(docs)
        ctx.metrics["gate_decision"] = "skip_large"
        return GATE_PROCEED

    # ①b Skip judging for explicitly unbounded aggregation / analysis queries
    # where the vector recall IS the whole topic scope — a per-doc relevance
    # judge is both pointless and expensive. Creative/DAG sub-tasks have no
    # perception artifact (they didn't run ParseStep) — they go through the
    # normal gate path below, NOT through this skip.
    #
    # EXIST_CHECK is excluded: "有没有" must be decided by a relevance judge,
    # otherwise every candidate (e.g. an unrelated 美食 collection) reaches the
    # generator and the answer drifts off-topic / fabricates a "yes".
    perception = ctx.artifacts.get("perception")
    if perception is not None \
            and not getattr(perception, "explicit_limit", False) \
            and perception.intent != Intent.EXIST_CHECK:
        log.info("Gate: unbounded/aggregation query, skipping judge (keep all %d docs)", len(docs))
        ctx.metrics["gate_total"] = len(docs)
        ctx.metrics["gate_decision"] = "skip_unbounded"
        return GATE_PROCEED

    # ② Batch relevance judgment (three-tier)
    ctx.llm_call_count += 1
    # Judge under the user's own API key (NOT the evaluator/Zhipu config);
    # only embedding should use Zhipu. Missing user key raises in _call_llm.
    judged = await judge_batch(docs, query, ai_config=ctx.input.get("ai_config"))

    if judged is None:
        # Fail-open: both parse attempts failed → keep all + degraded flag
        log.warning("Gate: judge_batch failed (fail-open), keeping all %d docs", len(docs))
        ctx.flags["gate_degraded"] = True
        ctx.metrics["gate_fail_open"] = True
        ctx.metrics["gate_decision"] = "proceed_fail_open"
        return GATE_PROCEED

    kept = [d for d, v in zip(docs, judged)
            if v in ("relevant", "topic_relevant")]
    relevant_count = sum(1 for v in judged if v == "relevant")
    topic_count = sum(1 for v in judged if v == "topic_relevant")
    irrelevant_count = sum(1 for v in judged if v == "irrelevant")
    ctx.metrics["gate_total"] = len(docs)
    ctx.metrics["gate_relevant"] = relevant_count
    ctx.metrics["gate_topic_relevant"] = topic_count
    ctx.metrics["gate_irrelevant"] = irrelevant_count

    if kept:
        # Keep relevant + topic_relevant; drop only fully irrelevant docs.
        # For summary/aggregation queries, topic_relevant members are retained
        # so the whole主题范畴 reaches the generator instead of being mis-killed.
        retrieval.fused_items = kept
        # EXIST_CHECK: total_count 必须反映"实际相关的内容数"，否则候选集
        # 大小（含大量不相关条目）会让 LLM 误判"有"。同步为 judge 后保留数。
        if perception is not None and perception.intent == Intent.EXIST_CHECK:
            retrieval.total_count = len(kept)
        # ── 评论区兜底（comment fallback drill）──
        # 用户原则：内容正文找不到问题的语义时，再去该内容的评论向量找。
        # judge 三档语义：relevant=正文直接回答问题；topic_relevant=同主题
        # 但正文不回答具体问题。relevant_count==0 且仍有 topic_relevant 命中
        # = "主题定位成功，但正文缺问题答案语义"（如『第二个美食的地点在
        # 哪里』，地点只在评论区）——取这些条目的评论全文附到条目上，由
        # 生成 LLM 在评论里找答案。不做场景关键词预判（不预设"地点/地址"
        # 该查评论），触发条件只看"正文是否回答了问题"这一判定结果。
        if relevant_count == 0 and topic_count > 0:
            await _drill_comments(kept, ctx)
        log.info("Gate: %d relevant, %d topic_relevant, %d irrelevant -> keeping %d",
                 relevant_count, topic_count, irrelevant_count, len(kept))
        ctx.metrics["gate_decision"] = "proceed"
        return GATE_PROCEED

    # ② All irrelevant: 评论兜底优先（用户原则：内容找不到问题语义时，
    # 再去内容的评论向量找）。all-irrelevant 也是一种"正文找不到问题语义"
    # ——候选集虽被 judge 全判 irrelevant（正文不回答问题），但候选本身是
    # 向量检索召回的（语义相关），其评论区可能含答案（如"地点在哪里"→
    # 评论里的地址）。先尝试评论兜底，成功则直接 proceed 不走 fallback。
    # 用原始 docs（retrieval.fused_items，judge 前的全集），取 rrf_score
    # 最高的 N 条钻取评论。attached > 0 时把这些条目（含评论文本）放回
    # retrieval.fused_items，标记 low_confidence，直接 proceed。
    if relevant_count == 0 and topic_count == 0 and len(docs) > 0:
        await _drill_comments(docs, ctx)
        drilled = [d for d in docs if d.get("comments_text")]
        if drilled:
            retrieval.fused_items = drilled
            ctx.flags["gate_degraded"] = True
            ctx.metrics["gate_decision"] = "proceed_comment_fallback"
            log.info("Gate: all irrelevant -> comment fallback rescued %d items", len(drilled))
            return GATE_PROCEED

    # ③ All irrelevant: check on-demand image re-parse
    # (No time budget — pipelines are not time-limited; only the attempt cap.)
    if ctx.counters["ondemand"] < 1:
        has_unparsed = _check_unparsed_images(retrieval)
        if has_unparsed:
            log.info("Gate: all irrelevant, attempting on-demand image re-parse")
            ctx.counters["ondemand"] += 1
            ctx.metrics["gate_ondemand_triggered"] = True
            try:
                parsed_count = await _ondemand_parse_and_reindex(retrieval, ctx)
                ctx.metrics["gate_ondemand_parsed"] = parsed_count
                if parsed_count > 0:
                    return GATE_RE_RETRIEVE
            except Exception as e:
                log.warning("Gate: on-demand parse failed: %s", e)

    # ③ Query rewrite + re-retrieve (≤1, filters unchanged)
    if ctx.counters["rewrite"] < 1:
        log.info("Gate: all irrelevant, attempting query rewrite")
        ctx.counters["rewrite"] += 1
        ctx.metrics["gate_rewrite_triggered"] = True
        try:
            rewritten = await _rewrite_query(query, ctx)
            if rewritten and rewritten != query:
                ctx.input["_rewritten_query"] = rewritten
                return GATE_REWRITE
        except Exception as e:
            log.warning("Gate: query rewrite failed: %s", e)

    # ④ Degrade: proceed with original docs, low confidence
    log.info("Gate: fallback chain exhausted, degrading to low_confidence")
    ctx.flags["low_confidence"] = True
    ctx.metrics["gate_decision"] = "proceed_degraded"
    # EXIST_CHECK: 判定无任何相关内容 → total_count=0，prompt 据此回答"没有"，
    # 避免把候选集大小当"有"误导 LLM。
    if perception is not None and perception.intent == Intent.EXIST_CHECK:
        retrieval.total_count = 0
    return GATE_PROCEED


async def _drill_comments(docs: list[dict], ctx: Any) -> None:
    """评论区兜底：为 topic_relevant 条目取回评论全文，附到 item["comments_text"]。

    触发条件已在调用方保证（judge 无 relevant、仅 topic_relevant = 正文
    找不到问题语义）。这里只负责取数：评论向量按内容聚合（{note_id}#
    comments，document=评论全文），按 content_id 从 ChromaDB get 即可。
    "在评论里找答案"由生成 LLM 阅读完成（prompt 渲染见
    ask_orchestrator._build_system_prompt 的评论区标注）。

    预算控制：最多前 comment_drill_max_items 条（按 rrf_score 排序），
    每条截断 comment_drill_max_chars 字符。comment_drill_max_items=0
    可整体关闭兜底（回滚开关）。失败仅告警，不影响主流程。
    """
    cfg = get_config()
    max_items = cfg.gate.comment_drill_max_items
    max_chars = cfg.gate.comment_drill_max_chars
    if max_items <= 0:
        return
    targets = sorted(
        docs,
        # rerank_score 优先（RAG 两阶段检索方案 B）：精排条目按云端相关度定序；
        # 未精排条目（评论命中 / 降级回退 / 非 rerank 路径）rerank_score 缺省，
        # 回退 rrf_score——与改造前行为一致。注意 is not None 而非 or：
        # relevance_score 可能为 0（相关度最低），or 会误判为缺省。
        key=lambda d: (d.get("rerank_score")
                       if d.get("rerank_score") is not None
                       else d.get("rrf_score", d.get("score", 0.0))),
        reverse=True,
    )[:max_items]
    ids = [d.get("content_id") for d in targets if d.get("content_id") is not None]
    if not ids:
        return

    from omnibox_agent.agent.loop import run_blocking
    from omnibox_agent.services.chroma_store import get_comment_docs
    log.info("Gate: comment drill triggered — querying comments for %d content_ids: %s",
             len(ids), ids[:5])
    try:
        # get 按主键过滤取 5 条文档，毫秒级；丢线程池避免占 event loop
        comment_docs = await run_blocking(get_comment_docs, ids)
    except Exception as e:
        log.warning("Gate: comment drill fetch failed: %s", e)
        return

    log.info("Gate: comment drill fetched %d/%d comment docs", len(comment_docs), len(ids))
    attached = 0
    for d in targets:
        text = comment_docs.get(d.get("content_id"))
        if text and text.strip():
            d["comments_text"] = text[:max_chars]
            attached += 1
    if attached:
        ctx.metrics["comment_drill_items"] = attached
        log.info("Gate: comment fallback drill — attached comments to %d/%d items "
                 "(body lacks answer semantics)", attached, len(targets))
        from omnibox_agent.core.trace_recorder import trace_event
        trace_event("qa.comment_drill", phase="qa",
                    data={"items": attached, "trigger": "no_relevant_only_topic"})
    else:
        log.info("Gate: comment drill — 0 comments attached (no comment vectors for these items)")


def _check_unparsed_images(retrieval: Any) -> bool:
    """Check if any retrieved items have unparsed images.

    v4.1 §5.2: Only images (not videos — videos are async, can't wait in query).
    The ingestion pipeline writes `parsed` and `has_image` into ChromaDB metadata
    (see _build_metadata), and vector_search._parse_results passthroughs them
    into the fused_items dicts. If any candidate has images that are not yet
    parsed, the on-demand re-parse fallback can enrich its content.
    """
    for item in (retrieval.fused_items or []):
        if item.get("has_image") and not item.get("parsed"):
            log.debug("Gate: found unparsed image in doc content_id=%s", item.get("content_id"))
            return True
    return False


async def _ondemand_parse_and_reindex(retrieval: Any, ctx: Any) -> int:
    """Query-time on-demand image parsing + reindex.

    v4.1 §5.2: Parse unparsed images (≤6, dual budget), reindex via
    reindex_note (shared per-note lock), then retrieval will pick them up.

    Returns:
        Number of images successfully parsed.
    """
    cfg = get_config()
    max_images = cfg.gate.max_ondemand_images

    # Load NoteRecords for items with unparsed images
    # This requires the note_loader — in the current architecture, we need
    # to load from MySQL. For now, this is a stub that will be connected
    # when the ingestion pipeline's note_loader is fully wired.
    #
    # TODO: Wire to note_loader.load_note_by_id for each content_id
    #       Then call parse_and_reindex from ingestion.py
    log.debug("on-demand parse: max_images=%d (stub — needs note_loader integration)", max_images)
    return 0


async def _rewrite_query(query: str, ctx: Any) -> str | None:
    """Rewrite the query for better retrieval (filters unchanged).

    v4.1 §5.1③: One rewrite attempt, same filters.
    Uses the evaluator model to generate a semantically different but
    intent-preserving query.
    """
    cfg = get_config()

    messages = [
        {"role": "system", "content": (
            "你是一个查询改写器。用户查询在现有收藏库中未找到相关结果。"
            "请改写查询以提高召回率：使用同义词、更具体或更泛化的表达。"
            "保持原意不变。只输出改写后的查询，不要其他文字。")},
        {"role": "user", "content": f"原查询: {query}\n\n改写后的查询:"},
    ]

    ctx.llm_call_count += 1
    try:
        # Rewrite runs under the user's own api-key (never the system key).
        user_ai_config = ctx.input.get("ai_config") if ctx else None
        rewritten = await generate(
            messages, ai_config=user_ai_config,
            temperature=0.3, max_tokens=2048, timeout=None,
            no_thinking=True,  # QA utility call — disable thinking for speed
        )
        return rewritten.strip() if rewritten else None
    except Exception as e:
        log.warning("Query rewrite LLM call failed: %r", e)
        return None


# ── CRAG Knowledge Refinement (§5.5) ─────────────────────────────────────

async def refine_docs(docs: list[dict], query: str, ctx: Any = None) -> list[dict]:
    """CRAG-style knowledge refinement: sentence-level denoising.

    v4.1 §5.5:
      - Only trigger for docs with content > REFINEMENT_MIN_CHARS (300)
      - Short docs (typical Xiaohongshu notes ≤300 chars) pass through unchanged
      - Each qualifying doc: split sentences → batch judge → reassemble
      - Fail-open: if judge fails, keep all sentences (same as gate)

    Args:
        docs: List of retrieval result dicts with "content_id" and content fields
        query: User query
        ctx: Optional AgentContext for metrics tracking

    Returns:
        Refined docs list (same length, content may be compressed)
    """
    from omnibox_agent.services.llm_service import batch_judge_sentences

    cfg = get_config()
    min_chars = cfg.gate.refinement_min_chars
    refined_count = 0

    result = []
    for doc in docs:
        # Get the content text — try summary field, then title
        content = doc.get("summary", "") or doc.get("title", "")
        # Also check content_map for full content
        if not content and "content_id" in doc:
            # In the pipeline, content_map has full details
            pass

        if len(content) <= min_chars:
            # Short doc: pass through, zero cost
            result.append(doc)
            continue

        sentences = _split_sentences(content)
        if len(sentences) <= 2:
            result.append(doc)
            continue

        # Batch judge sentences (single LLM call per doc) — runs under the
        # user's own api-key (never the system key).
        if ctx:
            ctx.llm_call_count += 1
        user_ai_config = ctx.input.get("ai_config") if ctx else None
        kept = await batch_judge_sentences(sentences, query, ai_config=user_ai_config)

        if kept and len(kept) < len(sentences):
            # Some sentences filtered out — update content
            refined_text = " ".join(kept)
            refined_doc = dict(doc)
            refined_doc["summary"] = refined_text
            result.append(refined_doc)
            refined_count += 1
            if ctx:
                ctx.metrics.setdefault("refine_triggered", 0)
                ctx.metrics["refine_triggered"] += 1
                ctx.metrics.setdefault("refine_compressed", 0)
                ctx.metrics["refine_compressed"] += 1
        else:
            # No filtering happened (all kept or fail-open) — original doc
            result.append(doc)

    if ctx and refined_count > 0:
        log.info("refine_docs: %d/%d docs refined", refined_count, len(docs))

    return result


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (Chinese + English aware).

    Uses a simple regex-based splitter that handles:
      - Chinese sentence enders: 。！？；
      - English sentence enders: . ! ? ;
      - Newlines as soft boundaries
    """
    import re
    # Split on sentence-ending punctuation, keeping the punctuation
    parts = re.split(r'(?<=[。！？；.!?;])\s*', text)
    # Also split on newlines
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Further split very long parts on newlines
        for sub in part.split('\n'):
            sub = sub.strip()
            if sub:
                result.append(sub)
    return result


# ── Token budget fitting (§6) ────────────────────────────────────────────

def fit_budget(docs: list[dict], budget: int, top_k_complete: int = 1) -> list[dict]:
    """Fit docs into a token budget, sorted by relevance score.

    v4.1 §6: Top-1 always complete. Others truncated at tail if needed.
    Uses a simple char-based approximation (1 Chinese char ≈ 1.5 tokens).
    """
    if not docs:
        return []

    # Sort by relevance score descending. rerank_score 优先（RAG 两阶段检索
    # 方案 B）：精排条目按云端相关度定序；未精排条目（评论命中 / 降级回退 /
    # 非 rerank 路径）rerank_score 缺省，回退 rrf_score（RRF fusion ordering:
    # main/media primary before comment supplement）再回退 raw per-vector
    # similarity `score` —— a comment-matched note's raw similarity is often
    # higher than a primary note's, and sorting by it would let comment
    # supplements crowd out primary matches inside the token budget.
    # 注意 is not None 而非 or：relevance_score 可能为 0（相关度最低），
    # or 会误判为缺省、错误回退到 rrf_score。
    sorted_docs = sorted(
        docs,
        key=lambda d: (d.get("rerank_score")
                       if d.get("rerank_score") is not None
                       else d.get("rrf_score", d.get("score", 0.0))),
        reverse=True,
    )

    result = []
    used = 0
    for i, doc in enumerate(sorted_docs):
        content = doc.get("summary", "") or doc.get("title", "")
        char_count = len(content)
        # Rough token estimate: 1 Chinese char ≈ 1.5 tokens, 1 English word ≈ 1 token
        token_est = int(char_count * 1.2)

        if i < top_k_complete:
            # 前 top_k_complete 条无条件完整（默认 1 = 原行为，simple QA 零回归）
            result.append(doc)
            used += token_est
        elif used + token_est > budget:
            # Truncate this doc
            remaining_chars = int((budget - used) / 1.2)
            if remaining_chars > 50:  # Only include if meaningful amount left
                truncated = dict(doc)
                truncated["summary"] = content[:remaining_chars] + "..."
                result.append(truncated)
            break
        else:
            result.append(doc)
            used += token_est

    return result
