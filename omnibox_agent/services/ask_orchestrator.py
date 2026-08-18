"""Ask Orchestrator: vector-first retrieval + RRF scoring + system prompt building.

Core flow:
  Query Understanding -> Structured Filtering -> Vector Search (embedding)
  -> RRF Scoring (tag boost + freshness) -> Prompt Assembly

v4.1 flow fix: retrieval is driven by vector search only. The FULLTEXT / LIKE /
recent channels were removed from the flow because they produced keyword-gated
statistics (total_count / platform_dist) that contradicted the vector recall
(e.g. "共找到 0 条" while 58 items were shown). Statistics are now derived
from the actual fused result set so the generator always sees consistent numbers.

Non-streaming answer generation (handle_ask) was removed: 生产路径全流式，
最终回答由 stream_qa_pipeline / stream_chat 逐 token 输出。
"""

import json
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any

from omnibox_agent.core.config import get_config
from omnibox_agent.models.query import Intent, QueryUnderstandingResult
from omnibox_agent.services.vector_search import vector_search
from omnibox_agent.services.retrieval_store import (
    count_with_filters,
    get_content_by_ids,
)

log = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))

# ── 时间列举是否"带主题" ──
# 纯时间列举（"最近收藏了什么/今天收藏的"）里没有实质主题词，语义召回抓不住
# 刚收藏的条目，必须全库召回 + 按时间排，才能兜住新收藏。
# 带主题的时间列举（"最近收藏的美食"）相反：主题约束优先于时间，得先靠语义
# 召回收敛到主题候选，再在这些候选内按时间排——不能全库一股脑按时间堆，
# 否则旧的美食会被刚收藏的无主题视频挤到更后面，主题维度被时间淹没。
_COLLECTION_GENERIC_WORDS = {
    "收藏", "内容", "干货", "最近", "最新", "近期", "新收藏",
    "今天", "昨天", "前天", "本周", "这周", "上周", "本月",
    "上月", "这个月", "上个月", "一周", "一个月", "几天",
    "多少", "几条", "几个", "哪些", "有什么",
}


def _has_meaningful_topic(qu_result: Any) -> bool:
    """时间列举类查询是否带实质主题词。

    只检 keywords：若全部命中"收藏/时间/数量"这类泛化词，视为纯时间列举。
    """
    kws = {(w or "").strip() for w in (getattr(qu_result, "keywords", None) or [])}
    kws.discard("")
    if not kws:
        return False
    return any(w not in _COLLECTION_GENERIC_WORDS for w in kws)



from omnibox_agent.agent.context import RetrievalOutput

def retrieve_pipeline(
    qu_result: QueryUnderstandingResult,
    account_ids: list[str],
    request_input: dict,
    ai_config: dict | None = None,
    cfg: Any = None,
) -> RetrievalOutput:
    """Vector-first retrieval pipeline (the canonical flow).

    Called by RetrieveStep, qa_complex and creative_solver.

    v4.1 flow fix: retrieval is driven by vector search (embedding) only.
    FULLTEXT / LIKE / recent channels are removed from the flow — they produced
    keyword-gated statistics that contradicted the vector recall. total_count /
    platform_dist are now derived from the fused result set, so the generator
    always sees numbers consistent with the items it is given.
    """
    if cfg is None:
        cfg = get_config().retrieval

    # Defensive: query understanding may be missing (e.g. ParseStep failed
    # non-critically and never wrote the artifact). Never let a None crash
    # retrieval -- fall back to an empty understanding built from the query.
    if qu_result is None:
        log.warning("retrieve_pipeline: qu_result is None, using empty understanding")
        qu_result = QueryUnderstandingResult(
            resolved_query=request_input.get("query", "") or "",
        )

    favorite_only = request_input.get("favorite_only", True)
    time_start = qu_result.time_range_start
    time_end = qu_result.time_range_end
    platform = qu_result.platform

    # R10：会话指代查询的检索词优先用 LLM 消解的 resolved_query（如
    # 「第三点有哪些推荐的」→「AI/职场 相关收藏推荐」）——编号/指代在
    # QU 的规则消解里不一定能解开，直接用消解词做向量检索最可靠。
    conv_resolved = (request_input.get("_conv_resolved_query") or "").strip()
    if conv_resolved:
        search_query = conv_resolved
    else:
        search_query = qu_result.embedding_query or qu_result.resolved_query or request_input.get("query", "")

    # Determine candidate n
    is_recency_or_exist = (
        qu_result.recency
        or qu_result.intent == Intent.EXIST_CHECK
        or qu_result.intent == Intent.COUNT
    )
    candidate_n = cfg.recency_top_n if is_recency_or_exist else cfg.candidate_n

    # Useful flags shared by candidate sizing and final sort.
    time_listing = qu_result.recency or qu_result.time_range_start or qu_result.time_range_end
    has_topic = _has_meaningful_topic(qu_result)

    # 时间列举查询的粗召回预算：
    #  - 纯时间列举（无实质主题，如"最近收藏了什么"）：语义 top-k 选候选会把
    #    "刚收藏但与泛问题语义不近"的条目截掉（实测新收藏排 71/73，进不了
    #    top-50 候选），而下游的时间排序只能在候选集内重排，救不回被截断的
    #    条目。故 n_results 抬到用户收藏总数（含时间窗/平台过滤），保证
    #    "召回全集 → 按时间排序 → 门控/token 预算收口" 的顺序成立。
    #  - 带主题的时间列举（如"最近收藏的美食"）：同样把召回预算抬到全量，
    #    保证"散步在收藏库里的所有美食"都进候选池（不被语义 top-k 截掉）；
    #    主题约束由下方排序块的"相关性地板"剪尾保证——时间排序只在这些
    #    主题候选内进行，不会把最近收藏的无主题视频卷到最前。
    if time_listing:
        full_count = count_with_filters(
            account_ids=account_ids,
            time_start=qu_result.time_range_start,
            time_end=qu_result.time_range_end,
            platform=platform,
            favorite_only=favorite_only,
        )
        if full_count > 0:
            candidate_n = max(candidate_n, full_count)

    # Unbounded mode: the user did NOT ask for a specific count or time window.
    # Aggregation / summary queries want breadth (the whole主题范畴), not a
    # hard top_n cap. Widen recall and the fusion truncation accordingly.
    # Also force unbounded when the user asks to classify the collection
    # ("将收藏分类") — classification needs the whole set, not a top_n slice.
    want_classify = getattr(qu_result, "want_classify", False)
    limit_count = getattr(qu_result, "limit_count", None)

    # ── 精召回 topk（第二阶段：排序后截取）──
    # 两档 topk（用户指令）：
    #   粗召回（candidate_n，向量召回阶段）= 50~100
    #   精召回（排序后截取）= 20~50（cfg.refine_top_n）
    # 顺序务必是"先召回全集/全库 → 排序 → 再精截取"，而非"先粗截取后排序"
    # （否则刚收藏的项目会在语义粗截时被挤掉，时间排序救不回）。
    # 时间列举（recency/时间窗）：纯时间列举已全库抬成 candidate_n（上方块）；
    #     带主题的时间列举保持主题候选预算，二者排序后都按精召回 topk 截取。
    #   分类(want_classify) / 聚合 / 分析（非显式限定）：宽阔召回不精截，
    #     由门控 + token 预算收口（分类需全量分组、分析需全范畴）。
    #   用户问题自带条数（limit_count，如"给我10条"）优先级最高，覆盖精召回默认。
    if limit_count and limit_count > 0:
        candidate_n = max(candidate_n, limit_count)
        eff_top_n = limit_count
    elif time_listing:
        eff_top_n = cfg.refine_top_n
    elif want_classify or not qu_result.explicit_limit:
        candidate_n = max(candidate_n, cfg.unbounded_candidate_n)
        eff_top_n = None
    else:
        eff_top_n = cfg.refine_top_n

    user_id = request_input.get("user_id", "")

    # Vector search: main + media primary, comments supplementary — same
    # pipeline step, but with INDEPENDENT top-k budgets (用户指令：评论向量
    # 不要限制 top-k):
    #   - 主通道: main + media，按 candidate_n 语义（主要还是看主/媒体向量）
    #   - 评论通道: 独立预算 comment_candidate_n，不与主通道共享 top-k 池，
    #     否则主/媒体命中会挤占评论召回（即"限制 top-k"的根因）
    # 评论命中标记 is_comment_match，RRF 降权 —— 只补位，不干扰主命中。
    vector_results: list[dict] = []
    comment_results: list[dict] = []
    try:
        comment_weight = cfg.rrf_comment_weight
        vector_results = vector_search(
            query=search_query,
            user_id=user_id,
            n_results=candidate_n,
            time_start=time_start,
            time_end=time_end,
            platform=platform,
            favorite_only=favorite_only,
            vec_types=["main", "media"],
            max_pool=False,  # main+media 不去重，各自进 RRF 后分数累加
        )
        if comment_weight > 0:
            comment_results = vector_search(
                query=search_query,
                user_id=user_id,
                n_results=cfg.comment_candidate_n,
                time_start=time_start,
                time_end=time_end,
                platform=platform,
                favorite_only=favorite_only,
                vec_types=["comments"],
                max_pool=True,
            )
        log.debug("Vector search: %d primary (main+media), %d comment-supplement (independent top-k)",
                  len(vector_results), len(comment_results))
    except Exception as e:
        log.warning("Vector search failed: %s", e)

    # RRF scoring (vector-only; the BM25 channel was removed from the flow)
    fused = _rrf_fusion(
        vector_results=vector_results,
        bm25_results=[],
        comment_results=comment_results,
        qu_tags=qu_result.tags,
        k=cfg.rrf_k,
        vector_weight=cfg.rrf_vector_weight,
        bm25_weight=cfg.rrf_bm25_weight,
        comment_weight=cfg.rrf_comment_weight,
        tag_boost_factor=cfg.tag_boost_factor,
        freshness_lambda=cfg.freshness_lambda,
        has_time_range=time_start is not None,
        is_exist_check=qu_result.intent == Intent.EXIST_CHECK,
    )

    # §12.2 长期记忆 RRF 软加权：post-fusion reweight——命中偏好平台/标签的
    # 条目 rrf_score × 1.1（±10% 软加权，不硬过滤；不动 _rrf_fusion 本体）。
    try:
        rrf_boost = ((request_input or {}).get("long_term") or {}).get("rrf_boost") or {}
        pref_platforms = {str(p).lower() for p in (rrf_boost.get("platforms") or []) if p}
        pref_tags = {str(t).lower() for t in (rrf_boost.get("tags") or []) if t}
        if pref_platforms or pref_tags:
            for item in fused:
                pn = str(item.get("platform_name")
                         or item.get("platform") or "").lower()
                item_tags = {t.lower() for t in _parse_tags(item.get("tags", ""))}
                if pn in pref_platforms or (item_tags & pref_tags):
                    item["rrf_score"] = float(item.get("rrf_score", 0.0)) * 1.1
    except Exception as e:
        log.debug("LT rrf soft-boost skipped: %s", e)

    # Sort：时间列举（纯时间/带主题）按收藏时间倒序；其余按 RRF 相关度。
    # 精召回 topk（refine_top_n）在排序后施加，保证时间列举类结果是"最近在前"。
    #
    # 带主题的时间列举（如"最近收藏的美食"）：候选池已抬到全量以保证"所有
    # 美食都在池里"，但语义召回会带入池尾上少量与主题不近的条目。这里先用
    # 语义相关性地板把尾巴剪掉，确保时间排序只在"美食"这类主题候选内进行，
    # 而不是把最近收藏的无主题内容卷到最前（用户指令：不能一股脑按时间排序）。
    if time_listing:
        pool = fused
        if has_topic and pool:
            best = max((it.get("score", 0) or 0) for it in pool)
            floor = max(0.12, best - 0.30)
            pool = [it for it in pool if (it.get("score", 0) or 0) >= floor]
        pool.sort(key=lambda x: x.get("collected_at", ""), reverse=True)
        fused = pool
    else:
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    # Statistics derived from the fused set (consistent with what is shown).
    total_count = len(fused)
    if qu_result.intent == Intent.COUNT:
        # Pure count queries report the user's library size with filters but
        # WITHOUT keywords — a plain filtered COUNT, no FULLTEXT/LIKE — so
        # "how many" stays accurate instead of keyword-gated at 0.
        total_count = count_with_filters(
            account_ids=account_ids,
            time_start=time_start,
            time_end=time_end,
            platform=platform,
            favorite_only=favorite_only,
        )
    platform_dist: dict[str, int] = {}
    for item in fused:
        pn = item.get("platform_name") or item.get("platform") or "unknown"
        platform_dist[pn] = platform_dist.get(pn, 0) + 1

    # 不限制 topK（eff_top_n=None）时返回全部召回，由门控 + token 预算收口。
    # 有界模式（用户指定条数）下：主通道按 eff_top_n 截断；评论补充通道
    # 不截断、附加在主命中之后（用户指令：评论向量不要限制 top-k）。
    if eff_top_n is None:
        top_n = fused
    else:
        top_n = (
            [it for it in fused if not it.get("is_comment_match")][:eff_top_n]
            + [it for it in fused if it.get("is_comment_match")]
        )
    top_ids = [item["content_id"] for item in top_n]
    content_details = get_content_by_ids(top_ids, account_ids)
    content_map = {c["id"]: c for c in content_details}

    # Enrich fused items with summary text so the CRAG gate / refinement can
    # judge on real content. Fused dicts only carry `title` by default, so the
    # gate would otherwise see empty bodies and mis-judge everything.
    for item in top_n:
        cid = item["content_id"]
        detail = content_map.get(cid)
        # 修复：fused item 的 content_id 是 str（Chroma metadata），content_map
        # 的 key 是 int（MySQL id），类型不匹配会导致 get 恒 None → summary 永远
        # 补不上，正文只能残留在 document 字段。此处对 str 数字做 int 回退。
        if detail is None and isinstance(cid, str) and cid.isdigit():
            detail = content_map.get(int(cid))
        if detail:
            summary = detail.get("summary", "") or ""
            item.setdefault("summary", summary)
            item.setdefault("content", summary)

    # Ask 追踪：检索完成（§4.2 qa.retrieve）—— 记录命中数
    from omnibox_agent.core.trace_recorder import trace_event
    trace_event("qa.retrieve", phase="qa",
                data={
                    "rewritten_query": (getattr(qu_result, "rewritten_query", None) or "")[:200]
                                     if qu_result else "",
                    "hit_count": len(top_n),
                    "top_k": len(top_n),
                    "platforms": platform_dist,
                })

    return RetrievalOutput(
        vector_results=vector_results,
        fulltext_results=[],
        fused_items=top_n,
        total_count=total_count,
        platform_dist=platform_dist,
        content_map=content_map,
    )


def _rrf_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    qu_tags: list[str],
    k: int = 60,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    comment_results: list[dict] | None = None,
    comment_weight: float = 0.3,
    tag_boost_factor: float = 0.15,
    freshness_lambda: float = 0.01,
    has_time_range: bool = False,
    is_exist_check: bool = False,
) -> list[dict]:
    """RRF (Reciprocal Rank Fusion) with tag boost and conditional freshness.

    Comment supplement channel: comment-vector hits join the fusion with a
    lower weight (comment_weight < vector_weight) and are marked
    is_comment_match=True. Because their per-rank contribution is smaller
    than every main/media hit's, comment-only matches always rank BELOW
    primary matches — they supplement the pool, never displace it. A note
    matched by both channels keeps its primary metadata and scores higher.

    Returns list of dicts with content_id, rrf_score, and merged metadata.
    """
    scores: dict[int, float] = {}
    metadata: dict[int, dict] = {}

    # Vector channel (main/media — the primary signal)
    # main 和 media 不做 max_pool 去重，各自独立进 RRF：
    #   - 同一 content_id 的 main 向量 rank=3 → rrf_main = w/(k+4)
    #   - 同一 content_id 的 media 向量 rank=20 → rrf_media = w/(k+21)
    #   - 该内容总分 = rrf_main + rrf_media（scores dict 累加）
    # 这就是"main 和 media 放一起当一个内容计算分数"——两个视角各自匹配，
    # 都贡献分数。标题和图片都匹配查询的内容天然比只匹配标题的排更前。
    for rank, item in enumerate(vector_results):
        cid = item["content_id"]
        rrf = vector_weight / (k + rank + 1)
        scores[cid] = scores.get(cid, 0.0) + rrf
        # 合并 document：第一条（如 main）进 metadata，后续（如 media）的
        # document 补到 media_text / main_text，供 prompt 渲染。
        if cid not in metadata:
            metadata[cid] = item
        else:
            existing = metadata[cid]
            doc = (item.get("document") or "").strip()
            vt = item.get("vec_type", "")
            if doc and vt == "media":
                existing.setdefault("media_text", doc)
            elif doc and vt == "main":
                existing.setdefault("main_text", doc)

    # BM25 channel
    for rank, item in enumerate(bm25_results):
        cid = item["content_id"]
        rrf = bm25_weight / (k + rank + 1)
        scores[cid] = scores.get(cid, 0.0) + rrf
        if cid not in metadata:
            metadata[cid] = item

    # Comment supplement channel (never primary, never displacing).
    # comment_weight <= 0 disables the channel entirely (rollback switch).
    if comment_weight > 0:
        for rank, item in enumerate(comment_results or []):
            cid = item["content_id"]
            rrf = comment_weight / (k + rank + 1)
            scores[cid] = scores.get(cid, 0.0) + rrf
            if cid not in metadata:
                meta = dict(item)
                meta["is_comment_match"] = True
                metadata[cid] = meta
            # Note already matched via main/media → keep primary metadata and
            # do NOT flag it as a comment match (primary evidence exists).

    # Tag soft boost
    if qu_tags:
        for cid in list(scores.keys()):
            meta = metadata.get(cid, {})
            item_tags = _parse_tags(meta.get("tags", ""))
            if _any_tag_match(qu_tags, item_tags):
                scores[cid] *= (1.0 + tag_boost_factor)

    # Conditional freshness decay (only when no explicit time range and not exist_check)
    now = datetime.now(CST)
    if not has_time_range and not is_exist_check:
        for cid in list(scores.keys()):
            collected = metadata.get(cid, {}).get("collected_at", "")
            if collected:
                try:
                    dt = datetime.fromisoformat(collected)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=CST)
                    days_ago = (now - dt).total_seconds() / 86400.0
                    if days_ago >= 0:
                        decay = math.exp(-freshness_lambda * days_ago)
                        scores[cid] *= decay
                except (ValueError, TypeError):
                    pass

    # Build result list
    result = []
    for cid, rrf_score in scores.items():
        meta = dict(metadata.get(cid, {}))
        meta["content_id"] = cid
        meta["rrf_score"] = rrf_score
        result.append(meta)

    return result


def _parse_tags(tags_str: str) -> list[str]:
    """Parse tags from JSON array string or comma-separated string."""
    if not tags_str:
        return []
    try:
        parsed = json.loads(tags_str)
        if isinstance(parsed, list):
            return [str(t) for t in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    # Try comma-separated
    return [t.strip() for t in tags_str.split(",") if t.strip()]


def _any_tag_match(qu_tags: list[str], item_tags: list[str]) -> bool:
    """Check if any query tag matches any item tag."""
    qu_set = {t.lower() for t in qu_tags}
    item_set = {t.lower() for t in item_tags}
    return bool(qu_set & item_set)


def _build_system_prompt(
    qu_result: QueryUnderstandingResult,
    top_items: list[dict],
    content_map: dict[int, dict],
    total_count: int,
    platform_dist: dict[str, int],
) -> str:
    """Build enhanced system prompt with retrieval context."""
    intent = qu_result.intent

    # 注入当前时间（CST）：统一从 system_context 取（P0 收口）。回答 LLM 需要
    # 知道"今天"是几号，才能判断条目收藏时间是否在今天/昨天/最近N天内。
    from omnibox_agent.services.system_context import time_fact

    parts = ["你是 OmniHub Ask，回答必须严格基于用户的私人收藏库。\n"]
    parts.append(time_fact())

    # Retrieval stats
    parts.append(f"【检索统计】共找到 {total_count} 条相关内容")
    # 检索时间范围：用户明确给了时间窗（今天/最近一周等）时告知本次检索的
    # 具体起止，LLM 据此核对条目收藏时间并组织回答。
    ts, te = qu_result.time_range_start, qu_result.time_range_end
    if ts is not None or te is not None:
        ts_s = ts.strftime("%Y-%m-%d %H:%M") if ts else "-"
        te_s = te.strftime("%Y-%m-%d %H:%M") if te else "-"
        parts.append(f"（检索时间范围：{ts_s} ~ {te_s}，仅返回该时间段内的收藏）")
    # 平台分布展示规则：
    #  - 用户明确排除平台分组（"不按平台分类"）→ 永不展示
    #  - 用户要求按主题/标签/类型等非平台维度分类 → 不展示（避免带偏）
    #  - 用户要求按平台分类，或只是普通计数/列举（无分类意图）→ 展示
    # 这样普通"我有多少收藏"仍保留平台分布，但分类场景不再默认按平台带偏。
    show_platform_dist = (
        bool(platform_dist)
        and not getattr(qu_result, "exclude_platform", False)
        and (
            getattr(qu_result, "classify_by", None) == "platform"
            or not getattr(qu_result, "want_classify", False)
        )
    )
    if show_platform_dist:
        dist_str = "，".join(f"{plat}: {cnt}条" for plat, cnt in sorted(platform_dist.items()))
        parts.append(f"（平台分布: {dist_str}）")
    parts.append("\n")

    # Items
    if top_items:
        parts.append("【召回到的相关条目】\n")
        for i, item in enumerate(top_items):
            cid = item["content_id"]
            detail = content_map.get(cid, {})
            title = detail.get("title") or item.get("title", "无标题")
            platform = detail.get("platform_name") or item.get("platform_name", "")
            author = detail.get("author_name") or item.get("author_name", "")
            summary = detail.get("summary") or ""
            collected = item.get("collected_at", "")
            tags = item.get("tags", "")

            # Truncate summary
            if len(summary) > 200:
                summary = summary[:200]

            # 向量库 document：media 向量的图片解析内容（价目表、菜品名等）
            # 和 main 向量的正文。这些是向量库里实际嵌入的文本，比 MySQL summary
            # 更完整（summary 可能为空），渲染进 prompt 让 LLM 能看到图片解析结果。
            media_text = str(item.get("media_text") or "").strip()
            main_text = str(item.get("main_text") or "").strip()
            # main_text 和 summary 可能重复（都来自正文），只补充非重复部分
            if main_text and main_text != summary and summary not in main_text:
                if len(main_text) > 200:
                    main_text = main_text[:200]
            else:
                main_text = ""
            if media_text:
                # 截断：单条 media document 可能含多张图解析，取前 300 字符
                if len(media_text) > 300:
                    media_text = media_text[:300]

            # 评论区兜底检索：该条目正文未含问题答案语义（gate 判定无
            # relevant、仅 topic_relevant）时附上的评论全文。答案可能
            # 藏在评论区（如"地点在哪里"→评论里的地址），生成时须阅读。
            comments_text = str(item.get("comments_text") or "").strip()

            # 评论补充命中：正文未直接匹配查询，仅评论区提及。标注出来，
            # 让 LLM 知道该条目的正文可信度低于主/媒体命中。
            match_tag = "（评论补充命中：正文未直接匹配，仅评论区提及）" if item.get("is_comment_match") else ""

            parts.append(f"{i + 1}. [{platform} / @{author}] {title}{match_tag}")
            parts.append(f"  content_id: {cid}")
            if summary:
                parts.append(f"  摘要: {summary[:200]}")
            if main_text:
                parts.append(f"  正文: {main_text}")
            if media_text:
                parts.append(f"  图片解析: {media_text}")
            if comments_text:
                parts.append(f"  评论区: {comments_text}")
            if tags:
                parts.append(f"  标签: {tags}")
            parts.append(f"  收藏时间: {collected[:10]}")
            parts.append("")
    else:
        parts.append("【召回结果】未找到相关条目。\n")

    # 分类标签分布：当用户要求分类且非平台维度时，作为分组提示提供给 LLM。
    tag_dist: dict[str, int] = {}
    for _it in top_items:
        _d = content_map.get(_it.get("content_id"), {})
        _raw = _d.get("tags") or _it.get("tags", "")
        for _t in _parse_tags(_raw):
            if _t:
                tag_dist[_t] = tag_dist.get(_t, 0) + 1
    top_tags = sorted(tag_dist.items(), key=lambda x: -x[1])[:12]

    # Intent-specific instructions
    parts.append("【回答要求】")
    parts.append("- 使用 Markdown 格式回答，遵循以下排版规范：")
    parts.append("  * 标题：用 `##` 作主标题，`###` 作子标题，不要跳级；单条回答最多两层标题")
    parts.append("  * 列表：并列要点用 `-` 无序列表；有顺序/步骤用 `1.` 有序列表；列表项之间不空行")
    parts.append("  * 加粗：仅对关键结论、数字、平台名/作者名等需要强调的信息用 `**加粗**`，不要整句加粗")
    parts.append("  * 引用：引用条目原文用 `> ` 引用块，引用块内不再嵌套列表")
    parts.append("  * 代码/链接：路径、命令、ID 等用反引号 `` ` `` 包裹")
    parts.append("  * 分隔：不同主题之间用 `---` 分隔线隔开；段落之间空一行，保持留白")
    parts.append("  * 表格：分类统计、多维度对比优先用 Markdown 表格呈现")
    parts.append("  * 禁止：不输出 HTML 标签、不滥用代码块包裹普通文本、不为单句话加标题")
    parts.append("- **超链接规则（重要）**：当你在回答中提到具体的收藏条目时，必须使用 Markdown 超链接格式：")
    parts.append("  `[条目标题](content://条目的content_id)`")
    parts.append("  例如：`[收藏条目标题](content://123)` — 用户点击即可跳转到该条目详情页")
    parts.append("  注意：content_id 已在上方每个条目中标注，请使用真实的 content_id 值")
    parts.append("  对于非收藏条目的普通链接（如外部网址），仍使用 `[文本](url)` 格式")
    parts.append("- **正文格式（重要）**：不要在正文中用列表（`-` 或 `1.`）形式罗列收藏项目。")
    parts.append("  所有提到的收藏项目应该用内联 `[标题](content://content_id)` 格式在叙述中提及，")
    parts.append("  或者完全省略（因为小程序的胶囊会自动展示所有提到的项目）。")
    parts.append("  正文应该是一个连贯的叙述段落，而不是项目列表。分类介绍用文字描述，")
    parts.append("  不要用 `- **分类名**：item1、item2、item3` 这种形式。")
    if intent == Intent.EXIST_CHECK:
        parts.append(f"- 用户问\"有没有\"相关内容：共找到 {total_count} 条，请回答\"有\"或\"没有\"")
        parts.append("- 如果 total_count > 0，说\"有\"并列出前几条")
        parts.append("- 如果 total_count = 0，说\"未在你的收藏中找到相关内容\"")
    elif intent == Intent.COUNT:
        parts.append(f"- 用户问数量：直接回答共 {total_count} 条")
        if getattr(qu_result, "exclude_platform", False):
            parts.append("- 用户要求不按平台分类：不要展示平台分布，改为按主题/标签/内容类型补充说明")
        else:
            parts.append("- 如果用户问了平台分布，可以补充说明")
    elif intent == Intent.GENERAL_LIST:
        parts.append("- 这是泛化列举查询，请概括性介绍用户的收藏内容")
        parts.append("- 分类介绍时优先按主题/标签/内容类型分组，不要默认按平台分组")
    else:
        parts.append("- 仅基于以上条目回答用户问题")
        parts.append("- 引用具体条目时，使用超链接格式 `[条目标题](content://content_id)` 而非纯序号标注")
        if not getattr(qu_result, "explicit_limit", False):
            # Unbounded aggregation / analysis queries (e.g. 口味/菜系偏好分析):
            # the whole召回集 reaches the LLM, so allow a fuller structured
            # analysis instead of the 300-char terse cap.
            parts.append("- 这是分析/聚合类问题，请基于召回的条目进行结构化分析，"
                         "可用表格/列表归纳偏好与规律，篇幅以内容充分为准")
        else:
            parts.append("- 控制在 300 字以内，简明扼要")
        parts.append("- 信息不足时直接说\"未在你的收藏中找到相关信息\"")
        parts.append("- 不要编造未在条目中出现的信息")
        parts.append("- 标记为『评论补充命中』的条目仅凭评论区命中，正文可能未直接涉及该主题；"
                     "引用时须以正文/摘要为准，评论信息只能作为辅助线索，不得当作条目正文内容描述")
        if any(item.get("comments_text") for item in top_items):
            parts.append("- 带『评论区』信息的条目：其正文被判定未直接回答问题，答案可能藏在评论区；"
                         "回答前务必阅读这些评论内容，若答案在评论中请如实引用并注明\"来自该条目的评论区\"，"
                         "不要因正文缺失而回答\"没有相关信息\"")

    # 分类指令：用户要求"将收藏分类"时，强制按主题/标签/内容类型分组，
    # 并在明确"不按平台分类"时禁止平台分组（覆盖上面的平台偏向）。
    classify_inst = _classify_instruction(qu_result, top_tags)
    if classify_inst:
        parts.append(classify_inst)

    return "\n".join(parts)


def _classify_instruction(qu_result: "QueryUnderstandingResult", top_tags: list) -> str | None:
    """Build a classification instruction honoring the user's grouping intent.

    Returns None when the user did NOT ask to classify. Otherwise instructs the
    LLM to group by theme/tag/type (never platform unless explicitly requested),
    and forbids platform grouping when the user said "不按平台分类".
    """
    if not getattr(qu_result, "want_classify", False):
        return None

    tag_hint = ""
    if top_tags:
        tag_str = "，".join(f"{t}: {c}" for t, c in top_tags)
        tag_hint = f" 可参考的标签分布（按条数降序）: {tag_str}。"

    if getattr(qu_result, "exclude_platform", False):
        return ("- 用户要求对收藏**分类**，且明确**不要按平台**分类。"
                "请改用【主题/标签/内容类型】等非平台维度对条目分组，"
                "禁止以平台作为分组依据，也不要展示平台分布表。"
                "先给出收藏总数，再按非平台维度分类列举，并给出每类条数。" + tag_hint)

    dim = getattr(qu_result, "classify_by", None)
    if dim == "theme":
        return "- 用户要求按【主题/话题】对收藏分类。先给总数，再按主题分组列举，不要按平台分组。" + tag_hint
    if dim == "tag":
        return "- 用户要求按【标签】对收藏分类。先给总数，再按标签分组列举，不要按平台分组。" + tag_hint
    if dim == "type":
        return "- 用户要求按【内容类型】对收藏分类（如美食/穿搭/数码等）。先给总数，再按内容类型分组列举，不要按平台分组。" + tag_hint
    if dim == "platform":
        return ("- 用户要求按【平台】对收藏分类。先给总数，再按平台分组列举"
                "（可配合上方平台分布）。")
    # 未指定维度：默认主题/标签，不要平台
    return ("- 用户要求对收藏**分类**但未指定维度。请默认按【主题/标签/内容类型】分组，"
            "不要按平台分组，也不要展示平台分布表。先给总数，再分类列举。" + tag_hint)
