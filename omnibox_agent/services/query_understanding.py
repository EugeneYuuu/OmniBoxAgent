"""Query Understanding service: LLM-driven query parsing with rule fallback.

Extracts: keywords, intent, time_range, platform, tags, recency flag,
resolvedQuery (with context), embeddingQuery (clean for vector search).
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone

from omnibox_agent.core.config import get_config
from omnibox_agent.models.query import (
    Intent,
    QueryUnderstandingResult,
)

log = logging.getLogger(__name__)

# Timezone for China (Asia/Shanghai = UTC+8)
CST = timezone(timedelta(hours=8))

# Known platform names and aliases
PLATFORM_ALIASES: dict[str, str] = {
    "抖音": "douyin",
    "快手": "kuaishou",
    "小红书": "xiaohongshu",
    "b站": "bilibili",
    "bilibili": "bilibili",
    "微博": "weibo",
    "知乎": "zhihu",
    "公众号": "wechat",
    "微信公众号": "wechat",
    "微信": "wechat",
    "头条": "toutiao",
}

# Time window regex patterns
TIME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"最近\s*(\d+)\s*天"), "days"),
    (re.compile(r"最近\s*(\d+)\s*周"), "weeks"),
    (re.compile(r"最近\s*(\d+)\s*个?月"), "months"),
    (re.compile(r"最近\s*一\s*天"), "1_day"),
    (re.compile(r"最近\s*一\s*周"), "1_week"),
    (re.compile(r"最近\s*一\s*个?月"), "1_month"),
    (re.compile(r"今天"), "today"),
    (re.compile(r"昨天"), "yesterday"),
    (re.compile(r"本周"), "this_week"),
    (re.compile(r"上周"), "last_week"),
    (re.compile(r"本月"), "this_month"),
    (re.compile(r"上月"), "last_month"),
    (re.compile(r"今年"), "this_year"),
    (re.compile(r"去年"), "last_year"),
]

# Recency intent keywords
RECENCY_KEYWORDS = {"最近", "最新", "近期", "新收藏", "刚收藏", "近些", "新"}


async def parse_query(
    query: str,
    history: list[dict] | None = None,
    ai_config: dict | None = None,
) -> QueryUnderstandingResult:
    """Parse user query with optional 12h conversation context.

    Uses LLM for full understanding, falls back to rule-based parsing
    on timeout/failure.

    Args:
        query: Raw user query string.
        history: Recent conversation messages (role, content, ts).

    Returns:
        QueryUnderstandingResult with parsed intent and filters.
    """
    # Filter history to last 12 hours
    recent_history = _filter_recent_history(history) if history else []

    # Try LLM-based parsing
    try:
        result = await _llm_parse(query, recent_history, ai_config)
        if result:
            # 时间窗兜底（确定性优先）：查询含显式时间表达（今天/昨天/最近N天/
            # 本周/本月等）时，一律以规则按当前时刻精确换算的窗口为准，覆盖 LLM
            # 可能漏填/算错的 timeRange——修复「我今天刚收藏的内容」这类查询在
            # LLM 不知道当前日期时返回 null 时间窗、导致检索不做时间过滤的问题。
            _rule_start, _rule_end = _parse_time_window(query)
            if _rule_start is not None or _rule_end is not None:
                result.time_range_start = _rule_start
                result.time_range_end = _rule_end
                result.explicit_limit = True
                log.debug("QU: time window from rules (query=%s): %s ~ %s",
                          query[:40], _rule_start, _rule_end)
            # Fix: analysis/aggregation queries must not be treated as recency.
            # The LLM often sets recency=True for "分析我收藏的美食/口味/菜系"
            # style questions; only keep recency when the query has an explicit
            # recency keyword ("最近/最新/近期/新收藏" etc.).
            if not _detect_recency(query) and _is_analysis_query(query):
                result.recency = False
                log.debug("QU: cleared recency for analysis query: %s", query[:40])
            return result
    except Exception as e:
        log.warning("LLM query understanding failed: %r, using rule fallback", e)

    # Rule-based fallback
    result = _rule_parse(query)
    # Apply context resolution even for rule fallback
    if recent_history:
        result.resolved_query = _resolve_context(query, recent_history)

    # Derive embeddingQuery from result
    if not result.embedding_query:
        result.embedding_query = _clean_for_embedding(result.resolved_query or query)

    # Derive resolvedQuery if not set
    if not result.resolved_query:
        result.resolved_query = query

    # Set keywords from resolved query
    if not result.keywords:
        result.keywords = _extract_keywords(result.resolved_query)

    return result


def _filter_recent_history(history: list[dict]) -> list[dict]:
    """Filter history to messages within last 12 hours, most recent first."""
    now = datetime.now(CST)
    cutoff = now - timedelta(hours=12)
    recent = []
    for msg in history:
        ts = msg.get("ts")
        if ts:
            try:
                msg_time = datetime.fromtimestamp(ts / 1000, tz=CST)
                if msg_time >= cutoff:
                    recent.append(msg)
            except (ValueError, OSError):
                pass
    # Sort by ts ascending (oldest first)
    recent.sort(key=lambda m: m.get("ts", 0))
    return recent


async def _llm_parse(
    query: str,
    history: list[dict],
    ai_config: dict | None = None,
) -> QueryUnderstandingResult | None:
    """Use fast LLM model to parse query with context.

    Per the user directive, QU must run under the user's own api-key (never the
    system key). Without a user key we skip LLM parsing and fall back to rules.
    """
    cfg = get_config().qu
    api_key = (ai_config or {}).get("apiKey")
    if not api_key:
        log.warning("QU: no user API key, skipping LLM parse (rule fallback)")
        return None

    # 注入当前日期（CST）：统一从 system_context 取，避免散落拼接（P0 收口）。
    # 否则 LLM 不知道"今天/昨天/最近N天"是几号，无法换算成 timeRange 绝对日期。
    from omnibox_agent.services.system_context import time_qu_hint

    today_hint = time_qu_hint()

    system_prompt = (
        "你是一个查询理解助手。根据用户问题和最近对话上下文,解析以下信息并以JSON格式返回。\n\n"
        + today_hint
        + """返回格式（严格的JSON对象,不要有多余内容）:
{
  "keywords": ["关键词1", "关键词2"],
  "intent": "search_and_summarize" | "exist_check" | "count" | "general_list",
  "resolvedQuery": "结合上下文消解后的完整查询",
  "embeddingQuery": "去除功能词(如\"帮我找\"/\"有没有\"/\"收藏的\")后的干净查询词",
  "recency": true/false,
  "timeRange": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} 或 null,
  "platform": "douyin/kuaishou/xiaohongshu/bilibili/weibo/zhihu/wechat/toutiao" 或 null,
  "tags": ["标签1", "标签2"],
  "wantClassify": true/false,
  "classifyBy": "platform" | "theme" | "tag" | "type" | "topic" | "content" 或 null,
  "excludePlatform": true/false
}

意图说明:
- search_and_summarize: 搜索并总结（默认）
- exist_check: 问"有没有"/"是否存在"
- count: 问"有多少"/"几个"
- general_list: 泛化列举,如"我收藏了什么"/"有哪些内容"（无具体主题时用此意图）

recency为true当用户明确要求按收藏时间排序("最近"/"最新"/"近期"等),此时timeRange通常为null（排序意图不加时间窗）。

timeRange只在用户明确给出时间范围时才填写（如"最近一周"/"上个月"）。

上下文窗口仅限最近12小时内对话。请基于上下文进行指代消解（省略主体、代词回指等，从历史中补全被省略或指代的主体）。

主题延续（重要）：当用户的新问题省略了前面对话中已经确立的主题时，必须在 resolvedQuery 与 embeddingQuery 中补全省略的主题，而不是当作全新独立主题。判定标准：若当前问题只出现天数、数量、方式等维度词而未出现任何具体主题名词，而前面对话已确立某个主题，则这些维度词承继的是前面那个主题——resolvedQuery 与 embeddingQuery 都应写成「历史主题 + 当前维度词」，禁止只保留当前维度词而丢掉历史主题。

注意：
- 平台名称统一为英文小写：douyin/kuaishou/xiaohongshu/bilibili/weibo/zhihu/wechat/toutiao
- embeddingQuery应该是最简洁的关键词组合，不含"帮我找"等请求用语

分类偏好（重要）：
- wantClassify: 用户是否要求对收藏进行"分类/归类/分组/盘点/梳理"（即明确说出分类、归类、分组、盘点、梳理等动作）。
- classifyBy: 用户指定的分组维度。theme=按主题/话题, tag=按标签, type=按内容类型(如美食/穿搭/数码), platform=按平台。若用户只说"分类"但未指定维度、且未排除平台，留 null（由回答端默认按主题/标签分组，而不是平台）。
- excludePlatform: 用户明确要求"不按平台/不要按平台/不是按平台分类/别按平台"时为 true。此时 classifyBy 一般留 null（端到端默认按主题/标签分组）。
"""
    )

    messages = _build_qu_messages(system_prompt, query, history)
    return await _call_qu_api(messages, cfg, ai_config)


def _build_qu_messages(
    system_prompt: str,
    query: str,
    history: list[dict],
) -> list[dict]:
    """Build messages for QU LLM call, including context."""
    msgs = [{"role": "system", "content": system_prompt}]

    # Add recent history as context (limit to ~2000 chars)
    total_chars = 0
    context_msgs = []
    for msg in reversed(history):
        content = msg.get("content", "")
        if len(content) > 200:
            content = content[:200]
        msg_text = f"[{msg.get('role', 'user')}]: {content}"
        if total_chars + len(msg_text) > 2000:
            break
        context_msgs.insert(0, msg_text)
        total_chars += len(msg_text)

    if context_msgs:
        context_str = "【最近对话上下文】\n" + "\n".join(context_msgs)
        msgs.append({"role": "user", "content": context_str})

    msgs.append({"role": "user", "content": f"【用户问题】\n{query}"})
    return msgs


async def _call_qu_api(
    messages: list[dict],
    cfg,
    ai_config: dict | None = None,
) -> QueryUnderstandingResult | None:
    """Call QU LLM (LangChain ChatOpenAI) with per-user config.

    Uses ONLY the user's own api-key (never the system key). Missing key should
    not reach here — callers skip LLM parsing when there is no user key.
    """
    cfg_ai = ai_config or {}
    base_url = cfg_ai.get("baseUrl") or cfg.base_url
    model = cfg_ai.get("modelName") or cfg.model
    api_key = cfg_ai.get("apiKey")
    if not api_key:
        log.warning("QU: no user API key, skipping LLM parse")
        return None

    from omnibox_agent.services.llm_langchain import _call_llm

    try:
        # read 无超时（给 LLM 思考时间）；connect 10s 快速失败（ChatOpenAI 内部实现）。
        # QU 是工具型调用：no_thinking=True 禁用思考；max_tokens=4096（用户定）。
        # 不强制 response_format=json_object（与旧实现一致，兼容不支持该参数的模型），
        # 依赖 system prompt 约束 + _parse_qu_response 兜底抽取 JSON。
        content = await _call_llm(
            messages,
            model=model,
            base_url=base_url,
            api_key=api_key,
            ai_config=None,
            temperature=0.1,
            max_tokens=4096,
            no_thinking=True,
        )
        if not content:
            return None
        return _parse_qu_response(content, query=extract_query_from_messages(messages))
    except Exception as e:
        log.warning("QU API call failed: %r", e)
        return None


def _parse_qu_response(content: str, query: str = "") -> QueryUnderstandingResult | None:
    """Parse LLM response JSON into QueryUnderstandingResult."""
    try:
        # Extract JSON from response
        json_str = _extract_json(content)
        data = json.loads(json_str)

        keywords = data.get("keywords", [])
        intent_str = data.get("intent", "search_and_summarize")
        intent = _normalize_intent(intent_str)

        resolved_query = data.get("resolvedQuery", "")
        embedding_query = data.get("embeddingQuery", "")
        recency = data.get("recency", False)
        platform = data.get("platform")
        tags = data.get("tags", [])

        # Classification preference
        want_classify = bool(data.get("wantClassify", False))
        raw_classify_by = data.get("classifyBy")
        classify_by = raw_classify_by if raw_classify_by in (
            "platform", "theme", "tag", "type", "topic", "content"
        ) else None
        exclude_platform = bool(data.get("excludePlatform", False))
        # A user who excludes platform grouping should never be forced into a
        # platform classification even if "平台" appears inside a negation.
        if exclude_platform and classify_by == "platform":
            classify_by = None

        # Parse time range
        time_start = None
        time_end = None
        tr = data.get("timeRange")
        if tr and isinstance(tr, dict):
            time_start = _parse_date(tr.get("start"))
            time_end = _parse_date(tr.get("end"))

        return QueryUnderstandingResult(
            keywords=keywords,
            intent=intent,
            resolved_query=resolved_query,
            embedding_query=embedding_query,
            recency=recency,
            time_range_start=time_start,
            time_range_end=time_end,
            platform=platform,
            tags=tags,
            want_classify=want_classify,
            classify_by=classify_by,
            exclude_platform=exclude_platform,
            limit_count=_extract_limit_count(query),
            # COUNT/EXIST_CHECK 是明确的单主题/定点查询：必须走有界检索
            # 与 relevance 判断（"有没有"需要判断是否相关），不能当无界聚合。
            explicit_limit=(intent in (Intent.COUNT, Intent.EXIST_CHECK))
            or (time_start is not None or time_end is not None)
            or _extract_limit_count(query) is not None,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        log.warning("Failed to parse QU response: %s, content=%s", e, content[:200])
        return None


def extract_query_from_messages(messages: list[dict]) -> str:
    """Extract the last user query from messages."""
    for msg in reversed(messages):
        if msg.get("role") == "user" and "用户问题" in msg.get("content", ""):
            return msg["content"].replace("【用户问题】\n", "")
    return ""


# ---- Intent alias normalization ----

_INTENT_ALIASES: dict[str, Intent] = {
    # 模型常返回的变体 → 标准枚举
    "query_collection_count": Intent.COUNT,
    "count_query": Intent.COUNT,
    "query_count": Intent.COUNT,
    "collection_count": Intent.COUNT,
    "count": Intent.COUNT,
    "exist_check": Intent.EXIST_CHECK,
    "existence_check": Intent.EXIST_CHECK,
    "query_collection_exist": Intent.EXIST_CHECK,
    "has_query": Intent.EXIST_CHECK,
    "general_list": Intent.GENERAL_LIST,
    "query_collection_list": Intent.GENERAL_LIST,
    "collection_list": Intent.GENERAL_LIST,
    "list_query": Intent.GENERAL_LIST,
    "search_and_summarize": Intent.SEARCH_AND_SUMMARIZE,
    "search_summarize": Intent.SEARCH_AND_SUMMARIZE,
    "query_collection_search": Intent.SEARCH_AND_SUMMARIZE,
    "search": Intent.SEARCH_AND_SUMMARIZE,
}


def _normalize_intent(intent_str: str) -> Intent:
    """Map raw LLM intent string to the canonical Intent enum.

    DeepSeek 等模型常返回非标准枚举(如 query_collection_count),
    这里做别名归一,未知值回退到 search_and_summarize。
    """
    key = (intent_str or "").strip().lower()
    return _INTENT_ALIASES.get(key, Intent.SEARCH_AND_SUMMARIZE)


# ---- Rule-based fallback ----

def _rule_parse(query: str) -> QueryUnderstandingResult:
    """Rule-based query parsing as fallback when LLM is unavailable."""
    intent = _detect_intent(query)
    recency = _detect_recency(query)
    time_start, time_end = _parse_time_window(query)
    platform = _detect_platform(query)
    tags = _detect_tags(query)
    keywords = _extract_keywords(query)
    want_classify, classify_by, exclude_platform = _detect_classify(query)

    return QueryUnderstandingResult(
        keywords=keywords,
        intent=intent,
        resolved_query=query,
        embedding_query=_clean_for_embedding(query),
        recency=recency,
        time_range_start=time_start,
        time_range_end=time_end,
        platform=platform,
        tags=tags,
        want_classify=want_classify,
        classify_by=classify_by,
        exclude_platform=exclude_platform,
        limit_count=_extract_limit_count(query),
        explicit_limit=(intent in (Intent.COUNT, Intent.EXIST_CHECK))
        or (time_start is not None or time_end is not None)
        or _extract_limit_count(query) is not None,
    )


def _extract_limit_count(query: str) -> int | None:
    """提取用户显式指定的结果条数（"给我10条"/"推荐3个"/"前5篇"）。

    返回 None 表示未指定条数 → 检索不限制 topK。
    仅匹配「数字 + 量词(条/个/篇/份/则/张/部/款/首/本/段)」；
    "几条/几个"(COUNT 意图) 因"几"非数字而不会误判。
    """
    m = re.search(r"(\d+)\s*(条|个|篇|份|则|张|部|款|首|本|段)", query)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except ValueError:
        return None
    return n if n >= 1 else None


def _detect_intent(query: str) -> Intent:
    """Detect query intent from patterns."""
    # Exist check
    if re.search(r"有没有|是否存在|有过|收藏过", query):
        return Intent.EXIST_CHECK
    # Count
    if re.search(r"多少|几个|几条|统计|计数|一共有", query):
        return Intent.COUNT
    # General list
    if re.search(r"我收藏了?什么|有哪些|全部的?|所有的?|收藏了?哪些", query):
        return Intent.GENERAL_LIST
    if len(query.strip()) <= 3:
        return Intent.GENERAL_LIST
    return Intent.SEARCH_AND_SUMMARIZE


def _detect_recency(query: str) -> bool:
    """Detect if query asks for recency-based sorting."""
    for kw in RECENCY_KEYWORDS:
        if kw in query:
            return True
    return False


# Analysis verbs — "分析我的口味/菜系偏好" is an aggregation task, NOT a
# recency/sorting request. Used to correct QU's recency misjudgment.
_ANALYSIS_PATTERNS = re.compile(
    r"分析|总结|归纳|偏好|口味|菜系|特点|风格|趋势|盘点|攻略|对比|推荐|评价|感受"
)


def _is_analysis_query(query: str) -> bool:
    """Heuristic: does the query ask for analysis/synthesis of the collection?"""
    return bool(_ANALYSIS_PATTERNS.search(query))


def _parse_time_window(query: str) -> tuple[datetime | None, datetime | None]:
    """Parse explicit time window from query.

    Returns (time_start, time_end) in CST.
    """
    now = datetime.now(CST)

    # Try numeric patterns first
    for pattern, unit in TIME_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue

        if unit in ("days", "weeks", "months"):
            n = int(match.group(1))
            if unit == "days":
                start = now - timedelta(days=n)
            elif unit == "weeks":
                start = now - timedelta(weeks=n)
            else:
                start = now - timedelta(days=n * 30)
            return (start.replace(hour=0, minute=0, second=0, microsecond=0), now)

        elif unit == "1_day":
            start = now - timedelta(days=1)
            return (start.replace(hour=0, minute=0, second=0, microsecond=0), now)
        elif unit == "1_week":
            start = now - timedelta(weeks=1)
            return (start.replace(hour=0, minute=0, second=0, microsecond=0), now)
        elif unit == "1_month":
            start = now - timedelta(days=30)
            return (start.replace(hour=0, minute=0, second=0, microsecond=0), now)

        elif unit == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return (start, now)
        elif unit == "yesterday":
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(hour=23, minute=59, second=59, microsecond=999999)
            return (start, end)
        elif unit == "this_week":
            # Monday of current week
            days_since_monday = now.weekday()
            start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            return (start, now)
        elif unit == "last_week":
            days_since_monday = now.weekday() + 7
            start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            return (start, end)
        elif unit == "this_month":
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return (start, now)
        elif unit == "last_month":
            first_of_this_month = now.replace(day=1)
            last_month_last_day = first_of_this_month - timedelta(days=1)
            start = last_month_last_day.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = last_month_last_day.replace(hour=23, minute=59, second=59, microsecond=999999)
            return (start, end)
        elif unit == "this_year":
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            return (start, now)
        elif unit == "last_year":
            start = now.replace(year=now.year - 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = now.replace(year=now.year - 1, month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
            return (start, end)

    return (None, None)


def _detect_platform(query: str) -> str | None:
    """Detect platform mention in query."""
    for name, code in PLATFORM_ALIASES.items():
        if name in query.lower():
            return code
    return None


_CLASSIFY_VERBS = re.compile(r"分类|归类|分组|盘点|梳理|整理")
_EXCLUDE_PLATFORM = re.compile(
    r"不(要|是)?\s*(按|用|以)?\s*平台|别\s*(按|用|以)?\s*平台|非\s*平台|不要.*平台.*分类"
)
_DIM_THEME = re.compile(r"按\s*(主题|话题|内容方向|方向)")
_DIM_TAG = re.compile(r"按\s*(标签|tag|关键字|关键词)", re.IGNORECASE)
_DIM_TYPE = re.compile(r"按\s*(内容)?\s*(类型|类别|品类)")
_DIM_PLATFORM = re.compile(r"按\s*平台")


def _detect_classify(query: str) -> tuple[bool, str | None, bool]:
    """Detect classification intent for "将收藏分类" style queries.

    Returns (want_classify, classify_by, exclude_platform):
      - want_classify: query asks to classify/group the collection.
      - classify_by: explicit dimension (theme/tag/type/platform) if stated.
      - exclude_platform: user explicitly says NOT to classify by platform.

    The model default (no dimension) is theme/tag grouping, never platform —
    this is enforced downstream in the answer prompt.
    """
    want_classify = bool(_CLASSIFY_VERBS.search(query))
    exclude_platform = bool(_EXCLUDE_PLATFORM.search(query))

    classify_by: str | None = None
    if _DIM_THEME.search(query):
        classify_by = "theme"
    elif _DIM_TAG.search(query):
        classify_by = "tag"
    elif _DIM_TYPE.search(query):
        classify_by = "type"
    elif _DIM_PLATFORM.search(query):
        classify_by = "platform"

    # "不按平台" must not be turned into a platform classification.
    if exclude_platform and classify_by == "platform":
        classify_by = None

    return want_classify, classify_by, exclude_platform


def _detect_tags(query: str) -> list[str]:
    """Extract hashtag-like tags from query."""
    tags = re.findall(r"#(\S+)", query)
    return tags[:5]


def _extract_keywords(query: str) -> list[str]:
    """Extract significant keywords from query for search."""
    stop_words = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不",
        "这", "那", "吗", "呢", "啊", "吧", "请", "帮", "让",
        "可以", "能", "会", "要", "想", "有没有", "告诉我",
        "收藏", "什么", "怎么", "哪些", "那个", "这个",
        "最近", "最新", "关于",
    }

    # 优先用 jieba 切中文词；不可用时退化为空白切分
    words: list[str] = []
    try:
        import jieba
        words = list(jieba.cut(query.strip()))
    except Exception:
        cleaned = re.sub(r"[^\w\s]", " ", query)
        words = cleaned.split()

    keywords = []
    seen = set()
    for w in words:
        w = w.strip()
        if len(w) < 2 or w.lower() in stop_words or w in seen:
            continue
        seen.add(w)
        keywords.append(w)
        if len(keywords) >= 5:
            break

    return keywords


def _clean_for_embedding(query: str) -> str:
    """Clean query for embedding: remove function/request words.

    Only keeps the core semantic content.
    """
    patterns_to_remove = [
        r"帮我找(下|一下)?",
        r"(请)?帮我(搜索|查找|检索|看看?|找下)",
        r"有没有(关于)?",
        r"是否(有|存在)?",
        r"我(想|要)知道",
        r"告诉我",
        r"收藏的?",
        r"有哪些",
        r"几个",
        r"多少",
        r"相关的?",
    ]

    cleaned = query
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else query


def _resolve_context(query: str, history: list[dict]) -> str:
    """Simple context resolution for pronoun references.

    Handles cases like "那北京的呢?" by appending the last query's topic.
    """
    # Check if query looks like a follow-up (contains relative pronouns or very short)
    follow_up_patterns = [
        r"^那\S*的呢?",
        r"^(还|也|再)\S*的呢?",
        r"^(这个|那个|上面|前面).*",
    ]

    is_follow_up = any(re.match(p, query) for p in follow_up_patterns)
    is_follow_up = is_follow_up or len(query.strip()) <= 5

    if is_follow_up and history:
        # Extract last user question from history
        for msg in reversed(history):
            if msg.get("role") == "user":
                prev_query = msg.get("content", "")
                # Simple: prepend context words from previous query
                context_keywords = _clean_for_embedding(prev_query)
                if context_keywords:
                    return f"{context_keywords} {query}"
                break

    return query


# ---- helpers ----

def _extract_json(text: str) -> str:
    """Extract JSON object from LLM response text."""
    # Try to find JSON block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # Try to find bare JSON
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse date string to datetime in CST."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=CST)
        return dt
    except (ValueError, TypeError):
        return None
