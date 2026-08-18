"""AskAgent pipeline steps (v4.1).

Steps: Parse → Guard → Retrieve → Gate → Reason → Act(streaming) → done

各 Step 被 LangGraph QA 子图（graph_qa.py）的节点按需实例化并执行。
生产路径全流式：act_node 内直接调 stream_chat 逐 token 输出；
done 事件由 stream_pipeline handler 组装。
"""

from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

from omnibox_agent.agent.base import Agent, AgentMeta
from omnibox_agent.agent.context import (
    AgentContext,
    RetrievalOutput,
    ReasoningOutput,
)
from omnibox_agent.agent.loop import (
    PipelineStep,
    PipelineAborted,
    run_blocking,
)

if TYPE_CHECKING:
    from omnibox_agent.core.config import Config

log = logging.getLogger(__name__)


# ---- Step 1: Parse (Query Understanding) ----

class ParseStep(PipelineStep):
    """Run query understanding (LLM or rule-based). Non-critical.

    Writes ctx.artifacts["perception"] -> QueryUnderstandingResult.
    """

    async def execute(self, ctx: AgentContext) -> None:
        from omnibox_agent.services.query_understanding import parse_query
        from omnibox_agent.models.query import QueryUnderstandingResult

        query = ctx.input.get("query", "")
        history = ctx.input.get("history", [])
        ai_config = ctx.input.get("ai_config", {})

        # R10：会话指代查询的 LLM 消解词（如「第三点有哪些推荐的」→「AI/职场
        # 相关收藏推荐」）——QU 的规则消解解不开编号/指代，直接用消解词构建
        # perception，让整条 simple 链路（检索/Gate/回答）都基于消解后的语义。
        # explicit_limit=True：这是明确的单主题检索（用户索要该主题的条目），
        # 不是聚合/分析查询——必须让 Gate 走相关性过滤，避免「相关收藏推荐」
        # 这类词触发 unbounded skip 导致全部候选混入（美食/旅行带偏回答）。
        conv_resolved = (ctx.input.get("_conv_resolved_query") or "").strip()
        if conv_resolved:
            result = QueryUnderstandingResult(
                keywords=_extract_keywords(conv_resolved),
                resolved_query=conv_resolved,
                embedding_query=_clean_for_embedding(conv_resolved),
                explicit_limit=True,
            )
            ctx.artifacts["perception"] = result
            log.info("ParseStep: using conversation-resolved query: %s", conv_resolved[:60])
            return

        try:
            result = await parse_query(query, history, ai_config)
        except Exception as e:
            log.warning("ParseStep failed: %s, using rule fallback", e)
            # Build a minimal fallback result
            result = QueryUnderstandingResult(
                keywords=_extract_keywords(query),
                resolved_query=query,
                embedding_query=_clean_for_embedding(query),
            )

        # §12.2 长期记忆 QU 先验：QU 显式判定 > 用户 query 显式指定 > 偏好先验。
        # 只填空值（本次未被显式指定的字段）；recall 侧已保证 query 文本含
        # 平台名时不下发 platform 先验（双保险）。不改 parse_query 签名。
        try:
            prior = (ctx.input.get("long_term") or {}).get("qu_prior") or {}
            if prior.get("platform") and not (result.platform or "").strip():
                result.platform = prior["platform"]
            if prior.get("classify_by") and not (result.classify_by or "").strip():
                result.classify_by = prior["classify_by"]
        except Exception as e:
            log.debug("QU prior fill skipped: %s", e)

        ctx.artifacts["perception"] = result


# ---- Step 2: Guard ----

class GuardStep(PipelineStep):
    """Authorization / pre-flight check. Critical.

    Resolves account_ids, writes ctx.artifacts["guard"].
    If no accounts: PipelineAborted(code="guard").
    """

    def __init__(self):
        super().__init__(name="GuardStep", critical=True)

    async def execute(self, ctx: AgentContext) -> None:
        from omnibox_agent.services.retrieval_store import get_account_ids

        user_id = ctx.input.get("user_id", "")
        account_ids = await run_blocking(get_account_ids, user_id)

        if not account_ids:
            raise PipelineAborted("No authorized accounts", code="guard")

        ctx.artifacts["guard"] = {"account_ids": account_ids}


# ---- Step 3: Retrieve ----

class RetrieveStep(PipelineStep):
    """Hybrid search + RRF fusion. Non-critical.

    Reads: ctx.artifacts["perception"], ctx.artifacts["guard"]
    Writes: ctx.artifacts["retrieval"] -> RetrievalOutput

    v4.1: Supports re-retrieval after gate rewrite (reads _rewritten_query).
    """

    def __init__(self, config: Config):
        super().__init__(name="RetrieveStep", critical=False)
        self._config = config

    async def execute(self, ctx: AgentContext) -> None:
        from omnibox_agent.services.ask_orchestrator import retrieve_pipeline

        perception = ctx.artifacts.get("perception")
        # Defensive: if ParseStep failed non-critically and never wrote a
        # perception artifact, build a minimal default so retrieval (and the
        # downstream ReasonStep) never receive None.
        if perception is None:
            from omnibox_agent.models.query import QueryUnderstandingResult
            perception = QueryUnderstandingResult(
                resolved_query=ctx.input.get("query", "") or "",
            )
            ctx.artifacts["perception"] = perception

        # v4.1: Check for rewritten query from gate
        rewritten_query = ctx.input.pop("_rewritten_query", None)
        if rewritten_query:
            # Update perception with rewritten query for this retrieval pass
            from omnibox_agent.models.query import QueryUnderstandingResult
            perception = QueryUnderstandingResult(
                resolved_query=rewritten_query,
                embedding_query=rewritten_query,
                keywords=perception.keywords if perception else [],
                intent=perception.intent if perception else None,
                tags=perception.tags if perception else [],
                platform=perception.platform if perception else None,
                time_range_start=perception.time_range_start if perception else None,
                time_range_end=perception.time_range_end if perception else None,
                recency=perception.recency if perception else False,
                # 重建时保留原始范围标记：EXIST_CHECK/COUNT/时间窗的有界语义
                # 不能被 rewrite 悄悄重置回 unbounded（否则 gate 又会跳过判断）。
                explicit_limit=perception.explicit_limit if perception else False,
            )
            ctx.artifacts["perception"] = perception
            log.info("RetrieveStep: using rewritten query: %s", rewritten_query[:50])

        guard = ctx.artifacts.get("guard", {})
        account_ids = guard.get("account_ids", [])
        ai_config = ctx.input.get("ai_config", {})

        try:
            result = await run_blocking(
                retrieve_pipeline,
                perception,
                account_ids,
                ctx.input,
                ai_config,
                self._config.retrieval,
            )
        except Exception as e:
            log.warning("RetrieveStep failed: %s", e)
            result = RetrievalOutput()

        ctx.artifacts["retrieval"] = result


# ---- Step 3.5: Gate (v4.1 Quality Gate + CRAG Refinement) ----

class GateStep(PipelineStep):
    """v4.1 Layer 2: Batch relevance gate + CRAG refinement + fallback chain.

    Reads: ctx.artifacts["retrieval"], ctx.input["query"]
    Writes: ctx.artifacts["retrieval"] (filtered), ctx.flags, ctx.metrics

    Gate flow (§5.1):
      1. Batch judge all candidates (single LLM call)
      2. Any relevant → filter to relevant, proceed
      3. All irrelevant → on-demand parse → re-retrieve
      4. Still irrelevant → rewrite → re-retrieve
      5. Still irrelevant → degrade (low_confidence)

    After gate passes: CRAG refinement on >300 char docs (§5.5).
    Then fit_budget (§6) for token budget control.

    This step is non-critical: if gate fails, proceed with original docs.
    """

    def __init__(self, config: Config):
        super().__init__(name="GateStep", critical=False)
        self._config = config

    async def execute(self, ctx: AgentContext) -> None:
        from omnibox_agent.services.quality_gate import (
            quality_gate, refine_docs, fit_budget, GATE_PROCEED,
            GATE_RE_RETRIEVE, GATE_REWRITE,
        )
        log.debug("GateStep: gate=%s", self._config.gate.enabled)

        # v4.1: Gate toggle (rollback switch — design doc §13 风险: 回滚 = 关闭门控开关)
        if not self._config.gate.enabled:
            log.debug("GateStep: gate disabled (GATE_ENABLED=false), skipping")
            ctx.metrics["gate_decision"] = "disabled"
            return

        retrieval = ctx.artifacts.get("retrieval")
        if retrieval is None or not retrieval.fused_items:
            log.debug("GateStep: no retrieval results, skipping gate")
            ctx.metrics["gate_decision"] = "skip_empty"
            return

        # Run quality gate
        decision = await quality_gate(ctx)

        # R10：会话指代查询（_conv_referential）跳过 Gate 的 rewrite/re-retrieve
        # 循环——检索词已是 LLM 消解的 resolved_query（「第三点」→「AI/职场」），
        # Gate 重写只会把查询带回原始语义（「第三点有哪些推荐」）导致召回噪音。
        # 这类查询的检索语义由 judge 保证：Gate 若判 all irrelevant，保留消解词
        # 召回的条目，不触发 rewrite。
        # 不硬编码截断 top 8（用户指令：默认不限制 top-k）——保留全部召回，
        # 仅按 token 预算收口（替代原先 [:8] 兼任的隐式收口作用，否则全量
        # 召回会直接进 prompt）。
        if ctx.input.get("_conv_referential") and decision in (GATE_RE_RETRIEVE, GATE_REWRITE):
            retrieval = ctx.artifacts.get("retrieval")
            if retrieval is not None and retrieval.fused_items:
                perception = ctx.artifacts.get("perception")
                if perception and (not perception.explicit_limit
                                   or getattr(perception, "want_classify", False)):
                    budget = self._config.generation.unbounded_context_token_budget
                else:
                    budget = self._config.generation.context_token_budget
                retrieval.fused_items = fit_budget(retrieval.fused_items, budget)
                retrieval.total_count = len(retrieval.fused_items)
            log.info("GateStep: conversation-referential query, skipping gate rewrite (decision=%s), "
                     "keeping %d resolved items (token-budgeted, no top-8 cap)",
                     decision, len(retrieval.fused_items) if retrieval is not None else 0)
            ctx.metrics["gate_decision"] = f"skip_conv_ref_{decision}"
            return

        if decision in (GATE_RE_RETRIEVE, GATE_REWRITE):
            # v4.1 flow fix: actually re-retrieve instead of proceeding with the
            # (irrelevant) original docs. Previously this was a no-op that only
            # logged "proceeding with current docs", making the gate fallback
            # chain (on-demand parse / query rewrite -> re-retrieve) non-functional.
            from omnibox_agent.services.ask_orchestrator import retrieve_pipeline
            from omnibox_agent.models.query import QueryUnderstandingResult, Intent

            perception = ctx.artifacts.get("perception")
            guard = ctx.artifacts.get("guard", {})
            account_ids = guard.get("account_ids", [])
            ai_config = ctx.input.get("ai_config", {})

            if decision == GATE_REWRITE and ctx.input.get("_rewritten_query"):
                rewritten = ctx.input["_rewritten_query"]
                perception = QueryUnderstandingResult(
                    resolved_query=rewritten,
                    embedding_query=rewritten,
                    keywords=perception.keywords if perception else [],
                    intent=perception.intent if perception else None,
                    tags=perception.tags if perception else [],
                    platform=perception.platform if perception else None,
                    time_range_start=perception.time_range_start if perception else None,
                    time_range_end=perception.time_range_end if perception else None,
                    recency=perception.recency if perception else False,
                    # 与 RetrieveStep 一致：重建时保留 explicit_limit，防止
                    # EXIST_CHECK/COUNT 的定点语义在 rewrite 后回退为 unbounded。
                    explicit_limit=perception.explicit_limit if perception else False,
                )
                ctx.artifacts["perception"] = perception
                log.info("GateStep: re-retrieving with rewritten query: %s", rewritten[:50])
                ctx.metrics["gate_rewrite"] = True
            else:
                log.info("GateStep: re-retrieving after on-demand reindex")
                ctx.metrics["gate_re_retrieve"] = True

            try:
                new_retrieval = await run_blocking(
                    retrieve_pipeline,
                    perception,
                    account_ids,
                    ctx.input,
                    ai_config,
                    self._config.retrieval,
                )
                ctx.artifacts["retrieval"] = new_retrieval
                ctx.metrics["gate_re_retrieved"] = True
                log.info("GateStep: re-retrieval returned %d fused items",
                         len(new_retrieval.fused_items))

                # EXIST_CHECK：rewrite/reindex 后必须重新过 relevance 判断。
                # 否则 re-retrieve 的新结果未经过滤、total_count 仍是候选集
                # 大小，LLM 会基于大量不相关内容误判"有"（猫咪查询返回美食
                # 攻略的根因）。计数器（rewrite/ondemand）已耗尽，重判最坏
                # 只走到 degrade（total_count=0），不会死循环。
                if (perception is not None
                        and getattr(perception, "intent", None) == Intent.EXIST_CHECK
                        and new_retrieval.fused_items):
                    from omnibox_agent.services.quality_gate import quality_gate
                    await quality_gate(ctx)
            except Exception as e:
                log.warning("GateStep: re-retrieval failed: %r", e)

        # CRAG refinement on gated docs
        retrieval = ctx.artifacts.get("retrieval")
        if retrieval and retrieval.fused_items:
            query = ctx.input.get("query", "")
            perception = ctx.artifacts.get("perception")
            # Aggregation / summary queries (no explicit limit) get a wider
            # context budget so the whole主题范畴 reaches the generator.
            # Classification requests ("将收藏分类") also need the wide budget
            # so the whole collection reaches the generator for grouping.
            if perception and (not perception.explicit_limit
                               or getattr(perception, "want_classify", False)):
                budget = self._config.generation.unbounded_context_token_budget
            else:
                budget = self._config.generation.context_token_budget
            try:
                refined = await refine_docs(retrieval.fused_items, query, ctx)
                retrieval.fused_items = refined
            except Exception as e:
                log.warning("GateStep: CRAG refinement failed: %s", e)

            # Fit token budget
            retrieval.fused_items = fit_budget(retrieval.fused_items, budget)

        # Record gate metrics
        ctx.metrics["gate_elapsed"] = round(ctx.elapsed(), 2)
        ctx.metrics["llm_calls_so_far"] = ctx.llm_call_count


# ---- Step 4: Reason (Prompt building) ----

class ReasonStep(PipelineStep):
    """Build system prompt and messages. Non-critical.

    Reads: ctx.artifacts["perception"], ctx.artifacts["retrieval"]
    Writes: ctx.artifacts["reasoning"] -> ReasoningOutput
    """

    async def execute(self, ctx: AgentContext) -> None:
        from omnibox_agent.services.ask_orchestrator import _build_system_prompt
        from omnibox_agent.services.session_store import session_memory_suffix
        from omnibox_agent.models.query import QueryUnderstandingResult, Intent

        perception = ctx.artifacts.get("perception")
        retrieval: RetrievalOutput = ctx.artifacts.get("retrieval", RetrievalOutput())
        query = ctx.input.get("query", "")

        # Default perception if missing
        if perception is None:
            perception = QueryUnderstandingResult(resolved_query=query)

        system_prompt = _build_system_prompt(
            qu_result=perception,
            top_items=retrieval.fused_items,
            content_map=retrieval.content_map,
            total_count=retrieval.total_count,
            platform_dist=retrieval.platform_dist,
        )

        # §5.2：技能指令注入（非命中时 skills 为 None，行为不变）
        skills = ctx.artifacts.get("skills")
        if skills is not None and getattr(skills, "instructions", ""):
            from omnibox_agent.agent.graph_skill import build_skill_instructions
            system_prompt += build_skill_instructions(skills.instructions)

        # 记忆系统：会话摘要注入 system prompt + 近期消息注入 messages（§4.3）。
        # 未启用/失败时 session_context 为 None，行为与现状逐字节等价。
        session_context = ctx.input.get("session_context")
        if session_context:
            system_prompt += session_memory_suffix(session_context)

        # 长期记忆：L1 画像 + L2/L3 召回注入（§12.2；未启用时 long_term 为空，行为不变）
        long_term = ctx.input.get("long_term")
        if long_term:
            from omnibox_agent.services.memory_manager import (
                user_profile_suffix, recalled_memories_suffix)
            system_prompt += user_profile_suffix(long_term)
            system_prompt += recalled_memories_suffix(long_term)

        # R10：会话指代查询（如「第三点有哪些推荐的」）——用户问的是会话里编号/
        # 指代的内容（「第三点 = AI/职场」）。消解结果在 _conv_resolved_query，
        # 检索条目即该主题内容。必须在 system prompt 显式告知 LLM 这条对应关系，
        # 否则 LLM 只看到"检索到 AI 条目"却不理解它们就是用户问的"第三点"，
        # 会答成"没找到第三点"。
        conv_resolved = (ctx.input.get("_conv_resolved_query") or "").strip()
        if ctx.input.get("_conv_referential") and conv_resolved:
            system_prompt += (
                f"\n\n【重要】用户的问题「{query}」是在问当前会话里之前提到的"
                f"内容（编号/指代）。该内容已被解析为「{conv_resolved}」，"
                "下方「召回到的相关条目」就是这一主题下用户收藏的真实内容。"
                "请直接基于这些条目回答/推荐，不要再说「找不到第三点/未指明」之类的话。\n"
            )

        messages = [{"role": "system", "content": system_prompt}]
        if session_context:
            messages.extend(session_context.get("recent") or [])
        # R10：会话指代查询走 simple 检索路径时，即使 session_context 缺失
        # （fresh session 建树失败等），也要把会话历史喂给 LLM——否则 LLM 无法
        # 理解「第三点 = AI/职场」的编号对应关系，会答成"问题太简短没有指明"。
        if not session_context and ctx.input.get("_conv_referential"):
            _hist = [
                m for m in (ctx.input.get("history") or [])
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                and (m.get("content") or "").strip()
            ]
            if _hist:
                messages.extend(_hist[-6:])  # 最近 6 轮足够理解编号对应
        messages.append({"role": "user", "content": query})

        ctx.artifacts["reasoning"] = ReasoningOutput(
            system_prompt=system_prompt,
            messages=messages,
            intent=perception.intent.value if perception else "",
        )


# ---- Fallback helpers ----


def _fallback_answer(
    qu_result: Any,
    top_items: list[dict],
    content_map: dict,
    total_count: int,
) -> str:
    """Rule-based fallback answer when LLM is unavailable."""
    from omnibox_agent.models.query import Intent

    intent = qu_result.intent if qu_result else Intent.SEARCH_AND_SUMMARIZE

    if intent == Intent.COUNT:
        return f"你的收藏库中共有 {total_count} 条相关内容。"

    if intent == Intent.EXIST_CHECK:
        if total_count > 0:
            items_info = []
            for item in top_items[:3]:
                detail = content_map.get(item.get("content_id"), {})
                cid = item.get("content_id")
                title = detail.get("title", "无标题")
                items_info.append(f"[{title}](content://{cid})")
            return f"是的，你的收藏中有相关内容，比如：{'、'.join(items_info)}。"
        return "未在你的收藏中找到相关内容。"

    if not top_items:
        return "未在你的收藏中找到相关内容，建议换个问法或添加更多收藏。"

    items_info = []
    for item in top_items[:5]:
        detail = content_map.get(item.get("content_id"), {})
        cid = item.get("content_id")
        title = detail.get("title", "无标题")
        items_info.append(f"- [{title}](content://{cid})")

    title_str = "\n".join(items_info)
    return f"根据你的收藏库，找到了 {total_count} 条相关内容，其中最相关的是：\n{title_str}"


# ---- Keyword extraction (for parse fallback) ----


def _extract_keywords(query: str) -> list[str]:
    """Extract keywords from query for search."""
    stop_words = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不",
        "这", "那", "吗", "呢", "啊", "吧", "请", "帮", "让",
        "可以", "能", "会", "要", "想", "有没有", "告诉我",
        "收藏", "什么", "怎么", "哪些", "那个", "这个",
        "最近", "最新", "关于",
    }
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
    """Clean query for embedding: remove function words."""
    patterns = [
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
    for p in patterns:
        cleaned = re.sub(p, "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else query


# ---- AskAgent factory ----


def create_ask_agent(config: Config, mcp_registry=None) -> Agent:
    """Create an AskAgent registration entry.

    QA 编排由 LangGraph 子图 run_qa_graph 接管，Agent 实例仅用于
    harness 注册（健康检查 + 路由存在性判断）。各 Step 类由 graph
    节点按需实例化，此处不再预构造 steps 列表。
    """
    return Agent(
        meta=AgentMeta(
            name="ask",
            description="Ask Agent: natural language Q&A over user's personal collection (v4.1)",
            capabilities=["search", "summarize", "qa", "gate", "refine"],
        ),
        config=config,
    )
