"""v4.1 §9.1-9.2: Creative Planner — task type routing + plan generation.

Three-layer protection (§9.1):
  1. CREATIVE_MODE = off → all queries go to QA path
  2. Plan output invalid (LLM failure/empty) → fallback to QA
  3. Single section + no variants → fallback to QA (it's just a QA query)

SubTask structure (§9.2): Planner generates query/filters/constraints/requires/produces.
produces uses structured [PRODUCES] block contract for reliable extraction.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from omnibox_agent.core.config import get_config
from omnibox_agent.models.note import SubTask, PlanOutput
from omnibox_agent.services.llm_service import generate

log = logging.getLogger(__name__)


# ── Task Type Router (§9.1) ─────────────────────────────────────────────

def route_task(query: str, plan_output: PlanOutput | None, ctx: Any = None) -> str:
    """Two-layer protection: returns 'qa' | 'creative'.

    Layer 3 (single-section fallback) removed: if the complexity classifier
    already decided "complex", we trust that decision and do NOT re-classify
    as "simple" inside the DAG — that would waste the entire PLAN phase (~17s).

      1. CREATIVE_MODE == "off" → "qa"
      2. Plan invalid/None → "qa"
      3. Otherwise → "creative" (trust the complexity classifier)

    Also checks whitelist if mode == "whitelist".
    """
    cfg = get_config()
    mode = cfg.creative.mode

    # Layer 1: Feature toggle
    if mode == "off":
        return "qa"

    # Layer 2: Plan validity
    if plan_output is None or not plan_output.valid or not plan_output.tasks:
        return "qa"

    # Whitelist check
    if mode == "whitelist" and ctx:
        user_id = ctx.input.get("user_id", "")
        whitelist_str = _get_creative_whitelist()
        if whitelist_str and str(user_id) not in whitelist_str:
            return "qa"

    return "creative"


def _get_creative_whitelist() -> str:
    """Get whitelist user IDs from env."""
    import os
    return os.getenv("CREATIVE_WHITELIST", "")


# ── Planner (§9.2) ──────────────────────────────────────────────────────

async def plan(query: str, ctx: Any = None) -> PlanOutput:
    """LLM-based task decomposition for creative queries.

    §9.2: Planner reads the user query and outputs SubTasks with:
      - id: unique section identifier
      - type: "section" or "retrieval_variant"
      - query: retrieval query for this sub-task
      - filters: optional structured filters
      - constraints: optional content constraints
      - requires: shared_state keys this task depends on
      - produces: shared_state keys this task writes

    Output contract: JSON array of task objects.
    Parse failures → PlanOutput(valid=False) → route_task returns "qa".

    Returns:
        PlanOutput with tasks list and validity flag.
    """
    messages = [
        {"role": "system", "content": _build_planner_prompt()},
        {"role": "user", "content": f"用户查询: {query}\n\n请拆分为子任务。"
         "注意:每个子任务的query必须包含用户查询的核心关键词,"
         "不要替换为更具体的子类型;若会话历史显示该查询省略了主题(只给了维度词、"
         "未说主题名词),请先把历史主题补全进query再拆分。"},
    ]

    # §5.3：技能指令注入（规划阶段）。防御性读取，skills 为 None 时不注入。
    try:
        if ctx is not None:
            skills = ctx.artifacts.get("skills")
            if skills is not None and getattr(skills, "instructions", ""):
                from omnibox_agent.agent.graph_skill import build_skill_instructions
                messages[0]["content"] += build_skill_instructions(
                    skills.instructions, "【技能指令-规划阶段】")
    except Exception as e:
        log.debug("Planner skill injection skipped: %s", e)

    # §4.3 记忆系统：会话摘要 + 近期对话注入（DAG 规划阶段；未启用时 session_context 为 None，无影响）。
    # 仅注入 summary 不够：会话未触发 compaction 时 summary 为空，拆解会丢失跨轮主题
    # （当前查询只给维度词、省略了历史主题时尤其明显），故补注入 recent 逐条历史。
    try:
        if ctx is not None:
            sctx = ctx.input.get("session_context")
            if sctx:
                from omnibox_agent.services.session_store import (
                    session_memory_suffix, session_history_suffix)
                messages[0]["content"] += session_memory_suffix(sctx)
                messages[0]["content"] += session_history_suffix(
                    sctx, exclude_query=query)
    except Exception as e:
        log.debug("Planner memory injection skipped: %s", e)

    # §12.2 长期记忆：L1 画像 + L2/L3 召回注入（规划阶段；未启用时无影响）
    try:
        if ctx is not None:
            lt = ctx.input.get("long_term")
            if lt:
                from omnibox_agent.services.memory_manager import (
                    user_profile_suffix, recalled_memories_suffix)
                messages[0]["content"] += user_profile_suffix(lt)
                messages[0]["content"] += recalled_memories_suffix(lt)
    except Exception as e:
        log.debug("Planner LT injection skipped: %s", e)

    try:
        if ctx:
            ctx.llm_call_count += 1
        # Use the user's own API key (NOT the evaluator/Zhipu config) for
        # planning — only embedding should use Zhipu. Missing user key raises
        # in _call_llm (per the user directive, never the system key).
        ai_config = ctx.input.get("ai_config") if ctx else None
        raw = await generate(
            messages, ai_config=ai_config,
            temperature=0.3, max_tokens=2048, timeout=None,
            # Planner 是纯结构化 JSON 拆解任务：关闭 thinking（no_thinking=True
            # → disable_thinking 注入 thinking={"type":"disabled"}）。推理模型
            # （如 deepseek-v4-flash）在 thinking 模式下 content 可能为空
            # （输出全进 reasoning_content 或被 max_tokens 占满），导致 plan
            # 解析失败整体降级 QA。结构化输出不需要思考，直接关掉最稳。
            no_thinking=True,
        )
        return _parse_plan(raw, query)
    except Exception as e:
        log.warning("Planner failed: %r — will fallback to QA", e)
        from omnibox_agent.core.trace_recorder import trace_event
        trace_event("creative.plan.llm_failed", phase="creative", level="warn",
                    data={"reason": repr(e)[:500]})
        return PlanOutput(valid=False, error=repr(e))


def _build_planner_prompt() -> str:
    """Build the system prompt for the planner LLM."""
    return (
        "你是一个任务规划器。将用户的复杂查询拆分为多个子任务。\n\n"
        "重要背景:用户的收藏来自小红书、抖音等平台,是个人笔记/帖子,"
        "不是专业数据库。检索query必须足够宽泛以匹配这些非结构化内容。\n\n"
        "子任务类型:\n"
        '- "section": 独立内容章节,各自检索+生成,最后合成\n'
        '- "retrieval_variant": 检索变体,只扩大召回不生成章节,结果进背景池\n\n'
        "每个子任务必须包含:\n"
        '- id: 唯一标识(英文)\n'
        '- type: "section" 或 "retrieval_variant"\n'
        '- query: 该子任务的检索查询词\n'
        '- filters: 结构化过滤(可选)\n'
        '- constraints: 内容约束(可选)\n'
        '- requires: 依赖的其他子任务produces键(数组)\n'
        '- produces: 本任务产出的共享状态键(数组)\n\n'
        "规则:\n"
        "0. 【主题补全·最高优先】先结合会话历史（system 中的 <session_history>）判断"
        "当前查询是否省略了主题：只有当历史里正围绕某个具体主题展开、且当前查询只给出"
        "维度词而未出现任何新的主题名词时，才把历史中的该主题补全进子任务 query"
        "（写成「历史主题 + 当前维度词」），严禁照抄省略了主题的原始查询。"
        "若当前查询已明确说出自己的主题名词，则忽略历史主题，不要强加。\n"
        "1. section间默认独立,只在真正需要上游数据时才用requires/produces建立依赖。"
        "内容生成类章节不需要互相依赖\n"
        "2. retrieval_variant不应有requires(它不生成内容)\n"
        "3. 子任务数2-5个,section至少2个才值得走创作型\n"
        "4. 【最重要·集合论】用户查询是一个集合(全集),每个子任务都是该集合内的"
        "一个检索窗口。\n"
        "   query必须完整保留用户原始查询的核心主题词(停留在全集内),只能在其后追加"
        "维度修饰词,绝不能把核心主题替换成它的子集。\n"
        "   子集=全集的具体细分。检索子集只能召回子集内的内容,必然漏掉全集中"
        "其他数据——这是逻辑错误,不是检索质量问题。\n"
        "   维度=观察全集的角度,维度横跨全集,检索维度才能覆盖用户收藏。\n"
        "   判断标准: 子任务query必须包含原始查询的核心名词,否则该子任务无效。\n"
        "   自检方法: 把子任务query与原始查询对比——若query缺失了原始查询的核心名词,"
        "并引入了更具体的细分概念,即为错误的子集化,必须修正。\n"
        "5. 至少有一个section的query直接使用用户原始查询的核心关键词"
        "(去掉「做一个」「帮我」等动词后的部分)\n"
        "6. 按观察全集的通用维度拆分(如按价格区间/使用场景/目标人群/区域/清单整理等"
        "横跨全集的视角),不要按子集窄化拆分。每个子任务query必须以原始核心主题词开头。\n\n"
        "只输出JSON数组,不要其他文字。格式:\n"
        '[{"id":"a","type":"section","query":"...","filters":{},"constraints":{},"requires":[],"produces":[]}]'
    )


def _parse_plan(raw: str, original_query: str) -> PlanOutput:
    """Parse LLM output into PlanOutput.

    Handles JSON wrapped in code blocks or plain text.
    解析结果（成功：任务拆解列表；失败：原因）通过日志 + trace 事件上报：
      creative.plan.parsed       — 解析成功（含 task_count + tasks 详情）
      creative.plan.parse_failed — 解析失败（含 reason + 原始输出片段）
    """
    from omnibox_agent.core.trace_recorder import trace_event

    if not raw or not raw.strip():
        log.warning("Planner: empty output — plan parse failed, fallback to QA")
        trace_event("creative.plan.parse_failed", phase="creative", level="warn",
                    data={"reason": "empty planner output"})
        return PlanOutput(valid=False, error="empty planner output")

    # Extract JSON from possible code block
    text = raw.strip()
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # Try to find JSON array
    arr_match = re.search(r'\[.*\]', text, re.DOTALL)
    if arr_match:
        text = arr_match.group(0)

    try:
        tasks_data = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("Planner JSON parse failed: %s — raw: %s", e, raw[:500])
        trace_event("creative.plan.parse_failed", phase="creative", level="warn",
                    data={"reason": f"json decode: {e}", "raw": raw[:500]})
        return PlanOutput(valid=False, error=f"json parse: {e}")

    if not isinstance(tasks_data, list) or not tasks_data:
        log.warning("Planner: parsed output is not a non-empty array — raw: %s", raw[:300])
        trace_event("creative.plan.parse_failed", phase="creative", level="warn",
                    data={"reason": "not a non-empty array", "raw": raw[:300]})
        return PlanOutput(valid=False, error="not a non-empty array")

    tasks = []
    for td in tasks_data:
        if not isinstance(td, dict):
            continue
        try:
            task = SubTask(
                id=str(td.get("id", "")),
                type=td.get("type", "section"),
                query=str(td.get("query", "")),
                filters=td.get("filters", {}) or {},
                constraints=td.get("constraints", {}) or {},
                requires=list(td.get("requires", []) or []),
                produces=list(td.get("produces", []) or []),
            )
            if task.id and task.query:
                tasks.append(task)
        except Exception as e:
            log.debug("Skip invalid task: %s", e)

    if not tasks:
        log.warning("Planner: no valid tasks parsed from %d entries — raw: %s",
                    len(tasks_data) if isinstance(tasks_data, list) else -1, raw[:300])
        trace_event("creative.plan.parse_failed", phase="creative", level="warn",
                    data={"reason": "no valid tasks parsed", "raw": raw[:300]})
        return PlanOutput(valid=False, error="no valid tasks parsed")

    # Programmatic guard: ensure at least one task has a broad query
    # that preserves the original query's core keywords.
    tasks = _ensure_broad_search_task(tasks, original_query)

    log.info("Planner: %d tasks — %s", len(tasks),
             [(t.id, t.type) for t in tasks])
    trace_event("creative.plan.parsed", phase="creative", data={
        "task_count": len(tasks),
        "tasks": [{"id": t.id, "type": t.type, "query": (t.query or "")[:100]}
                  for t in tasks],
    })
    return PlanOutput(tasks=tasks, valid=True)


def _ensure_broad_search_task(
    tasks: list[SubTask], original_query: str
) -> list[SubTask]:
    """Ensure EVERY section task's query stays inside the query's superset.

    Set-theory principle: the user's query is the SUPERSET. A sub-type is a
    SUBSET of it. Retrieving a subset can only ever return subset content —
    it can never surface the rest of the superset's data, so a subset query
    is a logic error, not a retrieval-quality problem.

    Therefore EVERY section query must preserve the core phrase (stay in the
    superset) + optional dimension modifiers. If the LLM narrowed any section
    to a subset, rewrite that section's query back to
    "core [+ 保留原 query 中合法的维度词]" — keep id/produces/requires so the
    DAG contract is untouched.

    Previously this returned early as soon as ONE task contained the core
    phrase, leaving other subset-narrowed sections untouched. Now every
    section is checked.
    """
    # v3.5：先提取【用户澄清补充】标记【之前】的原始查询核心，只保留原始查询。
    # 否则 core 会含补充片段，导致子任务干净 query（如"上海旅游攻略"）因不含
    # 补充内容而被误判为 subset 重写成雷同 query，最终所有 section 检索词
    # 变成同一批内容，与各自维度不匹配 → 子任务 LLM 诚实写 empty-apology。
    _clean_query = re.split(r"[【\[［]", original_query, 1)[0].strip() or original_query
    # Strip common Chinese verbs/particles to extract core keywords
    core = re.sub(
        r'^(做一个|做份|做个|做一份|做一份|帮我|请|帮我规划|帮我整理|帮我总结|'
        r'帮我分析|规划|整理|总结|分析|生成|制作|写一份|写个|写一个)',
        '', _clean_query,
    ).strip()
    if not core:
        core = _clean_query

    # Legitimate dimension words (cross-cutting views of the superset —
    # NOT subsets). Kept when rewriting a narrowed query so sections keep
    # some differentiation instead of all becoming identical.
    dimension_words = [
        "平价", "外卖", "特色", "清单", "路线", "避坑", "区域",
        "招牌", "网红", "小众", "经典", "打卡", "探店", "周边",
        "深夜", "宝藏", "便宜", "排队", "合集", "攻略",
    ]

    # 核心名词（v3.5）：is_broad 用"子任务 query 是否包含去虚字后的核心串"判断，
    # 而不是要求包含完整 core 短语（含"做个"等动词前缀）——否则像
    # "上海旅游攻略 美食 餐厅 小吃"这种带维度、但缺动词前缀的合法子任务会被
    # 误判为 subset 而重写成无维度的超集，导致所有 section 检索词雷同。
    # 去虚字（的/了/是…）后，"上海的旅游攻略"→"上海旅游攻略"，子任务包含它
    # 即认为仍在全集内；真 subset（如"北京旅游攻略"）不含则会被重写。
    _VIRTUAL_CHARS = "的了地得是在和与或很也"
    _core_clean = "".join(c for c in core if c not in _VIRTUAL_CHARS)

    # Core words (the significant terms of the stripped original query)
    core_words = [w for w in core.split() if len(w) >= 2]
    if not core_words:
        core_words = [core]

    def is_broad(t: SubTask) -> bool:
        # 子任务包含去虚字后的核心串 → 在用户查询全集内（维度词不影响）。
        if _core_clean and _core_clean in t.query:
            return True
        if core in t.query:
            return True
        return all(w in t.query for w in core_words)

    rewritten = 0
    for t in tasks:
        if t.type != "section":
            continue
        if is_broad(t):
            continue
        # Narrowed to a subset — rewrite into the superset, keeping any
        # legitimate dimension words found in the original query.
        dims = [w for w in dimension_words if w in t.query]
        new_query = core if not dims else f"{core} {' '.join(dims)}"
        log.info(
            "Planner: rewrote task %s query %r → %r (subset-narrowed, "
            "rewriting into superset)",
            t.id, t.query, new_query,
        )
        t.query = new_query
        rewritten += 1

    # If nothing was rewritten and no section is broad at all (shouldn't
    # happen — every section either kept core or was rewritten), append a
    # broad fallback section as a last resort.
    if rewritten == 0 and not any(is_broad(t) for t in tasks if t.type == "section"):
        broad_task = SubTask(
            id="broad_search",
            type="section",
            query=core,
            filters={},
            constraints={},
            requires=[],
            produces=[],
        )
        tasks.append(broad_task)
        log.info(
            "Planner: added broad_search task (query=%r) — "
            "no section preserved the core phrase",
            core,
        )
    return tasks


# ── produces Block Parser (§9.2) ────────────────────────────────────────

def parse_produces_block(raw_output: str, expected_keys: list[str]) -> tuple[str, dict[str, Any]]:
    """Extract [PRODUCES] block from sub-agent output.

    §9.2: Sub-agent prompt asks for structured block at end:
      ...section text...
      [PRODUCES]{"stay_area": "静安寺"}[/PRODUCES]

    Returns:
        Tuple of (clean_section_text, produces_dict).
        Missing keys → not in dict → downstream treats as "dependency permanently missing".
    """
    produces_data: dict[str, Any] = {}
    clean_text = raw_output

    # Match [PRODUCES]...[/PRODUCES]
    pattern = r'\[PRODUCES\]\s*(.*?)\s*\[/PRODUCES\]'
    match = re.search(pattern, raw_output, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            produces_data = json.loads(json_str)
            if not isinstance(produces_data, dict):
                produces_data = {}
        except json.JSONDecodeError:
            log.debug("PRODUCES block JSON parse failed: %s", json_str[:100])
            produces_data = {}

        # Remove the block from the section text
        clean_text = re.sub(pattern, '', raw_output).strip()

    # Only keep expected keys
    if expected_keys:
        produces_data = {k: v for k, v in produces_data.items() if k in expected_keys}

    return clean_text, produces_data


# ── Dynamic Rounds (§9.6) ───────────────────────────────────────────────

def compute_max_rounds(plan_output: PlanOutput) -> int:
    """§9.6: Dynamic round ceiling based on plan complexity.

    Formula:
      base = 1
      +1 if len(plan) >= 4
      +1 if any section type
      +1 if any requires (dependency chain)
      Cap at ABSOLUTE_MAX_ROUNDS (10).
    """
    cfg = get_config()
    suggested = 1
    if len(plan_output) >= 4:
        suggested += 1
    if any(t.type == "section" for t in plan_output):
        suggested += 1
    if any(t.requires for t in plan_output):
        suggested += 1
    return min(suggested, cfg.creative.absolute_max_rounds)
