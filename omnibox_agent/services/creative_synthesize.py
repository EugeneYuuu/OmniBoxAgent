"""v4.1 §9.7-9.8: Creative Synthesize — mixed merge + budget + conflict reconciliation.

§9.7: All-empty short circuit — skip Synthesize, return degraded response.
§9.8: Synthesize — merge variant background + section text, token budget,
       conflict pairs with arbitration instructions, note_id dedup, partial/missing labels.

Confidence rule (§9.7): takes the worst tier across all sub-results.
  - All sufficient → normal
  - Any sparse/degraded/partial → low
  - All empty → empty
"""

from __future__ import annotations

import logging
from typing import Any

from omnibox_agent.core.config import get_config
from omnibox_agent.models.note import SubTask, SubResult, ConflictPair, ReflectResult
from omnibox_agent.services.llm_service import generate

log = logging.getLogger(__name__)


# ── All-empty Short Circuit (§9.7) ──────────────────────────────────────

def should_short_circuit(results: dict[str, SubResult]) -> bool:
    """§9.7: If all sub-results are empty, skip Synthesize entirely.

    Running Synthesize on empty content wastes an LLM call and risks hallucination.
    """
    if not results:
        return True
    return all(r.coverage_status == "empty" for r in results.values())


def build_short_circuit_response(results: dict[str, SubResult]) -> dict[str, Any]:
    """Build the degraded response for all-empty short circuit."""
    missing = [tid for tid, r in results.items() if r.coverage_status == "empty"]
    return {
        "answer": "收藏中暂无与该问题相关的内容。建议补充收藏相关笔记后再试。",
        "confidence": "empty",
        "missing": missing,
        "partial": False,
    }


# ── Confidence Aggregation (§9.7) ───────────────────────────────────────

def aggregate_confidence(
    results: dict[str, SubResult],
    ctx: Any = None,
) -> str:
    """§9.7: Confidence = worst tier across all sub-results.

    - All sufficient → normal
    - Any sparse/degraded/partial → low
    - All empty → empty
    """
    if not results:
        return "empty"

    all_empty = all(r.coverage_status == "empty" for r in results.values())
    if all_empty:
        return "empty"

    for r in results.values():
        if r.coverage_status == "sparse" or r.degraded_reason or r.confidence == "low":
            return "low"

    if ctx and ctx.flags.get("partial"):
        return "low"
    if ctx and ctx.flags.get("gate_degraded"):
        return "low"

    return "normal"


# ── Synthesize (§9.8) ───────────────────────────────────────────────────

async def synthesize(
    plan_tasks: list[SubTask],
    results: dict[str, SubResult],
    variant_pool: list[dict],
    conflicts: list[ConflictPair],
    ctx: Any,
    token_cb: Any = None,
) -> dict[str, Any]:
    """§9.8: Synthesize final answer from all sub-results.

    Steps:
      1. Sort sections by plan order
      2. Format variant pool as background text (dedup by note_id)
      3. Token budget: system 500, background 1/4, sections share rest
      4. Pass conflicts with arbitration instructions to LLM
      5. Mark missing/partial sections

    token_cb（可选）：流式回调，逐 token 调用 `await token_cb(tok)`，
    让调用方实时把最终答案推给客户端（豆包式流式合成）。未传则走非流式。

    Returns:
        Dict with: answer, confidence, sources, missing, partial
    """
    cfg = get_config()
    budget = cfg.generation.synthesize_token_budget

    # 1. Sort sections by plan order
    order = {t.id: i for i, t in enumerate(plan_tasks)}
    section_results = sorted(
        (r for r in results.values() if r.section_text),
        key=lambda r: order.get(r.sub_task_id, 999),
    )

    # 2. Format variant background
    bg_text = _format_background(variant_pool) if variant_pool else ""

    # 3. Token budget allocation
    sections_text, bg_text = _fit_synthesize_input(
        [r.section_text for r in section_results],
        bg_text,
        budget,
    )

    # 4. Missing sections
    missing = [
        tid for tid, r in results.items()
        if r.coverage_status == "empty" or r.degraded_reason
    ]

    # 5. Partial flag
    partial = ctx.flags.get("partial", False)

    # 6. Build Synthesize prompt
    prompt = _build_synthesize_prompt(
        sections_text, bg_text, conflicts, missing, partial, ctx
    )

    # §5.3：技能指令注入（合成阶段）。防御性读取，skills 为 None 时不注入。
    try:
        skills = ctx.artifacts.get("skills")
        if skills is not None and getattr(skills, "instructions", ""):
            from omnibox_agent.agent.graph_skill import build_skill_instructions
            prompt[0]["content"] += build_skill_instructions(
                skills.instructions, "【技能指令-合成阶段】")
    except Exception as e:
        log.debug("Synthesize skill injection skipped: %s", e)

    # 7. LLM call — runs under the user's own api-key (never the system key)
    #    有 token_cb 时改为流式（逐 token 实时下发），无则非流式聚合。
    ctx.llm_call_count += 1
    answer = ""
    try:
        if token_cb is not None:
            from omnibox_agent.services.llm_service import stream_chat
            async for tok in stream_chat(
                prompt, ai_config=ctx.input.get("ai_config"),
                temperature=0.7, max_tokens=20480,
                no_thinking=False,  # synthesize 保留 thinking + 20480
            ):
                answer += tok
                await token_cb(tok)
        else:
            answer = await generate(
                prompt, ai_config=ctx.input.get("ai_config"),
                temperature=0.7, max_tokens=20480, timeout=None,
            )
    except Exception as e:
        log.error("Synthesize LLM failed: %r — using fallback merge", e)
        if not answer:
            answer = _fallback_synthesize(section_results, bg_text, missing, partial)
            if token_cb is not None:
                await token_cb(answer)

    # 8. Collect all sources
    all_sources = set()
    for r in results.values():
        all_sources.update(r.sources)
    if variant_pool:
        for d in variant_pool:
            cid = d.get("content_id")
            if cid:
                all_sources.add(str(cid))

    # 9. Confidence
    confidence = aggregate_confidence(results, ctx)

    return {
        "answer": answer.strip(),
        "confidence": confidence,
        "sources": list(all_sources),
        "missing": missing,
        "partial": partial,
    }


# ── Helpers ─────────────────────────────────────────────────────────────

def _format_background(variant_pool: list[dict]) -> str:
    """Format variant pool results as background context text."""
    if not variant_pool:
        return ""

    parts = []
    for d in variant_pool[:10]:  # Cap at 10 for background
        title = d.get("title", "")
        summary = d.get("summary", "") or d.get("content", "")
        cid = d.get("content_id", "")
        if title or summary:
            parts.append(f"- {title}: {summary[:200]}\n  content_id: {cid}")

    return "\n".join(parts) if parts else ""


def _fit_synthesize_input(
    sections: list[str],
    bg_text: str,
    budget: int,
) -> tuple[list[str], str]:
    """Token budget allocation for Synthesize input.

    §9.8: system 500, background 1/4, sections share rest.
    Sufficient sections prioritized.
    """
    system_reserve = 500
    bg_budget = (budget - system_reserve) // 4
    sections_budget = budget - system_reserve - bg_budget

    # Truncate background
    if bg_text and len(bg_text) > bg_budget:
        bg_text = bg_text[:bg_budget] + "..."

    # Fit sections
    fitted = []
    used = 0
    for s in sections:
        char_count = len(s)
        if used + char_count > sections_budget and fitted:
            # Truncate last fitting section
            remaining = sections_budget - used
            if remaining > 100:
                fitted.append(s[:remaining] + "...")
            break
        fitted.append(s)
        used += char_count

    return fitted, bg_text


def _build_synthesize_prompt(
    sections_text: list[str],
    bg_text: str,
    conflicts: list[ConflictPair],
    missing: list[str],
    partial: bool,
    ctx: Any,
) -> list[dict]:
    """Build the Synthesize LLM prompt.

    §9.8: Conflicts passed with arbitration instructions.
    Missing/partial sections explicitly marked (but hidden from the user-facing
    answer — the model must NOT write retrieval-process/debugging prose).
    """
    # Format sections
    sections_str = ""
    for i, s in enumerate(sections_text):
        sections_str += f"\n--- 素材 {i+1} ---\n{s}\n"

    # Format background
    bg_str = f"\n--- 补充素材 ---\n{bg_text}\n" if bg_text else ""

    # Format conflicts
    conflict_str = ""
    if conflicts:
        parts = []
        for c in conflicts:
            parts.append(
                f"- 素材 {c.sections} 冲突: {c.issue}. 仲裁: {c.arbitrate}"
            )
        conflict_str = f"\n--- 冲突调和指令 ---\n" + "\n".join(parts) + "\n"

    # Missing / partial are passed for internal awareness only — the output
    # must NOT expose them as "检索情况/分析限制/优化建议" debugging prose.
    notes_str = ""
    if missing or partial:
        notes_str = (
            "\n--- 内部提示（仅供你判断语气，不要向用户复述） ---\n"
            "部分维度的素材不完整。请在回答中直接、自然地回应：若有相关内容就讲，"
            "若无则简单说一句\"这部分在你的收藏中暂时没有相关内容\"，不要展开说明"
            "检索过程、数据缺失原因、改进建议等元信息。"
        )

    system_prompt = (
        "你是一名专业的私人收藏顾问，直接面向用户回答。\n"
        "规则:\n"
        "1. 基于提供的素材，用自然、亲切、面向用户的口吻回答用户的问题\n"
        "2. 直接给结论和内容，不要出现\"数据检索情况\"\"分析限制\"\"优化建议\"\""
        "检索策略\"\"关键词\"等过程性、诊断性表述\n"
        "3. 素材不足时，坦率说明收藏中暂无相关内容，一两句话带过，不要长篇解释原因\n"
        "4. 不要编造素材中没有的信息\n"
        "5. **Markdown 格式（重要）**：回答必须使用 Markdown 结构化输出——用标题"
        "（##/###）划分板块、用加粗突出重点、必要时用列表（`-` 或 `1.`）组织条目。"
        "回答是\"给用户看的结论\"，要层次清晰、易于扫读，不是\"给开发者看的报告\"\n"
        "6. **超链接规则（重要）**：当你在回答中提到具体的收藏条目时，必须使用 Markdown 超链接格式：\n"
        "   `[条目标题](content://条目的content_id)`\n"
        "   例如：`[收藏条目标题](content://123)` — 用户点击即可跳转到该条目详情页\n"
        "   注意：content_id 已在素材中标注，请使用真实的 content_id 值\n"
        "   对于非收藏条目的普通链接（如外部网址），仍使用 `[文本](url)` 格式\n"
        "   在列表项中提到收藏条目时同样使用 `[标题](content://id)` 超链接格式。"
    )

    user_prompt = (
        f"以下是可用的素材，请合成一篇直接回答用户问题的回答:\n"
        f"{sections_str}"
        f"{bg_str}"
        f"{conflict_str}"
        f"{notes_str}"
    )

    # §4.3 记忆系统：会话摘要 + 近期对话注入（合成阶段；未启用时 session_context 为 None，无影响）。
    # 补注入 recent：合成阶段需感知跨轮主题，否则会话短、无 summary 时会把多主题素材并列。
    try:
        sctx = ctx.input.get("session_context") if ctx else None
        if sctx:
            from omnibox_agent.services.session_store import (
                session_memory_suffix, session_history_suffix)
            system_prompt += session_memory_suffix(sctx)
            system_prompt += session_history_suffix(
                sctx, exclude_query=(ctx.input.get("query") if ctx else None))
    except Exception:
        pass

    # §12.2 长期记忆：L1 画像 + L2/L3 召回注入（合成阶段；未启用时无影响）。
    # L1 interaction（作答风格）经画像注入，回答长度/分点习惯稳定输出。
    try:
        lt = ctx.input.get("long_term") if ctx else None
        if lt:
            from omnibox_agent.services.memory_manager import (
                user_profile_suffix, recalled_memories_suffix)
            system_prompt += user_profile_suffix(lt)
            system_prompt += recalled_memories_suffix(lt)
    except Exception:
        pass

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _fallback_synthesize(
    section_results: list[SubResult],
    bg_text: str,
    missing: list[str],
    partial: bool,
) -> str:
    """Fallback: simple concatenation when LLM synthesis fails."""
    parts = []
    for r in section_results:
        if r.section_text:
            parts.append(r.section_text)

    if bg_text:
        parts.append(f"\n参考信息:\n{bg_text}")

    if missing:
        parts.append(f"\n注: 以下部分信息暂缺: {', '.join(missing)}")

    if partial:
        parts.append("\n注: 以下为部分完成的结果，部分章节内容暂缺。")

    return "\n\n".join(parts) if parts else "合成失败,无可用内容。"
