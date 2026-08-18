"""Compaction 服务（MEMORY_SYSTEM_DESIGN.md v1.1 §4.2）。

纯逻辑：estimate_tokens / should_compact / find_cut_point / find_last_compaction
LLM 摘要：llm_generate_summary —— **必须使用用户自己的 API Key**（全项目政策红线，
与 query_understanding 一致，见设计 §4.8）；无 key 时调用方直接跳过压缩。
编排：run_compaction —— done 事件后异步执行，best-effort，异常只记日志。
上树：_apply_compaction —— 事务内 INSERT compaction 节点 + parent_id 重定向
（内容只增不改，结构指针可变更，§3.2 / §4.2）。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import text

from omnibox_agent.core.config import get_config
from omnibox_agent.core.database import get_session as db_session
from omnibox_agent.agent.loop import run_blocking
from omnibox_agent.services import session_store

log = logging.getLogger(__name__)

# 摘要输入保护：压缩区总字符上限（超出的旧历史直接丢弃，可被摘要替代）
_MAX_SUMMARY_INPUT_CHARS = 20000

# 收藏助手六段式摘要指令（设计 §7：重点追踪引用条目）
_SUMMARY_SYSTEM_PROMPT = (
    "你是 OmniHub Ask 的会话记忆压缩器。把给定的对话历史压缩为结构化摘要，"
    "供后续轮次恢复上下文。\n"
    "输出六段式摘要：\n"
    "1. 【用户目标】用户正在做的事 / 总目标\n"
    "2. 【已确认偏好】用户明确表达过的偏好（平台、主题、口味、筛选习惯等）\n"
    "3. 【筛选条件】已确认的时间范围、平台、标签、数量等约束（若已被推翻需注明）\n"
    "4. 【引用条目】对话中提到过的收藏条目，用 content_id 标注\n"
    "5. 【未解决问题】尚未回答 / 等待用户补充的问题\n"
    "6. 【其他重要背景】其他影响后续回答的事实\n"
    "要求：只总结对话中实际出现的事实，禁止推断或编造；引用条目必须保留 content_id；"
    "某段无内容写“无”；总字数不超过 {budget} 字。"
)

# 长期记忆候选追加指令（仅 lt_enabled=True 时拼接；未启用用户 prompt 逐字节不变，§11.2）
_LT_CANDIDATES_MARKER_OPEN = "<long_term_candidates>"
_LT_CANDIDATES_MARKER_CLOSE = "</long_term_candidates>"
_LT_SUMMARY_APPENDIX = (
    "\n在六段式摘要正文之后，另起一行追加一个长期记忆候选块：\n"
    "<long_term_candidates>\n"
    '[{"mem_type": "preference|fact|episodic", "content": "一句话跨会话记忆", '
    '"importance": 0.5}, ...]\n'
    "</long_term_candidates>\n"
    "候选要求：只提取用户明确表达过的稳定偏好、个人事实、重要事件（如\"整理收藏时按主题分组\""
    "\"在备考考研\"）；身份证/手机号/住址/健康状况/财务等敏感信息禁止输出；"
    "没有值得记录的内容时输出空数组 []。"
)


# ---- 纯逻辑 ----

def estimate_tokens(text: str | None) -> int:
    """中英分语言启发式：中文≈0.7 token/字，ASCII≈4 chars/token（设计 §6）。

    后续可替换为模型真实 tokenizer；本函数与存储列 token_est 解耦。
    """
    if not text:
        return 0
    zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - zh
    return int(zh * 0.7 + other / 4) + 1


def should_compact(entries: list[Any], threshold_tokens: int) -> bool:
    """路径 token 总和（自最近 compaction 之后）超过阈值即触发。"""
    total = sum(getattr(e, "token_est", 0) or 0 for e in entries)
    return total > threshold_tokens


def find_cut_point(entries: list[Any], keep_recent_tokens: int) -> tuple[int, list[Any]]:
    """从最新往回找切割点：返回 (cut_index, kept_entries)。

    - entries[:cut_index] 被压缩；entries[cut_index:] 保留。
    - 保留区以 user 消息开头（一问一答完整，§4.2 切割点策略）。
    - 压缩区至少包含一条 message（避免只重压 compaction 摘要节点）。
    - cut_index == 0 表示无可压缩内容，调用方应直接 return。
    """
    if not entries:
        return 0, []
    total = 0
    cut = 0
    for i in range(len(entries) - 1, -1, -1):
        total += getattr(entries[i], "token_est", 0) or 0
        if total > keep_recent_tokens:
            cut = i + 1
            break
    # 保留区必须以 user 消息开头：若 cut 落在 assistant 上，**向前**推进到下一个
    # user 边界（压缩稍多、保留区更小，且仍 ≤ keep_recent 预算）。
    # ⚠️ 方向必须向前：向后挪会让保留区超过预算，甚至归零导致永不压缩（E2E 实证）。
    j = cut
    while 0 < j < len(entries) and (entries[j].entry_type != "message" or entries[j].role != "user"):
        j += 1
    if j == len(entries):  # 保留区以 user 开头失败（cut==len 或路径无 user）→ 从末尾前移到最近 user
        j -= 1
        while j >= 0 and (entries[j].entry_type != "message" or entries[j].role != "user"):
            j -= 1
    cut = j
    if cut <= 0:
        return 0, []
    if not any(e.entry_type == "message" for e in entries[:cut]):
        return 0, []
    return cut, entries[cut:]


def find_last_compaction(entries: list[Any]) -> str | None:
    """路径上最近一次 compaction 的摘要内容（作为增量 previousSummary，§4.2）。"""
    for e in reversed(entries):
        if e.entry_type == "compaction":
            return e.content
    return None


def split_lt_candidates(raw: str) -> tuple[str, list[dict]]:
    """从 LLM 输出剥离长期记忆候选 JSON 块（§11.2 实现约束）。

    LLM 输出 = 摘要正文 + <long_term_candidates>[...]</long_term_candidates>。
    返回 (净化后摘要, 候选列表)。⚠️ 若不剥离，候选 JSON 会污染 compaction 节点
    content（经 <session_memory> 注入后续轮次 + 作为 previous_summary 回喂下一次
    压缩，逐轮累积）。

    宽容解析（红线）：标记不存在 / JSON 损坏 / 字段异常 → 候选丢弃、
    返回原样全文作摘要（退化为无 LT 提取，绝不丢摘要主输出）。
    """
    if not raw:
        return raw or "", []
    i = raw.find(_LT_CANDIDATES_MARKER_OPEN)
    if i < 0:
        return raw, []
    j = raw.find(_LT_CANDIDATES_MARKER_CLOSE, i)
    if j < 0:
        # 有开标记无闭标记 → 视为 LLM 输出损坏，整体按普通摘要落库
        return raw, []
    blob = raw[i + len(_LT_CANDIDATES_MARKER_OPEN):j].strip()
    candidates: list[dict] = []
    try:
        parsed = json.loads(blob)
        if isinstance(parsed, list):
            for item in parsed:
                if (isinstance(item, dict) and isinstance(item.get("content"), str)
                        and item["content"].strip()
                        and item.get("mem_type") in ("preference", "fact", "episodic")):
                    candidates.append({
                        "mem_type": item["mem_type"],
                        "content": item["content"].strip()[:500],
                        "importance": max(0.0, min(1.0, float(item.get("importance", 0.5)))),
                    })
    except (ValueError, TypeError):
        return raw, []  # JSON 损坏：整个输出按普通摘要落库（不丢摘要）
    summary = (raw[:i] + raw[j + len(_LT_CANDIDATES_MARKER_CLOSE):]).strip()
    return summary, candidates


# ---- 摘要生成（用户自己的 API Key） ----

async def llm_generate_summary(
    to_compact: list[Any],
    previous_summary: str | None,
    ai_config: dict | None,
    max_summary_tokens: int = 1200,
    lt_enabled: bool = False,
) -> str:
    """用用户的 API Key 生成六段式结构化摘要。无 key → RuntimeError（调用方跳过压缩）。

    lt_enabled=True 时（仅长期记忆启用用户，按用户门控传入，§11.2）：摘要 prompt
    末尾追加"长期记忆候选"指令——同一次 LLM 调用两件事；输出含候选 JSON 块，
    由调用方（run_compaction）split_lt_candidates 剥离后再上树。
    默认 False = 现状：prompt 与摘要输出逐字节不变（红线）。
    """
    from omnibox_agent.services.llm_langchain import _call_llm

    if not ai_config or not ai_config.get("apiKey"):
        raise RuntimeError("compaction requires user api key")

    # 输入预算保护：压缩区超长时从旧到新截断（旧历史可被摘要替代）
    parts: list[str] = []
    used = 0
    for e in to_compact:
        t = (e.content or "").strip()
        if not t:
            continue
        if parts and used + len(t) > _MAX_SUMMARY_INPUT_CHARS:
            break
        parts.append(f"[{e.role or e.entry_type}] {t}")
        used += len(t)
    history_text = "\n".join(parts) if parts else "（无正文内容）"

    prev = ""
    if previous_summary:
        prev = f"\n\n<previous_summary>\n{previous_summary}\n</previous_summary>"

    system_prompt = _SUMMARY_SYSTEM_PROMPT.format(budget=max_summary_tokens)
    if lt_enabled:
        system_prompt += _LT_SUMMARY_APPENDIX

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请压缩以下对话历史：\n{history_text}{prev}"},
    ]
    content = await _call_llm(
        messages,
        ai_config=ai_config,
        temperature=0.3,
        max_tokens=max_summary_tokens,
        no_thinking=True,
    )
    return content or ""


# ---- 上树（同步事务：INSERT compaction + parent_id 重定向） ----

def _apply_compaction(
    session_id: str,
    user_id: str,
    entries: list[Any],
    cut_point: int,
    kept: list[Any],
    summary: str,
) -> bool:
    """事务执行设计 §4.2 步骤 3-5：

    1. INSERT compaction 节点 C（parent = 被压缩区起点的 parent，通常为 None → 主链新根）
    2. 首条保留节点 parent → C（主链变为 ...→ C → 保留区 → leaf）
    3. 被压缩区整体挂到 C 下作旁支（保留可达性 / 审计）
    kept 为空（极端）时：leaf 直接指向 C。
    """
    entry_id = f"e_{uuid.uuid4().hex}"
    first = entries[0]
    now = datetime.now(session_store.CST)
    meta = {
        "compacted_from": first.entry_id,
        "compacted_to": entries[cut_point - 1].entry_id if cut_point > 0 else None,
    }
    s = db_session()
    try:
        s.execute(
            text(
                "INSERT INTO agent_session_entry "
                "(entry_id, session_id, user_id, parent_id, entry_type, role, status, "
                " content, meta, request_id, token_est, created_at) "
                "VALUES (:eid, :sid, :uid, :pid, 'compaction', NULL, 'complete', "
                "        :content, :meta, NULL, :tokens, :now)"
            ),
            {
                "eid": entry_id,
                "sid": session_id,
                "uid": user_id,
                "pid": first.parent_id,
                "content": summary,
                "meta": json.dumps(meta, ensure_ascii=False),
                "tokens": estimate_tokens(summary),
                "now": now,
            },
        )
        if kept:
            # 主链重定向：首条保留节点挂到 C 下
            s.execute(
                text(
                    "UPDATE agent_session_entry SET parent_id = :pid "
                    "WHERE entry_id = :eid AND session_id = :sid AND user_id = :uid"
                ),
                {"pid": entry_id, "eid": entries[cut_point].entry_id,
                 "sid": session_id, "uid": user_id},
            )
            # 被压缩区整体挂到 C 下作旁支（审计 / 回溯）
            s.execute(
                text(
                    "UPDATE agent_session_entry SET parent_id = :pid "
                    "WHERE entry_id = :eid AND session_id = :sid AND user_id = :uid"
                ),
                {"pid": entry_id, "eid": first.entry_id,
                 "sid": session_id, "uid": user_id},
            )
        else:
            # 极端：整段都被压缩 → leaf 直接指向 C
            s.execute(
                text(
                    "UPDATE agent_session SET leaf_entry_id = :eid, last_active_at = :now "
                    "WHERE session_id = :sid AND user_id = :uid AND leaf_entry_id = :exp"
                ),
                {"eid": entry_id, "now": now, "sid": session_id, "uid": user_id,
                 "exp": entries[-1].entry_id},
            )
        s.commit()
        log.info("compaction applied: session=%s cut=%d kept=%d summary_tokens=%d",
                 session_id, cut_point, len(kept), estimate_tokens(summary))
        return True
    except Exception as e:
        s.rollback()
        log.warning("_apply_compaction failed (best-effort): %s", e)
        return False
    finally:
        s.close()


# ---- 编排（done 事件后异步执行） ----

async def run_compaction(
    session_id: str,
    user_id: str,
    ai_config: dict | None = None,
    model_name: str | None = None,
    lt_enabled: bool = False,
) -> list[dict]:
    """done 事件处理时**同步**执行压缩（与请求同协程，确保数据一致，§4.2）。

    设计（v1.1 生产实证修正）：由 done 持久化钩子在发 done 帧**之前**
    `await` 调用——压缩完成后树才进入一致状态，后续请求看到的必然是压缩后视图。
    best-effort：异常只记日志；触发压缩的回合流结束会延迟一次摘要 LLM 调用
    （仅超阈值时；无用户 key 直接跳过）。

    model_name：用户模型名（ai_config.modelName），用于按模型窗口计算触发阈值；
    缺省时按 default_window_tokens 计算。

    lt_enabled（§11.2）：仅长期记忆启用用户传 True（persist_event 按用户门控）；
    默认 False = 现状（should_compact / find_cut_point 等纯逻辑零改动）。
    返回长期记忆候选列表（lt_enabled=False 或未触发压缩时为 []），由
    MemoryManager 写入 L2/L3（存储层边界：本模块不持有 long_term_store 引用）。
    """
    cfg = get_config().memory
    if not cfg.enabled:
        return []
    if not ai_config or not ai_config.get("apiKey"):
        log.warning("compaction skipped: no user api key (session=%s)", session_id)
        return []
    try:
        async with session_store.per_session_lock(session_id):
            # 1. 读路径（同步 DB，线程池执行；归属校验在 get_path_entries_sync 内）
            entries = await run_blocking(
                session_store.get_path_entries_sync, session_id, user_id
            )
            if not entries:
                return []
            threshold = cfg.should_compact_at(model_name)
            if not should_compact(entries, threshold):
                return []
            cut_point, kept = find_cut_point(entries, cfg.keep_recent_tokens)
            if cut_point <= 0:
                return []
            to_compact = entries[:cut_point]
            previous_summary = find_last_compaction(entries)

            # 2. LLM 生成摘要（异步；用户 key，政策红线）；lt 顺带候选（§11.2）
            raw = await llm_generate_summary(
                to_compact, previous_summary, ai_config,
                max_summary_tokens=cfg.max_summary_tokens,
                lt_enabled=lt_enabled,
            )
            if not raw or not raw.strip():
                return []
            # ⚠️ 候选必须剥离后再落库：否则 JSON 块污染 compaction content，
            # 经 <session_memory> 注入后续轮次并逐轮累积（§11.2 实现约束）
            summary, candidates = split_lt_candidates(raw) if lt_enabled else (raw, [])
            if not summary:
                # 摘要被剥空（LLM 只输出了候选块的极端）→ 退化为原始输出落库
                summary = raw
                candidates = []

            # 3. 上树（同步事务，线程池执行；锁内保证与 append 互斥，§4.7）
            await run_blocking(
                _apply_compaction, session_id, user_id, entries, cut_point, kept, summary
            )
            return candidates
    except Exception as e:
        log.warning("run_compaction failed (best-effort, session=%s): %s", session_id, e)
        return []


__all__ = [
    "estimate_tokens",
    "should_compact",
    "find_cut_point",
    "find_last_compaction",
    "split_lt_candidates",
    "llm_generate_summary",
    "run_compaction",
]
