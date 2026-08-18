"""Ask 中间澄清（Clarify）判定与上下文快照（docs/clarify-mid-ask-design.md §4）。

职责：
- need_clarification 判定（rule-based 前置过滤 + LLM-as-a-Judge，§4.2）
- 构建 resume 所需的上下文快照（top_items + content_map + plan，§4.1.1 / §5.3.5）

Agent 只产出不含时间字段的 clarify 信号（§4.3.3），时间字段由后端注入；
Agent 不参与超时逻辑（时间权威唯一在后端）。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, date, time
from typing import Any

import threading
import time as _time
from collections import OrderedDict

from omnibox_agent.core.config import get_config
from omnibox_agent.models.query import Intent
from omnibox_agent.services.llm_service import _call_llm, disable_thinking

log = logging.getLogger(__name__)


# ── 澄清会话计数器（Agent 侧权威） ──
#
# 计数语义：clarify_count = 本 Stream（同一 clarify_session_id 问答链路）
# 已触发过的澄清次数。0 = 还没澄清过，3 = 已澄清 3 次 → 命中上限。
#
# 设计动机：
#   旧实现把 clarify_count 的权威放在"后端每次请求透传字段"。若后端按
#   user session 累计而非按 clarify_session，会出现跨 Stream 累计过快、
#   新提问也被挡的问题。Agent 侧改为 per clarify_session_id 内部计数：
#     • 有 clarify_session_id → 同一 Stream 共享计数，达上限才挡
#     • 无 clarify_session_id → 单次新 Stream，计数恒为 0（不写入缓存，
#       避免 dict 膨胀；只有真的抛出 ClarifySignal 时才会写入并分配计数）
#
# LRU + TTL + 硬上限，与 note_lock.py 风格一致。
_MAX_SESSIONS = 10000
_PRUNE_INTERVAL_S = 3600  # 1 小时尝试一次清理
_DEFAULT_TTL_S = 24 * 3600  # 会话计数 24h 过期（足够覆盖长时间中断的澄清）

_counter_lock = threading.Lock()
_session_counts: "OrderedDict[str, tuple[int, float]]" = OrderedDict()
_last_prune = _time.monotonic()


class ClarifySessionCounter:
    """澄清会话计数入口（Agent 侧权威）。两层上限：
    • 总上限（total）：同一 Stream（clarify_session_id 相同的整段问答链路）内最多 N 次
    • 分节点上限（per phase）：同一澄清点（"qa" / "plan" / "reflect" / "synthesize"）内最多 N 次

    phase 命名契约：
      "qa"          = Simple QA Reason 节点
      "plan"        = DAG PLAN 澄清点
      "reflect"     = DAG REFLECT 澄清点
      "synthesize"  = DAG SYNTHESIZE 澄清点

    线程安全；LRU 逐出 + 按 TTL 过期。
    """

    @staticmethod
    def get_state(session_id: str | None) -> dict:
        """读取计数快照。无 id / 不存在 → 返回空态。

        Returns: {"total": int, "phase_counts": dict[str,int], "is_ephemeral": bool}
          is_ephemeral=True 表示这个计数是临时值，不会被写入缓存（session_id 为空时使用）。
        """
        if not session_id:
            return {"total": 0, "phase_counts": {}, "is_ephemeral": True}
        with _counter_lock:
            ClarifySessionCounter._maybe_prune()
            entry = _session_counts.get(session_id)
            now = _time.monotonic()
            if entry is None:
                return {"total": 0, "phase_counts": {}, "is_ephemeral": False}
            total, phase_counts, ts = entry
            if now - ts > _DEFAULT_TTL_S:
                _session_counts.pop(session_id, None)
                return {"total": 0, "phase_counts": {}, "is_ephemeral": False}
            _session_counts.move_to_end(session_id)
            return {
                "total": total,
                "phase_counts": dict(phase_counts),
                "is_ephemeral": False,
            }

    @staticmethod
    def get(session_id: str | None) -> int:
        """兼容旧接口：返回 total_count。"""
        return ClarifySessionCounter.get_state(session_id)["total"]

    @staticmethod
    def incr(session_id: str | None, phase: str = "qa") -> dict:
        """计数+1（总计数器必+1，当前 phase 计数器也+1）。返回新状态快照。

        只有在"澄清事件即将发生"（捕获 ClarifySignal、即将 yield clarify 帧）
        之前才应调用，否则会让"未真正触发的 LLM 判定假阳性"污染计数。
        """
        if not phase:
            phase = "qa"
        if not session_id:
            # 无 session：只读路径，incr 返回"临时+1"但不持久化（下一次 get 仍为 0）。
            # 该场景下整个 Stream 只发生一次 HTTP 请求，所以不会连续澄清，
            # 返回一个递增后的假值方便审计。
            return {"total": 1, "phase_counts": {phase: 1}, "is_ephemeral": True}
        with _counter_lock:
            ClarifySessionCounter._maybe_prune()
            entry = _session_counts.get(session_id)
            now = _time.monotonic()
            if entry is None or (now - entry[2] > _DEFAULT_TTL_S):
                total = 1
                phase_counts: dict[str, int] = {phase: 1}
            else:
                total, phase_counts, _ts = entry
                total += 1
                phase_counts = dict(phase_counts)
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            _session_counts[session_id] = (total, phase_counts, now)
            _session_counts.move_to_end(session_id)
            while len(_session_counts) > _MAX_SESSIONS:
                _session_counts.popitem(last=False)
            return {
                "total": total,
                "phase_counts": dict(phase_counts),
                "is_ephemeral": False,
            }

    @staticmethod
    def try_incr(session_id: str | None, phase: str = "qa",
                 max_total: int = 5, max_phase: int = 3) -> dict | None:
        """v2（R3/TOCTOU）：原子"检查上限 + 占位 +1"。

        在判定需要澄清的**发出前瞬间**调用：若 total 或 phase 已达上限则返回
        None（本次放弃澄清，走正常作答），否则 +1 并返回新状态快照。
        调用方把返回的快照透传给澄清事件，避免"judge 检查计数、pipeline 捕获
        信号后再 incr"的检查/计数分离竞态（并发同 id 请求可能双双通过检查）。
        无 session_id 时返回临时快照（不落库，语义同 incr）。
        """
        if not phase:
            phase = "qa"
        if not session_id:
            return {"total": 1, "phase_counts": {phase: 1}, "is_ephemeral": True}
        with _counter_lock:
            ClarifySessionCounter._maybe_prune()
            entry = _session_counts.get(session_id)
            now = _time.monotonic()
            if entry is None or (now - entry[2] > _DEFAULT_TTL_S):
                total = 0
                phase_counts: dict[str, int] = {}
            else:
                total, phase_counts, _ts = entry
                phase_counts = dict(phase_counts)
            if total >= max_total:
                return None
            if phase_counts.get(phase, 0) >= max_phase:
                return None
            total += 1
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            _session_counts[session_id] = (total, phase_counts, now)
            _session_counts.move_to_end(session_id)
            while len(_session_counts) > _MAX_SESSIONS:
                _session_counts.popitem(last=False)
            return {
                "total": total,
                "phase_counts": phase_counts,
                "is_ephemeral": False,
            }

    @staticmethod
    def reset(session_id: str | None) -> None:
        """v2（D2）：新提问重置该链路的计数。

        无 resume_context 的请求视为新提问——即使后端复用同一个
        clarify_session_id，也不让上一个提问的澄清计数跨问题累计。
        """
        if not session_id:
            return
        with _counter_lock:
            _session_counts.pop(session_id, None)

    @staticmethod
    def clear() -> None:
        """仅用于测试 / 手工重置。"""
        with _counter_lock:
            _session_counts.clear()

    @staticmethod
    def _maybe_prune() -> None:
        """调用方需持锁。定期清理 TTL 过期条目，避免 dict 无限增长。"""
        global _last_prune
        now = _time.monotonic()
        if now - _last_prune < _PRUNE_INTERVAL_S:
            return
        _last_prune = now

        pruned = 0
        to_remove = []
        for sid, entry in _session_counts.items():
            if len(entry) < 3:
                to_remove.append(sid)
                continue
            _total, _phase, ts = entry
            if now - ts > _DEFAULT_TTL_S:
                to_remove.append(sid)
        for sid in to_remove:
            _session_counts.pop(sid, None)
            pruned += 1

        while len(_session_counts) > _MAX_SESSIONS:
            _session_counts.popitem(last=False)
            pruned += 1

        if pruned > 0:
            log.debug("Pruned %d clarify session counters (%d remaining)",
                      pruned, len(_session_counts))


class ClarifyDecision:
    """澄清判定的结果。字段与 docs §4.3.3 Agent 端输出结构对齐（不含时间字段）。"""

    def __init__(
        self,
        need: bool,
        question: str = "",
        options: list[dict] | None = None,
        recommended_key: str | None = None,
        importance: str = "low",
        allow_custom_input: bool = True,
        custom_input_placeholder: str = "",
        reason: str = "",
        fallback_answer: str = "",
        context: dict | None = None,
        option_intents: dict | None = None,
    ):
        self.need = need
        self.question = question
        self.options = options or []
        self.recommended_key = recommended_key
        self.importance = importance
        self.allow_custom_input = allow_custom_input
        self.custom_input_placeholder = custom_input_placeholder
        self.reason = reason
        self.fallback_answer = fallback_answer
        self.context = context or {}
        # v3.1 增量 resume：选项 key → 恢复意图（canonical 见 _RESUME_INTENTS）。
        # 仅 agent 内部使用（随澄清时入 _dag_resume_states 缓存），不进事件帧。
        self.option_intents = option_intents or {}

    def to_event_data(self) -> dict:
        """转成 clarify 帧的 data（§4.3.3，不含时间字段，含 resume 上下文）。"""
        data = {
            "question": self.question,
            "options": self.options,
            "recommendedKey": self.recommended_key,
            "importance": self.importance,
            "allowCustomInput": self.allow_custom_input,
            "customInputPlaceholder": self.custom_input_placeholder,
            "reason": self.reason,
            "fallbackAnswer": self.fallback_answer,
        }
        if self.context:
            data["context"] = _jsonable(self.context)
        return data


class ClarifySignal(Exception):
    """DAG 管线在 Plan/Reflect/Synthesize 澄清点触发：携带澄清决策 + 阶段 + 上下文快照。

    由 stream_creative_pipeline 捕获并转成 clarify NDJSON 事件（结束流等待用户回答）。
    """

    def __init__(self, decision: ClarifyDecision, phase: str, context: dict):
        super().__init__(f"clarify@{phase}")
        self.decision = decision
        self.phase = phase
        self.context = context


# ── v3.1 增量 resume（incremental DAG resume）──
#
# 背景：DAG 澄清 resume 原本全量重跑（plan/solve 全部重来）。reflect 强制澄清的
# 答案大多只需局部动作（"直接用当前版本"只需合成、"以A为准"只需带裁决合成、
# "我来补充要求"只需重做 poor 任务），全量重跑纯属浪费。
#
# 机制：澄清下发时把完整 plan/results（活对象，同进程引用）+ forced_signals +
# option_intents 存入 agent 侧 token 缓存；token 内嵌澄清 context 随后端
# resumeContext 回传，resume 时据此映射为增量 DAG 初始状态（合成 reflect_result
# 复用 solve/synthesize 现有的 replan_actions/conflicts 通道）。
# 快照里的 results 正文被截断（800字）不可还原，故必须 agent 侧另存完整版。

# 选项恢复意图（canonical）：映射用户答案 → 增量 resume 动作
_RESUME_INTENTS = {
    "supplement",        # 补充要求 → 重做 poor 任务（regenerate + feedback）
    "retry_differently", # 换角度重写 → 重做 poor 任务
    "accept_current",    # 接受当前版本 → 直接合成
    "prefer_first",      # 矛盾以第一处为准 → 直接合成（带裁决）
    "prefer_second",     # 矛盾以第二处为准 → 直接合成（带裁决）
    "keep_both",         # 两处保留并列标注 → 直接合成（带裁决）
}

_DAG_RESUME_TTL_S = 24 * 3600   # 与澄清计数同寿命（覆盖长时间中断的澄清-恢复）
_MAX_DAG_RESUME = 10000
_dag_resume_states: "OrderedDict[str, dict]" = OrderedDict()
_dag_resume_lock = threading.Lock()


def save_dag_resume_state(
    *,
    phase: str,
    query: str,
    plan_output: Any = None,
    results: dict | None = None,
    forced_signals: dict | None = None,
    option_intents: dict | None = None,
) -> str:
    """澄清下发时缓存完整 DAG 状态，返回 token（内嵌澄清 context 供 resume 回传）。

    与 ClarifySessionCounter 同构：进程内 OrderedDict + 锁 + TTL/容量清理。
    存活对象引用（PlanOutput/SubResult），无需序列化还原。
    """
    import uuid

    token = uuid.uuid4().hex[:20]
    now = _time.monotonic()
    with _dag_resume_lock:
        # TTL + 容量清理（与计数器同策略）
        stale = [
            k for k, v in _dag_resume_states.items()
            if now - v.get("ts", now) > _DAG_RESUME_TTL_S
        ]
        for k in stale:
            _dag_resume_states.pop(k, None)
        while len(_dag_resume_states) >= _MAX_DAG_RESUME:
            _dag_resume_states.popitem(last=False)
        _dag_resume_states[token] = {
            "phase": phase,
            "query": query,
            "plan_output": plan_output,
            "results": results,
            "forced_signals": forced_signals,
            "option_intents": option_intents or {},
            "ts": now,
        }
        _dag_resume_states.move_to_end(token)
    return token


def get_dag_resume_state(token: str | None) -> dict | None:
    """按 token 取回缓存的 DAG 状态；miss/过期返回 None（调用方回退全量重跑）。"""
    if not token:
        return None
    with _dag_resume_lock:
        entry = _dag_resume_states.get(token)
        if entry is None:
            return None
        if _time.monotonic() - entry.get("ts", 0) > _DAG_RESUME_TTL_S:
            _dag_resume_states.pop(token, None)
            return None
        return entry


def build_incremental_resume(
    entry: dict | None,
    answer_type: str | None,
    answer_key: str | None,
    answer_text: str,
) -> dict | None:
    """把 reflect 强制澄清的用户答案映射为增量 resume 的 DAG 初始状态片段。

    返回 {"skip_to": "solve"|"synthesize", "plan_output", "results", "reflect_result"}；
    不可映射（非 reflect 澄清 / 无强制信号 / 意图未知 / 数据缺失）返回 None，
    调用方回退全量重跑（augmented query 路径）。

    意图 → 动作：
      supplement / retry_differently → 局部 SOLVE（poor 任务 regenerate + feedback）
        再经单次 REFLECT 把关（轮次预置使必转 SYNTHESIZE，不再 replan/二次澄清）
      accept_current / prefer_first / prefer_second / keep_both / 自由输入裁决
        → 直接 SYNTHESIZE（矛盾以 ConflictPair.arbitrate 注入用户裁决）
    """
    if not entry or entry.get("phase") != "reflect":
        return None
    fs = entry.get("forced_signals") or {}
    if not fs:
        return None
    poor = fs.get("poor_tasks") or []
    conflicts = fs.get("conflicts") or []
    answer_text = (answer_text or "").strip()

    # 意图解析：自由输入 → poor 类视作补充要求；纯矛盾类视作用户自定义裁决
    if (answer_type or "custom") == "custom" or not answer_key:
        if poor:
            intent = "supplement"
        elif conflicts:
            intent = "arbitrate_custom"
        else:
            return None
    else:
        intent = (entry.get("option_intents") or {}).get(str(answer_key))
    if intent is None:
        return None

    plan_output = entry.get("plan_output")
    results = entry.get("results")
    if plan_output is None or results is None:
        return None

    from omnibox_agent.models.note import ConflictPair, ReflectResult, SubTaskOverride

    if intent in ("supplement", "retry_differently"):
        if not poor:
            return None
        overrides: dict = {}
        for t in poor:
            tid = str(t.get("id"))
            if intent == "supplement":
                fb = (f"用户澄清补充：{answer_text}" if answer_text
                      else "按用户此前的澄清补充重写本部分。")
            else:
                fb = ("用户要求换一个角度重写本部分。"
                      + (f"补充说明：{answer_text}" if answer_text else ""))
            overrides[tid] = SubTaskOverride(
                query="", feedback=fb, mode="regenerate", strategy=1,
            )
        rr = ReflectResult()
        rr.replan_actions = overrides  # solve_node 现有通道：删旧结果+带 feedback 重做
        return {"skip_to": "solve", "plan_output": plan_output,
                "results": results, "reflect_result": rr}

    # 合成类意图：accept_current / prefer_* / keep_both / arbitrate_custom
    if intent in ("prefer_first", "prefer_second", "keep_both") and not conflicts:
        return None  # 裁决类意图但无矛盾快照 → 语义不成立，回退全量
    pairs = []
    for c in conflicts:
        secs = [str(s) for s in (c.get("sections") or [])]
        if intent == "prefer_first":
            arb = (f"用户裁决：以「{secs[0]}」部分为准，其余口径向它统一。"
                   if secs else "用户裁决：以第一处为准，其余口径向它统一。")
        elif intent == "prefer_second":
            base = secs[1] if len(secs) > 1 else (secs[-1] if secs else "另一处")
            arb = f"用户裁决：以「{base}」部分为准，其余口径向它统一。"
        elif intent == "keep_both":
            arb = "用户裁决：两处口径均保留，并列呈现并以中性表述标注分歧。"
        elif intent == "arbitrate_custom":
            arb = (f"用户裁决：{answer_text}" if answer_text
                   else "用户已给出自定义裁决，请按素材自然调和。")
        else:  # accept_current：接受现状，不显式暴露冲突
            arb = "用户已确认接受当前版本，请按素材自然调和，不要在答案中显式讨论冲突。"
        pairs.append(ConflictPair(
            sections=secs, issue=str(c.get("issue") or ""), arbitrate=arb,
        ))
    rr = ReflectResult()
    rr.conflicts = pairs  # synthesize_node 现有通道：冲突调和指令进合成 prompt
    return {"skip_to": "synthesize", "plan_output": plan_output,
            "results": results, "reflect_result": rr}


def _jsonable(value: Any) -> Any:
    """递归把非 JSON 可序列化对象（datetime/date 等）转成字符串。

    澄清事件会被序列化为 NDJSON 透传给后端；DAG 上下文快照来自 dataclasses，
    常含 datetime 字段（如 collected_at），必须在此处提前清洗，否则
    json.dumps 抛 TypeError 中断整个 clarify 事件。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def build_dag_clarify_context(
    phase: str,
    query: str,
    plan_output: Any = None,
    results: dict | None = None,
    variant_pool: list | None = None,
    replan_rounds_so_far: int = 0,
) -> dict:
    """构建 DAG 澄清的 resume 上下文快照（§5.3.5 complex/DAG 内容）。

    phase: "plan" | "reflect" | "synthesize"
    记录 plan 状态、已完成子任务结果等，供 resume 时从断点继续。

    replan_rounds_so_far（v3）：澄清触发时 DAG 已运行的 reflect 轮数
    （ctx.metrics.creative_rounds）。写入快照供 resume 重跑时累加为强制澄清的
    轮次基准——否则 resume 重跑 round_num 从 0 重置，跨澄清的"已重规划 N 轮"
    累计会丢失，导致 reflect 强制澄清永远不可达。
    """
    import dataclasses

    ctx: dict = {"dag": True, "phase": phase, "query": query}
    if replan_rounds_so_far:
        ctx["creative_rounds_so_far"] = int(replan_rounds_so_far)

    if plan_output is not None and getattr(plan_output, "tasks", None):
        ctx["plan"] = _jsonable([
            dataclasses.asdict(t)
            for t in plan_output.tasks
            if dataclasses.is_dataclass(t)
        ])
    if results:
        # 子结果正文较大，快照仅截断保留，避免 clarify 事件 payload 过大
        #（DAG resume 以澄清后的 augmented query 重跑，快照主要用于审计/续接上下文）
        snapshot = {}
        for tid, r in results.items():
            if not dataclasses.is_dataclass(r):
                continue
            d = dataclasses.asdict(r)
            text = d.get("section_text") or ""
            if len(text) > 800:
                d["section_text"] = text[:800] + "…（截断）"
            snapshot[str(tid)] = _jsonable(d)
        ctx["results"] = snapshot
    if variant_pool:
        ctx["variant_pool"] = _jsonable(variant_pool)
    return ctx


# ── 上下文快照 ──

def build_clarify_context(
    retrieval: Any | None,
    qu_result: Any | None,
    plan: list | None = None,
    sub_results: dict | None = None,
) -> dict:
    """构建 resume 所需的精简上下文快照（§5.3.5）。

    simple QA：top_items + content_map 精简版（正文截断 500 字）
    complex/DAG：额外缓存 plan 状态与已完成子任务结果。

    序列化后内联进后端 draft 的 retrievedContextJson；体量小时无需额外 Store。
    """
    top_items = []
    content_map = {}
    if retrieval is not None:
        for item in getattr(retrieval, "fused_items", []) or []:
            try:
                top_items.append(item)
            except Exception:
                continue
        cm = getattr(retrieval, "content_map", {}) or {}
        for cid, detail in cm.items():
            d = dict(detail or {})
            summary = d.get("summary") or ""
            if len(summary) > 500:
                d["summary"] = summary[:500]
            content_map[str(cid)] = d

    qu = {}
    if qu_result is not None:
        try:
            qu = {
                "intent": getattr(qu_result, "intent", None),
                "resolved_query": getattr(qu_result, "resolved_query", "") or "",
                "explicit_limit": bool(getattr(qu_result, "explicit_limit", False)),
                "want_classify": bool(getattr(qu_result, "want_classify", False)),
                "classify_by": getattr(qu_result, "classify_by", None),
            }
        except Exception:
            qu = {}

    ctx = {
        "top_items": top_items,
        "content_map": content_map,
        "qu": qu,
    }
    if plan is not None:
        ctx["plan"] = plan
    if sub_results is not None:
        ctx["sub_results"] = sub_results
    return ctx


# ── 前置过滤（rule-based，零 LLM 成本，§4.2） ──

# 会话内指代（conversation-referential）查询：答案在对话历史里，不在收藏库检索的
# 歧义维度上。如「上面说的什么」「最近一次回答的内容」「你刚才说的」「总结一下
# 我们刚才聊的」——这类问题指向会话自身的上一轮内容，澄清没有意义，前置过滤
# 直接跳过 judge（§4.2 零 LLM 成本）。R6 新增。
#
# 两级指代：
#   • 强指代（上面/刚才/最近一次/你X/我们的对话…）：窗口内出现内容词或提问词即可；
#   • 弱指代（上次/上回/之前）：必须出现会话动词（说/讲/回答…），避免把
#     「上次去的那家店叫什么」这类指向收藏/行程实体的查询误判为会话指代。
_CONVERSATION_REFERRAL_RE = re.compile(
    r"(?:"
    # 强指代 + 内容词/提问词（窗口 15 字内）
    r"(?:上面|上边|以上|前面|刚才|刚刚|最近一次|上一次|上一条|上一轮|上一段|上一句|"
    r"刚才那条|上面那条|你上面|你刚才|你说的|你说了|你说|你讲的|你讲了|你回答的|"
    r"你回复的|你的回答|你的回复|你总结的|你概括的|你提到的|你提及的|我们刚才|我们之前|我们的对话|"
    r"这次对话|当前会话|本次对话|这个会话)"
    r".{0,15}?"
    r"(?:说|讲|回答|回复|内容|总结|概括|摘要|意思|什么|啥|哪些|哪儿|哪里|哪|"
    r"怎么|聊|讨论|重复|复述|提到|提及|那个|这个)"
    r")"
    r"|(?:"
    # 弱指代（上次/上回/之前）：必须紧跟会话动词
    r"(?:上次|上回|之前)"
    r".{0,15}?"
    r"(?:说|讲|回答|回复|内容|总结|概括|摘要|意思|聊|讨论|重复|复述|提到|提及)"
    r")"
    # 纯会话指代短语（无指代词也成立）
    r"|(?:再说一遍|再说一次|重新说|重新回答|重复一遍|重复一下|复述一遍|复述一下|"
    r"继续说下去|接着说|继续刚才|继续说说)"
)


def _is_conversation_referential(query: str) -> bool:
    """查询是否明显指向「会话自身内容」（上一轮回答/最近一次回答等）。

    这类问题的答案在对话历史里，与收藏库检索无关——澄清没有意义，
    直接跳过 judge（零 LLM 成本，§4.2 前置过滤）。

    v3/R9 起仅作为**零成本短路**：命中直接判 true（不调 LLM）；未命中的
    模糊地带交给 judge_conversation_referential 做语义判定，不再依赖枚举覆盖。
    """
    return bool(_CONVERSATION_REFERRAL_RE.search(query or ""))


async def judge_conversation_referential(
    query: str,
    history: list,
    ai_config: dict | None,
    *,
    rule_hint: bool | None = None,
) -> dict:
    """R9：语义判定「查询是否指向当前会话自身内容」+「是否需要结合检索」。

    替代关键词枚举：用户表达会话指代的方式千变万化（「上面说的什么」
    「你刚才那套方案」「我们聊到哪了」「第二次讲的那部分」…），枚举永远
    有漏网之鱼。这里用 LLM-as-a-Judge 做语义判断。

    返回 {"referential": bool, "need_retrieval": bool, "source": str, "resolved_query": str}：
      referential   = 查询是否指向会话自身内容（上一轮回答/最近对话）
      need_retrieval= 若是会话指代，是否需要结合收藏库检索补充细节
                     （「总结上面的内容」不需要；「上面说的那家店在哪」需要；
                      「第三点有哪些推荐的」需要——用户索要该点的实际内容）
      resolved_query= need_retrieval=True 时，LLM 结合最近对话把编号/指代消解成
                      **可检索的独立查询**（「第三点有哪些推荐的」+ 上轮回答
                      「1.美食 2.旅行 3.AI/职场」→「AI/职场 相关收藏推荐」），
                      供向量检索使用；无需检索时为 ""。
      source        = "rule" | "llm"（判定来源，审计用）

    短路优化：rule_hint=True（正则命中强信号）→ 直接 referential=True，
    不调 LLM（省一次调用）；否则才走 LLM。LLM 失败/无历史 → 回退规则结果，
    保证行为不退化。
    """
    cfg = get_config().clarify
    rule_hit = rule_hint if rule_hint is not None else _is_conversation_referential(query)
    if rule_hit:
        return {"referential": True, "need_retrieval": True, "source": "rule",
                "resolved_query": ""}

    # 无对话历史 → 不可能是会话指代（无「上面」可指）
    if not history:
        return {"referential": False, "need_retrieval": False, "source": "rule",
                "resolved_query": ""}

    recent = _history_brief(history)
    system = (
        "你是 OmniHub Ask 的查询意图判定器。用户基于自己的私人收藏库提问，"
        "系统会展示一段**最近对话**（用户与助手的问答记录）。\n"
        "请判断用户的新查询是否**指向会话自身内容**——即用户在问上一轮回答/"
        "最近对话里说过的东西（含指代词、省略主语、追问上文、编号指代、回指上文"
        "某一部分等），而不是在问收藏库里的新内容。\n"
        "【判定要点】\n"
        "- referential=true：查询明显指代会话历史（含指代词、省略主语、追问上文、"
        "编号指代、回指上文某一部分）\n"
        "- 若 referential=true，再判断 need_retrieval：用户是否想**看到/获得**该内容"
        "的实际条目（如要求列出、推荐、给出具体内容、地址、详情等）？想获得实际内容 → "
        "need_retrieval=true；只是回顾/复述/总结刚才说过的话 → need_retrieval=false。\n"
        "- 若 need_retrieval=true，必须给出 resolvedQuery：结合最近对话把编号/指代"
        "消解成**可独立检索的查询词**。\n"
        "  **编号指代必须消解成主题词**：当用户以编号指代上轮回答中的某一要点时，"
        "应结合上轮列出的要点找到该编号对应的主题名，作为 resolvedQuery；"
        "**禁止保留『第几点』等编号原文**——编号在向量检索里没有语义。\n"
        "  **指代某条具体内容（如上面提的某条/某家店）** → resolvedQuery 用上轮"
        "提到的主题名/名称，不能是指代原文。\n"
        "  need_retrieval=false 时 resolvedQuery 为空字符串。\n"
        "- 拿不准时 referential 偏向 false（宁可按普通收藏库查询处理，也不要误吞）。\n"
        "输出严格 JSON：{{\"referential\": bool, \"need_retrieval\": bool, \"resolvedQuery\": str}}"
    )
    user = f"【最近对话】\n{recent or '（无）'}\n\n【用户新查询】\n{query}"
    try:
        raw = await _call_llm(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            ai_config=ai_config,
            model=(ai_config or {}).get("modelName") or cfg.judge_model,
            temperature=0.1,
            max_tokens=cfg.judge_max_tokens,
            response_format={"type": "json_object"},
            no_thinking=True,
        )
    except Exception as e:
        log.warning("Conversation-referential judge LLM failed, rule fallback: %s", e)
        return {"referential": False, "need_retrieval": False, "source": "rule",
                "resolved_query": ""}

    parsed = _parse_judge(raw)
    if parsed is None:
        return {"referential": False, "need_retrieval": False, "source": "rule",
                "resolved_query": ""}
    ref = bool(parsed.get("referential", False))
    need_retr = bool(parsed.get("need_retrieval", False)) if ref else False
    resolved = str(parsed.get("resolvedQuery") or "").strip() if need_retr else ""
    return {"referential": ref, "need_retrieval": need_retr, "source": "llm",
            "resolved_query": resolved}


def _should_skip_judge(
    query: str,
    qu_result: Any,
    phase: str,
    total_count: int,
    phase_count: int,
    max_total_per_stream: int,
    max_per_phase: int,
) -> str | None:
    """返回跳过原因；None 表示需要继续 LLM 判定。

    两层上限（两者命中任一则跳过）：
      1. total_count >= max_total_per_stream → 整流到顶
      2. phase_count >= max_per_phase       → 当前节点（澄清点）到顶
    phase 仅用于日志记录，实际上限由调用方按 pipeline/Qa 分档计算后传入 max_per_phase。

    注（v2/R1）：already_clarified 历史守卫已废弃——同一请求内 judge 只执行一次
    （澄清即中断图），不存在"同一轮内重复追问"；而"上一轮已澄清、本轮无 resume"
    的请求按 D2 语义是新提问，应允许澄清（由 D2 的 reset + 两层上限约束滥用）。
    防重复追问的职责改由原子占位 try_incr（R3）承担。

    注（v3/R6）：会话内指代查询（「上面说的什么」等）在 query_too_short 之后、
    LLM 判定之前直接跳过——答案在对话历史里，澄清毫无意义。
    """
    if not query or len(query.strip()) < 4:
        return "query_too_short"
    if _is_conversation_referential(query):
        return "conversation_referential"
    if total_count >= max_total_per_stream:
        return f"clarify_total_limit_reached(total={total_count})"
    if phase_count >= max_per_phase:
        return f"clarify_phase_limit_reached(phase={phase}, count={phase_count})"
    if qu_result is not None:
        intent = getattr(qu_result, "intent", None)
        if intent in (Intent.COUNT, Intent.EXIST_CHECK):
            return "pure_factual"
    return None


# ── LLM 判定 ──

async def judge_need_clarification(
    query: str,
    qu_result: Any,
    retrieval: Any | None,
    history: list,
    ai_config: dict | None,
    phase: str = "qa",
    total_count: int = 0,
    phase_count: int = 0,
    max_total_per_stream: int = 5,
    max_per_phase: int = 3,
    enabled: bool = True,
    is_resume: bool = False,
    supplement: str = "",
) -> ClarifyDecision | None:
    """判断是否需要澄清（§4.2）。

    两层上限：
      • max_total_per_stream=5：同一 Stream（clarify_session_id 相同链路）内最多 5 次
      • max_per_phase=3：Simple QA "qa" 节点内最多 3 次；DAG 各节点传 2

    is_resume（v2/D1-D3）：resume 请求标记，用于日志/审计（守卫已废弃，见 _should_skip_judge 注）。
    supplement（v2/D1-D3）：用户对上次澄清的补充答案，喂给判定器，防止
      resume 用同一 query 再 judge 时对同一歧义反复追问。
    history（v2/R5）：最近几轮对话会压缩成简短摘要拼进判定 prompt，
      让 judge 能感知"已澄清过什么 / 用户答了什么"，不只看单条 query。

    返回：
      - None：不需要澄清（或被前置过滤跳过）——走正常作答
      - ClarifyDecision：需要澄清
    """
    cfg = get_config().clarify
    if not enabled:
        return None

    final_total_cap = max_total_per_stream or cfg.effective_max_total()

    skip = _should_skip_judge(
        query, qu_result, phase,
        total_count=total_count,
        phase_count=phase_count,
        max_total_per_stream=final_total_cap,
        max_per_phase=max_per_phase,
    )
    if skip:
        log.info("Clarify judge skipped: %s (query=%r, phase=%s)", skip, query[:30], phase)
        return None

    # 只对"晚·检索后"主路径做 LLM 判定：把检索到的内容摘要喂给判定器，
    # 判断是否答案依赖未指定的维度/参数/范围。
    context_brief = _retrieval_brief(retrieval)

    system = (
        "你是 OmniHub Ask 的澄清判定器。用户基于自己的私人收藏库提问，"
        "系统已检索到若干候选内容。你需要判断：是否因为查询存在歧义、"
        "缺少关键实体、或答案依赖用户未指定的维度/参数/范围，导致无法给出准确答案。\n"
        "若确实需要向用户追问一次，则输出 need=true 并给出澄清问题与选项；否则 need=false。\n"
        "【推荐选项要求】recommendedKey 必须从 options 中选取，不可为空；"
        "推荐要均衡覆盖大多数用户常见意图，不可偏向极端选项；"
        "不确定时选破坏性最小、可逆的选项。\n"
        "importance：low=有合理默认值；medium=影响答案方向但有可接受兜底；high=缺此信息无法回答。\n"
        "【用户是普通人，请用自然语言，选项 2~5 个，互斥且尽量穷尽。】\n"
        "输出严格 JSON："
        '{"need": bool, "confidence": 0-1, "question": str, '
        '"options": [{"key":"A","label":"..","description":".."}], '
        '"recommendedKey": str, "importance": "low|medium|high", '
        '"allowCustomInput": true, "customInputPlaceholder": str, '
        '"reason": str, "fallbackAnswer": str, "needed_field": str}'
    )

    user = (
        f"用户查询：{query}\n\n"
        f"检索到的候选内容：\n{context_brief or '（无检索结果）'}\n\n"
    )
    recent = _history_brief(history)
    if recent:
        user += f"最近对话：\n{recent}\n\n"
    if supplement:
        user += (
            f"用户已补充：{supplement}\n\n"
            "（请基于用户补充重新判断：若原有歧义已解决则 need=false；"
            "仅当仍存在新的、影响答案的歧义时才 need=true）\n\n"
        )
    user += "请判断是否需要向用户澄清一次。"

    try:
        raw = await _call_llm(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            ai_config=ai_config,
            # 复用 complexity_classifier 同规格：优先用用户自己的模型，
            # 避免把 judge_model（默认 glm-4-flash）发给用户的 DeepSeek baseUrl 导致 400。
            model=(ai_config or {}).get("modelName") or cfg.judge_model,
            temperature=0.2,
            max_tokens=cfg.judge_max_tokens,
            response_format={"type": "json_object"},
            no_thinking=True,
        )
    except Exception as e:
        log.warning("Clarify judge LLM failed, fallback to no clarify: %s", e)
        return None

    parsed = _parse_judge(raw)
    if parsed is None:
        return None

    need = bool(parsed.get("need", False))
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    if not need or confidence < cfg.confidence_threshold:
        return None

    options = _normalize_options(parsed.get("options"))
    if not options:
        return None

    recommended = parsed.get("recommendedKey") or options[0]["key"]
    if recommended not in [o["key"] for o in options]:
        recommended = options[0]["key"]

    importance = parsed.get("importance") or "low"
    if importance not in ("low", "medium", "high"):
        importance = "low"

    fallback = parsed.get("fallbackAnswer") or ""
    # medium 场景必须在正文首行追加假设标注（§4.3.6）
    if importance == "medium" and fallback:
        recommended_label = next((o["label"] for o in options if o["key"] == recommended), recommended)
        fallback = f"> ⚠️ 假设：{recommended_label}。如需调整请重新描述。\n\n{fallback}"

    # 若 importance 为 low/medium 但无 fallbackAnswer，按后端校验规则升级为 high
    if importance in ("low", "medium") and not fallback:
        importance = "high"

    return ClarifyDecision(
        need=True,
        question=parsed.get("question") or "",
        options=options,
        recommended_key=recommended,
        importance=importance,
        allow_custom_input=bool(parsed.get("allowCustomInput", True)),
        custom_input_placeholder=parsed.get("customInputPlaceholder") or "",
        reason=parsed.get("reason") or "",
        fallback_answer=fallback,
    )


async def judge_dag_clarification(
    query: str,
    phase: str,
    history: list,
    ai_config: dict | None,
    total_count: int = 0,
    phase_count: int = 0,
    max_total_per_stream: int = 5,
    max_per_phase: int = 2,
    enabled: bool = True,
    plan_summary: str = "",
    result_summary: str = "",
    is_resume: bool = False,
    supplement: str = "",
    forced_signals: dict | None = None,
    veto_allowed: bool = True,
) -> ClarifyDecision | None:
    """DAG 路径澄清判定（§4.2 complex 三澄清点：Plan / Reflect / Synthesize）。

    两层上限：
      • max_total_per_stream=5：同一 Stream 内最多 5 次总澄清
      • max_per_phase=2：Plan / Reflect / Synthesize 各节点分别最多 2 次（Simple QA 用 3）

    is_resume（v2/D1-D3）：resume 重跑标记，用于日志/审计（守卫已废弃，见 _should_skip_judge 注）。
    supplement（v2/D1-D3）：用户对上次澄清的补充答案，喂给判定器，防止对同一歧义反复追问。
    history（v2/R5）：最近几轮对话压缩进判定 prompt，让 judge 感知已澄清内容与用户回答。

    forced_signals（v3/Reflect 强制澄清）：由 graph_creative._detect_forced_clarify 生成的
    结构化信号（reasons/round_num/poor_tasks/conflicts）。非空时进入强制模式：
      • 前置过滤（含两层计数上限）与 enabled 开关仍然生效——cap 满则降级为不澄清；
      • 跳过 confidence 门控，need 视为 true，LLM 只负责把结构化问题转成自然语言措辞；
      • 仅当 supplement 非空且 LLM 判定用户补充已解决问题时允许否决（need=false），
        防止对"直接用当前版本"这类回答重复追问；
      • LLM 调用失败/解析失败/选项为空时，降级为规则模板澄清（_build_forced_template_decision），
        不因措辞失败丢失强制语义。

    复用 same 判定规格（rule-based 前置过滤 + LLM-as-a-Judge），
    针对 DAG 阶段加入阶段上下文提示，帮助判定"查询歧义 / 缺关键约束 / 依赖未指定维度"。
    """
    cfg = get_config().clarify
    if not enabled:
        return None

    final_total_cap = max_total_per_stream or cfg.effective_max_total()

    skip = _should_skip_judge(
        query, None, phase,
        total_count=total_count,
        phase_count=phase_count,
        max_total_per_stream=final_total_cap,
        max_per_phase=max_per_phase,
    )
    if skip:
        log.info("DAG clarify judge skipped: %s (query=%r, phase=%s)", skip, query[:30], phase)
        return None

    forced = forced_signals is not None

    if forced:
        # v3 强制澄清：规则已判定必须澄清（replan 多轮仍 poor/conflicts），
        # LLM 不判断 need（仅 resume 且补充已解决问题时允许否决），只负责措辞
        problem_brief = _build_forced_problem_brief(forced_signals)
        system = (
            "你是 OmniHub Ask 复杂任务的澄清判定器。用户基于自己的私人收藏库提出一个较复杂的问题，"
            "系统正在用 Plan-Solve-Reflect-Synthesize 流程处理。\n"
            "【强制澄清模式】系统已自动重规划多轮，仍存在下方列出的结构化问题，无法自行修复，"
            "已决定向用户澄清。你的任务是把这些问题转成一个清晰、自然的澄清问题与互斥选项，"
            "而不是判断是否需要澄清。\n"
            "need 输出恒为 true；仅当\"用户已补充\"的内容明确解决了所有列出的问题时才允许 need=false。\n"
            "【面向用户表达】问题要说明具体是哪个主题/哪两部分、遇到的困难是什么，"
            "并明确期望用户提供什么信息或做什么选择。禁止使用『质量不理想/达标、子任务、"
            "自动重规划、sparse/poor』等内部术语，一律用自然、通俗的大白话向用户转述，"
            "不要照搬下方技术化的结构化描述。\n"
            "【禁止猜测原因】不要猜测为什么会整理不理想（不要归咎于内容数量、信息充足度等"
            "具体原因），这些猜测可能完全错误。只需说明『这部分我反复整理了几次，结果都"
            "不太理想，想请你帮我定个方向』即可。\n"
            "【禁止复述】绝对不要在问题中重复完整的用户查询原文（下方有参考），"
            "只需用简短的主题名或『这部分/这几部分』指代即可。\n"
            "【选项要求】选项必须结合下方列出的具体问题与主题来生成，label 和 description 都要"
            "具体到让用户一眼知道选了会发生什么。**不要**生搬硬套『补充你的要求』『换个角度再试』"
            "『就用现在的』这种通用话术，而是把它们落到具体内容上：\n"
            "  · 补充类选项（intent=supplement）：label 直接写成用户可能的具体补充方向，"
            "description 说明补充后我会怎么改。结合主题给出 1~2 个用户最可能考虑的"
            "具体要求（如侧重某类预算、某类偏好、某个具体子方向），label 落在具体内容上，"
            "description 写『我按这个方向重写这部分内容』。\n"
            "  · 换角度类选项（intent=retry_differently）：label 直接给出一个具体的新整理角度，"
            "description 说明为何可能更合适。结合主题给出 1~2 个具体角度（如按价格区间、"
            "按使用场景、按区域/分类维度等）。\n"
            "  · 用现在类选项（intent=accept_current）：label 用『就用现在的版本』，description 用"
            "一两句话概括当前整理结果的实际内容（基于下方问题里提到的主题/结果），并说明选了会在"
            "结果里标注哪里不够好。这是兜底选项，务必保留并标注『（推荐）』。\n"
            "选项总数 2~5 个，尽量穷尽；若上面的具体选项无法穷尽，可再补一个通用的『我自己补充』"
            "（intent=supplement）选项。每个选项的 label 都要具体、可读，不要把补充方向/角度"
            "塞进 description 却让 label 空泛。\n"
            f"【不要重复历史】用户已就一些方面给过答复（见『用户已补充』），"
            "不要再问用户已经回答过的内容；只针对本次列出的、尚未解决的具体问题提问。\n"
            "【推荐选项要求】recommendedKey 必须从 options 中选取，不可为空；"
            "推荐要均衡覆盖大多数用户常见意图，不可偏向极端选项。\n"
            "importance：建议 high（系统已多轮无法自行解决）。\n"
            "【optionIntents 要求】为每个选项标注恢复语义，取值只能是："
            "质量类问题用 supplement（用户补充要求重做）/ retry_differently（换角度重写）/ "
            "accept_current（接受当前版本直接合成）；"
            "矛盾类问题用 prefer_first（以第一处为准）/ prefer_second（以第二处为准）/ "
            "keep_both（都保留标注分歧）。每个选项 key 都必须有对应意图。\n"
            "【用户是普通人，请用自然语言，选项 2~5 个，互斥且尽量穷尽。】\n"
            "输出严格 JSON："
            '{"need": bool, "confidence": 0-1, "question": str, '
            '"options": [{"key":"A","label":"..","description":".."}], '
            '"optionIntents": {"A":"supplement","B":"..","C":".."}, '
            '"recommendedKey": str, "importance": "low|medium|high", '
            '"allowCustomInput": true, "customInputPlaceholder": str, '
            '"reason": str, "fallbackAnswer": str, "needed_field": str}'
        )
    else:
        phase_hint = {
            "plan": "你正处于任务拆解（Plan）阶段：若查询本身存在歧义、缺少关键约束，导致无法确定子任务划分，需要向用户澄清一次。",
            "reflect": "你正处于结果反思（Reflect/Replan）阶段：若子任务结果汇总后发现信息不足或有矛盾，需要用户补充才能继续，可向用户澄清一次。",
            "synthesize": "你正处于最终合成（Synthesize）阶段：若最终答案依赖用户未指定的维度/参数/范围，需向用户确认一次。",
        }.get(phase, "复杂任务拆解执行中。")
        system = (
            "你是 OmniHub Ask 复杂任务的澄清判定器。用户基于自己的私人收藏库提出一个较复杂的问题，"
            "系统正在用 Plan-Solve-Reflect-Synthesize 流程处理。\n"
            f"{phase_hint}\n"
            "判断是否需要向用户追问一次以获得准确答案。若需要，输出 need=true 并给出澄清问题与选项；"
            "否则 need=false。\n"
            "【推荐选项要求】recommendedKey 必须从 options 中选取，不可为空；"
            "推荐要均衡覆盖大多数用户常见意图，不可偏向极端选项。\n"
            "importance：low=有合理默认值；medium=影响答案方向但有可接受兜底；high=缺此信息无法回答。\n"
            "【用户是普通人，请用自然语言，选项 2~5 个，互斥且尽量穷尽。】\n"
            "输出严格 JSON："
            '{"need": bool, "confidence": 0-1, "question": str, '
            '"options": [{"key":"A","label":"..","description":".."}], '
            '"recommendedKey": str, "importance": "low|medium|high", '
            '"allowCustomInput": true, "customInputPlaceholder": str, '
            '"reason": str, "fallbackAnswer": str, "needed_field": str}'
        )

    user = f"用户查询：{query}\n\n当前阶段：{phase}"
    if forced:
        user += f"\n{problem_brief}"
    if plan_summary:
        user += f"\n已拆解的子任务：\n{plan_summary}"
    if result_summary:
        user += f"\n子任务结果概要：\n{result_summary}"
    recent = _history_brief(history)
    if recent:
        user += f"\n最近对话：\n{recent}"
    if supplement:
        user += (
            f"\n用户已补充：{supplement}\n"
            "（请基于用户补充重新判断：若原有歧义已解决则 need=false；"
            "仅当仍有新的、影响答案的歧义时才 need=true）"
        )
    user += "\n\n请判断是否需要向用户澄清一次。"

    try:
        raw = await _call_llm(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            ai_config=ai_config,
            # 复用 complexity_classifier 同规格：优先用用户自己的模型，
            # 避免把 judge_model（默认 glm-4-flash）发给用户的 DeepSeek baseUrl 导致 400。
            model=(ai_config or {}).get("modelName") or cfg.judge_model,
            temperature=0.2,
            max_tokens=cfg.judge_max_tokens,
            response_format={"type": "json_object"},
            no_thinking=True,
        )
    except Exception as e:
        if forced:
            # 强制模式：措辞失败不丢失强制语义，降级为规则模板澄清
            log.warning("Forced clarify judge LLM failed, fallback to template: %s", e)
            return _build_forced_template_decision(forced_signals)
        log.warning("DAG clarify judge LLM failed, fallback to no clarify: %s", e)
        return None

    parsed = _parse_judge(raw)
    if parsed is None:
        if forced:
            log.warning("Forced clarify judge output unparseable, fallback to template")
            return _build_forced_template_decision(forced_signals)
        return None

    need = bool(parsed.get("need", False))
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    if forced:
        if not need:
            # v3.2 否决权收紧：仅当上次澄清就是 reflect 强制澄清（同一问题问过、
            # 用户答过、仍失败）时才允许 supplement 否决，防对同一问题反复追问。
            # plan 阶段的偏好型答案（如"综合以上因素"）对 poor/conflicts 无修复
            # 作用，无权否决质量类强制澄清——否则多轮失败后被静默带伤合成。
            if supplement and veto_allowed:
                log.info("Forced clarify vetoed by judge: supplement resolves problems")
                return None
            # 无补充（或否决权不适用）时 LLM 误判 need=false 不采纳，走模板保住强制语义
            return _build_forced_template_decision(forced_signals)
        # 强制模式跳过 confidence 门控（confidence 仅作审计字段）
    elif not need or confidence < cfg.confidence_threshold:
        return None

    options = _normalize_options(parsed.get("options"))
    if not options:
        if forced:
            return _build_forced_template_decision(forced_signals)
        return None

    recommended = parsed.get("recommendedKey") or options[0]["key"]
    if recommended not in [o["key"] for o in options]:
        recommended = options[0]["key"]

    importance = parsed.get("importance") or "low"
    if importance not in ("low", "medium", "high"):
        importance = "low"

    fallback = parsed.get("fallbackAnswer") or ""
    if importance == "medium" and fallback:
        recommended_label = next((o["label"] for o in options if o["key"] == recommended), recommended)
        fallback = f"> ⚠️ 假设：{recommended_label}。如需调整请重新描述。\n\n{fallback}"

    if importance in ("low", "medium") and not fallback:
        importance = "high"

    # v3.1：强制模式解析 optionIntents（key 需在选项内、值需 canonical，非法项丢弃）
    option_intents: dict = {}
    if forced:
        raw_oi = parsed.get("optionIntents")
        if isinstance(raw_oi, dict):
            valid_keys = {o["key"] for o in options}
            option_intents = {
                str(k): str(v) for k, v in raw_oi.items()
                if str(k) in valid_keys and str(v) in _RESUME_INTENTS
            }

    return ClarifyDecision(
        need=True,
        question=parsed.get("question") or "",
        options=options,
        recommended_key=recommended,
        importance=importance,
        allow_custom_input=bool(parsed.get("allowCustomInput", True)),
        custom_input_placeholder=parsed.get("customInputPlaceholder") or "",
        reason=parsed.get("reason") or "",
        fallback_answer=fallback,
        option_intents=option_intents,
    )


# ── Reflect 强制澄清（v3）辅助 ──

def _build_forced_problem_brief(fs: dict) -> str:
    """把 reflect 强制澄清信号转成给 LLM 的问题清单（含用户可读的转述要求）。

    供 judge 理解具体问题，并据此生成面向用户（普通人）的澄清措辞——
    禁止出现"质量不理想 / poor / 子任务 / 自动重规划"等术语，要说明是
    哪个主题、遇到什么具体困难，期望用户怎么选。
    """
    round_num = int(fs.get("round_num") or 1)
    lines = [
        "【请用用户能看懂的话转述以下问题】"
        f"（系统已尝试 {max(round_num - 1, 0)} 轮仍未能解决；"
        "禁止使用『质量不理想/达标、子任务、自动重规划、sparse/poor』等内部术语，"
        "要用具体主题名说出来，不要照搬下面对内结构化的原文）"
    ]
    poor_tasks = fs.get("poor_tasks") or []
    if poor_tasks:
        seen = set()
        unique_topics = []
        for t in poor_tasks:
            topic = str(t.get("topic") or t.get("query") or "").strip()
            if topic and topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)
        if unique_topics:
            lines.append("整理不理想的部分（按主题名说，别用编号）：")
            for topic in unique_topics[:5]:
                lines.append(f"- 主题：{topic}")
        else:
            lines.append("整理不理想的部分：（未命名）")
    conflicts = fs.get("conflicts") or []
    if conflicts:
        lines.append("两部分说法不一致，需用户裁决：")
        for c in conflicts:
            secs = " / ".join(str(s) for s in (c.get("sections") or [])[:3])
            issue = str(c.get("issue") or "").strip()
            lines.append(f"- 涉及：「{secs}」" + (f"；不一致点：{issue}" if issue else ""))
    return "\n".join(lines)


def _build_forced_template_decision(fs: dict) -> ClarifyDecision:
    """强制澄清的规则模板兜底：LLM 措辞失败/被否时也不丢失强制语义。

    conflicts 优先（用户裁决价值最高），否则用 poor 模板；
    importance=high 且必给 fallbackAnswer，保证用户跳过时兜底合成不卡死。
    """
    reasons = fs.get("reasons") or []
    conflicts = fs.get("conflicts") or []
    poor_tasks = fs.get("poor_tasks") or []

    if "conflicts" in reasons and conflicts:
        first = conflicts[0]
        sections = [str(s) for s in (first.get("sections") or [])][:2]
        issue = str(first.get("issue") or "").strip()
        part_a = sections[0] if sections else "其中一部分"
        part_b = sections[1] if len(sections) > 1 else "另一部分"
        question = (
            f"我在整理时发现「{part_a}」和「{part_b}」两部分的说法不太一致"
            + (f"（{issue}）" if issue else "")
            + "。你希望以哪个为准？"
        )
        options = [
            {
                "key": "A",
                "label": f"按「{part_a}」的说法",
                "description": f"其他部分都以「{part_a}」为准，说法统一",
            },
            {
                "key": "B",
                "label": f"按「{part_b}」的说法",
                "description": f"其他部分都以「{part_b}」为准，说法统一",
            },
            {
                "key": "C",
                "label": "两处都列出来",
                "description": "两处说法都保留，并在结果里注明这里不一致（推荐）",
            },
        ]
        recommended = "C"
        fallback = "将按系统规则取用一处，并在结果中标注这里存在分歧。"
        reason = "整理时发现两部分说法不一致，需要你确认以哪个为准。"
        option_intents = {"A": "prefer_first", "B": "prefer_second", "C": "keep_both"}
    else:
        seen = set()
        unique_topics = []
        for t in poor_tasks[:5]:
            topic = str(t.get("topic") or t.get("query") or "").strip()
            if topic and topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)

        _GENERIC_FALLBACKS = {"这部分", "这几部分", "内容", "部分"}
        is_generic = not unique_topics or all(t in _GENERIC_FALLBACKS for t in unique_topics)
        if is_generic:
            unique_topics = [t for t in unique_topics if t not in _GENERIC_FALLBACKS]
            if not unique_topics:
                topic_phrase = "这部分"
            else:
                topic_phrase = f"「{'」「'.join(unique_topics[:3])}」这几部分"
        elif len(unique_topics) == 1:
            topic_phrase = f"「{unique_topics[0]}」这部分"
        elif len(unique_topics) <= 3:
            topic_phrase = f"「{'」「'.join(unique_topics)}」这几部分"
        else:
            topic_phrase = f"「{'」「'.join(unique_topics[:3])}」等{len(unique_topics)}部分"

        # 选项描述里嵌入主题名，让"补什么/换什么角度/当前结果"落到具体内容上。
        # 单主题用具体主题名；多主题用"第一个主题名 + 等这些内容"，保持单复数一致。
        if not unique_topics:
            ref_phrase = "这部分"
            unit = "这部分"
        elif len(unique_topics) == 1:
            ref_phrase = f"「{unique_topics[0]}」"
            unit = "这部分"
        else:
            ref_phrase = f"「{unique_topics[0]}」等这些内容"
            unit = "这些内容"
        question = (
            f"{topic_phrase}我试了几次都没整理出满意的结果，"
            "想请你帮我把方向定一下。你希望我怎么处理？"
        )
        options = [
            {
                "key": "A",
                "label": "补充你的要求",
                "description": (
                    f"告诉我{ref_phrase}你更看重哪些点（比如按什么标准、"
                    f"侧重哪些方面、要怎么组织），我照着你的要求重写{unit}"
                ),
            },
            {
                "key": "B",
                "label": "换个角度再试",
                "description": (
                    f"我换一种整理思路重新做{ref_phrase}（比如换个分类维度、"
                    "换个讲述顺序、换一种侧重点），看哪种更接近你想要的"
                ),
            },
            {
                "key": "C",
                "label": "就用现在的",
                "description": (
                    f"保留当前整理的{ref_phrase}结果直接出最终回答，"
                    "整理得不够好的地方我会在结果里标注出来（推荐）"
                ),
            },
        ]
        recommended = "C"
        fallback = "将直接采用当前版本合成，并对整理不理想的部分做标注。"
        reason = "这部分内容反复整理仍不满意，需要你指定处理方式。"
        option_intents = {"A": "supplement", "B": "retry_differently", "C": "accept_current"}

    return ClarifyDecision(
        need=True,
        question=question,
        options=options,
        recommended_key=recommended,
        importance="high",
        allow_custom_input=True,
        custom_input_placeholder="补充你对这些内容的具体要求…",
        reason=reason,
        fallback_answer=fallback,
        option_intents=option_intents,
    )


# ── 辅助 ──

def _history_brief(history: list, max_rounds: int = 3, max_chars: int = 600) -> str:
    """v2（R5）：把最近几轮对话压缩成简短摘要，喂给澄清判定器。

    judge 的 LLM prompt 原本不含 history——resume 二次判定时只有 query + 补充，
    无法感知"已澄清过什么 / 用户答了什么"。这里取最近 max_rounds 轮
    （user/assistant 消息，单条截断 120 字符，总计 ≤ max_chars），
    按时间正序输出。非 user/assistant 消息（如澄清帧的 meta）跳过。
    """
    if not history:
        return ""
    picked: list[str] = []
    seen_user_rounds = 0
    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = str(msg.get("content") or "").replace("\n", " ").strip()[:120]
        if not content:
            continue
        picked.append(f"- {role}: {content}")
        if role == "user":
            seen_user_rounds += 1
        if seen_user_rounds >= max_rounds:
            break
    if not picked:
        return ""
    picked.reverse()
    return "\n".join(picked)[:max_chars]


def _retrieval_brief(retrieval: Any | None) -> str:
    """把检索结果压缩成简短描述供判定器参考。"""
    if retrieval is None:
        return ""
    items = getattr(retrieval, "fused_items", []) or []
    content_map = getattr(retrieval, "content_map", {}) or {}
    lines = []
    for i, item in enumerate(items[:8]):
        cid = item.get("content_id")
        detail = content_map.get(cid, {})
        title = detail.get("title") or item.get("title", "")
        summary = detail.get("summary") or ""
        if len(summary) > 120:
            summary = summary[:120]
        lines.append(f"{i+1}. {title} | {summary}")
    return "\n".join(lines)


def _normalize_options(raw: Any) -> list[dict]:
    if not isinstance(raw, list):
        return []
    options = []
    for i, o in enumerate(raw[:5]):
        if not isinstance(o, dict):
            continue
        key = str(o.get("key") or f"OPT{i}")
        label = str(o.get("label") or "").strip()
        if not label:
            continue
        options.append({
            "key": key,
            "label": label,
            "description": str(o.get("description") or "") or None,
        })
    return options


def _parse_judge(raw: str) -> dict | None:
    if not raw:
        return None
    text = raw.strip()
    # 去掉可能的 ```json ... ``` 包裹
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text
        if text.startswith("json"):
            text = text[4:].lstrip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        # 尝试提取首个 { ... } 块
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
        return None


def build_resume_supplement(request_input: dict) -> str:
    """v2（D1/D3）：把用户对上次澄清的补充答案整理成 judge 的 supplement 输入。

    防"resume 用同一 query 再 judge 时对同一歧义反复追问"——judge 的
    LLM prompt 原本不含 history，必须显式把用户补充喂进去。
    """
    answer_text = request_input.get("answer_text") or ""
    answer_key = request_input.get("answer_key")
    if answer_key:
        return f"{answer_text}（对应选项 key={answer_key}）"
    return answer_text or ""
