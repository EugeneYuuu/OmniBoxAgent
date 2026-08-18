"""主题锚定（V3）单元测试。

覆盖 resume-memory-loss-fix-design.md §7.5：
  - resolve_topic_anchor：§4.4 边界矩阵 8 行逐一验证
  - _session_topic / _topic_nouns：主题提取与 resume 包裹清理
  - _ensure_broad_search_task：anchor 场景差异化子任务不 rewrite；非 anchor 与现状等价；
    漏主题子任务 rewrite 仍保留主题

运行：.venv/bin/python -m pytest tests/test_topic_anchor.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnibox_agent.services.creative_planner import (
    _ensure_broad_search_task,
    _session_topic,
    _topic_nouns,
    resolve_topic_anchor,
)
from omnibox_agent.models.note import SubTask


def _msgs(*lines):
    """构造会话消息；忽略空行。"""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": c}
        for i, c in enumerate(lines)
        if c
    ]


def _subtask(tid: str, q: str) -> SubTask:
    return SubTask(id=tid, type="section", query=q, filters={},
                   constraints={}, requires=[], produces=[])


BALI = _msgs(
    "帮我做个巴厘岛的旅游攻略",
    "您希望这份巴厘岛攻略侧重哪些方面呢？",
    "经典全面型",
    "这是你的巴厘岛经典全面型旅游攻略...巴厘岛美食、巴厘岛交通...",
)
BJ = _msgs("北京旅游攻略", "北京经典景点", "北京")


def test_matrix_anchor_resume():
    """巴厘岛会话 + resume 补充包裹 → anchor（修复主目标）。"""
    r = resolve_topic_anchor("我要玩5天4晚\n[用户补充]经典打卡型", BALI)
    assert r["action"] == "anchor"
    assert r["session_topic"] == "巴厘岛"


def test_matrix_anchor_plain():
    """巴厘岛会话 + 首跑省略主题 → anchor。"""
    r = resolve_topic_anchor("我要玩5天4晚", BALI)
    assert r["action"] == "anchor"
    assert r["session_topic"] == "巴厘岛"


def test_matrix_override_self_topic():
    """巴厘岛会话 + 明确新主题 → override（显式主题守卫）。"""
    r = resolve_topic_anchor("4天3晚上海攻略", BALI)
    assert r["action"] == "override"
    assert r["session_topic"] == "巴厘岛"


def test_matrix_none_same_topic():
    """北京会话 + 北京自带主题 → none（已自带同主题）。"""
    r = resolve_topic_anchor("北京5天4晚怎么玩", BJ)
    assert r["action"] == "none"


def test_matrix_none_no_history():
    """首轮无历史 + 5天4晚 → none（全新规划，不锚定）。"""
    r = resolve_topic_anchor("5天4晚攻略", [])
    assert r["action"] == "none"


def test_matrix_anchor_generic_tail():
    """泛化收尾"对吧"→ anchor（延续会话主题）。"""
    r = resolve_topic_anchor("对吧", BALI)
    assert r["action"] == "anchor"
    assert r["session_topic"] == "巴厘岛"


def test_matrix_none_ambiguous():
    """双主题并打 → 显著门放弃 anchors → none（防误锚）。"""
    ambi = _msgs("巴厘岛攻略", "上海攻略", "巴厘岛", "上海")
    r = resolve_topic_anchor("还有呢", ambi)
    assert r["action"] == "none"


def test_matrix_none_missing_memory():
    """缺记忆（两者皆空）→ none（安全退化为现状）。"""
    r = resolve_topic_anchor("5天4晚", [])
    assert r["action"] == "none"


def test_topic_nouns_strips_want_to():
    """"我要玩5天4晚"清洗后不得残留 own_topic（否则误判 override）。"""
    assert _topic_nouns("我要玩5天4晚", first_line_only=True) == set()


def test_topic_nouns_first_line_only():
    """query 走首行（剥离补充尾巴），消息走全文。"""
    assert "巴厘岛" in _topic_nouns("帮我做个巴厘岛的旅游攻略", first_line_only=True)


def test_broad_anchor_keeps_diff():
    """anchor 场景：差异化子任务含会话主题 → 不 rewrite，保留多样性。"""
    tasks = [_subtask("a", "巴厘岛 乌布 景点"), _subtask("b", "巴厘岛 美食 路边摊")]
    out = _ensure_broad_search_task(tasks, "巴厘岛 我要玩5天4晚", anchored_topic="巴厘岛")
    assert out[0].query == "巴厘岛 乌布 景点"
    assert out[1].query == "巴厘岛 美食 路边摊"


def test_broad_no_anchor_regression():
    """非 anchor：与现状一致，缺 core 的子任务被 rewrite 回 core。"""
    tasks = [
        _subtask("a", "上海旅游攻略 美食 餐厅"),
        _subtask("b", "上海 城市地标 拍照"),
    ]
    out = _ensure_broad_search_task(tasks, "上海旅游攻略")
    assert out[0].query == "上海旅游攻略 美食 餐厅"
    # b 不含 core（上海旅游攻略），被 rewrite 回 core
    assert out[1].query.startswith("上海旅游攻略")
    assert "上海 城市地标" not in out[1].query


def test_broad_anchor_missing_topic_rewrite_keeps_topic():
    """anchor 场景：漏主题子任务被 rewrite，但主题必须保留（不擦除）。"""
    tasks = [_subtask("c", "海边 拍照 打卡")]
    out = _ensure_broad_search_task(tasks, "巴厘岛 我要玩5天4晚", anchored_topic="巴厘岛")
    assert "巴厘岛" in out[0].query