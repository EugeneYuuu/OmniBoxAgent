"""System context assembly: 统一注入 LLM 的「当前时间」等系统事实。

目的（P0 收口）：
  「当前时间」此前散落在 query_understanding（QU 换算规则）与
  ask_orchestrator（回答 prompt 事实行）两处，各自计算、格式易漂移。
  本模块提供单一出口，后续新增系统事实（用户所在地、时区偏好等）只改这里。

设计原则（见 docs/prompt-injection-vs-tooling-analysis.md）：
  确定性换算交给代码，每次请求都必需的系统事实用注入，不为此调外部工具。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# 权威时区：Asia/Shanghai（UTC+8）。与 query_understanding.CST、ask_orchestrator.CST 等值。
CST = timezone(timedelta(hours=8))

_WEEKDAYS = "一二三四五六日"


def _now() -> datetime:
    return datetime.now(CST)


def time_fact() -> str:
    """回答 prompt 用：一行「当前时间」事实 + 使用提示（结尾含换行）。

    回答 LLM 需要知道「今天」是几号，才能判断条目收藏时间是否在今天/昨天/
    最近N天内；此前从未注入，LLM 只能猜测，导致「无法获取当前的时间」。
    """
    now = _now()
    return (
        f"【当前时间】{now:%Y-%m-%d %H:%M}（星期{_WEEKDAYS[now.weekday()]}，北京时间）。\n"
        "用户提到\"今天/昨天/最近\"等相对时间时，以当前时间为基准判断收藏时间是否在范围内，\n"
        "不要因为不知道当前日期而说\"无法确认是不是今天收藏\"之类的话。\n"
    )


def time_qu_hint() -> str:
    """QU prompt 用：含「今天/昨天/最近N天」换算规则的时间提示（结尾含两个换行）。

    QU 需要把相对时间换算成 timeRange 的绝对日期，必须显式告知当前日期，
    否则 LLM 面对「我今天刚收藏的内容」无法算出「今天」是几号。
    """
    now = _now()
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    return (
        f"当前日期：{today}（星期{_WEEKDAYS[now.weekday()]}，北京时间UTC+8）。\n"
        "用户提到的相对时间必须换算成绝对日期填入 timeRange"
        '（格式 "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"）：\n'
        f'- "今天" → start={today} 00:00:00，end=当前时刻\n'
        f'- "昨天" → start={yesterday} 00:00:00，end={yesterday} 23:59:59\n'
        '- "最近N天/周/月" → start=当前时刻往前推N个单位，end=当前时刻\n'
        '- "本周" → start=本周一00:00:00，end=当前时刻\n'
        '- "本月" → start=本月1日00:00:00，end=当前时刻\n\n'
    )
