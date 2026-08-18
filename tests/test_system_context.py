"""system_context 单测：验证「当前时间」统一出口（P0 收口）。

覆盖：
  - time_fact() 含当前日期/星期，结尾换行
  - time_qu_hint() 含今天/昨天绝对日期，且昨天 = 今天 - 1 天
  - 两者日期一致（同一出口，无漂移）
  - CST 时区 = UTC+8
"""

from __future__ import annotations

from datetime import datetime, timedelta

from omnibox_agent.services.system_context import (
    CST,
    time_fact,
    time_qu_hint,
)


def test_cst_is_utc_plus_8():
    assert CST.utcoffset(None) == timedelta(hours=8)


def test_time_fact_contains_today_and_weekday():
    now = datetime.now(CST)
    fact = time_fact()
    assert f"{now:%Y-%m-%d}" in fact
    assert f"星期{'一二三四五六日'[now.weekday()]}" in fact
    assert fact.endswith("\n")


def test_time_qu_hint_contains_today_and_yesterday():
    now = datetime.now(CST)
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    hint = time_qu_hint()
    assert f"当前日期：{today}" in hint
    assert f'start={today} 00:00:00' in hint          # 今天起点
    assert f'start={yesterday} 00:00:00' in hint      # 昨天起点
    assert hint.endswith("\n\n")


def test_fact_and_hint_share_same_date():
    """同一出口生成，两者日期必须一致（无漂移）。"""
    now = datetime.now(CST)
    today = now.strftime("%Y-%m-%d")
    assert today in time_fact()
    assert f"当前日期：{today}" in time_qu_hint()
