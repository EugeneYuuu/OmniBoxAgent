"""独立 skill 节点（docs/skill-support-design.md §5.4），供 QA 与 Creative 子图共用。

skill_node 接受 ctx 入参的独立异步函数，不依赖具体 GraphState TypedDict。
非 critical：任何失败降级为空（ctx.artifacts["skills"] = None），绝不阻断主流程。
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def build_skill_instructions(instructions: str,
                             header: str = "【技能指令】") -> str:
    """构建【技能指令】注入区块（docs/skill-support-design.md §5.2 / §10.1）。

    header 用于区分注入阶段（QA 默认 / Creative 规划的「-规划阶段」）。
    优先级声明（§10.1）：
      - 技能指令优先级高于通用回答要求，如冲突以技能指令为准
      - 多技能冲突时，优先遵循得分最高的技能（指令按匹配得分降序排列）
      - 用户在本轮 query 中显式要求忽略技能指令时，以用户为准
    """
    if not instructions:
        return ""
    return (
        f"\n\n{header}\n"
        "（以下技能指令优先级高于通用回答要求，如有冲突以技能指令为准；"
        "如多个技能指令相互冲突，优先遵循得分最高的技能）\n"
        f"{instructions}\n"
        "（若用户在本轮 query 中明确要求忽略上述技能指令，以用户为准）"
    )


async def skill_node(ctx: Any, query: str,
                     skill_manager: Any | None,
                     progress_cb=None) -> None:
    """执行渐进式技能匹配，结果写入 ctx.artifacts["skills"]。

    未命中 / 功能关闭 / 失败 → 置 None（ReasonStep 照常构建 prompt，行为不变）。
    用户级开关（F4，USER_FLAG_PLATFORM_DESIGN.md §4.5）：skill_injection flag
    关闭 → 跳过注入；默认 True，行为与现状一致（uid 取 ctx.input["user_id"]，
    QA/Creative 两子图的 ctx.input 同为 request_input）。
    """
    try:
        if skill_manager is None or not skill_manager.config.enabled:
            ctx.artifacts["skills"] = None
            return
        from omnibox_agent.services.flag_service import is_enabled as _flag_enabled
        uid = (ctx.input or {}).get("user_id") or ""
        if not await _flag_enabled(uid, "skill_injection"):
            ctx.artifacts["skills"] = None
            return
        if progress_cb:
            await progress_cb("skill", "正在匹配技能...")
        resolution = await skill_manager.resolve(query)
        ctx.artifacts["skills"] = resolution
    except Exception as e:
        log.warning("skill_node degraded to None: %s", e)
        ctx.artifacts["skills"] = None