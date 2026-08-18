"""SKILL 数据模型（docs/skill-support-design.md §2）。

Skill: 技能元信息常驻内存，instructions 惰性读盘（渐进式加载关键）。
SkillResolution: 单次解析（匹配）的结果类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


@dataclass
class Skill:
    name: str                 # 唯一标识，如 "report_writing"
    description: str          # 一句话描述（用于语义匹配 & Level0 标签）
    tags: list[str] = field(default_factory=list)   # 关键词标签（Level0 轻量匹配）
    version: str = "1.0"
    path: str = ""            # 技能目录绝对路径（content/source 两种模式）
    instructions: str = ""    # SKILL.md 全文（懒加载，未命中不读）
    resources: list[str] = field(default_factory=list)  # 资源清单（相对路径）
    enabled: bool = True
    created_at: str = ""      # ISO 8601
    updated_at: str = ""      # ISO 8601

    # —— 序列化 ——
    def to_meta(self) -> dict:
        """轻量元信息：不含 instructions，常驻内存 / 列表接口用。

        ⚠️ 必须包含 path 与 resources，否则 source 模式技能目录在
        skills/ 之外时重启后 path 丢失、read_instructions 失效。
        """
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "version": self.version,
            "path": self.path,
            "resources": self.resources,
            "enabled": self.enabled,
            "resource_count": len(self.resources),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_full(self) -> dict:
        meta = self.to_meta()
        meta["instructions"] = self.instructions
        return meta

    @classmethod
    def from_meta(cls, data: dict) -> "Skill":
        """从 skills.json 反序列化轻量元信息（不含 instructions）。"""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []) or [],
            version=data.get("version", "1.0"),
            path=data.get("path", ""),
            resources=data.get("resources", []) or [],
            enabled=data.get("enabled", True),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class SkillResolution:
    """单次渐进式解析结果（§4.6）。"""
    selected: list[Skill]            # 命中的技能（已含懒加载指令）
    instructions: str                # 拼接好的指令文本（供 prompt 注入）
    matched_by: str                  # "keyword" | "embedding" | "llm"
    candidates: list[str]            # 参与匹配/精排的候选技能名（可观测性）
    match_score: float | None        # 最高匹配分
    degraded: bool = False           # Level1 不可用/降级标记
    resources_injected: list[str] = field(default_factory=list)  # 成功注入的资源相对路径

    def to_snapshot(self) -> dict:
        """可序列化的 resume 快照（§5.5）：供 clarify resume_context 携带。"""
        return {
            "skills": [s.name for s in self.selected],
            "instructions": self.instructions,
            "matched_by": self.matched_by,
            "resources_injected": self.resources_injected,
        }