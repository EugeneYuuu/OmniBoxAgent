"""SkillStore — skills.json 注册表 + 磁盘目录操作（docs/skill-support-design.md §3.2）。

仿 McpStore，提供：
- load / save：JSON 持久化（save 原子写，防崩溃损坏注册表）
- scan：扫描 skills/ 目录发现新技能（不覆盖已有注册）
- read_instructions / list_resources / read_resource：磁盘读取（懒加载）
- create_skill_dir / delete_skill_dir / cleanup_zombies：目录操作（含安全校验）
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from omnibox_agent.skills.model import Skill
from omnibox_agent.skills import validator

log = logging.getLogger(__name__)

SKILL_FILE_NAME = "SKILL.md"
RESOURCES_DIR = "resources"


class SkillStore:
    def __init__(self, registry_path: Path | str | None = None,
                 skills_dir: Path | str | None = None):
        self.skills_dir = Path(skills_dir).resolve() if skills_dir else None
        self.registry_path = Path(registry_path).resolve() if registry_path else None

    # —— 持久化 ——

    def load(self) -> list[Skill]:
        """读 skills.json（含 from_meta 反序列化）。文件不存在/损坏返回空列表。"""
        if self.registry_path is None or not self.registry_path.exists():
            return []
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return []
            return [Skill.from_meta(item) for item in data if isinstance(item, dict)]
        except Exception as e:
            log.warning("Failed to load skills registry %s: %s", self.registry_path, e)
            return []

    def save(self, skills: list[Skill]) -> None:
        """写 skills.json（只写 to_meta，不含 instructions）。原子写。"""
        if self.registry_path is None:
            return
        payload = [s.to_meta() for s in skills]
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(self.registry_path.parent),
                                        prefix=".skills.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.registry_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # —— 磁盘操作 ——

    def scan(self) -> list[Skill]:
        """扫描 skills/ 目录，自动发现新 SKILL.md。

        - 技能名已在注册表中 → 跳过（不覆盖已有注册），但校验其 SKILL.md 仍存在
        - 技能名未注册 → 返回新 Skill（仅含 path/name，无 instructions）
        - 目录存在但 SKILL.md 缺失 → 不作为新技能注册
        """
        if self.skills_dir is None:
            return []
        if not self.skills_dir.exists():
            return []
        discovered: list[Skill] = []
        for child in self.skills_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if validator.validate_name(name):
                continue
            skill_md = child / SKILL_FILE_NAME
            if not skill_md.exists():
                continue
            discovered.append(Skill(
                name=name,
                description="",
                tags=[],
                path=str(child.resolve()),
            ))
        return discovered

    def _skill_dir(self, skill: Skill) -> Path | None:
        """解析技能目录：优先 skill.path，为空时 fallback 到 skills_dir/name（部署可移植）。"""
        if skill.path:
            return Path(skill.path)
        if self.skills_dir:
            return self.skills_dir / skill.name
        return None

    def read_instructions(self, skill: Skill) -> str:
        """懒加载 SKILL.md 全文。调用方应包 asyncio.to_thread。"""
        skill_dir = self._skill_dir(skill)
        if skill_dir is None:
            return ""
        skill_md = skill_dir / SKILL_FILE_NAME
        if not skill_md.exists():
            return ""
        return skill_md.read_text(encoding="utf-8")

    def list_resources(self, skill: Skill) -> list[str]:
        """枚举 resources/ 下的相对路径清单。"""
        skill_dir = self._skill_dir(skill)
        if skill_dir is None:
            return []
        res_dir = skill_dir / RESOURCES_DIR
        if not res_dir.is_dir():
            return []
        rels = []
        for f in sorted(res_dir.rglob("*")):
            if f.is_file():
                rels.append(str(f.relative_to(res_dir)))
        return rels

    def read_resource(self, skill: Skill, rel_path: str) -> str:
        """读 resources/<rel_path>（仅技能目录内，只读文本）。"""
        skill_dir = self._skill_dir(skill)
        if skill_dir is None:
            raise FileNotFoundError("skill path empty and skills_dir not configured")
        res_dir = skill_dir / RESOURCES_DIR
        target = (res_dir / rel_path).resolve()
        # 安全校验：必须落在技能目录内
        skill_root = skill_dir.resolve()
        try:
            target.relative_to(skill_root)
        except ValueError:
            raise FileNotFoundError(f"resource path escapes skill dir: {rel_path}")
        if not target.is_file():
            raise FileNotFoundError(f"resource not found: {rel_path}")
        return target.read_text(encoding="utf-8")

    # —— 目录操作 ——

    def create_skill_dir(self, name: str, instructions: str) -> str:
        """为 content 模式创建 skills/<name>/ 目录并写 SKILL.md，返回绝对路径。"""
        if self.skills_dir is None:
            raise ValueError("skills_dir not configured")
        err = validator.validate_name(name)
        if err:
            raise ValueError(err)
        skill_dir = (self.skills_dir / name).resolve()
        # 防路径穿越：最终路径必须落在 skills_dir 内（name 白名单已保证，双保险）
        if not validator.is_under_roots(skill_dir, [self.skills_dir]):
            raise ValueError("skill dir escapes skills_dir")
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / SKILL_FILE_NAME).write_text(instructions, encoding="utf-8")
        return str(skill_dir.resolve())

    def delete_skill_dir(self, name: str) -> None:
        """删除 skills/<name>/ 整个目录。resolve 后必须落在 skills_dir 内。"""
        if self.skills_dir is None:
            raise ValueError("skills_dir not configured")
        target = (self.skills_dir / name).resolve()
        if not validator.is_under_roots(target, [self.skills_dir]):
            raise ValueError("refusing to delete outside skills_dir")
        if target.is_dir():
            import shutil
            shutil.rmtree(target)

    def cleanup_zombies(self, skills: list[Skill]) -> list[Skill]:
        """清理僵尸记录：注册表中有但磁盘目录/ SKILL.md 缺失的，从列表中移除。"""
        alive = []
        for s in skills:
            if not s.path:
                # content 模式可推导；source 模式标记 unresolved（不参与匹配）
                if self.skills_dir is not None:
                    derived = (self.skills_dir / s.name) / SKILL_FILE_NAME
                    if derived.exists():
                        s.path = str(derived.parent.resolve())
                        alive.append(s)
                        continue
                log.warning("Skill %s has no path, dropping", s.name)
                continue
            skill_md = Path(s.path) / SKILL_FILE_NAME
            if not skill_md.exists():
                log.warning("Skill %s SKILL.md missing, dropping zombie", s.name)
                continue
            alive.append(s)
        return alive