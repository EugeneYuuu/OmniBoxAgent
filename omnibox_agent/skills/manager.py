"""SkillManager — 运行时 CRUD + 渐进式匹配（docs/skill-support-design.md §3.3 / §4）。

设计要点：
- 写操作（add/update/remove/reload）在 asyncio.Lock 内串行，完成后原子替换快照。
- 读路径（resolve）不持锁，读取不可变快照，网络调用（embedding/LLM）互不阻塞。
- description 向量缓存：startup 批量向量化；增删改时同步更新，失败置 degraded。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from omnibox_agent.skills.model import Skill, SkillResolution
from omnibox_agent.skills.store import SkillStore
from omnibox_agent.skills import validator
from omnibox_agent.skills import loader as skill_loader

log = logging.getLogger(__name__)


class SkillManager:
    def __init__(self, config: Any):
        self.config = config
        self._write_lock = asyncio.Lock()
        self._store = SkillStore(
            registry_path=_registry_path(config),
            skills_dir=_skills_dir(config),
        )
        self._skills: dict[str, Skill] = {}            # 快照（不可变替换）
        self._description_vectors: dict[str, list[float]] = {}
        self._level1_ready: bool = False
        self._instructions_cache: dict[str, tuple[float, str]] = {}
        self._resources_cache: dict[str, tuple[float, list[str]]] = {}

    # ── 生命周期 ──

    async def startup(self) -> None:
        """合并加载：注册表 + 磁盘扫描 + 清理僵尸 + 向量缓存。"""
        try:
            registered = self._store.load()
        except Exception as e:
            log.warning("SkillStore.load failed: %s", e)
            registered = []

        # 磁盘扫描发现新技能（不覆盖已有注册）
        try:
            discovered = self._store.scan()
        except Exception as e:
            log.warning("SkillStore.scan failed: %s", e)
            discovered = []

        known = {s.name: s for s in registered}
        for s in discovered:
            if s.name not in known:
                known[s.name] = s

        merged = list(known.values())

        # 清理僵尸（磁盘已删 / 缺 SKILL.md）
        try:
            merged = self._store.cleanup_zombies(merged)
        except Exception as e:
            log.warning("SkillStore.cleanup_zombies failed: %s", e)

        # 写回清理后的注册表
        try:
            self._store.save(merged)
        except Exception as e:
            log.warning("SkillStore.save failed: %s", e)

        # 构建 enabled 快照
        snapshot = {s.name: s for s in merged if s.enabled}
        self._skills = snapshot

        # 刷新 resources 清单
        await self._refresh_resources()

        # 批量向量化 description
        await self._refresh_vectors()

    async def shutdown(self) -> None:
        try:
            self._store.save(list(self._skills.values()))
        except Exception as e:
            log.warning("SkillStore.save on shutdown failed: %s", e)

    # ── 运行时 CRUD（写锁内）──

    async def add_skill(self, name: str, *,
                        source: str | None = None,
                        description: str = "",
                        tags: list[str] | None = None,
                        instructions: str = "") -> dict:
        async with self._write_lock:
            err = validator.validate_name(name)
            if err:
                raise ValueError(err)
            if name in self._skills:
                raise ValueError(f"skill '{name}' already exists")
            if any(s.name == name for s in self._store.load()):
                raise ValueError(f"skill '{name}' already exists")

            paths = self._store.skills_dir
            roots = validator.resolve_allowed_roots(
                str(paths) if paths else self.config.dir,
                self.config.allowed_source_roots,
            )

            if source:
                err = validator.validate_source(source, roots)
                if err:
                    raise ValueError(err)
                skill_path = str(Path(source).resolve())
                raw = self._store.read_instructions(Skill(name=name, path=skill_path))
                meta, body = validator.parse_front_matter(raw)
                if not description:
                    description = str(meta.get("description", ""))
                if tags is None:
                    tags = meta.get("tags", []) or []
                if not instructions:
                    instructions = body
            else:
                if not description:
                    raise ValueError("content 模式必须提供 description（Level0 匹配必需）")
                if not tags:
                    raise ValueError("content 模式必须提供 tags（Level0 匹配必需）")
                if not instructions:
                    raise ValueError("content 模式必须提供 instructions")
                skill_path = self._store.create_skill_dir(name, instructions)

            if validator.has_override_prefix(instructions):
                raise ValueError("instructions 首行含疑似指令覆写前缀，已拒绝注册")

            now = _now_iso()
            skill = Skill(
                name=name,
                description=description,
                tags=tags or [],
                path=skill_path,
                instructions=instructions,
                enabled=True,
                created_at=now,
                updated_at=now,
            )
            skill.resources = self._store.list_resources(skill)

            new_skills = dict(self._skills)
            new_skills[name] = skill
            self._skills = new_skills
            self._store.save(list(new_skills.values()))
            await self._upsert_vector(skill)
            return {"ok": True, "skill": skill.to_meta()}

    async def update_skill(self, name: str, **fields) -> dict:
        async with self._write_lock:
            skill = self._skills.get(name)
            if skill is None:
                raise KeyError(f"skill '{name}' not found")

            allowed = {"description", "tags", "enabled", "version"}
            changed = {k: v for k, v in fields.items() if k in allowed}
            if not changed:
                return {"ok": True, "skill": skill.to_meta()}

            desc_changed = "description" in changed and changed["description"] != skill.description
            for k, v in changed.items():
                setattr(skill, k, v)
            skill.updated_at = _now_iso()

            new_skills = dict(self._skills)
            new_skills[name] = skill
            self._skills = new_skills
            self._store.save(list(new_skills.values()))

            if desc_changed:
                await self._upsert_vector(skill)
            if changed.get("enabled") is False:
                self._description_vectors.pop(name, None)
                self._level1_ready = bool(self._description_vectors)
            return {"ok": True, "skill": skill.to_meta()}

    async def remove_skill(self, name: str, *,
                           delete_files: bool = False) -> dict:
        async with self._write_lock:
            skill = self._skills.get(name)
            if skill is None:
                raise KeyError(f"skill '{name}' not found")

            if delete_files:
                # 仅允许删除 skills_dir 内的副本；source 模式指向外部目录时仅移除注册
                if self._store.skills_dir is not None:
                    skill_root = Path(skill.path).resolve() if skill.path else None
                    if skill_root is not None:
                        try:
                            skill_root.relative_to(self._store.skills_dir.resolve())
                            self._store.delete_skill_dir(name)
                        except ValueError:
                            log.info("Skill %s points outside skills_dir, only unregistering", name)

            new_skills = dict(self._skills)
            new_skills.pop(name, None)
            self._skills = new_skills
            self._description_vectors.pop(name, None)
            self._level1_ready = bool(self._description_vectors)
            self._store.save(list(new_skills.values()))
            return {"ok": True, "removed": name}

    async def reload_skill(self, name: str) -> dict:
        async with self._write_lock:
            skill = self._skills.get(name)
            if skill is None:
                raise KeyError(f"skill '{name}' not found")
            raw = await asyncio.to_thread(self._store.read_instructions, skill)
            skill.instructions = raw
            skill.resources = self._store.list_resources(skill)
            skill.updated_at = _now_iso()
            new_skills = dict(self._skills)
            new_skills[name] = skill
            self._skills = new_skills
            self._store.save(list(new_skills.values()))
            await self._upsert_vector(skill)
            return {"ok": True, "skill": skill.to_meta()}

    async def reload_all(self) -> list[dict]:
        async with self._write_lock:
            results = []
            names = list(self._skills.keys())
            for name in names:
                skill = self._skills.get(name)
                if skill is None:
                    continue
                raw = await asyncio.to_thread(self._store.read_instructions, skill)
                skill.instructions = raw
                skill.resources = self._store.list_resources(skill)
                skill.updated_at = _now_iso()
                results.append(skill.to_meta())
            self._store.save(list(self._skills.values()))
            await self._refresh_vectors()
            return results

    # ── 查询 ──

    def list_skills(self) -> list[dict]:
        return [s.to_meta() for s in self._skills.values()]

    def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    async def get_full(self, name: str) -> dict | None:
        skill = self._skills.get(name)
        if skill is None:
            return None
        if not skill.instructions:
            skill.instructions = await asyncio.to_thread(self._store.read_instructions, skill)
        return skill.to_full()

    # ── 渐进式解析 ──

    async def resolve(self, query: str) -> SkillResolution | None:
        """对 query 做渐进式匹配。不持全局锁（读不可变快照）。"""
        if not self.config.enabled or not self._skills:
            return None

        # §4.2：Level1 惰性重试——embedding 启动时不可用，则在写锁内重试一次向量化，
        # 仍失败则本轮仅用 Level0（resolve 走快照读，重试罕见且只影响向量缓存）。
        if not self._level1_ready and self.config.match_mode in ("embedding", "hybrid", "llm"):
            try:
                async with self._write_lock:
                    if not self._level1_ready:
                        await self._refresh_vectors()
            except Exception as e:
                log.warning("Level1 lazy retry failed: %s", e)

        rctx = skill_loader.ResolverContext(
            skills=list(self._skills.values()),
            skill_map=dict(self._skills),
            vectors=self._description_vectors,
            level1_ready=self._level1_ready,
            config=self.config,
            resources_cache=self._resources_cache,
            load_instructions=self._load_instructions,
        )

        async def embed_one(text: str) -> list[float] | None:
            from omnibox_agent.services.embedding_service import embed_text
            return await asyncio.to_thread(embed_text, text)

        async def llm_select(q: str, candidates: list[str], max_inject: int) -> list[str]:
            return await self._llm_select(q, candidates, max_inject)

        return await skill_loader.resolve_skill(
            query, rctx, embed_one_text=embed_one, llm_select=llm_select,
        )

    async def _load_instructions(self, name: str) -> str:
        """懒加载任意技能指令（技能间依赖解析用），带 mtime 缓存。"""
        skill = self._skills.get(name)
        if skill is None:
            return ""
        body = skill.instructions or ""
        if body:
            return body
        try:
            body = await asyncio.to_thread(self._store.read_instructions, skill)
            skill.instructions = body
            return body
        except Exception as e:
            log.warning("Failed to load instructions for %s: %s", name, e)
            return ""

    # ── 内部辅助 ──

    async def _llm_select(self, query: str, candidates: list[str], max_inject: int) -> list[str]:
        """Level2 LLM 仲裁：从候选技能中选最相关的 1~max_inject 个。"""
        try:
            from omnibox_agent.services.llm_service import generate
            from omnibox_agent.core.config import get_config

            cfg = get_config()
            selector_model = self.config.selector_model or cfg.evaluator.model
            selector_base_url = self.config.selector_base_url or cfg.evaluator.base_url
            selector_api_key = self.config.selector_api_key or cfg.evaluator.api_key
            ai_config = {
                "modelName": selector_model,
                "baseUrl": selector_base_url,
                "apiKey": selector_api_key,
            }
            detail = "\n".join(
                f"- {name}: {self._skills[name].description if name in self._skills else ''}"
                for name in candidates
            )
            system = (
                "你是技能匹配器。根据用户 query，从候选技能中选出最相关的 1~%d 个。"
                "如果都不相关，返回空列表。输出严格 JSON：{\"selected\": [\"name1\"]}"
            ) % max_inject
            user = f"用户 query：{query}\n\n候选技能：\n{detail}"
            raw = await generate(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                ai_config=ai_config,
                temperature=0.0, max_tokens=256, timeout=30,
                no_thinking=True,
            )
            return _parse_llm_selection(raw, candidates)
        except Exception as e:
            log.warning("Level2 LLM arbitration failed: %s", e)
            return candidates[:max_inject]

    async def _refresh_vectors(self) -> None:
        """批量向量化所有 enabled 技能的 description。失败不阻断启动。"""
        enabled = [s for s in self._skills.values() if s.description]
        if not enabled:
            self._description_vectors = {}
            self._level1_ready = False
            return
        try:
            from omnibox_agent.services.embedding_service import embed_texts
            texts = [s.description for s in enabled]
            vectors = await asyncio.to_thread(embed_texts, texts)
            ready = {}
            for s, v in zip(enabled, vectors):
                if v:
                    ready[s.name] = v
            self._description_vectors = ready
            self._level1_ready = bool(ready)
        except Exception as e:
            log.warning("Skill vector refresh failed: %s", e)
            self._description_vectors = {}
            self._level1_ready = False

    async def _upsert_vector(self, skill: Skill) -> None:
        """单技能向量增删。description 为空或失败 → 移除向量项。"""
        if not skill.description:
            self._description_vectors.pop(skill.name, None)
            self._level1_ready = bool(self._description_vectors)
            return
        try:
            from omnibox_agent.services.embedding_service import embed_text
            vec = await asyncio.to_thread(embed_text, skill.description)
            if vec:
                ready = dict(self._description_vectors)
                ready[skill.name] = vec
                self._description_vectors = ready
            else:
                ready = dict(self._description_vectors)
                ready.pop(skill.name, None)
                self._description_vectors = ready
            self._level1_ready = bool(self._description_vectors)
        except Exception as e:
            log.warning("Skill vector upsert failed for %s: %s", skill.name, e)
            ready = dict(self._description_vectors)
            ready.pop(skill.name, None)
            self._description_vectors = ready
            self._level1_ready = bool(self._description_vectors)

    async def _refresh_resources(self) -> None:
        """刷新 resources 清单缓存（与磁盘文件保持一致）。"""
        for s in self._skills.values():
            s.resources = self._store.list_resources(s)


def _registry_path(config: Any) -> Path:
    skills_dir = Path(config.dir).resolve()
    return skills_dir / "skills.json"


def _skills_dir(config: Any) -> Path:
    return Path(config.dir).resolve()


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).astimezone().isoformat()


def _parse_llm_selection(raw: str, candidates: list[str]) -> list[str]:
    """解析 Level2 LLM JSON 输出 {"selected": [...]}。失败返回候选前 max_inject。"""
    import json
    if not raw:
        return candidates
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text
        if text.startswith("json"):
            text = text[4:].lstrip()
    try:
        data = json.loads(text)
        selected = data.get("selected", []) if isinstance(data, dict) else []
        if isinstance(selected, list):
            valid = [s for s in selected if s in candidates]
            return valid
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                selected = data.get("selected", []) if isinstance(data, dict) else []
                if isinstance(selected, list):
                    valid = [s for s in selected if s in candidates]
                    return valid
            except json.JSONDecodeError:
                pass
    return candidates