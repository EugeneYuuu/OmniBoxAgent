"""SKILL 渐进式匹配 resolver（docs/skill-support-design.md §4）。

两级召回 + 一级精排：
- Level0 关键词/标签召回（纯内存，永远先跑）
- Level1 语义向量匹配（1 次 embedding，兜底召回 + 精排）
- Level2 LLM 仲裁（仅 hybrid/llm 模式且候选过命中时触发）

命中后懒加载 instructions 并解析 {{resource:<相对路径>}} 引用（§4.7）。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from omnibox_agent.skills.model import Skill, SkillResolution
from omnibox_agent.skills import validator

log = logging.getLogger(__name__)

# 中文/英文分词：中文用 jieba，英文按空格
_ENG_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize_query(query: str) -> list[str]:
    """分词：中文用 jieba（长度≥2），英文按空格（长度≥2）。"""
    tokens: list[str] = []
    try:
        import jieba
        for w in jieba.cut(query):
            w = w.strip()
            if len(w) >= 2:
                tokens.append(w)
    except Exception:
        pass
    for w in _ENG_WORD_RE.findall(query):
        if len(w) >= 2:
            tokens.append(w)
    # 去重保序
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def level0_recall(query_tokens: list[str], skills: list[Skill],
                  min_hits: int = 1) -> list[tuple[Skill, int]]:
    """Level0 关键词/标签子串召回。返回 (skill, 命中词数) 列表。

    命中阈值：至少命中 `min_hits` 个 tag 或 description 关键词（keyword 置信下限）。
    """
    hits: list[tuple[Skill, int]] = []
    for s in skills:
        if not s.enabled:
            continue
        hay_tags = " ".join(s.tags)
        hay_desc = s.description or ""
        score = 0
        for tok in query_tokens:
            if tok in hay_tags or tok in hay_desc:
                score += 1
        if score >= min_hits:
            hits.append((s, score))
    return hits


def cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class ResolverContext:
    """resolver 依赖的不可变快照 + 配置。"""
    skills: list[Skill]                     # enabled 技能快照
    skill_map: dict[str, Skill]             # name -> Skill（技能间依赖解析用）
    vectors: dict[str, list[float]]         # name -> description 向量
    level1_ready: bool
    config: Any                             # SkillConfig
    resources_cache: dict[str, tuple[float, list[str]]]  # path -> (mtime, 清单)
    load_instructions: Any = None           # async (name) -> str，懒加载任意技能指令


async def resolve_skill(
    query: str,
    rctx: ResolverContext,
    embed_one_text: Any = None,   # async callable: (text) -> list[float] | None
    llm_select: Any = None,       # async callable: (query, candidates, max_inject) -> list[str]
) -> SkillResolution | None:
    """对 query 做渐进式匹配，命中返回 SkillResolution，未命中返回 None。

    embed_one_text / llm_select 为可注入的异步回调（生产环境由 SkillManager 提供，
    内部已包 asyncio.to_thread）。便于测试注入假实现。
    """
    cfg = rctx.config
    query_tokens = tokenize_query(query)

    # ---- Level0（永远先跑）----
    l0 = level0_recall(query_tokens, rctx.skills, cfg.keyword_min_hits)
    l0_candidates = [s for s, _ in l0]
    l0_scores = {s.name: sc for s, sc in l0}

    merged: list[Skill] = list(l0_candidates)
    matched_by = "keyword"
    level1_used = False
    degraded = False

    # ---- Level1 兜底召回（L0 候选=0 且 mode 需要语义）----
    if not merged and cfg.match_mode in ("embedding", "hybrid", "llm"):
        if rctx.level1_ready and embed_one_text is not None:
            try:
                q_vec = await embed_one_text(query)
                if q_vec:
                    scored = []
                    for s in rctx.skills:
                        v = rctx.vectors.get(s.name)
                        if not v:
                            continue
                        sim = cosine_sim(q_vec, v)
                        if sim >= cfg.similarity_threshold:
                            scored.append((s, sim))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    # 合并 candidate 名（可观测性）
                    for s, sc in scored[:cfg.select_top_k]:
                        if s not in merged:
                            merged.append(s)
                            l0_scores[s.name] = sc
                    if scored:
                        level1_used = True
                        matched_by = "embedding"
                else:
                    degraded = True
            except Exception as e:
                log.warning("Level1 embedding failed: %s", e)
                degraded = True
        else:
            degraded = True

    if not merged:
        return None

    # 候选名清单（可观测性）
    candidates = [s.name for s in merged]

    # ---- 收敛：候选 ≤ max_inject 直接命中 ----
    selected = merged[:cfg.max_inject]

    # ---- 精排：候选 > max_inject ----
    if len(merged) > cfg.max_inject:
        mode = cfg.match_mode
        if mode == "keyword":
            # 按 Level0 命中词数降序
            merged.sort(key=lambda s: (l0_scores.get(s.name, 0), candidates.index(s.name)),
                        reverse=True)
            selected = merged[:cfg.max_inject]
        elif mode == "embedding":
            # Level1 cosine 降序（需有向量）
            if rctx.level1_ready:
                merged.sort(
                    key=lambda s: (cosine_sim(_query_vec_if_any(q_vec, s.name, rctx),
                                              rctx.vectors.get(s.name) or []),
                                   candidates.index(s.name)),
                    reverse=True)
            selected = merged[:cfg.max_inject]
        elif mode in ("hybrid", "llm"):
            # Level1 top-k → Level2 LLM 仲裁
            if llm_select is not None:
                top_k = merged[:cfg.select_top_k]
                try:
                    chosen = await llm_select(query, [s.name for s in top_k], cfg.max_inject)
                    chosen_set = set(chosen)
                    selected = [s for s in top_k if s.name in chosen_set]
                    if selected:
                        matched_by = "llm"
                    else:
                        selected = top_k[:cfg.max_inject]
                except Exception as e:
                    log.warning("Level2 LLM arbitration failed, degrade: %s", e)
                    selected = merged[:cfg.max_inject]
            else:
                selected = merged[:cfg.max_inject]

    # ---- 收尾：剪裁有效注入 ----
    selected = selected[:cfg.max_inject]
    if not selected:
        return None

    # ---- 懒加载指令 + 资源/技能依赖注入 ----
    instructions_parts: list[str] = []
    resources_injected: list[str] = []
    for s in sorted(selected, key=lambda x: l0_scores.get(x.name, 0.0) or 0.0, reverse=True):
        resolved, injected = await _resolve_instruction(s, rctx, visited=set(), depth=0)
        if not resolved:
            continue
        resources_injected.extend(injected)
        instructions_parts.append(resolved)

    if not instructions_parts:
        return None

    # 预算：总字符上限，超出按匹配得方案从末尾截断
    joined = "\n\n---\n\n".join(instructions_parts)
    if len(joined) > cfg.max_instruction_chars:
        joined = _truncate_instructions(joined, cfg.max_instruction_chars)

    top_score = max(l0_scores.get(s.name, 0.0) or 0.0 for s in selected) if selected else None

    return SkillResolution(
        selected=selected,
        instructions=joined,
        matched_by=matched_by,
        candidates=candidates,
        match_score=top_score,
        degraded=degraded,
        resources_injected=resources_injected,
    )


def _query_vec_if_any(q_vec: Any, name: str, rctx: ResolverContext) -> list[float]:
    """返回 query 向量（精排用）。q_vec 可能为 None。"""
    return q_vec if q_vec is not None else []


async def _resolve_instruction(skill: Skill, rctx: ResolverContext,
                               visited: set[str], depth: int) -> tuple[str, list[str]]:
    """解析单个技能指令：懒加载 + 资源注入 + 技能间依赖注入（递归）。

    返回 (解析后的指令, 成功注入的资源相对路径列表)。
    技能间依赖：`{{skill:<name>}}` 引用另一技能的指令片段（§12），递归解析，
    带循环检测（visited）与深度上限（config.max_dep_depth）。
    """
    body = skill.instructions or ""
    if not body and rctx.load_instructions is not None:
        try:
            body = await rctx.load_instructions(skill.name) or ""
        except Exception as e:
            log.debug("Lazy load instructions failed for %s: %s", skill.name, e)
            body = ""
    if not body:
        return "", []

    injected: list[str] = []

    # 1) 资源引用
    body = _resolve_resources(body, skill, rctx, injected)

    # 2) 技能间依赖（递归）
    refs = validator.extract_skill_refs(body)
    for ref in refs:
        if ref in visited:
            body = body.replace("{{skill:%s}}" % ref,
                                f"[技能 {ref} 依赖循环，已跳过]")
            continue
        if depth >= rctx.config.max_dep_depth:
            body = body.replace("{{skill:%s}}" % ref,
                                f"[技能 {ref} 依赖层级过深，已跳过]")
            continue
        dep = rctx.skill_map.get(ref)
        if dep is None or not dep.enabled:
            body = body.replace("{{skill:%s}}" % ref,
                                f"[技能 {ref} 不存在或未启用]")
            continue
        dep_resolved, dep_injected = await _resolve_instruction(
            dep, rctx, visited | {skill.name}, depth + 1)
        if not dep_resolved:
            dep_resolved = "[技能 %s 无指令内容]" % ref
        body = body.replace("{{skill:%s}}" % ref, dep_resolved)
        injected.extend(dep_injected)

    return body, injected


def _resolve_resources(body: str, skill: Skill, rctx: ResolverContext,
                       injected: list[str]) -> str:
    """把 body 中的 {{resource:<rel_path>}} 引用解析为文件内容注入。

    非法引用 / 缺失 / 解码失败 / 超限 → 占位提示，指令本体照常注入（非 critical）。
    资源内容不嵌套解析（v1 禁嵌套），原样注入。
    """
    refs = validator.extract_resource_refs(body)
    if not refs:
        return body

    from omnibox_agent.skills.store import SkillStore

    cfg = rctx.config
    # 复用 store 的技能目录读取（安全校验内置）
    store = SkillStore(skills_dir=None)
    for rel in refs:
        if not validator.validate_resource_ref(rel):
            body = body.replace("{{resource:%s}}" % rel,
                                f"[资源 {rel} 不可用/非法引用]")
            continue
        if not skill.path:
            body = body.replace("{{resource:%s}}" % rel,
                                f"[资源 {rel} 不可用/目录缺失]")
            continue
        try:
            content = store.read_resource(skill, rel)
        except Exception as e:
            log.debug("Resource %s read failed: %s", rel, e)
            body = body.replace("{{resource:%s}}" % rel,
                                f"[资源 {rel} 不可用]")
            continue
        if len(content) > cfg.max_resource_chars:
            content = content[:cfg.max_resource_chars] + "...[资源已截断]"
        body = body.replace("{{resource:%s}}" % rel, content)
        injected.append(rel)
    return body


def _truncate_instructions(text: str, max_chars: int) -> str:
    """超预算截断（保留前段，尾部加标记）。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[技能指令已截断]"