"""SKILL 校验与 front matter 解析（docs/skill-support-design.md §10.1 / §8）。

职责：
- name 白名单校验（防路径穿越）
- source 根目录校验（Path.resolve() 后必须落在允许根内）
- SKILL.md 的 YAML front matter 宽容解析（缺省字段取默认值）
- 指令覆写前缀黑名单启发式（非完备防护，信任边界主要靠鉴权）
- 资源引用格式校验（{{resource:<相对路径>}}）
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# 名称白名单：禁 `.` `/` `\` 空白，长度 1~64
NAME_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_-]{0,63}$")

# front matter 分隔
_FM_START = re.compile(r"^\s*---\s*$")
_FM_END = re.compile(r"^\s*\.\.\.\s*$|^\s*---\s*$")
_FM_KEY = re.compile(r"^\s*([\w-]+)\s*:\s*(.*)$")

# 指令覆写前缀黑名单（启发式）
_OVERRIDE_PREFIXES = (
    "system",
    "ignore previous",
    "ignore all previous",
    "forget everything",
    "you are now",
    "override",
)


def validate_name(name: str) -> str | None:
    """校验技能名白名单。合法返回 None，否则返回错误信息。"""
    if not name or not NAME_RE.match(name):
        return "name 必须匹配白名单 ^[a-zA-Z0-9_][a-zA-Z0-9_-]{0,63}$（禁 . / \\ 与空白）"
    return None


def resolve_allowed_roots(config_dir: str, configured: list[str]) -> list[Path]:
    """计算允许的源根列表（默认 = skills_dir）。"""
    roots = [Path(config_dir).resolve()]
    for r in configured:
        if not r:
            continue
        try:
            roots.append(Path(r).resolve())
        except Exception:
            log.warning("Invalid allowed_source_root %r", r)
    return roots


def is_under_roots(path: Path, roots: list[Path]) -> bool:
    """path 是否位于任一允许根内（resolve 后）。"""
    try:
        resolved = path.resolve()
    except Exception:
        return False
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def validate_source(source: str, roots: list[Path]) -> str | None:
    """校验 source 目录位于允许根内。合法返回 None，否则返回错误信息。"""
    if not source:
        return "source 不能为空"
    try:
        resolved = Path(source).resolve()
    except Exception as e:
        return f"source 路径解析失败: {e}"
    if not is_under_roots(resolved, roots):
        return "source 必须位于 allowed_source_roots（默认 skills_dir）内"
    if not resolved.is_dir():
        return f"source 目录不存在: {source}"
    return None


def parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """宽容解析 SKILL.md 的 YAML front matter。

    返回 (meta, body)。meta 含 name/description/tags/version 等；
    解析失败或缺失 front matter 时返回 ({}, 全文)。
    """
    lines = content.splitlines()
    if not lines or not _FM_START.match(lines[0]):
        return {}, content

    meta: dict[str, Any] = {}
    body_lines: list[str] = []
    i = 1
    closed = False
    while i < len(lines):
        line = lines[i]
        if _FM_END.match(line):
            closed = True
            i += 1
            break
        m = _FM_KEY.match(line)
        if m:
            key = m.group(1)
            val = _parse_scalar(m.group(2).strip())
            meta[key] = val
        i += 1

    if not closed:
        # 未正确闭合 → 视为无 front matter，全文作为 body
        return {}, content

    body_lines = lines[i:]
    return meta, "\n".join(body_lines)


def _parse_scalar(raw: str) -> Any:
    """把 front matter 值解析为标量 / 列表 / 字符串。"""
    if not raw:
        return ""
    if raw.startswith("[") and raw.endswith("]"):
        items = [x.strip().strip("\"'") for x in raw[1:-1].split(",") if x.strip()]
        return items
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered.startswith('"') and lowered.endswith('"'):
        return lowered[1:-1]
    if lowered.startswith("'") and lowered.endswith("'"):
        return lowered[1:-1]
    return raw


def has_override_prefix(instructions: str) -> bool:
    """启发式检查指令首行是否含覆写前缀。返回 True 表示疑似注入。"""
    if not instructions:
        return False
    first_line = instructions.strip().splitlines()[0].lower()
    return any(first_line.startswith(p) for p in _OVERRIDE_PREFIXES)


RESOURCE_REF_RE = re.compile(r"\{\{resource:([^{}]*?)\}\}")
SKILL_REF_RE = re.compile(r"\{\{skill:([a-zA-Z0-9_][a-zA-Z0-9_-]{0,63})\}\}")

# 资源引用允许的字符（相对路径，禁止绝对路径与 ..）
_RESOURCE_REF_OK = re.compile(r"^[^/\\][^.*?]*$")


def validate_resource_ref(rel_path: str) -> bool:
    """校验资源引用为相对路径（非绝对路径、不含 .. 与通配符）。"""
    if not rel_path:
        return False
    stripped = rel_path.strip()
    if stripped != rel_path:
        return False
    if stripped.startswith("/") or stripped.startswith("\\"):
        return False
    if ".." in stripped.split("/"):
        return False
    if "*" in stripped or "?" in stripped:
        return False
    return True


def extract_resource_refs(instructions: str) -> list[str]:
    """提取指令中所有 {{resource:...}} 引用的相对路径。"""
    return [m.group(1).strip() for m in RESOURCE_REF_RE.finditer(instructions)]


def extract_skill_refs(instructions: str) -> list[str]:
    """提取指令中所有 {{skill:<name>}} 引用的技能名（技能间依赖，§12）。"""
    return [m.group(1).strip() for m in SKILL_REF_RE.finditer(instructions)]