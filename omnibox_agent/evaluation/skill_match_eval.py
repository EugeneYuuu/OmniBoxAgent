"""SKILL 匹配质量评估基线（docs/skill-support-design.md §8 / §12）。

golden set（query → 期望技能名），跑 SkillManager.resolve()，
计算 precision / recall / F1，并输出三级匹配（matched_by）分布。

用法：
  .venv/bin/python -m omnibox_agent.evaluation.skill_match_eval
  .venv/bin/python -m omnibox_agent.evaluation.skill_match_eval --golden ./skill_golden.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from omnibox_agent.core.config import SkillConfig
from omnibox_agent.skills.manager import SkillManager


# 默认 golden set：覆盖三级命中的典型场景（中英文、近义、多技能）
DEFAULT_GOLDEN = [
    {"query": "帮我写一份周报", "expected": ["report_writing"]},
    {"query": "生成一份数据分析报告", "expected": ["report_writing"]},
    {"query": "写周报", "expected": ["report_writing"]},
    {"query": "翻译这段英文", "expected": ["translation"]},
    {"query": "英译中", "expected": ["translation"]},
    {"query": "帮我总结这篇论文", "expected": ["paper_summary"]},
    {"query": "会议的待办事项怎么整理", "expected": ["meeting_notes"]},
    {"query": "写个开场白", "expected": ["greeting"]},
    {"query": "帮我介绍下这个产品", "expected": ["greeting"]},
    {"query": "如何做时间管理", "expected": []},
]


DEFAULT_SKILLS = [
    {
        "name": "report_writing",
        "description": "根据数据生成结构化报告，支持周报、数据分析报告撰写",
        "tags": ["报告", "周报", "写作", "数据分析"],
        "instructions": "你是一个专业的报告撰写助手，输出结构化 Markdown 报告。",
    },
    {
        "name": "translation",
        "description": "中英文互译，保持语境与专业术语准确",
        "tags": ["翻译", "英文", "中英互译"],
        "instructions": "你是专业翻译，输出准确、自然的目标语言译文。",
    },
    {
        "name": "paper_summary",
        "description": "对学术论文进行结构化摘要，提炼核心贡献与结论",
        "tags": ["论文", "摘要", "学术"],
        "instructions": "你是学术论文摘要助手，先提炼核心贡献再给结论。",
    },
    {
        "name": "meeting_notes",
        "description": "整理会议纪要，提取待办事项与决策项",
        "tags": ["会议", "纪要", "待办"],
        "instructions": "你是会议纪要助手，输出待办事项与决策项清单。",
    },
    {
        "name": "greeting",
        "description": "生成友好的问候与开场白，适用于产品介绍与日常对话开场",
        "tags": ["问候", "开场", "介绍"],
        "instructions": "你是一个友好、热情的开场白助手。",
    },
]


async def _build_manager(tmpdir: Path, match_mode: str, embed_off: bool = False) -> SkillManager:
    cfg = SkillConfig(
        enabled=True,
        dir=str(tmpdir / "skills"),
        match_mode=match_mode,
        max_inject=3,
        max_instruction_chars=6000,
        select_top_k=6,
        similarity_threshold=0.5,
    )
    mgr = SkillManager(cfg)
    # 预写 skills.json / 目录，startup 会 load + scan + 向量化
    for s in DEFAULT_SKILLS:
        await mgr.add_skill(s["name"], description=s["description"],
                            tags=s["tags"], instructions=s["instructions"])
    # 关闭 Level1/2 网络依赖（评估聚焦匹配逻辑，不依赖 embedding API）
    mgr._description_vectors = {}
    mgr._level1_ready = False
    return mgr


def _metric(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


async def _evaluate(mgr: SkillManager, golden: list[dict]) -> dict:
    tp = fp = fn = 0
    matched_by_dist: dict[str, int] = {}
    per_query = []
    for item in golden:
        query = item["query"]
        expected = set(item.get("expected", []))
        res = await mgr.resolve(query)
        actual = set(s.name for s in res.selected) if res else set()
        if res:
            matched_by_dist[res.matched_by] = matched_by_dist.get(res.matched_by, 0) + 1

        q_tp = len(actual & expected)
        q_fp = len(actual - expected)
        q_fn = len(expected - actual)
        tp += q_tp
        fp += q_fp
        fn += q_fn
        per_query.append({
            "query": query,
            "expected": sorted(expected),
            "actual": sorted(actual),
            "matched_by": res.matched_by if res else None,
            "tp": q_tp, "fp": q_fp, "fn": q_fn,
        })
    overall = _metric(tp, fp, fn)
    return {"overall": overall, "matched_by_dist": matched_by_dist, "per_query": per_query}


async def main() -> None:
    parser = argparse.ArgumentParser(description="SKILL 匹配质量评估")
    parser.add_argument("--golden", type=str, default=None, help="golden set JSON 文件路径")
    parser.add_argument("--match-mode", type=str, default="keyword",
                        help="匹配模式（keyword/embedding/hybrid），离线默认 keyword")
    args = parser.parse_args()

    golden = DEFAULT_GOLDEN
    if args.golden:
        data = json.loads(Path(args.golden).read_text(encoding="utf-8"))
        golden = data.get("golden", data) if isinstance(data, dict) else data

    with tempfile.TemporaryDirectory() as tmp:
        mgr = await _build_manager(Path(tmp), args.match_mode)
        result = await _evaluate(mgr, golden)

    print(f"=== SKILL 匹配评估（match_mode={args.match_mode}）===")
    print("样本数:", len(golden))
    print("整体指标:", result["overall"])
    print("匹配手段分布:", result["matched_by_dist"])
    print("\n逐条明细:")
    for q in result["per_query"]:
        print(f"  [{q['matched_by'] or '-'}] {q['query']} "
              f"expected={q['expected']} actual={q['actual']} "
              f"(tp={q['tp']} fp={q['fp']} fn={q['fn']})")


if __name__ == "__main__":
    asyncio.run(main())