"""v4.1 data models for ingestion pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MediaStatus(str, Enum):
    """Parsing status of a media item."""
    PENDING = "pending"      # Not yet parsed
    PARSED = "parsed"        # Successfully parsed
    FAILED = "failed"        # Parse attempted but failed
    SKIPPED = "skipped"      # Skipped (e.g. video not supported in query-time)


@dataclass
class ImageRef:
    """Reference to an image in a note."""
    id: str = ""
    url: str = ""
    note_id: str = ""
    status: MediaStatus = MediaStatus.PENDING
    cdn_hash: str = ""          # CDN hash for cross-note dedup
    parsed_text: str = ""       # Raw MCP output (before summary)

    def mark_parsed(self, text: str):
        self.status = MediaStatus.PARSED
        self.parsed_text = text

    def mark_failed(self):
        self.status = MediaStatus.FAILED


@dataclass
class VideoRef:
    """Reference to a video in a note."""
    id: str = ""
    url: str = ""
    note_id: str = ""
    status: MediaStatus = MediaStatus.PENDING
    task_id: str = ""           # ai-video-notes task ID
    parsed_text: str = ""       # Summarized notes text (after worker fetches)

    def mark_parsed(self, text: str):
        self.status = MediaStatus.PARSED
        self.parsed_text = text

    def mark_failed(self):
        self.status = MediaStatus.FAILED


@dataclass
class CommentRef:
    """Reference to a comment from content_comments table."""
    comment_id: str = ""
    content: str = ""
    author_name: str = ""
    like_count: int = 0
    create_time: int = 0


@dataclass
class NoteRecord:
    """A note as seen by the ingestion pipeline.

    Maps from MySQL content_items + media metadata.
    This is the central data structure for the v4.1 ingestion layer.
    """
    id: str = ""                         # content_id from MySQL
    user_id: str = ""                    # user_id (for filtering)
    account_ids: list[str] = field(default_factory=list)
    title: str = ""
    content: str = ""                    # Main text content
    summary: str = ""                    # MySQL summary field
    platform: str = ""
    platform_name: str = ""
    author_name: str = ""
    cover_url: str = ""
    original_url: str = ""
    collected_at: str = ""
    tags: str = ""
    images: list[ImageRef] = field(default_factory=list)
    videos: list[VideoRef] = field(default_factory=list)
    comments: list[CommentRef] = field(default_factory=list)

    @property
    def main_text(self) -> str:
        """Combined text for vec_main embedding (title + content)."""
        return f"{self.title}\n{self.content}".strip()

    @property
    def comments_text(self) -> str:
        """Comments text for vec_comments embedding.

        Concatenates all comments with author context into a single block.
        Individual comments are too short for meaningful vector search,
        and lack post context when isolated. Combining them preserves the
        discussion context while capturing social proof / extra perspectives.
        """
        if not self.comments:
            return ""
        lines = []
        for c in self.comments[:50]:  # Cap at 50 comments to avoid excessive length
            author = f"@{c.author_name}" if c.author_name else ""
            lines.append(f"{author}: {c.content}".strip())
        # Further cap total chars to ~8000 (embedding model context window)
        combined = "\n".join(lines)
        if len(combined) > 8000:
            combined = combined[:8000]
        return combined

    def all_media_parsed(self) -> bool:
        """Check if all images and videos have been parsed."""
        for img in self.images:
            if img.status == MediaStatus.PENDING:
                return False
        for vid in self.videos:
            if vid.status == MediaStatus.PENDING:
                return False
        return True

    def collect_parsed_media(self) -> list[str]:
        """Collect all successfully parsed media texts (post-summary)."""
        texts: list[str] = []
        for img in self.images:
            if img.status == MediaStatus.PARSED and img.parsed_text:
                texts.append(img.parsed_text)
        for vid in self.videos:
            if vid.status == MediaStatus.PARSED and vid.parsed_text:
                texts.append(vid.parsed_text)
        return texts

    def has_unparsed_images(self) -> bool:
        """Check if any images are still pending (for query-time on-demand parse)."""
        return any(img.status == MediaStatus.PENDING for img in self.images)

    def unparsed_images(self) -> list[ImageRef]:
        """Get all pending images."""
        return [img for img in self.images if img.status == MediaStatus.PENDING]


@dataclass
class SubTask:
    """v4.1 Section 9.2: SubTask for creative Plan-Solve-Reflect."""
    id: str = ""
    type: str = "section"               # "section" or "retrieval_variant"
    query: str = ""                     # Planner-generated retrieval query
    filters: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)   # shared_state keys to depend on
    produces: list[str] = field(default_factory=list)   # shared_state keys to write


@dataclass
class SubResult:
    """Result of a sub-task execution."""
    sub_task_id: str = ""
    section_text: str = ""
    confidence: str = "normal"          # "normal" / "low" / "empty"
    coverage_status: str = "sufficient" # "sufficient" / "sparse" / "empty"
    sources: list[str] = field(default_factory=list)
    dep_snapshot: dict[str, Any] = field(default_factory=dict)
    degraded_reason: str = ""           # "missing_dependency" etc.

    @staticmethod
    def degraded(task: SubTask, reason: str = "missing_dependency") -> SubResult:
        return SubResult(
            sub_task_id=task.id,
            section_text="",
            confidence="low",
            coverage_status="empty",
            degraded_reason=reason,
        )


@dataclass
class SubTaskOverride:
    """Re-Plan override for a specific sub-task (§9.3/9.5).

    Three replan strategies (decided by LLM in Reflect):
      Strategy 1: Only change execution strategy (broaden query, switch source)
                  → output contract unchanged, downstream B waits for A to re-run.
      Strategy 2: Changed task goal/output structure → cascade replan downstream.
      Strategy 3: Major restructure → freeze completed nodes, rebuild sub-DAG.
    """
    query: str = ""           # Rewritten query (for re_retrieve or replan)
    feedback: str = ""        # Specific fix instruction (for regenerate)
    mode: str = "regenerate"  # "regenerate" | "re_retrieve" | "replan" | "replan_dag"
    strategy: int = 0         # 0=unspecified, 1=exec-only, 2=cascade, 3=full-replan


@dataclass
class HallucinationReport:
    """§9.4: Rule-based hallucination detection result (zero LLM)."""
    has_hallucination: bool = False
    unsupported_entities: list[str] = field(default_factory=list)
    number_mismatches: list[str] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)


@dataclass
class ConflictPair:
    """§9.4: Inter-section conflict detected by Reflect consistency dimension."""
    sections: list[str] = field(default_factory=list)   # e.g. ["route", "stay"]
    issue: str = ""                                     # Human-readable description
    arbitrate: str = ""                                 # Arbitration instruction for Synthesize


@dataclass
class ReflectResult:
    """§9.4: Four-dimensional evaluation result for one Re-Plan round."""
    # Per-sub-task coverage status
    coverage: dict[str, str] = field(default_factory=dict)    # task_id -> "sufficient"/"sparse"/"empty"
    # Compliance: task_id -> "compliant"/"violation"
    compliance: dict[str, str] = field(default_factory=dict)
    # Quality: task_id -> "good"/"poor"
    quality: dict[str, str] = field(default_factory=dict)
    # Hallucination reports: task_id -> HallucinationReport
    hallucinations: dict[str, HallucinationReport] = field(default_factory=dict)
    # Inter-section conflicts
    conflicts: list[ConflictPair] = field(default_factory=list)
    # Re-Plan actions: task_id -> SubTaskOverride (empty if no action needed)
    replan_actions: dict[str, SubTaskOverride] = field(default_factory=dict)
    # Overall assessment
    all_pass: bool = False
    all_empty: bool = False
    has_fixable: bool = False

    def get_problem_task_ids(self) -> list[str]:
        """Return task IDs that need Re-Plan (non-empty, non-conflict problems)."""
        problems = []
        for tid, cov in self.coverage.items():
            if cov == "sparse":
                problems.append(tid)
            elif cov == "empty":
                # empty is fixable via LLM-classified replan (Strategy 1/2/3)
                # — include if there's a replan action for this task
                if tid in self.replan_actions:
                    problems.append(tid)
            elif self.compliance.get(tid) == "violation":
                problems.append(tid)
            elif self.quality.get(tid) == "poor":
                problems.append(tid)
            elif self.hallucinations.get(tid) and self.hallucinations[tid].has_hallucination:
                problems.append(tid)
        return problems


@dataclass
class PlanOutput:
    """§9.2: Plan phase output — list of SubTasks + validity flag."""
    tasks: list[SubTask] = field(default_factory=list)
    valid: bool = False
    error: str = ""

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self):
        return len(self.tasks)


class CreativeState(str, Enum):
    """§10.2: Creative task state machine — 6 states, 9 transitions."""
    PLAN = "plan"
    SOLVE = "solve"
    REFLECT = "reflect"
    REPLAN = "replan"
    SYNTHESIZE = "synthesize"
    DONE = "done"


def compute_index_fingerprint(
    main_text: str,
    media_text: str,
    comments_text: str,
    all_parsed: bool,
    index_version: str,
    salt: str = "",
) -> str:
    """v4.1 Section 3.4: fingerprint = sha256(原文 + 解析摘要 + 评论 + parsed状态 + INDEX_VERSION).

    Changes in any of these → fingerprint changes → incremental re-embedding.
    """
    raw = f"{main_text}|{media_text}|{comments_text}|{all_parsed}|{index_version}|{salt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def dedupe_by_cdn_hash(images: list[ImageRef]) -> list[ImageRef]:
    """Deduplicate images by CDN hash — same image only parsed once."""
    seen: dict[str, ImageRef] = {}
    for img in images:
        key = img.cdn_hash or img.url
        if key not in seen:
            seen[key] = img
        else:
            # Point duplicate to the same ImageRef so parsed text propagates
            # when the original is parsed
            img.status = seen[key].status
            img.parsed_text = seen[key].parsed_text
    return list(seen.values())
