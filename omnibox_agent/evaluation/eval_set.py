"""v4.1 Evaluation set schema and persistence.

Design doc §5.4: 50-100 annotated queries for gate calibration + regression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalQuery:
    """A single evaluation query with expected relevant content IDs.

    Attributes:
        query: The user query string
        expected_content_ids: Content IDs in the user's collection that are
                              relevant to this query (ground truth)
        label: Optional category label (e.g., "food", "travel", "tech")
        user_id: Which user's collection to query (for multi-tenant)
        filters: Optional structured filters (platform, time_range, etc.)
        notes: Optional annotator notes
    """
    query: str
    expected_content_ids: list[int] = field(default_factory=list)
    label: str = ""
    user_id: str = ""
    filters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class EvalSet:
    """A collection of evaluation queries.

    Attributes:
        name: Eval set name (e.g., "default", "regression-v1")
        queries: List of EvalQuery
        description: Human-readable description
        created_at: ISO timestamp
        version: Schema version
    """
    name: str = "default"
    queries: list[EvalQuery] = field(default_factory=list)
    description: str = ""
    created_at: str = ""
    version: str = "1.0"

    def __len__(self) -> int:
        return len(self.queries)

    def add(self, query: EvalQuery) -> None:
        self.queries.append(query)

    def save(self, path: str | Path) -> None:
        """Save eval set to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "version": self.version,
            "queries": [
                {
                    "query": q.query,
                    "expected_content_ids": q.expected_content_ids,
                    "label": q.label,
                    "user_id": q.user_id,
                    "filters": q.filters,
                    "notes": q.notes,
                }
                for q in self.queries
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EvalSet:
        """Load eval set from JSON file."""
        path = Path(path)
        if not path.exists():
            return cls(name=path.stem)

        data = json.loads(path.read_text(encoding="utf-8"))
        queries = [
            EvalQuery(
                query=q["query"],
                expected_content_ids=q.get("expected_content_ids", []),
                label=q.get("label", ""),
                user_id=q.get("user_id", ""),
                filters=q.get("filters", {}),
                notes=q.get("notes", ""),
            )
            for q in data.get("queries", [])
        ]
        return cls(
            name=data.get("name", path.stem),
            queries=queries,
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            version=data.get("version", "1.0"),
        )

    @classmethod
    def create_template(cls, n: int = 50) -> EvalSet:
        """Create a template eval set with placeholder queries.

        Generates n empty query slots for manual annotation.
        """
        from datetime import datetime, timezone, timedelta
        CST = timezone(timedelta(hours=8))

        return cls(
            name="template",
            description="Template eval set — fill in queries and expected_content_ids",
            created_at=datetime.now(CST).isoformat(),
            version="1.0",
            queries=[
                EvalQuery(query=f"<query_{i+1}>", expected_content_ids=[], label="")
                for i in range(n)
            ],
        )
