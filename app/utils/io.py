"""File I/O for hackathon task and result JSON."""

from __future__ import annotations

import json
from pathlib import Path

from app.fireworks.models import ResultItem, TaskItem


def read_tasks(path: str | Path) -> list[TaskItem]:
    """Read and validate tasks from the input JSON file."""
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        msg = "tasks.json must be a JSON array"
        raise ValueError(msg)
    return [TaskItem.model_validate(item) for item in data]


def write_results(path: str | Path, results: list[ResultItem]) -> None:
    """Write results to the output JSON file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.model_dump() for item in results]
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
