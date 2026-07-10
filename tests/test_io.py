"""Tests for task and result I/O."""

import json
from pathlib import Path

from app.fireworks.models import ResultItem
from app.utils.io import read_tasks, write_results


def test_read_and_write_roundtrip(tmp_path: Path) -> None:
    input_file = tmp_path / "tasks.json"
    output_file = tmp_path / "results.json"

    tasks_data = [{"task_id": "t1", "prompt": "What is 2+2?"}]
    input_file.write_text(json.dumps(tasks_data), encoding="utf-8")

    tasks = read_tasks(input_file)
    assert len(tasks) == 1
    assert tasks[0].task_id == "t1"

    results = [ResultItem(task_id="t1", answer="4")]
    write_results(output_file, results)

    loaded = json.loads(output_file.read_text(encoding="utf-8"))
    assert loaded == [{"task_id": "t1", "answer": "4"}]
