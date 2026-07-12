#!/usr/bin/env python3
"""Evaluate practice-task answers with lightweight heuristics."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import Agent
from app.router import classify_task
from app.utils.io import read_tasks


def run_tests(code: str, fn: str, cases: list[dict]) -> bool:
    if not re.search(rf"def\s+{re.escape(fn)}\s*\(", code):
        return False
    harness = (
        code
        + "\n\nimport sys\n"
        + f"_tests = {cases!r}\n_fn = {fn}\n"
        + "_all_ok = True\n"
        + "for _t in _tests:\n"
        + "    try:\n"
        + "        if _fn(*_t['args']) != _t['expected']:\n"
        + "            _all_ok = False\n"
        + "    except Exception:\n"
        + "        _all_ok = False\n"
        + "print('PASS' if _all_ok else 'FAIL')\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
        handle.write(harness)
        path = handle.name
    try:
        result = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip().endswith("PASS")
    except Exception:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def check(task_id: str, answer: str) -> bool:
    answer = answer or ""
    if task_id == "factual-01":
        return "jakarta" in answer.lower() and "java" in answer.lower()
    if task_id == "factual-02":
        lower = answer.lower()
        return "hash" in lower and "o(1)" in lower.replace(" ", "")
    if task_id == "math-01":
        return answer.strip().replace(",", "") == "423"
    if task_id == "math-02":
        return answer.strip().replace(",", "") == "950.00"
    if task_id == "math-03":
        return answer.strip().replace(",", "") in {"5290", "5291", "5292", "5293"}
    if task_id == "sentiment-01":
        return "mixed" in answer.lower()
    if task_id == "sentiment-02":
        return "mixed" in answer.lower() and len(answer) > 15
    if task_id == "summarization-01":
        return 5 <= len(re.findall(r"[A-Za-z']+", answer)) <= 30
    if task_id == "summarization-02":
        bullets = [line for line in answer.splitlines() if line.strip().startswith("-")]
        return len(bullets) == 3
    if task_id in ("ner-01", "ner-02"):
        try:
            data = json.loads(answer)
        except json.JSONDecodeError:
            return False
        blob = json.dumps(data).lower()
        need = {
            "ner-01": ["satya", "microsoft", "seattle"],
            "ner-02": ["maria", "fireworks", "berlin"],
        }[task_id]
        return all(token in blob for token in need)
    if task_id.startswith("debugging-"):
        code = answer.split("\n\n")[0]
        mapping = {
            "debugging-01": ("average", [{"args": ([1, 2, 3],), "expected": 2.0}]),
            "debugging-02": ("dedupe", [{"args": ([1, 2, 2, 3, 1],), "expected": [1, 2, 3]}]),
            "debugging-03": ("second_largest", [{"args": ([5, 5, 3],), "expected": 3}]),
        }
        fn, cases = mapping[task_id]
        return run_tests(code, fn, cases)
    if task_id == "logic-01":
        return answer.strip().lower() in {"alicia", "diana"}
    if task_id == "logic-02":
        compact = answer.lower().replace(" ", "")
        return "jun-first" in compact and "lee-second" in compact and "samuel-third" in compact
    if task_id.startswith("codegen-"):
        mapping = {
            "codegen-01": ("merge_intervals", [{"args": ([[1, 3], [2, 6], [8, 10]],), "expected": [[1, 6], [8, 10]]}]),
            "codegen-02": ("is_palindrome", [{"args": ("A man, a plan, a canal: Panama",), "expected": True}]),
            "codegen-03": ("second_largest", [{"args": ([5, 5, 3],), "expected": 3}]),
        }
        fn, cases = mapping[task_id]
        return run_tests(answer, fn, cases)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "input/tasks.json"))
    parser.add_argument("--output", default=str(ROOT / "output/results.json"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    tasks = read_tasks(args.input)
    if args.run:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        results = Agent().process_tasks(tasks)
        out.write_text(json.dumps([r.model_dump() for r in results], indent=2))

    payload = json.loads(Path(args.output).read_text())
    answers = {item["task_id"]: item["answer"] for item in payload}
    by_category: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for task in tasks:
        category = classify_task(task).value
        ok = check(task.task_id, answers.get(task.task_id, ""))
        by_category[category].append((task.task_id, ok))

    total = len(tasks)
    passed = sum(1 for pairs in by_category.values() for _, ok in pairs if ok)
    report = {
        "total": total,
        "passed": passed,
        "accuracy_pct": round(100 * passed / total, 1) if total else 0.0,
        "by_category": {
            cat: {
                "passed": sum(1 for _, ok in pairs if ok),
                "total": len(pairs),
                "tasks": [{"task_id": tid, "pass": ok} for tid, ok in pairs],
            }
            for cat, pairs in sorted(by_category.items())
        },
    }
    print(json.dumps(report, indent=2))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
