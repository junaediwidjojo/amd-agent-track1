"""Tests for category-aware deterministic backend gating."""

from app.backend_selector import (
    BackendType,
    deterministic_min_confidence,
    has_deterministic_solver,
    select_backend_order,
)
from app.fireworks.models import TaskCategory, TaskItem
from app.runtime_manager import RuntimeManager


def test_deterministic_gates_are_category_aware() -> None:
    assert deterministic_min_confidence(TaskCategory.SENTIMENT) == 0.95
    assert deterministic_min_confidence(TaskCategory.NER) == 0.95
    assert deterministic_min_confidence(TaskCategory.SUMMARIZATION) == 0.9
    assert deterministic_min_confidence(TaskCategory.FACTUAL) == 0.9
    assert deterministic_min_confidence(TaskCategory.MATH) == 1.0
    assert deterministic_min_confidence(TaskCategory.LOGIC) == 1.0


def test_math_heuristic_solver_does_not_lock_deterministic() -> None:
    """Percent-remaining heuristic returns 0.95 — must not select deterministic."""
    task = TaskItem(
        task_id="m1",
        prompt="A store has 100 items and sells 15% of them. How many remain?",
    )
    assert has_deterministic_solver(TaskCategory.MATH, task) is False


def test_math_independent_solver_allows_deterministic() -> None:
    task = TaskItem(task_id="m2", prompt="What is 5 + 3?")
    assert has_deterministic_solver(TaskCategory.MATH, task) is True
    runtime = RuntimeManager(600.0)
    runtime.start()
    order = select_backend_order(TaskCategory.MATH, task, runtime)
    assert order[0] == BackendType.DETERMINISTIC
