"""Tests for TaskResult and two-pass scheduling helpers."""

from app.fireworks.models import TaskCategory
from app.task_result import TaskResult


def test_task_result_needs_pass2_when_low_confidence() -> None:
    tr = TaskResult(
        task_id="t1",
        answer="maybe",
        confidence=0.5,
        backend_used="local",
        pass_number=1,
        category=TaskCategory.FACTUAL,
        validation_passed=True,
    )
    assert tr.needs_pass2_retry(0.7)


def test_task_result_needs_pass2_when_validation_failed() -> None:
    tr = TaskResult(
        task_id="t2",
        answer="bad",
        confidence=0.9,
        backend_used="fireworks",
        pass_number=1,
        category=TaskCategory.MATH,
        validation_passed=False,
    )
    assert tr.needs_pass2_retry(0.7)


def test_task_result_skips_pass2_when_confident() -> None:
    tr = TaskResult(
        task_id="t3",
        answer="good",
        confidence=0.85,
        backend_used="fireworks",
        pass_number=1,
        category=TaskCategory.MATH,
        validation_passed=True,
    )
    assert not tr.needs_pass2_retry(0.7)
