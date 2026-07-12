"""Tests for agent orchestration."""

from unittest.mock import MagicMock, patch

from app.agent import Agent
from app.fireworks.client import APIExhaustedError
from app.fireworks.models import CompletionMetrics, TaskCategory, TaskItem
from app.task_executor import TaskExecutionResult
from app.verification import VerificationResult


def _mock_execution(answer: str = "test answer") -> TaskExecutionResult:
    return TaskExecutionResult(
        answer=answer,
        metrics=CompletionMetrics(total_tokens=10, model="model-a", backend="fireworks"),
        category=TaskCategory.FACTUAL,
        verification=VerificationResult(passed=True, confidence=0.9),
        backend="fireworks",
    )


def test_process_tasks_never_crashes_batch(mock_hybrid_provider: MagicMock) -> None:
    tasks = [
        TaskItem(task_id="1", prompt="What is the capital of France?"),
        TaskItem(task_id="2", prompt="Summarize this text in one sentence: hello world"),
    ]

    with patch("app.task_executor.TaskExecutor.execute", return_value=_mock_execution()):
        agent = Agent(provider=mock_hybrid_provider)
        results = agent.process_tasks(tasks)

    assert len(results) == 2
    assert all(r.answer for r in results)
    assert results[0].task_id == "1"
    assert results[1].task_id == "2"


def test_process_task_catches_api_exhausted(mock_hybrid_provider: MagicMock) -> None:
    task = TaskItem(task_id="err", prompt="What is Python?")

    with patch(
        "app.task_executor.TaskExecutor.execute",
        side_effect=APIExhaustedError("all models failed", "model-a"),
    ):
        agent = Agent(provider=mock_hybrid_provider)
        result, metrics, _ = agent.process_task(task)

    assert result.task_id == "err"
    assert "Unable" in result.answer


def test_process_task_catches_unexpected_exception(mock_hybrid_provider: MagicMock) -> None:
    task = TaskItem(task_id="err", prompt="What is Python?")

    with patch(
        "app.task_executor.TaskExecutor.execute",
        side_effect=RuntimeError("boom"),
    ):
        agent = Agent(provider=mock_hybrid_provider)
        result, _, _ = agent.process_task(task)

    assert result.task_id == "err"
    assert "Unable" in result.answer


def test_runtime_budget_skips_remaining_tasks(mock_hybrid_provider: MagicMock) -> None:
    tasks = [
        TaskItem(task_id="1", prompt="Question one"),
        TaskItem(task_id="2", prompt="Question two"),
    ]

    with (
        patch("app.task_executor.TaskExecutor.execute", return_value=_mock_execution("answer")),
        patch.object(Agent, "_runtime_budget_exceeded", side_effect=[False, True, False]),
        patch.object(Agent, "_run_pass2_retries"),
    ):
        agent = Agent(provider=mock_hybrid_provider, max_runtime_seconds=600.0)
        results = agent.process_tasks(tasks)

    assert agent.runtime_budget_exceeded is True
    assert results[0].answer == "answer"
    assert "runtime limit" in results[1].answer.lower()
