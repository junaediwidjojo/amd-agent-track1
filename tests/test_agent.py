"""Tests for agent orchestration."""

from unittest.mock import MagicMock, patch

from app.agent import Agent
from app.fireworks.client import APIExhaustedError
from app.fireworks.models import CompletionMetrics, CompletionResult, TaskItem


def test_process_tasks_never_crashes_batch(mock_hybrid_provider: MagicMock) -> None:
    tasks = [
        TaskItem(task_id="1", prompt="What is the capital of France?"),
        TaskItem(task_id="2", prompt="Summarize this text in one sentence: hello world"),
    ]

    mock_completion = CompletionResult(
        text="test answer",
        metrics=CompletionMetrics(total_tokens=10, model="model-a"),
    )

    with patch("app.handlers.base.BaseHandler.complete", return_value=mock_completion):
        agent = Agent(provider=mock_hybrid_provider)
        results = agent.process_tasks(tasks)

    assert len(results) == 2
    assert all(r.answer for r in results)
    assert results[0].task_id == "1"
    assert results[1].task_id == "2"


def test_process_task_catches_api_exhausted(mock_hybrid_provider: MagicMock) -> None:
    task = TaskItem(task_id="err", prompt="What is Python?")

    with patch(
        "app.handlers.base.BaseHandler.complete",
        side_effect=APIExhaustedError("all models failed", "model-a"),
    ):
        agent = Agent(provider=mock_hybrid_provider)
        result, metrics, _ = agent.process_task(task)

    assert result.task_id == "err"
    assert "Unable" in result.answer


def test_process_task_propagates_other_exceptions(mock_hybrid_provider: MagicMock) -> None:
    task = TaskItem(task_id="err", prompt="What is Python?")

    with patch(
        "app.handlers.base.BaseHandler.complete",
        side_effect=RuntimeError("boom"),
    ):
        agent = Agent(provider=mock_hybrid_provider)
        try:
            agent.process_task(task)
            assert False, "Expected RuntimeError to propagate"
        except RuntimeError:
            pass
