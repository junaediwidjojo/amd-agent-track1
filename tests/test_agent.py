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


def test_should_upgrade_pass2_prefers_usable_llm(mock_hybrid_provider: MagicMock) -> None:
    """Usable Fireworks Pass-2 should replace lower-confidence Pass-1 without +0.05 margin."""
    from app.task_result import TaskResult

    agent = Agent(provider=mock_hybrid_provider)
    prior = TaskResult(
        task_id="t1",
        answer="local answer",
        confidence=0.68,
        backend_used="local",
        pass_number=1,
        category=TaskCategory.FACTUAL,
        validation_passed=True,
    )
    candidate = TaskResult(
        task_id="t1",
        answer="llm answer that is better",
        confidence=0.70,
        backend_used="fireworks",
        pass_number=2,
        category=TaskCategory.FACTUAL,
        validation_passed=True,
    )
    assert agent._should_upgrade_pass2(prior, candidate) is True


def test_should_upgrade_pass2_rejects_placeholder_llm(mock_hybrid_provider: MagicMock) -> None:
    from app.task_result import TaskResult

    agent = Agent(provider=mock_hybrid_provider)
    prior = TaskResult(
        task_id="t2",
        answer="valid local answer",
        confidence=0.8,
        backend_used="deterministic",
        pass_number=1,
        category=TaskCategory.MATH,
        validation_passed=True,
    )
    candidate = TaskResult(
        task_id="t2",
        answer="Unable to process this task.",
        confidence=0.2,
        backend_used="fireworks",
        pass_number=2,
        category=TaskCategory.MATH,
        validation_passed=False,
    )
    assert agent._should_upgrade_pass2(prior, candidate) is False


def test_should_skip_pass2_for_successful_fireworks(mock_hybrid_provider: MagicMock) -> None:
    from app.task_result import TaskResult

    agent = Agent(provider=mock_hybrid_provider)
    prior = TaskResult(
        task_id="t3",
        answer="already good",
        confidence=0.85,
        backend_used="fireworks",
        pass_number=1,
        category=TaskCategory.FACTUAL,
        validation_passed=True,
    )
    assert agent._should_skip_pass2(prior) is True
