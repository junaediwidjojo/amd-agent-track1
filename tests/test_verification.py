"""Tests for output-quality verification engine."""

from app.fireworks.models import TaskCategory, TaskItem
from app.verification import verify_answer


def test_verify_math_passes_numeric() -> None:
    task = TaskItem(task_id="m1", prompt="Calculate 5 + 3. Return only the final numeric answer.")
    result = verify_answer("8", task, TaskCategory.MATH)
    assert result.passed
    assert result.confidence >= 0.7


def test_verify_math_accepts_numeric_when_solver_uncertain() -> None:
    task = TaskItem(
        task_id="m3",
        prompt="A complex word problem with ambiguous phrasing yields 42 as the answer.",
    )
    result = verify_answer("42", task, TaskCategory.MATH)
    assert result.passed


def test_verify_math_solver_mismatch() -> None:
    task = TaskItem(
        task_id="m2",
        prompt="A store has 100 items and sells 15% of them. How many remain?",
    )
    result = verify_answer("50", task, TaskCategory.MATH)
    assert not result.passed
    assert result.retry_reason == "solver_mismatch"


def test_verify_ner_valid_json() -> None:
    task = TaskItem(task_id="n1", prompt="Extract entities as JSON.")
    answer = '[{"text": "Alice", "type": "person"}]'
    result = verify_answer(answer, task, TaskCategory.NER)
    assert result.passed


def test_verify_ner_invalid_json() -> None:
    task = TaskItem(task_id="n2", prompt="Extract entities as JSON.")
    result = verify_answer("not json", task, TaskCategory.NER)
    assert not result.passed
    assert result.retry_reason == "invalid_json"


def test_verify_factual_rejects_generic() -> None:
    task = TaskItem(task_id="f1", prompt="What is the capital of France?")
    result = verify_answer("Unable to process this task.", task, TaskCategory.FACTUAL)
    assert not result.passed


def test_verify_summarization_one_sentence() -> None:
    task = TaskItem(task_id="s1", prompt="Summarize in one sentence: Hello world.")
    result = verify_answer("Hello world is a greeting.", task, TaskCategory.SUMMARIZATION)
    assert result.passed


def test_verify_code_syntax() -> None:
    task = TaskItem(task_id="c1", prompt="Write a function named foo.")
    result = verify_answer("def foo():\n    return 1", task, TaskCategory.CODE_GENERATION)
    assert result.passed


def test_verify_logic_solver_mismatch() -> None:
    task = TaskItem(
        task_id="l1",
        prompt=(
            "Three friends, Alice, Bob, and Carol, each own a different pet: cat, dog, fish. "
            "Alice does not own the cat. Bob owns the dog. Who owns the fish?"
        ),
    )
    result = verify_answer("Bob", task, TaskCategory.LOGIC)
    assert not result.passed
    assert result.retry_reason == "solver_mismatch"


def test_verify_summarization_rejects_generic() -> None:
    task = TaskItem(task_id="s2", prompt="Summarize in one sentence: Hello world.")
    result = verify_answer("Unable to process this task.", task, TaskCategory.SUMMARIZATION)
    assert not result.passed
