"""Tests for rule-based task router."""

from app.fireworks.models import TaskCategory, TaskItem
from app.router import classify_task


def test_summarization_routing() -> None:
    task = TaskItem(task_id="1", prompt="Summarize the following in one sentence: foo bar")
    assert classify_task(task) == TaskCategory.SUMMARIZATION


def test_ner_routing() -> None:
    task = TaskItem(task_id="2", prompt="Extract all named entities from the text below")
    assert classify_task(task) == TaskCategory.NER


def test_sentiment_routing() -> None:
    task = TaskItem(task_id="3", prompt="Classify the sentiment of this review: great product")
    assert classify_task(task) == TaskCategory.SENTIMENT


def test_code_generation_routing() -> None:
    task = TaskItem(task_id="4", prompt="Write a Python function that sorts a list")
    assert classify_task(task) == TaskCategory.CODE_GENERATION


def test_debugging_routing() -> None:
    task = TaskItem(task_id="5", prompt="This code has a bug, find and fix it: def f(): pass")
    assert classify_task(task) == TaskCategory.DEBUGGING


def test_math_routing() -> None:
    task = TaskItem(task_id="6", prompt="Calculate how many items remain after selling 15%")
    assert classify_task(task) == TaskCategory.MATH


def test_logic_routing() -> None:
    task = TaskItem(
        task_id="7",
        prompt=(
            "Three friends each own a different pet. "
            "Sam does not own the bird. Who owns the cat?"
        ),
    )
    assert classify_task(task) == TaskCategory.LOGIC


def test_factual_fallback() -> None:
    task = TaskItem(task_id="8", prompt="What is the capital of France?")
    assert classify_task(task) == TaskCategory.FACTUAL


def test_structured_extraction_json_object() -> None:
    task = TaskItem(
        task_id="17",
        prompt="Return ONLY a JSON object with these keys: summary, local_tasks, remote_tasks.",
    )
    assert classify_task(task) == TaskCategory.STRUCTURED_EXTRACTION


def test_structured_writing_proposal() -> None:
    task = TaskItem(
        task_id="19",
        prompt="Write a structured proposal with the following sections: 1. Executive Summary 2. Architecture Recommendation.",
    )
    assert classify_task(task) == TaskCategory.STRUCTURED_WRITING


def test_structured_writing_benchmark_recommendation() -> None:
    task = TaskItem(
        task_id="20",
        prompt="Based on benchmark results, recommend which tasks should always use the local model.",
    )
    assert classify_task(task) == TaskCategory.STRUCTURED_WRITING


def test_bullet_summary_not_structured() -> None:
    task = TaskItem(
        task_id="11",
        prompt="Explain in exactly three bullet points why unit testing is important.",
    )
    assert classify_task(task) == TaskCategory.SUMMARIZATION


def test_ner_array_kept_for_true_ner() -> None:
    task = TaskItem(
        task_id="5",
        prompt="Extract named entities as a JSON array with keys text and type.",
    )
    assert classify_task(task) == TaskCategory.NER
