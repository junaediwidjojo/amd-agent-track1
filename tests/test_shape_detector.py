"""Tests for output-shape detector."""

from app.fireworks.models import OutputShape
from app.shape_detector import detect_output_shape


def test_json_object_shape() -> None:
    prompt = "Return ONLY a JSON object with these keys: summary, local_tasks."
    assert detect_output_shape(prompt) == OutputShape.JSON_OBJECT


def test_json_array_shape() -> None:
    prompt = "Extract all email addresses and return only a JSON array."
    assert detect_output_shape(prompt) == OutputShape.JSON_ARRAY


def test_exact_sections_shape() -> None:
    prompt = "Your response must contain four sections titled: Bugs, Performance Improvements, Corrected Code, Explanation."
    assert detect_output_shape(prompt) == OutputShape.EXACT_SECTIONS


def test_exact_bullets_shape() -> None:
    prompt = "Explain in exactly three bullet points why testing is important."
    assert detect_output_shape(prompt) == OutputShape.EXACT_BULLETS


def test_code_only_shape() -> None:
    prompt = "Write a Python function named merge_intervals. Return only runnable code, no explanation."
    assert detect_output_shape(prompt) == OutputShape.CODE_ONLY


def test_numeric_only_shape() -> None:
    prompt = "Calculate the discount. Return only the final numeric answer."
    assert detect_output_shape(prompt) == OutputShape.NUMERIC_ONLY


def test_free_text_fallback() -> None:
    prompt = "What is the capital of France?"
    assert detect_output_shape(prompt) == OutputShape.FREE_TEXT
