"""Tests for deterministic code generation."""

from app.solvers.codegen_solver import solve_codegen


def test_merge_intervals() -> None:
    result = solve_codegen("Write a Python function named merge_intervals(intervals)")
    assert result is not None
    assert "def merge_intervals" in result[0]


def test_is_palindrome() -> None:
    result = solve_codegen("Write a Python function is_palindrome(text)")
    assert result is not None
    assert "def is_palindrome" in result[0]


def test_second_largest() -> None:
    result = solve_codegen("Write a Python function second_largest(nums)")
    assert result is not None
    assert "def second_largest" in result[0]
