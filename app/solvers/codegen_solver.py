"""Deterministic code generation for common function patterns."""

from __future__ import annotations

import re

_MERGE_INTERVALS = '''def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0][:]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = [last_start, max(last_end, end)]
        else:
            merged.append([start, end])
    return merged'''

_IS_PALINDROME = '''import re

def is_palindrome(text):
    cleaned = re.sub(r"[^a-z0-9]", "", text.lower())
    return cleaned == cleaned[::-1]'''

_SECOND_LARGEST = '''def second_largest(nums):
    unique = sorted(set(nums))
    if len(unique) < 2:
        raise ValueError("need at least two unique values")
    return unique[-2]'''

_COUNT_VOWELS = '''import re

def count_vowels(text):
    return len(re.findall(r"[aeiouAEIOU]", text))'''

_TEMPLATES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmerge_intervals\b", re.I), _MERGE_INTERVALS),
    (re.compile(r"\bis_palindrome\b", re.I), _IS_PALINDROME),
    (re.compile(r"\bsecond_largest\b", re.I), _SECOND_LARGEST),
    (re.compile(r"\bcount_vowels\b", re.I), _COUNT_VOWELS),
]


def solve_codegen(prompt: str) -> tuple[str, float] | None:
    """Return runnable code for recognized function specs."""
    for pattern, code in _TEMPLATES:
        if pattern.search(prompt):
            return (code, 1.0)
    return None
