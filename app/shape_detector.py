"""Deterministic output-shape detector — no LLM calls."""

from __future__ import annotations

import re

from app.fireworks.models import OutputShape


_JSON_OBJECT_PATTERNS = (
    r"\bjson\s+object\b",
    r"\breturn\s+only\s+a?\s*json\s+object\b",
    r"\bwith\s+these\s+keys\b",
    r"\bwith\s+keys\s*:\s*",
    r"\breturn\s+.*\bjson\s+object\b.*\bkeys\b",
)

_JSON_ARRAY_PATTERNS = (
    r"\bjson\s+array\b",
    r"\breturn\s+only\s+a?\s*json\s+array\b",
    r"\blist\b.*\bjson\b",
    r"\bjson\b.*\blist\b",
)

_CODE_ONLY_PATTERNS = (
    r"\breturn\s+only\s+runnable\s+code\b",
    r"\breturn\s+only\s+the\s+code\b",
    r"\bno\s+explanation\b.*\bcode\b",
    r"\bwrite\s+a?\s*(?:python\s+)?function\b",
    r"\bwrite\s+a?\s*sql\s+query\b",
    r"\bimplement\s+a?\s*function\b",
    r"\bcorrected\s+implementation\b",
)

_EXACT_BULLETS_PATTERNS = (
    r"\bexactly\s+\d+\s+bullet\b",
    r"\b\d+\s+bullet\s+points\b",
    r"\bthree\s+bullet\b",
    r"\bfour\s+bullet\b",
    r"\bfive\s+bullet\b",
    r"\bbullet\s+points\b",
)

_EXACT_SECTIONS_PATTERNS = (
    r"\b\d+\s+sections\s+titled\b",
    r"\bsections\s+titled\b",
    r"\bmust\s+contain\b.*\bsections?\b",
    r"\bwith\s+the\s+following\s+sections\b",
    r"\bsection\s+headers?\b",
    r"\btitled\s*:\s*Bugs\b",
    r"\btitled\s*:\s*Performance\b",
    r"\bExecutive\s+Summary\b",
    r"\bArchitecture\s+Recommendation\b",
    r"\bRollback\s+Plan\b",
    r"\bRouting\s+Strategy\b",
    r"\bMonitoring\s+Metrics\b",
)

_NUMERIC_ONLY_PATTERNS = (
    r"\breturn\s+only\s+the\s+final\s+numeric\s+answer\b",
    r"\bonly\s+the\s+final\s+numeric\s+answer\b",
    r"\bwhat\s+is\s+the\s+final\s+price\b",
    r"\bhow\s+many\s+.*\bremain\b",
    r"\bhow\s+many\s+.*\bleft\b",
    r"\bcalculate\b",
    r"\bwhat\s+is\s+\d",
    r"\bpercent\b",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def detect_output_shape(prompt: str) -> OutputShape:
    """Detect the required output shape from the task prompt."""
    lower = prompt.lower()

    # Order matters: more specific shapes first
    if _matches_any(lower, _JSON_OBJECT_PATTERNS):
        return OutputShape.JSON_OBJECT
    if _matches_any(lower, _JSON_ARRAY_PATTERNS):
        return OutputShape.JSON_ARRAY
    if _matches_any(lower, _EXACT_SECTIONS_PATTERNS):
        return OutputShape.EXACT_SECTIONS
    if _matches_any(lower, _EXACT_BULLETS_PATTERNS):
        return OutputShape.EXACT_BULLETS
    if _matches_any(lower, _CODE_ONLY_PATTERNS):
        return OutputShape.CODE_ONLY
    if _matches_any(lower, _NUMERIC_ONLY_PATTERNS):
        return OutputShape.NUMERIC_ONLY

    return OutputShape.FREE_TEXT
