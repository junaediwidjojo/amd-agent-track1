"""JSON and answer post-processing utilities."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.utils.logger import get_logger

_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.MULTILINE)

_REASONING_LEAK_PREFIXES = (
    "the user wants",
    "i need to",
    "let me",
)

_CATEGORY_EXPECTED_FORMAT: dict[str, str] = {
    "structured_extraction": "json",
    "ner": "json",
    "code_generation": "code",
    "debugging": "code",
}


def _looks_like_reasoning_leak(text: str) -> bool:
    lower = text.strip().lower()
    return any(lower.startswith(prefix) for prefix in _REASONING_LEAK_PREFIXES)


def _contains_expected_content(text: str, expected_format: str) -> bool:
    if expected_format == "json":
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            valid, _ = validate_json_string(stripped)
            if valid:
                return True
        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                valid, _ = validate_json_string(match.group(0))
                if valid:
                    return True
        return False
    if expected_format == "code":
        return bool(re.search(r"\bdef\s+\w+", text)) or "```" in text
    return True


def _warn_reasoning_leak(text: str, category: str | None) -> None:
    leak_prefix = _looks_like_reasoning_leak(text)
    expected_format = _CATEGORY_EXPECTED_FORMAT.get(category or "")
    missing_expected = bool(expected_format) and not _contains_expected_content(
        text, expected_format
    )
    if not leak_prefix and not missing_expected:
        return

    logger = get_logger(__name__)
    record = logger.makeRecord(
        logger.name,
        logging.WARNING,
        "",
        0,
        "reasoning_leak_detected",
        (),
        None,
    )
    record.extra_fields = {
        "category": category or "unknown",
        "text_preview": text[:200],
        "leak_prefix": leak_prefix,
        "missing_expected_format": missing_expected,
        "expected_format": expected_format or None,
    }
    logger.handle(record)


warn_if_reasoning_leak = _warn_reasoning_leak


def clean_answer(text: str, category: str | None = None) -> str:
    """Trim whitespace and strip markdown fences from model output."""
    cleaned = text.strip()
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    if category:
        _warn_reasoning_leak(cleaned, category)
    return cleaned


def extract_code_block(text: str) -> str:
    """Extract code from fenced blocks or return cleaned plain text."""
    match = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return clean_answer(text)


def validate_json_string(text: str) -> tuple[bool, Any]:
    """Return whether text is valid JSON and the parsed value."""
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        return False, None


def ensure_plain_text(text: str) -> str:
    """Ensure answer is plain text without markdown artifacts."""
    result = clean_answer(text)
    if result.startswith("{") or result.startswith("["):
        valid, _ = validate_json_string(result)
        if valid:
            return result
    return result
