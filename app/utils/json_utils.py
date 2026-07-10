"""JSON and answer post-processing utilities."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.MULTILINE)


def clean_answer(text: str) -> str:
    """Trim whitespace and strip markdown fences from model output."""
    cleaned = text.strip()
    cleaned = _FENCE_RE.sub("", cleaned).strip()
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
