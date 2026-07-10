"""Deterministic summarization helpers."""

from __future__ import annotations

import re


def _source_text(prompt: str) -> str:
    match = re.search(r":\s*(.+)$", prompt, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else prompt


def _shorten(words: list[str], limit: int = 12) -> str:
    trimmed = words[:limit]
    return " ".join(trimmed)


def solve_summarization(prompt: str) -> tuple[str, float] | None:
    lower = prompt.lower()
    source = _source_text(prompt)

    if "three bullet" in lower or "3 bullet" in lower:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source) if s.strip()]
        if len(sentences) >= 3:
            bullets = []
            for sentence in sentences[:3]:
                words = re.findall(r"[A-Za-z']+", sentence)
                if not words:
                    continue
                bullets.append(f"- {_shorten(words, 12)}")
            if len(bullets) == 3:
                return ("\n".join(bullets), 1.0)

    if "one sentence" in lower and "no more than" in lower:
        words = re.findall(r"[A-Za-z']+", source)
        limit_match = re.search(r"no more than (\d+) words", lower)
        limit = int(limit_match.group(1)) if limit_match else 25
        if words:
            return (_shorten(words, limit), 0.95)

    return None
