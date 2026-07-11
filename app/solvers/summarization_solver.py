"""Deterministic summarization helpers."""

from __future__ import annotations

import re


def _source_text(prompt: str) -> str:
    match = re.search(r":\s*(.+)$", prompt, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else prompt


_FILLER_WORDS = {
    "a", "an", "the", "it", "in", "on", "at", "to", "of", "for", "by", "with",
    "from", "into", "through", "during", "after", "over", "between", "out",
    "also", "additionally", "since", "that", "how", "they", "or", "and", "but",
    "if", "then", "so", "as", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "will", "would", "could", "may", "might",
    "must", "shall", "very", "really", "just", "still", "even", "when",
    "where", "which", "than", "too", "each", "few", "some", "such", "only",
}


def _shorten(words: list[str], limit: int = 12) -> str:
    while words and words[0].lower() in ("additionally", "also", "furthermore", "moreover", "it"):
        words = words[1:]

    if len(words) <= limit:
        return " ".join(words)

    trimmed = [w for w in words if w.lower() not in _FILLER_WORDS]
    if 4 <= len(trimmed) <= limit:
        return " ".join(trimmed)

    if len(words) > limit:
        trimmed = words[:limit]
        if trimmed[-1].endswith((",", ";", ":")):
            trimmed[-1] = trimmed[-1].rstrip(",;:")
        return " ".join(trimmed)

    return " ".join(words)


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
                line = _shorten(words, 12)
                if line:
                    line = line[0].upper() + line[1:]
                bullets.append(f"- {line}")
            if len(bullets) == 3:
                return ("\n".join(bullets), 1.0)

    if "one sentence" in lower and "no more than" in lower:
        words = re.findall(r"[A-Za-z']+", source)
        limit_match = re.search(r"no more than (\d+) words", lower)
        limit = int(limit_match.group(1)) if limit_match else 25
        if words:
            return (_shorten(words, limit), 0.95)

    return None
