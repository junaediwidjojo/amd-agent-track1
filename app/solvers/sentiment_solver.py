"""Deterministic sentiment solver using lexicon counting."""

from __future__ import annotations

import re

_POSITIVE_WORDS = {
    "great", "good", "excellent", "amazing", "love", "loved", "awesome", "fantastic",
    "wonderful", "best", "perfect", "nice", "happy", "satisfied", "recommend",
    "fast", "quick", "easy", "smooth", "beautiful", "impressive", "solid",
    "outstanding", "superb", "pleasant", "delightful", "brilliant", "cool",
    "superior", "remarkable", "stunning", "exceptional", "incredible", "marvelous",
    "clean", "bright", "spacious", "comfortable", "helpful", "friendly", "polite",
}

_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "hated", "worst", "poor", "disappointing",
    "slow", "difficult", "hard", "annoying", "frustrating", "broken", "defective",
    "useless", "cheap", "ugly", "unhappy", "unsatisfied", "regret", "problem",
    "issue", "bug", "crash", "fail", "failure", "error", "scratch", "scratches",
    "fragile", "flimsy", "uncomfortable", "painful", "ridiculous", "nonsense",
    "mediocre", "subpar", "unreliable", "inconvenient", "confusing", "complicated",
    "noisy", "small", "crowded", "dirty", "rude", "cold", "dissatisfied", "dislike",
    "horrible", "disgusting", "unpleasant", "dull", "boring", "tired",
}


def solve_sentiment(text: str) -> tuple[str, float] | None:
    """Determine sentiment using lexicon counting.

    Returns (label, confidence) or None if ambiguous.
    """
    words = re.findall(r"\b[a-z']+\b", text.lower())
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)

    if pos > 0 and neg > 0:
        return ("Mixed", 0.95)
    if pos > 0 and neg == 0:
        return ("Positive", 0.9)
    if neg > 0 and pos == 0:
        return ("Negative", 0.9)
    return None
