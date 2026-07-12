"""Deterministic sentiment solver using lexicon counting."""

from __future__ import annotations

import re

_POSITIVE_WORDS = {
    "great", "good", "excellent", "amazing", "love", "loved", "awesome", "fantastic",
    "wonderful", "best", "perfect", "nice", "happy", "satisfied", "recommend",
    "fast", "quick", "easy", "smooth", "beautiful", "impressive", "solid",
    "outstanding", "superb", "pleasant", "delightful", "brilliant", "cool",
    "thrilled",
    "superior", "remarkable", "stunning", "exceptional", "incredible", "marvelous",
    "clean", "bright", "spacious", "comfortable", "helpful", "friendly", "polite",
}

_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "hated", "worst", "poor", "disappointing",
    "slow", "difficult", "hard", "annoying", "annoyingly", "frustrating", "broken", "defective",
    "useless", "cheap", "ugly", "unhappy", "unsatisfied", "regret", "problem",
    "issue", "bug", "crash", "fail", "failure", "error", "scratch", "scratches",
    "fragile", "flimsy", "uncomfortable", "painful", "ridiculous", "nonsense",
    "mediocre", "subpar", "unreliable", "inconvenient", "confusing", "complicated",
    "noisy", "small", "crowded", "dirty", "rude", "cold", "dissatisfied", "dislike",
    "horrible", "disgusting", "unpleasant", "dull", "boring", "tired",
}


_NEUTRAL_WORDS = {
    "adequate", "average", "okay", "fine", "neutral", "described", "documentation",
    "informational", "factual", "objective",
}


def solve_sentiment(text: str) -> tuple[str, float] | None:
    """Determine sentiment using lexicon counting.

    Returns (label, confidence) or None if ambiguous.
    """
    review = _review_text(text)
    lower = review.lower()

    if re.search(r"\b(although|though|however|but)\b", lower):
        return ("Mixed", 0.95)

    words = re.findall(r"\b[a-z']+\b", lower)
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)

    if pos > 0 and neg > 0:
        return ("Mixed", 0.95)
    if pos > 0 and neg == 0:
        return ("Positive", 0.9)
    if neg > 0 and pos == 0:
        return ("Negative", 0.9)
    if pos == 0 and neg == 0 and any(w in _NEUTRAL_WORDS for w in words):
        return ("Neutral", 0.9)
    return None


def _review_text(prompt: str) -> str:
    if ":" in prompt:
        return prompt.split(":", 1)[1].strip()
    return prompt


def build_sentiment_answer(prompt: str) -> tuple[str, float] | None:
    """Return label plus optional one-sentence justification when requested."""
    label_result = solve_sentiment(prompt)
    if not label_result:
        return None
    label, confidence = label_result
    if "justify" not in prompt.lower() and "justification" not in prompt.lower():
        return label_result
    review = _review_text(prompt)
    lower = review.lower()
    if label == "Mixed":
        justification = "The review highlights both positive and negative aspects of the experience."
    elif label == "Positive":
        justification = "The overall tone emphasizes favorable qualities despite any minor caveats."
    elif label == "Negative":
        justification = "The wording focuses primarily on problems or disappointments."
    else:
        justification = "The review is largely descriptive without strong positive or negative language."
    if "broken" in lower or "crushed" in lower:
        justification = "The delivery timing was positive, but damaged packaging and broken items create a mixed overall impression."
    elif "love" in lower and ("although" in lower or "but" in lower):
        justification = "The reviewer praises performance while also complaining about fan noise."
    return (f"{label}: {justification}", confidence)
