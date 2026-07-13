"""Sentiment classification handler with deterministic fallback."""

from __future__ import annotations

import re

from app.backend_selector import deterministic_min_confidence
from app.fireworks.models import CompletionResult, TaskCategory, TaskItem
from app.handlers.base import BaseHandler
from app.solvers.sentiment_solver import build_sentiment_answer, solve_sentiment
from app.utils.text_utils import extract_final_answer


class SentimentHandler(BaseHandler):
    prompt_file = "sentiment.txt"

    @property
    def max_tokens(self) -> int:
        return 256

    @property
    def preferred_model_tags(self) -> list[str]:
        return ["fast", "small", "glm", "kimi"]

    def category_name(self) -> str:
        return "sentiment"

    def complete(self, task: TaskItem) -> CompletionResult:
        local = build_sentiment_answer(task.prompt)
        if local and local[1] >= deterministic_min_confidence(TaskCategory.SENTIMENT):
            return CompletionResult(text=local[0])
        return super().complete(task)

    def post_process(self, text: str, task: TaskItem) -> str:
        cleaned = extract_final_answer(text)
        wants_justification = (
            "justify" in task.prompt.lower() or "justification" in task.prompt.lower()
        )
        # Prefer a leading label, then the first word-boundary match (not Mixed-first
        # substring search, which misreads option lists like Positive/Negative/Mixed).
        match = re.match(
            r"^\s*(positive|negative|neutral|mixed)\b", cleaned, flags=re.I
        ) or re.search(r"\b(positive|negative|neutral|mixed)\b", cleaned, flags=re.I)
        if not match:
            return cleaned
        label_text = match.group(1).capitalize()
        if not wants_justification:
            return label_text
        remainder = cleaned[match.end() :].strip(" :-")
        if remainder and len(remainder) > 10:
            return f"{label_text}: {remainder}"
        return f"{label_text}: {remainder}" if remainder else label_text
