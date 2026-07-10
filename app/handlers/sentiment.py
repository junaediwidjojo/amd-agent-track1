"""Sentiment classification handler with deterministic fallback."""

from __future__ import annotations

import re

from app.fireworks.models import CompletionResult, TaskItem
from app.handlers.base import BaseHandler
from app.solvers.sentiment_solver import solve_sentiment
from app.utils.text_utils import extract_final_answer


class SentimentHandler(BaseHandler):
    prompt_file = "sentiment.txt"

    @property
    def max_tokens(self) -> int:
        return 128

    @property
    def preferred_model_tags(self) -> list[str]:
        return ["fast", "small", "kimi", "glm"]

    def category_name(self) -> str:
        return "sentiment"

    def complete(self, task: TaskItem) -> CompletionResult:
        local = solve_sentiment(task.prompt)
        if local and local[1] >= 0.9:
            return CompletionResult(text=local[0])
        return super().complete(task)

    def post_process(self, text: str, task: TaskItem) -> str:
        cleaned = extract_final_answer(text)
        lower = cleaned.lower()
        for label in ("mixed", "positive", "negative"):
            if label in lower:
                # Return only the label and optionally a short justification
                parts = cleaned.split("\n", 1)
                return parts[0].strip()
        match = re.search(r"\b(positive|negative|mixed)\b", lower)
        if match:
            return match.group(1).capitalize()
        return cleaned
