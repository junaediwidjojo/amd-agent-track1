"""Sentiment classification handler with deterministic fallback."""

from __future__ import annotations

import re

from app.fireworks.models import CompletionResult, TaskItem
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
        if local and local[1] >= 0.9:
            return CompletionResult(text=local[0])
        return super().complete(task)

    def post_process(self, text: str, task: TaskItem) -> str:
        cleaned = extract_final_answer(text)
        lower = cleaned.lower()
        wants_justification = "justify" in task.prompt.lower()
        for label in ("mixed", "positive", "negative", "neutral"):
            if label in lower:
                if wants_justification:
                    label_text = label.capitalize()
                    remainder = re.sub(rf"(?i)^{label}\W*", "", cleaned).strip(" :-")
                    if remainder and len(remainder) > 10:
                        return f"{label_text}: {remainder}"
                    return f"{label_text}: {remainder}" if remainder else label_text
                return label.capitalize()
        match = re.search(r"\b(positive|negative|mixed|neutral)\b", lower)
        if match:
            label_text = match.group(1).capitalize()
            if wants_justification:
                remainder = cleaned[match.end():].strip(" :-")
                if remainder:
                    return f"{label_text}: {remainder}"
            return label_text
        return cleaned
