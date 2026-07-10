"""Factual knowledge handler."""

from __future__ import annotations

import re

from app.fireworks.models import CompletionResult, TaskItem
from app.handlers.base import BaseHandler
from app.utils.json_utils import clean_answer


_IDENTITY_PATTERNS = (
    r"\bintroduce\s+who\s+you\s+are\b",
    r"\bwho\s+are\s+you\b",
    r"\bwhat\s+is\s+your\s+(?:role|purpose|identity)\b",
    r"\bdescribe\s+yourself\b",
)


class FactualHandler(BaseHandler):
    prompt_file = "factual.txt"

    @property
    def max_tokens(self) -> int:
        return 512

    @property
    def preferred_model_tags(self) -> list[str]:
        return ["fast", "small", "kimi", "glm"]

    def category_name(self) -> str:
        return "factual"

    def complete(self, task: TaskItem) -> CompletionResult:
        lower = task.prompt.lower()
        if any(re.search(p, lower) for p in _IDENTITY_PATTERNS):
            return CompletionResult(
                text="I am an AI agent built for the AMD Developer Hackathon Track 1, designed to process tasks using a hybrid local-and-cloud inference architecture."
            )
        return super().complete(task)

    def post_process(self, text: str, task: TaskItem) -> str:
        return clean_answer(text)
