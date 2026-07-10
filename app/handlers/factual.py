"""Factual knowledge handler."""

from __future__ import annotations

import re

from app.fireworks.models import CompletionResult, TaskItem
from app.handlers.base import BaseHandler
from app.solvers.factual_solver import solve_factual
from app.utils.json_utils import clean_answer
from app.utils.text_utils import strip_cot


class FactualHandler(BaseHandler):
    prompt_file = "factual.txt"

    @property
    def max_tokens(self) -> int:
        return 512

    @property
    def preferred_model_tags(self) -> list[str]:
        return ["fast", "small", "glm", "kimi"]

    def category_name(self) -> str:
        return "factual"

    def complete(self, task: TaskItem) -> CompletionResult:
        local = solve_factual(task.prompt)
        if local and local[1] >= 0.9:
            return CompletionResult(text=local[0])
        return super().complete(task)

    def post_process(self, text: str, task: TaskItem) -> str:
        cleaned = strip_cot(clean_answer(text, category=self.category_name()))
        if re.search(r"\bexplain\b", task.prompt, re.IGNORECASE):
            if re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned.strip()):
                fallback = solve_factual(task.prompt)
                if fallback:
                    return fallback[0]
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", cleaned)
                if s.strip() and len(s.strip()) > 20
            ]
            reasoning_prefixes = ("the user", "i need", "let me", "wait", "but wait", "*")
            good = [
                s for s in sentences
                if not s.lower().startswith(reasoning_prefixes)
                and not re.match(r"^\d+\.", s)
            ]
            if good:
                return " ".join(good[:3])
            fallback = solve_factual(task.prompt)
            if fallback:
                return fallback[0]
        return cleaned
