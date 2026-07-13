"""Math reasoning handler with deterministic fallback."""

from __future__ import annotations

import re

from app.backend_selector import deterministic_min_confidence
from app.fireworks.models import CompletionResult, TaskCategory, TaskItem
from app.handlers.base import BaseHandler
from app.solvers.math_solver import solve_math
from app.utils.text_utils import extract_final_answer


class MathHandler(BaseHandler):
    prompt_file = "math.txt"

    @property
    def max_tokens(self) -> int:
        return 512

    @property
    def preferred_model_tags(self) -> list[str]:
        return ["reason", "math", "glm", "kimi", "deepseek"]

    def category_name(self) -> str:
        return "math"

    def complete(self, task: TaskItem) -> CompletionResult:
        local = solve_math(task.prompt)
        if local and local[1] >= deterministic_min_confidence(TaskCategory.MATH):
            return CompletionResult(text=local[0])
        return super().complete(task)

    def post_process(self, text: str, task: TaskItem) -> str:
        cleaned = extract_final_answer(text)
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if numbers:
            return numbers[-1]
        return cleaned
