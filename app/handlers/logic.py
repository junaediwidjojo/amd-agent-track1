"""Logical reasoning handler with deterministic fallback."""

from __future__ import annotations

import re

from app.fireworks.models import CompletionResult, TaskItem
from app.handlers.base import BaseHandler
from app.solvers.logic_solver import solve_logic
from app.utils.text_utils import extract_final_answer


class LogicHandler(BaseHandler):
    prompt_file = "logic.txt"

    @property
    def max_tokens(self) -> int:
        return 128

    @property
    def preferred_model_tags(self) -> list[str]:
        return ["reason", "math", "glm", "kimi", "deepseek"]

    def category_name(self) -> str:
        return "logic"

    def complete(self, task: TaskItem) -> CompletionResult:
        local = solve_logic(task.prompt)
        if local and local[1] >= 0.9:
            return CompletionResult(text=local[0])
        return super().complete(task)

    def post_process(self, text: str, task: TaskItem) -> str:
        local = solve_logic(task.prompt)
        if local and local[1] >= 0.9:
            return local[0]
        if "name-place" in task.prompt.lower().replace(" ", ""):
            matches = re.findall(r"([A-Z][a-z]+)-(First|Second|Third)", text, re.IGNORECASE)
            if len(matches) >= 3:
                return ", ".join(f"{name}-{place.title()}" for name, place in matches)
        answer = extract_final_answer(text)
        if re.match(r"(?i)^(wait|but|let me|from|puzzle|the user)", answer.strip()):
            return ""
        answer = answer.strip().rstrip(".").rstrip(",")
        if re.search(r"who owns", task.prompt, re.IGNORECASE):
            for word in answer.split():
                if word[0].isupper() and word.isalpha():
                    return word
        words = answer.split()
        if len(words) > 1:
            for word in words:
                if word[0].isupper():
                    return word
            return words[0]
        return answer
