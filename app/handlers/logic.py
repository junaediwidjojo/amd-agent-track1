"""Logical reasoning handler with deterministic fallback."""

from __future__ import annotations

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
        return ["reason", "math", "kimi", "glm", "deepseek"]

    def category_name(self) -> str:
        return "logic"

    def complete(self, task: TaskItem) -> CompletionResult:
        local = solve_logic(task.prompt)
        if local and local[1] >= 0.9:
            return CompletionResult(text=local[0])
        return super().complete(task)

    def post_process(self, text: str, task: TaskItem) -> str:
        answer = extract_final_answer(text)
        # For logic puzzles, the answer is usually a single name/word.
        # Strip any trailing reasoning or punctuation beyond the first real word.
        answer = answer.strip().rstrip(".").rstrip(",")
        words = answer.split()
        if len(words) > 1:
            # Heuristic: if it looks like "Diana owns the dog", return just "Diana"
            # by taking the first capitalized word, otherwise the first word.
            for word in words:
                if word[0].isupper():
                    return word
            return words[0]
        return answer
