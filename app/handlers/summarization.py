"""Summarization handler."""

from __future__ import annotations

import re

from app.backend_selector import deterministic_min_confidence
from app.fireworks.models import CompletionResult, TaskCategory, TaskItem
from app.solvers.summarization_solver import solve_summarization
from app.handlers.base import BaseHandler
from app.utils.text_utils import extract_last_sentence, strip_cot
from app.utils.validators import fallback_summary, is_truncated_summary


class SummarizationHandler(BaseHandler):
    prompt_file = "summarization.txt"

    @property
    def max_tokens(self) -> int:
        return 512

    @property
    def preferred_model_tags(self) -> list[str]:
        return ["fast", "small", "kimi", "glm"]

    def complete(self, task: TaskItem) -> CompletionResult:
        local = solve_summarization(task.prompt)
        if local and local[1] >= deterministic_min_confidence(TaskCategory.SUMMARIZATION):
            return CompletionResult(text=local[0])
        return super().complete(task)

    def category_name(self) -> str:
        return "summarization"

    def post_process(self, text: str, task: TaskItem) -> str:
        cleaned = strip_cot(text)

        # Preserve bullet points if the prompt asks for them
        if re.search(r"\bbullet\s+points?\b", task.prompt, re.IGNORECASE):
            lines = [line for line in cleaned.splitlines() if line.strip().startswith(("- ", "* "))]
            if lines:
                return "\n".join(lines)

        # Summaries should be one sentence: take the last sentence if multi-sentence
        answer = extract_last_sentence(cleaned)
        # Detect truncation (incomplete sentence) and use fallback
        if is_truncated_summary(answer):
            fallback = fallback_summary(task.prompt)
            if fallback:
                return fallback
        return answer
