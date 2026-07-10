"""Named entity recognition handler with deterministic fallback."""

from __future__ import annotations

import json
import re

from app.fireworks.models import CompletionResult, TaskItem
from app.handlers.base import BaseHandler
from app.solvers.ner_solver import solve_ner
from app.utils.json_utils import validate_json_string
from app.utils.text_utils import strip_cot


class NerHandler(BaseHandler):
    prompt_file = "ner.txt"

    @property
    def max_tokens(self) -> int:
        return 512

    @property
    def preferred_model_tags(self) -> list[str]:
        return ["fast", "small", "kimi", "glm"]

    def category_name(self) -> str:
        return "ner"

    def complete(self, task: TaskItem) -> CompletionResult:
        local = solve_ner(task.prompt)
        if local and local[1] >= 0.85:
            return CompletionResult(text=local[0])
        return super().complete(task)

    def post_process(self, text: str, task: TaskItem) -> str:
        cleaned = strip_cot(text)
        valid, parsed = validate_json_string(cleaned)
        if valid:
            return json.dumps(parsed, ensure_ascii=False)

        array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if array_match:
            snippet = array_match.group(0)
            valid, parsed = validate_json_string(snippet)
            if valid:
                return json.dumps(parsed, ensure_ascii=False)

        return cleaned
