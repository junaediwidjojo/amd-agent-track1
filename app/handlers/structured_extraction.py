"""Structured extraction handler for JSON object/array outputs."""

from __future__ import annotations

import json
import re

from app.fireworks.models import CompletionResult, TaskItem
from app.handlers.base import BaseHandler
from app.utils.json_utils import clean_answer, validate_json_string


class StructuredExtractionHandler(BaseHandler):
    prompt_file = "structured_extraction.txt"

    @property
    def max_tokens(self) -> int:
        return 1024

    @property
    def preferred_model_tags(self) -> list[str]:
        return ["reason", "glm", "kimi"]

    def category_name(self) -> str:
        return "structured_extraction"

    def post_process(self, text: str, task: TaskItem) -> str:
        cleaned = clean_answer(text, category=self.category_name())

        # Try direct parse first
        valid, parsed = validate_json_string(cleaned)
        if valid:
            return json.dumps(parsed, ensure_ascii=False)

        # Try to extract the first JSON-like block
        obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        arr_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if obj_match:
            snippet = obj_match.group(0)
            valid, parsed = validate_json_string(snippet)
            if valid:
                return json.dumps(parsed, ensure_ascii=False)
        if arr_match:
            snippet = arr_match.group(0)
            valid, parsed = validate_json_string(snippet)
            if valid:
                return json.dumps(parsed, ensure_ascii=False)

        # Graceful degradation: return cleaned text
        return cleaned
