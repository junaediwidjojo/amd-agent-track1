"""Structured writing handler for sectioned outputs (proposals, reviews, reports)."""

from __future__ import annotations

import re

from app.fireworks.models import CompletionResult, TaskItem
from app.handlers.base import BaseHandler
from app.utils.json_utils import clean_answer


class StructuredWritingHandler(BaseHandler):
    prompt_file = "structured_writing.txt"

    @property
    def max_tokens(self) -> int:
        return 4096

    @property
    def preferred_model_tags(self) -> list[str]:
        return ["reason", "glm", "kimi", "deepseek"]

    def category_name(self) -> str:
        return "structured_writing"

    def post_process(self, text: str, task: TaskItem) -> str:
        # Strip boundary markdown fences only
        cleaned = clean_answer(text, category=self.category_name())

        # Extract expected section headers from the prompt
        expected_headers = self._extract_expected_headers(task.prompt)

        # If the prompt required specific sections, ensure they are present
        if expected_headers:
            missing = [h for h in expected_headers if h.lower() not in cleaned.lower()]
            if missing:
                # Append missing sections as placeholders at the end
                lines = cleaned.splitlines()
                for header in missing:
                    lines.append(f"\n{header}")
                    lines.append("(No content provided.)")
                cleaned = "\n".join(lines)

        return cleaned

    def _extract_expected_headers(self, prompt: str) -> list[str]:
        """Heuristic: extract section titles mentioned in the prompt."""
        headers: list[str] = []

        # Match patterns like "four sections titled: Bugs, Performance Improvements, ..."
        titled_match = re.search(
            r"sections?\s+titled\s*[:\-]?\s*(.+?)(?:\n|$)",
            prompt,
            re.IGNORECASE,
        )
        if titled_match:
            raw = titled_match.group(1)
            # Split by comma, 'and', or numbered list
            parts = re.split(r",|\band\b|\d+\.\s*", raw)
            for part in parts:
                header = part.strip().strip("'") .strip('"').strip()
                if header:
                    headers.append(header)
            return headers

        # Match numbered sections like "1. Executive Summary"
        numbered = re.findall(
            r"^\s*\d+\.\s+(.+)",
            prompt,
            re.MULTILINE | re.IGNORECASE,
        )
        for h in numbered:
            header = h.strip().strip("'") .strip('"').strip()
            if header and len(header) < 80:
                headers.append(header)

        return headers
