"""Code debugging handler."""

from __future__ import annotations

import re

from app.fireworks.models import TaskItem
from app.handlers.base import BaseHandler
from app.utils.text_utils import extract_first_code_block, is_valid_python, strip_cot
from app.utils.validators import generate_debug_explanation


class DebuggingHandler(BaseHandler):
    prompt_file = "debugging.txt"

    @property
    def max_tokens(self) -> int:
        return 512

    @property
    def preferred_model_tags(self) -> list[str]:
        return ["code", "kimi", "glm", "deepseek"]

    def _extract_explanation(self, text: str, code: str) -> str:
        """Extract a concise explanation sentence, stripping CoT noise."""
        after = text.split(code, 1)[-1] if code and code in text else text
        after = strip_cot(after)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", after) if s.strip()]
        # Clean bullet markers from sentences
        sentences = [re.sub(r"^[-*]\s+", "", s) for s in sentences]
        # Filter to sentences that explain a bug/fix
        for s in sentences:
            bug_keywords = (
                r"\b(bug|because|issue|fix|error|wrong|only returns|"
                r"missing|forgot|should|instead)\b"
            )
            if re.search(bug_keywords, s, re.IGNORECASE):
                return s
        if sentences:
            return sentences[0]
        return ""

    def category_name(self) -> str:
        return "debugging"

    def post_process(self, text: str, task: TaskItem) -> str:
        cleaned = strip_cot(text)

        # Detect if the task requires structured sections (e.g., challenge-18)
        section_requirement = re.search(
            r"(\d+)\s+sections\s+titled\s*[:\-]?\s*(.+?)(?:\n|$)",
            task.prompt,
            re.IGNORECASE,
        )
        if section_requirement:
            # Preserve the full structured response; don't extract code+explanation
            return cleaned

        code = extract_first_code_block(cleaned)
        # Validate code; if invalid, try a stricter line-based extraction
        if not is_valid_python(code):
            lines = cleaned.splitlines()
            code_lines: list[str] = []
            collecting = False
            base_indent = 0
            for line in lines:
                if not collecting and line.strip().startswith("def "):
                    collecting = True
                    base_indent = len(line) - len(line.lstrip())
                    code_lines.append(line)
                    continue
                if collecting:
                    if line.strip() == "":
                        code_lines.append(line)
                        continue
                    indent = len(line) - len(line.lstrip())
                    # Stop at a new def at same or lower indent, or at prose
                    if indent <= base_indent and line.strip():
                        if line.strip().startswith("def "):
                            break
                        # Non-code line at same or lower indent ends the block
                        code_keywords = (
                            r"^\s*(return |if |for |while |try |except |with |"
                            r"elif |else |pass |break |continue |raise |assert |"
                            r"print |\w+\s*=)"
                        )
                        if not re.match(code_keywords, line):
                            break
                    code_lines.append(line)
            candidate = "\n".join(code_lines).strip()
            if candidate and is_valid_python(candidate):
                code = candidate
        explanation = self._extract_explanation(cleaned, code)
        # Detect placeholder explanations and generate a real one
        placeholder_patterns = (
            "one sentence explaining",
            "explanation here",
            "placeholder",
            "the bug is that the function",
        )
        is_placeholder = (
            not explanation
            or len(explanation) < 20
            or any(p in explanation.lower() for p in placeholder_patterns)
        )
        if is_placeholder:
            explanation = generate_debug_explanation(task.prompt, code)
        # Return code even if explanation is missing; never return only explanation
        if code:
            if explanation:
                return f"{code}\n\n{explanation}"
            return code
        return cleaned
