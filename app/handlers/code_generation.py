"""Code generation handler."""

from __future__ import annotations

import re

from app.fireworks.models import TaskItem
from app.handlers.base import BaseHandler
from app.utils.text_utils import extract_first_code_block, is_valid_python, strip_cot


class CodeGenerationHandler(BaseHandler):
    prompt_file = "code_generation.txt"

    @property
    def max_tokens(self) -> int:
        return 1024

    @property
    def preferred_model_tags(self) -> list[str]:
        return ["code", "kimi", "glm", "deepseek"]

    def category_name(self) -> str:
        return "code_generation"

    def post_process(self, text: str, task: TaskItem) -> str:
        cleaned = strip_cot(text)
        code = extract_first_code_block(cleaned)
        if not is_valid_python(code):
            # Fallback: try to extract any def block more leniently
            def_match = re.search(r"(def\s+\w+\([^)]*\):.*?(?=\n\S|$))", cleaned, re.DOTALL)
            if def_match:
                candidate = def_match.group(1).strip()
                if is_valid_python(candidate):
                    code = candidate
            # Fallback 2: try to extract the first def block line-by-line
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
                        if indent <= base_indent and line.strip():
                            if line.strip().startswith("def "):
                                break
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
        return code if code else cleaned
