"""Aggressive chain-of-thought stripping and text extraction utilities."""

from __future__ import annotations

import ast
import re


def strip_cot(text: str) -> str:
    """Remove obvious chain-of-thought artifacts without stripping actual answers.

    Conservative: only strips markdown headers, numbered reasoning steps with bold
    labels, and explicit reasoning-marker bullet points. Preserves normal sentences
    that happen to start with words like "Because", "Actually", etc.
    """
    lines = text.splitlines()
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip markdown headers
        if re.match(r"^#{1,6}\s+", stripped):
            continue
        # Skip numbered markdown steps like "6.  **"
        if re.match(r"^\d+\.\s+\*\*?", stripped):
            continue
        if re.match(r"^\*\s+", stripped):
            continue
        # Skip numbered analysis steps like "1. **Analyze:**" or "2.  **Draft:**"
        if re.match(r"^\d+\.\s+\*\*.*\*\*$", stripped):
            continue
        # Skip bullet points that are pure reasoning markers
        if re.match(
            r"^[-*]\s+(Step\s+\d+:|Reasoning:|Thinking:|Draft\s+\d+:|Option\s+\d+:|Note:|Hint:)\s*",
            stripped,
            re.IGNORECASE,
        ):
            continue
        # Skip bare numbered lines like "5."
        if re.match(r"^\d+\.\s*$", stripped):
            continue
        # Skip lines that are just markdown formatting artifacts
        if stripped in {"*", "-", "```", "```python", "```json", "```text"}:
            continue
        cleaned_lines.append(line)
    result = "\n".join(cleaned_lines).strip()
    # Remove markdown fences from boundaries only (not inline backticks)
    result = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", result, flags=re.MULTILINE)
    return result.strip()


def extract_first_code_block(text: str) -> str:
    """Extract the first fenced code block, or the first def/class block, or cleaned text.

    Stops collecting code when it hits prose after the block (blank-line separator
    or lower-indent non-code line).
    """
    # Try fenced blocks and prefer the LAST one (model may reason before final code)
    matches = list(re.finditer(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()

    # Find the first def/class block, stopping at obvious prose.
    # Import blocks are intentionally excluded here because the primary use-case
    # is extracting a single runnable function for code-gen / debugging tasks.
    lines = text.splitlines()
    code_lines: list[str] = []
    collecting = False
    base_indent: int | None = None

    for i, line in enumerate(lines):
        if not collecting:
            if re.match(r"^\s*(def |class )", line):
                collecting = True
                base_indent = len(line) - len(line.lstrip())
                code_lines.append(line)
            continue

        # We are collecting
        if line.strip() == "":
            # Peek ahead: if the next non-blank line is prose at lower/equal indent, stop
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line.strip() == "":
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if base_indent is not None and next_indent <= base_indent:
                    code_keywords = (
                        r"^\s*(def |class |return |if |for |while |try |except |with |"
                        r"elif |else |pass |break |continue |raise |assert |print |"
                        r"\w+\s*=)"
                    )
                    if not re.match(code_keywords, next_line):
                        collecting = False
                        break
                break
            if not collecting:
                break
            code_lines.append(line)
            continue

        indent = len(line) - len(line.lstrip())

        # Stop if we hit a new def/class at same or lower indent
        if (
            base_indent is not None
            and indent <= base_indent
            and re.match(r"^\s*(def |class )", line)
        ):
            break

        # Stop if we hit prose at lower indent (not code-like)
        if base_indent is not None and indent < base_indent:
            code_keywords = (
                r"^\s*(return |if |for |while |try |except |with |elif |else |"
                r"pass |break |continue |raise |assert |print |\w+\s*=)"
            )
            if not re.match(code_keywords, line):
                break

        code_lines.append(line)

    if code_lines:
        candidate = "\n".join(code_lines).strip()
        if is_valid_python(candidate):
            return candidate

    return text.strip()


def extract_last_sentence(text: str) -> str:
    """Extract the last sentence from text, useful for summaries."""
    # Remove markdown fences first
    cleaned = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        return sentences[-1]
    return cleaned


def is_valid_python(code: str) -> bool:
    """Check if code is syntactically valid Python using ast.parse."""
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_final_answer(text: str) -> str:
    """For short-answer tasks, extract the final line or short phrase."""
    cleaned = strip_cot(text)
    reasoning_prefixes = (
        "the user wants",
        "i need to",
        "let me",
        "wait,",
        "but wait",
        "from the",
        "puzzle:",
    )
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return cleaned
    for line in reversed(lines):
        lower = line.lower()
        if any(lower.startswith(prefix) for prefix in reasoning_prefixes):
            continue
        if len(line) < 200:
            return line
    return cleaned
