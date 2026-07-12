"""Answer refinement before backend escalation."""

from __future__ import annotations

import json
import re

from app.fireworks.models import TaskCategory, TaskItem
from app.solvers.codegen_solver import solve_codegen
from app.solvers.debug_solver import try_fix_debug_task
from app.solvers.fallback import solver_fallback
from app.solvers.summarization_solver import solve_summarization
from app.utils.json_utils import validate_json_string
from app.utils.text_utils import extract_first_code_block, extract_last_sentence, is_valid_python
from app.utils.validators import fallback_summary, is_truncated_summary


def build_refinement_prompt(
    original_user: str,
    previous_answer: str,
    retry_reason: str | None,
    category: TaskCategory,
) -> str:
    """Build an improved prompt for a single retry attempt."""
    reason = retry_reason or "validation_failed"
    hints = {
        TaskCategory.SUMMARIZATION: "Return exactly one complete sentence with no markdown.",
        TaskCategory.NER: "Return only valid JSON with entity objects.",
        TaskCategory.CODE_GENERATION: "Return only runnable Python code with the required function.",
        TaskCategory.DEBUGGING: "Return fixed code then a one-sentence explanation.",
        TaskCategory.MATH: "Return only the final numeric answer.",
        TaskCategory.LOGIC: "Return only the final answer, no reasoning.",
        TaskCategory.STRUCTURED_EXTRACTION: "Return only valid JSON matching the required keys.",
        TaskCategory.FACTUAL: "Give a complete factual answer in plain text.",
        TaskCategory.SENTIMENT: "Return the sentiment label and justification if requested.",
        TaskCategory.STRUCTURED_WRITING: "Include all required sections with headers.",
    }
    hint = hints.get(category, "Fix the answer format and be concise.")
    return (
        f"{original_user}\n\n"
        f"Previous attempt failed ({reason}). {hint}\n"
        f"Previous answer: {previous_answer[:300]}"
    )


def refine_answer(
    answer: str,
    task: TaskItem,
    category: TaskCategory,
    retry_reason: str | None,
) -> str | None:
    """Attempt local refinement without calling an LLM."""
    if category == TaskCategory.SUMMARIZATION:
        return _refine_summary(answer, task, retry_reason)
    if category in (TaskCategory.NER, TaskCategory.STRUCTURED_EXTRACTION):
        return _refine_json(answer)
    if category == TaskCategory.CODE_GENERATION:
        return _refine_code(answer, task)
    if category == TaskCategory.DEBUGGING:
        fixed = try_fix_debug_task(task.prompt)
        if fixed:
            return fixed
        return _refine_code(answer, task)
    if category == TaskCategory.MATH:
        return solver_fallback(task, category)
    return None


def _refine_summary(answer: str, task: TaskItem, retry_reason: str | None) -> str | None:
    text = answer.strip()
    if is_truncated_summary(text) or retry_reason == "truncated_summary":
        fb = fallback_summary(task.prompt)
        if fb:
            return fb
    local = solve_summarization(task.prompt)
    if local:
        return local[0]
    cleaned = extract_last_sentence(text)
    if cleaned and not is_truncated_summary(cleaned):
        return cleaned
    return None


def _refine_json(answer: str) -> str | None:
    text = answer.strip()
    valid, parsed = validate_json_string(text)
    if valid:
        return json.dumps(parsed, ensure_ascii=False)

    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            valid, parsed = validate_json_string(match.group(0))
            if valid:
                return json.dumps(parsed, ensure_ascii=False)
    return None


def _refine_code(answer: str, task: TaskItem) -> str | None:
    code = extract_first_code_block(answer)
    if code and is_valid_python(code):
        return code
    local = solve_codegen(task.prompt)
    if local:
        return local[0]
    return code if code else None
