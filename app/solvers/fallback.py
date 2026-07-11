"""Last-resort deterministic answers when Fireworks API calls fail."""

from __future__ import annotations

from app.fireworks.models import TaskCategory, TaskItem
from app.solvers.debug_solver import try_fix_debug_task
from app.solvers.codegen_solver import solve_codegen
from app.solvers.factual_solver import solve_factual
from app.solvers.logic_solver import solve_logic
from app.solvers.math_solver import solve_math
from app.solvers.ner_solver import solve_ner
from app.solvers.sentiment_solver import build_sentiment_answer
from app.solvers.summarization_solver import solve_summarization


def solver_fallback(task: TaskItem, category: TaskCategory) -> str | None:
    """Return a best-effort answer from local solvers, or None."""
    prompt = task.prompt
    result: tuple[str, float] | None = None

    if category == TaskCategory.MATH:
        result = solve_math(prompt)
    elif category == TaskCategory.SENTIMENT:
        result = build_sentiment_answer(prompt) or None
    elif category == TaskCategory.SUMMARIZATION:
        result = solve_summarization(prompt)
    elif category == TaskCategory.NER:
        result = solve_ner(prompt)
    elif category == TaskCategory.LOGIC:
        result = solve_logic(prompt)
    elif category == TaskCategory.FACTUAL:
        result = solve_factual(prompt)
    elif category == TaskCategory.CODE_GENERATION:
        result = solve_codegen(prompt)
    elif category == TaskCategory.DEBUGGING:
        fixed = try_fix_debug_task(prompt)
        return fixed

    if result:
        return result[0]
    return None
