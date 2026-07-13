"""Dynamic backend selection based on category, runtime, and verification."""

from __future__ import annotations

from enum import Enum

from app.config import Settings, get_settings
from app.fireworks.models import TaskCategory, TaskItem
from app.runtime_manager import RuntimeManager
from app.solvers.codegen_solver import solve_codegen
from app.solvers.debug_solver import try_fix_debug_task
from app.solvers.factual_solver import solve_factual
from app.solvers.logic_solver import solve_logic
from app.solvers.math_solver import solve_math
from app.solvers.ner_solver import solve_ner
from app.solvers.sentiment_solver import build_sentiment_answer
from app.solvers.summarization_solver import solve_summarization


class BackendType(str, Enum):
    DETERMINISTIC = "deterministic"
    LOCAL = "local"
    FIREWORKS = "fireworks"
    FIREWORKS_STRONG = "fireworks_strong"


# Category-aware gates: math/logic/code require independently-checked solver
# confidence (1.0). Soft extractive categories allow 0.9. Brittle labels stay strict.
_DEFAULT_DETERMINISTIC_MIN_CONFIDENCE = 0.95


def deterministic_min_confidence(category: TaskCategory) -> float:
    """Minimum solver confidence to accept a deterministic backend answer."""
    if category in (TaskCategory.SENTIMENT, TaskCategory.NER):
        return 0.95
    if category in (TaskCategory.SUMMARIZATION, TaskCategory.FACTUAL):
        return 0.9
    if category in (
        TaskCategory.MATH,
        TaskCategory.LOGIC,
        TaskCategory.CODE_GENERATION,
        TaskCategory.DEBUGGING,
    ):
        # Only accept when the solver itself marks an independent check (1.0).
        return 1.0
    return _DEFAULT_DETERMINISTIC_MIN_CONFIDENCE


def _solver_result(category: TaskCategory, prompt: str) -> tuple[str, float] | None:
    """Return solver output with confidence, or None when unavailable."""
    if category == TaskCategory.MATH:
        return solve_math(prompt)
    if category == TaskCategory.SENTIMENT:
        return build_sentiment_answer(prompt)
    if category == TaskCategory.SUMMARIZATION:
        return solve_summarization(prompt)
    if category == TaskCategory.NER:
        return solve_ner(prompt)
    if category == TaskCategory.LOGIC:
        return solve_logic(prompt)
    if category == TaskCategory.FACTUAL:
        return solve_factual(prompt)
    if category == TaskCategory.CODE_GENERATION:
        return solve_codegen(prompt)
    if category == TaskCategory.DEBUGGING:
        fixed = try_fix_debug_task(prompt)
        return (fixed, 1.0) if fixed else None
    return None


def has_deterministic_solver(
    category: TaskCategory,
    task: TaskItem,
    *,
    min_confidence: float | None = None,
) -> bool:
    """Return True when a high-confidence local solver can answer reliably."""
    threshold = (
        deterministic_min_confidence(category)
        if min_confidence is None
        else min_confidence
    )
    result = _solver_result(category, task.prompt)
    return result is not None and result[1] >= threshold


def is_local_suitable(category: TaskCategory, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return category.value in settings.local_category_set


def select_backend_order(
    category: TaskCategory,
    task: TaskItem,
    runtime: RuntimeManager,
    *,
    settings: Settings | None = None,
) -> list[BackendType]:
    """Return ordered backends: deterministic → local → fireworks → strong."""
    settings = settings or get_settings()
    order: list[BackendType] = []

    if has_deterministic_solver(category, task):
        order.append(BackendType.DETERMINISTIC)

    if is_local_suitable(category, settings) and settings.enable_local_model:
        if runtime.allow_local():
            order.append(BackendType.LOCAL)

    if runtime.is_emergency():
        return order or [BackendType.FIREWORKS]

    order.append(BackendType.FIREWORKS)

    if runtime.allow_escalation() and category in (
        TaskCategory.CODE_GENERATION,
        TaskCategory.DEBUGGING,
        TaskCategory.MATH,
        TaskCategory.LOGIC,
        TaskCategory.STRUCTURED_EXTRACTION,
        TaskCategory.STRUCTURED_WRITING,
    ):
        order.append(BackendType.FIREWORKS_STRONG)

    return _dedupe(order)


def next_escalation(
    current: BackendType,
    category: TaskCategory,
    task: TaskItem,
    runtime: RuntimeManager,
) -> BackendType | None:
    """Return the next backend after escalation, or None."""
    order = select_backend_order(category, task, runtime)
    try:
        idx = order.index(current)
    except ValueError:
        return None
    if idx + 1 < len(order):
        return order[idx + 1]
    return None


def _dedupe(items: list[BackendType]) -> list[BackendType]:
    seen: set[BackendType] = set()
    out: list[BackendType] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
