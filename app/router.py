"""Rule-based task classifier — no LLM calls."""

from __future__ import annotations

import re

from app.fireworks.models import OutputShape, TaskCategory, TaskItem
from app.shape_detector import detect_output_shape

_SUMMARIZE_PATTERNS = (
    r"\bsummari[sz]e\b",
    r"\bin one sentence\b",
    r"\bcondense\b",
    r"\btl;?dr\b",
    r"\bshorten\b",
)

_NER_PATTERNS = (
    r"\bextract\b.*\bentit",
    r"\bnamed entit",
    r"\bner\b",
    r"\blabel\b.*\b(person|org|location|date)\b",
)

_SENTIMENT_PATTERNS = (
    r"\bsentiment\b",
    r"\bpositive\b.*\bnegative\b",
    r"\bclassify\b.*\breview\b",
    r"\btone of\b",
    r"\bhow does the (author|reader|user|customer|reviewer) feel\b",
    r"\bfeel\b.*\breview\b",
    r"\breview\b.*\bfeel\b",
    r"\bclassify\b.*\bsentiment\b",
)

_CODEGEN_PATTERNS = (
    r"\bwrite a python function\b",
    r"\bwrite a function\b",
    r"\bimplement a function\b",
    r"\bcreate a function\b",
    r"\bwrite code\b",
)

_DEBUG_PATTERNS = (
    r"\bbug\b",
    r"\bfix\b.*\bcode\b",
    r"\bdebug\b",
    r"\bhas a bug\b",
    r"\bcorrected implementation\b",
    r"\bfind and fix\b",
)

_MATH_PATTERNS = (
    r"\bcalculate\b",
    r"\bhow many\b",
    r"\bwhat is \d",
    r"\bpercent",
    r"\barithmetic\b",
    r"\bremain\b",
    r"\btotal cost\b",
    r"\b\d+\s*[\+\-\*/%]",
    r"\bsells?\s+\d+%",
)

_LOGIC_PATTERNS = (
    r"\bwho owns\b",
    r"\bwhich\b.*\b(satisfies|owns|has)\b",
    r"\blogic puzzle\b",
    r"\bdeduce\b",
    r"\bconstraint\b",
    r"\beach own(s)? a different\b",
    r"\bdoes not own\b",
    r"\bthree friends\b",
    r"\bpuzzle\b",
)

_STRUCTURED_WRITING_PATTERNS = (
    r"\bproposal\b",
    r"\bsections\b.*\btitled\b",
    r"\bmust\s+contain\b.*\bsections?\b",
    r"\bExecutive\s+Summary\b",
    r"\bArchitecture\s+Recommendation\b",
    r"\bRollback\s+Plan\b",
    r"\bMonitoring\s+Metrics\b",
    r"\bRouting\s+Strategy\b",
    r"\bbenchmark\s+results\b.*\brecommend\b",
    r"\brecommend\b.*\bbenchmark\b",
)

_FACTUAL_EXPLAIN_PATTERNS = (
    r"\bexplain\b",
    r"\bwhat is a\b",
    r"\bwhat are\b",
)

_STRUCTURED_EXTRACTION_PATTERNS = (
    r"\breturn\s+.*\bjson\s+object\b",
    r"\bextract\b.*\bkeys\b",
    r"\bwith\s+keys\b",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _detect_family(prompt: str) -> TaskCategory:
    """Detect task family using keyword heuristics."""
    lower = prompt.lower()

    if _matches_any(lower, _SENTIMENT_PATTERNS):
        return TaskCategory.SENTIMENT
    if _matches_any(lower, _DEBUG_PATTERNS):
        return TaskCategory.DEBUGGING
    if _matches_any(lower, _CODEGEN_PATTERNS):
        return TaskCategory.CODE_GENERATION
    if _matches_any(lower, _FACTUAL_EXPLAIN_PATTERNS):
        return TaskCategory.FACTUAL
    if _matches_any(lower, _MATH_PATTERNS):
        return TaskCategory.MATH
    if _matches_any(lower, _LOGIC_PATTERNS):
        return TaskCategory.LOGIC
    if _matches_any(lower, _SUMMARIZE_PATTERNS):
        return TaskCategory.SUMMARIZATION
    if _matches_any(lower, _NER_PATTERNS):
        return TaskCategory.NER

    return TaskCategory.FACTUAL


def classify_task(task: TaskItem) -> TaskCategory:
    """Classify a task into a handler category using two-phase routing."""
    prompt = task.prompt.lower()
    shape = detect_output_shape(task.prompt)
    family = _detect_family(prompt)

    # Phase 1: shape-driven routing for structured outputs
    if shape == OutputShape.JSON_OBJECT or shape == OutputShape.JSON_ARRAY:
        # If the family is NER, keep NER for true entity extraction regardless of JSON shape
        if family == TaskCategory.NER:
            return TaskCategory.NER
        return TaskCategory.STRUCTURED_EXTRACTION

    if shape == OutputShape.EXACT_SECTIONS:
        # Code/debug tasks with sections stay in their family (e.g. review code with sections)
        if family in (TaskCategory.CODE_GENERATION, TaskCategory.DEBUGGING):
            return family
        return TaskCategory.STRUCTURED_WRITING

    if shape == OutputShape.EXACT_BULLETS:
        # Tasks asking for bullet points that look like explanations or summaries
        # should route to summarization so bullet preservation works
        if family == TaskCategory.FACTUAL and _matches_any(prompt, (r"\bexplain\b",)):
            return TaskCategory.SUMMARIZATION

    if shape == OutputShape.CODE_ONLY:
        if family == TaskCategory.DEBUGGING:
            return TaskCategory.DEBUGGING
        return TaskCategory.CODE_GENERATION

    # Phase 2: structured-writing family patterns (proposals, benchmark recommendations)
    if _matches_any(prompt, _STRUCTURED_WRITING_PATTERNS):
        if family not in (TaskCategory.CODE_GENERATION, TaskCategory.DEBUGGING, TaskCategory.MATH, TaskCategory.LOGIC):
            return TaskCategory.STRUCTURED_WRITING

    if _matches_any(prompt, _STRUCTURED_EXTRACTION_PATTERNS):
        if family != TaskCategory.NER:
            return TaskCategory.STRUCTURED_EXTRACTION

    # Phase 3: default family routing
    return family
