"""Output-quality verification engine — validates answers, not routing patterns."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from app.fireworks.models import TaskCategory, TaskItem
from app.solvers.logic_solver import solve_logic
from app.solvers.math_solver import solve_math
from app.utils.json_utils import validate_json_string
from app.utils.validators import is_truncated_summary

_GENERIC_FAILURE_PHRASES = (
    "unable to process",
    "no answer generated",
    "i cannot",
    "i don't know",
    "as an ai",
)


@dataclass
class VerificationResult:
    """Result of validating a candidate answer."""

    passed: bool
    confidence: float
    retry_reason: str | None = None
    backend_recommendation: str | None = None


def verify_answer(
    answer: str,
    task: TaskItem,
    category: TaskCategory,
) -> VerificationResult:
    """Validate output quality for the given category."""
    validator = _VALIDATORS.get(category, _verify_factual)
    return validator(answer, task)


def _fail(
    reason: str,
    confidence: float = 0.0,
    backend: str | None = "fireworks",
) -> VerificationResult:
    return VerificationResult(
        passed=False,
        confidence=confidence,
        retry_reason=reason,
        backend_recommendation=backend,
    )


def _pass(confidence: float = 0.9) -> VerificationResult:
    return VerificationResult(passed=True, confidence=confidence)


def _is_generic_failure(text: str) -> bool:
    lower = text.strip().lower()
    return any(phrase in lower for phrase in _GENERIC_FAILURE_PHRASES)


def is_usable_answer(answer: str) -> bool:
    """True when an answer is non-empty and not a runtime/error placeholder."""
    text = (answer or "").strip()
    if not text or text == "No answer generated.":
        return False
    return not _is_generic_failure(text)


def _wants_one_sentence(prompt: str) -> bool:
    return bool(re.search(r"\bin one sentence\b|\bexactly one sentence\b", prompt, re.I))


def _wants_bullets(prompt: str) -> bool:
    return bool(re.search(r"\bbullet\s+points?\b", prompt, re.I))


def _extract_required_keys(prompt: str) -> list[str]:
    keys: list[str] = []
    for match in re.finditer(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', prompt):
        keys.append(match.group(1))
    for match in re.finditer(r"\bkeys?\s*:\s*([A-Za-z_,\s]+)", prompt, re.I):
        raw = match.group(1)
        keys.extend(k.strip() for k in raw.split(",") if k.strip())
    return list(dict.fromkeys(keys))


def _extract_function_name(prompt: str) -> str | None:
    match = re.search(
        r"(?:function|def)\s+(?:named\s+)?[`'\"]?(\w+)[`'\"]?",
        prompt,
        re.I,
    )
    if match:
        return match.group(1)
    match = re.search(r"`(\w+)\s*\(", prompt)
    return match.group(1) if match else None


def _verify_summarization(answer: str, task: TaskItem) -> VerificationResult:
    text = answer.strip()
    if not text:
        return _fail("empty_summary")
    if _is_generic_failure(text):
        return _fail("generic_failure", backend="fireworks")
    if re.search(r"#{1,6}\s|\*\*|```", text):
        return _pass(0.72)
    if is_truncated_summary(text):
        return _fail("truncated_summary", confidence=0.45, backend="local")

    if _wants_bullets(task.prompt):
        bullets = [ln for ln in text.splitlines() if ln.strip().startswith(("-", "*"))]
        if not bullets:
            return _fail("missing_bullets", confidence=0.45, backend="fireworks")
        return _pass(0.85)

    if _wants_one_sentence(task.prompt):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) > 3:
            return _fail("not_one_sentence", confidence=0.55, backend="local")
        if len(sentences) > 1:
            return _pass(0.72)
    return _pass(0.88)


def _verify_ner(answer: str, task: TaskItem) -> VerificationResult:
    text = answer.strip()
    valid, parsed = validate_json_string(text)
    if not valid:
        return _fail("invalid_json", confidence=0.1, backend="local")

    if isinstance(parsed, list):
        if not parsed:
            return _fail("empty_entities", confidence=0.2, backend="fireworks")
        for item in parsed:
            if not isinstance(item, dict) or not item:
                return _fail("invalid_entity_schema", confidence=0.3, backend="local")
        return _pass(0.92)

    if isinstance(parsed, dict):
        if not parsed:
            return _fail("empty_json_object", confidence=0.2, backend="fireworks")
        return _pass(0.9)

    return _fail("unexpected_json_type", confidence=0.2, backend="local")


def _verify_code(answer: str, task: TaskItem, *, require_explanation: bool = False) -> VerificationResult:
    text = answer.strip()
    if not text:
        return _fail("empty_code", backend="fireworks")
    if _is_generic_failure(text):
        return _fail("generic_failure", backend="fireworks")

    code = text.split("\n\n")[0] if require_explanation else text
    if not re.search(r"\bdef\s+\w+", code):
        return _fail("no_function", confidence=0.2, backend="fireworks")

    try:
        ast.parse(code)
    except SyntaxError:
        return _fail("syntax_error", confidence=0.1, backend="fireworks")

    if "re." in code and "import re" not in code:
        return _fail("missing_import", confidence=0.3, backend="local")

    fn_name = _extract_function_name(task.prompt)
    if fn_name and not re.search(rf"\bdef\s+{re.escape(fn_name)}\s*\(", code):
        return _fail("missing_required_function", confidence=0.3, backend="fireworks")

    if require_explanation:
        parts = text.split("\n\n", 1)
        if len(parts) < 2 or len(parts[1].strip()) < 15:
            return _fail("missing_explanation", confidence=0.5, backend="local")

    return _pass(0.9)


def _verify_codegen(answer: str, task: TaskItem) -> VerificationResult:
    return _verify_code(answer, task, require_explanation=False)


def _verify_debugging(answer: str, task: TaskItem) -> VerificationResult:
    if re.search(r"\bsections\s+titled\b", task.prompt, re.I):
        if len(answer.strip()) < 50:
            return _fail("incomplete_sections", confidence=0.3, backend="fireworks")
        return _pass(0.85)
    return _verify_code(answer, task, require_explanation=True)


def _verify_math(answer: str, task: TaskItem) -> VerificationResult:
    text = answer.strip()
    if not text:
        return _fail("empty_math", backend="fireworks")
    if _is_generic_failure(text):
        return _fail("generic_failure", backend="fireworks")

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not numbers:
        return _fail("no_numeric_answer", confidence=0.2, backend="fireworks")

    solver = solve_math(task.prompt)
    if solver:
        expected, solver_conf = solver[0].strip().replace(",", ""), solver[1]
        actual = numbers[-1]
        if actual == expected or actual.rstrip("0").rstrip(".") == expected.rstrip("0").rstrip("."):
            # Only treat agreement as lock-in-grade when the solver itself was
            # independently checked (conf 1.0). Heuristic matches stay softer so
            # a deterministic self-check cannot freeze a wrong answer at 0.95.
            if solver_conf >= 1.0:
                return _pass(0.95)
            return _pass(0.78)
        if solver_conf >= 0.95:
            return _fail("solver_mismatch", confidence=0.4, backend="fireworks_strong")
        return _pass(0.72)

    if len(text.split()) > 12:
        return _pass(0.65)
    return _pass(0.75)


def _verify_logic(answer: str, task: TaskItem) -> VerificationResult:
    text = answer.strip()
    if not text:
        return _fail("empty_logic", backend="fireworks")
    if _is_generic_failure(text):
        return _fail("generic_failure", backend="fireworks")
    if re.match(r"(?i)^(wait|but|let me|from the|puzzle|the user)", text):
        return _fail("reasoning_leak", confidence=0.1, backend="fireworks")

    solver = solve_logic(task.prompt)
    if solver:
        expected, solver_conf = solver[0].strip().lower(), solver[1]
        actual = text.lower().replace(" ", "")
        expected_compact = expected.replace(" ", "")
        if actual == expected or expected_compact in actual or actual in expected_compact:
            if solver_conf >= 1.0:
                return _pass(0.95)
            return _pass(0.78)
        if solver_conf >= 0.9:
            return _fail("solver_mismatch", confidence=0.35, backend="fireworks_strong")
        return _pass(0.7)

    if "name-place" in task.prompt.lower().replace(" ", ""):
        matches = re.findall(r"([A-Za-z]+)-(First|Second|Third)", text, re.I)
        if len(matches) < 3:
            return _fail("incomplete_name_place", confidence=0.45, backend="fireworks")
        names = [m[0].lower() for m in matches]
        if len(set(names)) != len(names):
            return _fail("duplicate_names", confidence=0.35, backend="fireworks")
        return _pass(0.9)

    words = text.replace(",", " ").split()
    if len(words) > 8:
        return _pass(0.65)
    return _pass(0.8)


def _verify_structured_extraction(answer: str, task: TaskItem) -> VerificationResult:
    text = answer.strip()
    valid, parsed = validate_json_string(text)
    if not valid:
        return _fail("invalid_json", confidence=0.1, backend="fireworks")

    required = _extract_required_keys(task.prompt)
    if required and isinstance(parsed, dict):
        missing = [k for k in required if k not in parsed]
        if missing:
            return _fail(f"missing_keys:{','.join(missing)}", confidence=0.3, backend="fireworks")

    if isinstance(parsed, (dict, list)) and not parsed:
        return _fail("empty_json", confidence=0.2, backend="fireworks")
    return _pass(0.88)


def _verify_factual(answer: str, task: TaskItem) -> VerificationResult:
    text = answer.strip()
    if not text or len(text) < 2:
        return _fail("empty_factual", backend="fireworks")
    if _is_generic_failure(text):
        return _fail("generic_failure", backend="fireworks")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return _pass(0.6)
    if len(text.split()) < 3 and re.search(r"\bexplain\b", task.prompt, re.I):
        return _fail("too_short_explanation", confidence=0.45, backend="fireworks")
    if len(text) > 1500:
        return _fail("too_long", confidence=0.5, backend="local")
    return _pass(0.82)


def _verify_sentiment(answer: str, task: TaskItem) -> VerificationResult:
    text = answer.strip()
    if not text:
        return _fail("empty_sentiment", backend="local")
    lower = text.lower()
    labels = ("positive", "negative", "neutral", "mixed")
    if not any(label in lower for label in labels):
        return _fail("no_sentiment_label", confidence=0.2, backend="local")

    wants_justification = "justify" in task.prompt.lower() or "justification" in task.prompt.lower()
    if wants_justification and len(text) < 20:
        return _fail("missing_justification", confidence=0.4, backend="fireworks")
    return _pass(0.88)


def _verify_structured_writing(answer: str, task: TaskItem) -> VerificationResult:
    text = answer.strip()
    if not text or len(text) < 30:
        return _fail("too_short_writing", backend="fireworks")
    if _is_generic_failure(text):
        return _fail("generic_failure", backend="fireworks")

    section_match = re.search(r"(\d+)\s+sections\s+titled", task.prompt, re.I)
    if section_match:
        required = int(section_match.group(1))
        headers = re.findall(r"^#+\s+.+|^([A-Z][A-Za-z\s]+):", text, re.M)
        if len(headers) < required:
            return _fail("missing_sections", confidence=0.3, backend="fireworks")
    return _pass(0.8)


_VALIDATORS = {
    TaskCategory.SUMMARIZATION: _verify_summarization,
    TaskCategory.NER: _verify_ner,
    TaskCategory.CODE_GENERATION: _verify_codegen,
    TaskCategory.DEBUGGING: _verify_debugging,
    TaskCategory.MATH: _verify_math,
    TaskCategory.LOGIC: _verify_logic,
    TaskCategory.STRUCTURED_EXTRACTION: _verify_structured_extraction,
    TaskCategory.FACTUAL: _verify_factual,
    TaskCategory.SENTIMENT: _verify_sentiment,
    TaskCategory.STRUCTURED_WRITING: _verify_structured_writing,
}
