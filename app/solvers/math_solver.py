"""Deterministic math solver for arithmetic word problems."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable

# Safe binary operators allowed in eval

def _normalize_number_text(text: str) -> str:
    """Remove thousands separators so 4,200 parses as one number."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def _format_number(value: float, *, decimals: int | None = None, whole: bool = False) -> str:
    if whole:
        return str(int(round(value)))
    if decimals is not None:
        return f"{value:.{decimals}f}"
    if value == int(value):
        return str(int(value))
    return str(value)


def _parse_output_constraints(text: str) -> tuple[int | None, bool]:
    lower = text.lower()
    if "nearest whole number" in lower or "rounded to the nearest whole number" in lower:
        return None, True
    decimals_match = re.search(r"rounded to (\w+) decimal places?", lower)
    if decimals_match:
        word = decimals_match.group(1)
        mapping = {"one": 1, "two": 2, "three": 3, "four": 4}
        if word in mapping:
            return mapping[word], False
        if word.isdigit():
            return int(word), False
    return None, False


_SAFE_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    """Safely evaluate a simple arithmetic expression."""
    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.BinOp):
            if type(n.op) not in _SAFE_OPS:
                raise ValueError(f"Unsupported operator: {type(n.op)}")
            left = _eval(n.left)
            right = _eval(n.right)
            return _SAFE_OPS[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp):
            if type(n.op) not in _SAFE_OPS:
                raise ValueError(f"Unsupported operator: {type(n.op)}")
            return _SAFE_OPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        # Legacy Python < 3.12 fallback for ast.Num
        if hasattr(n, "n"):
            return float(n.n)  # type: ignore[union-attr]
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        raise ValueError(f"Unsupported expression: {type(n)}")

    return _eval(node.body)


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text."""
    normalized = _normalize_number_text(text)
    found = re.findall(r"-?\d+(?:\.\d+)?", normalized)
    return [float(n) for n in found]


def _has_percent(text: str) -> bool:
    return "%" in text or re.search(r"\bpercent\b", text, re.IGNORECASE) is not None


def solve_math(text: str) -> tuple[str, float] | None:
    """Attempt to solve a math word problem deterministically.

    Returns (answer_string, confidence_estimate) or None if not solvable.
    """
    text_lower = text.lower()
    numbers = _extract_numbers(text)
    if not numbers:
        return None
    decimals, whole = _parse_output_constraints(text)

    growth_match = re.search(
        r"(?:had|has|with)\s+([\d,]+(?:\.\d+)?)\s+.*?grew?\s+(\d+(?:\.\d+)?)\s*%\s+month over month.*?"
        r"after\s+(\d+)\s+months?",
        text_lower,
        re.DOTALL,
    )
    if growth_match:
        start = float(_normalize_number_text(growth_match.group(1)))
        rate = float(growth_match.group(2)) / 100
        months = int(growth_match.group(3))
        result = start * ((1 + rate) ** months)
        return (_format_number(result, decimals=decimals, whole=whole), 1.0)

    discount_match = re.search(
        r"(?:price|cost|is)\s+(?:of\s+[^$]*)?\$?([\d,]+(?:\.\d+)?).*?"
        r"discounted by\s+(\d+(?:\.\d+)?)\s*%.*?"
        r"(?:coupon|additional|then).*?"
        r"(?:reduces?|subtract|minus|by)\s+(?:the\s+\w+\s+price\s+by\s+)?\$?([\d,]+(?:\.\d+)?)",
        text_lower,
        re.DOTALL,
    )
    if discount_match:
        price = float(_normalize_number_text(discount_match.group(1)))
        pct = float(discount_match.group(2))
        coupon = float(_normalize_number_text(discount_match.group(3)))
        result = price * (1 - pct / 100) - coupon
        return (_format_number(result, decimals=decimals, whole=whole), 1.0)


    # Direct arithmetic expression (e.g., "what is 5 + 3?")
    expr_match = re.search(r"what is ([\d\s+\-*/%.()]+)\?", text_lower)
    if expr_match:
        expr = expr_match.group(1).replace(" ", "")
        try:
            result = _safe_eval(expr)
            return (str(int(result) if result == int(result) else result), 1.0)
        except Exception:
            pass

    # Percentage patterns: "X% of Y"
    percent_of_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s+of\s+([\d,]+(?:\.\d+)?)", text_lower)
    if percent_of_match:
        pct = float(percent_of_match.group(1))
        base = float(_normalize_number_text(percent_of_match.group(2)))
        result = base * (pct / 100)
        return (str(int(result) if result == int(result) else result), 1.0)

    # Word problem with "sell/sold/uses/remains/left"
    if _has_percent(text) and len(numbers) >= 2:
        base_match = re.search(
            r"(?:has|have|starts? with|started with|store has|warehouse starts with)\s+([\d,]+(?:\.\d+)?)",
            text_lower,
        )
        base = float(_normalize_number_text(base_match.group(1))) if base_match else numbers[0]

        pct_match = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s+(?:are\s+)?(?:shipped|sold|removed|used)",
            text_lower,
        )
        if not pct_match:
            pct_match = re.search(r"(?:sell|sold|sells?|ship|shipped)\s+(\d+(?:\.\d+)?)\s*%", text_lower)
        if pct_match:
            pct = float(pct_match.group(1))
            remaining = base * (1 - pct / 100)
            after_pct = text_lower[pct_match.end():]
            extra_match = re.search(
                r"(?:another|additional|then|and)\s+(\d+(?:\.\d+)?)\b",
                after_pct,
            )
            if extra_match:
                remaining -= float(extra_match.group(1))
            if decimals is None and not whole and re.search(r"\b(boxes|items)\b", text_lower):
                remaining = round(remaining)
                whole = True
            return (_format_number(remaining, decimals=decimals, whole=whole), 0.95)

    # Simple subtraction/addition with "remain/left/total"
    if re.search(r"\b(how many remain|how many are left|how many total|what is the total)\b", text_lower):
        if len(numbers) >= 2:
            # Heuristic: if words like "sell", "remove", "subtract", "use" appear, subtract
            if re.search(r"\\b(sell|sold|ship|shipped|remove|removed|used|spend|subtract|give away)\\b", text_lower):
                result = numbers[0]
                for n in numbers[1:]:
                    result -= n
                return (str(int(result) if result == int(result) else result), 0.9)
            # Otherwise add
            result = sum(numbers)
            return (str(int(result) if result == int(result) else result), 0.9)

    # Multiplication: "each has X, there are Y, how many total?"
    if re.search(r"\b(each|per|every)\b.*\b(how many total| altogether)\b", text_lower):
        if len(numbers) >= 2:
            result = 1
            for n in numbers:
                result *= n
            return (str(int(result) if result == int(result) else result), 0.85)

    return None
