"""Deterministic math solver for arithmetic word problems."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable

# Safe binary operators allowed in eval
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
    found = re.findall(r"-?\d+(?:\.\d+)?", text)
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
    percent_of_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)", text_lower)
    if percent_of_match:
        pct = float(percent_of_match.group(1))
        base = float(percent_of_match.group(2))
        result = base * (pct / 100)
        return (str(int(result) if result == int(result) else result), 1.0)

    # Word problem with "sell/sold/uses/remains/left"
    if _has_percent(text) and len(numbers) >= 2:
        # Pattern: start with X, sell Y%, then maybe sell Z more, how many remain?
        # Find the base amount (usually the first number after "has" or at start)
        base_match = re.search(r"(?:has|have|started with|store has)\s+(\d+(?:\.\d+)?)", text_lower)
        base = float(base_match.group(1)) if base_match else numbers[0]

        # Find percentage sold
        pct_match = re.search(r"(?:sell|sold|sells?)\s+(\d+(?:\.\d+)?)\s*%", text_lower)
        if pct_match:
            pct = float(pct_match.group(1))
            remaining = base * (1 - pct / 100)
            # Look for an additional absolute amount after the percentage (e.g., "and 60 more" or "and 40 in the afternoon")
            after_pct = text_lower[pct_match.end():]
            extra_match = re.search(r"\band\s+(\d+(?:\.\d+)?)\b", after_pct)
            if extra_match:
                remaining -= float(extra_match.group(1))
            result = remaining
            return (str(int(result) if result == int(result) else result), 0.95)

    # Simple subtraction/addition with "remain/left/total"
    if re.search(r"\b(how many remain|how many are left|how many total|what is the total)\b", text_lower):
        if len(numbers) >= 2:
            # Heuristic: if words like "sell", "remove", "subtract", "use" appear, subtract
            if re.search(r"\b(sell|sold|remove|used|spend|subtract|give away)\b", text_lower):
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
