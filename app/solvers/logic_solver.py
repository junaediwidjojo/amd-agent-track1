"""Deterministic logic puzzle solver for small constraint problems."""

from __future__ import annotations

import itertools
import re
from typing import Any


def _extract_names(text: str) -> list[str]:
    """Heuristic: extract capitalized words that are likely person names."""
    # Look for patterns like "Three friends, A, B, and C," or "A, B, and C each own"
    comma_list_match = re.search(
        r"([A-Z][a-zA-Z]+(?:,\s+[A-Z][a-zA-Z]+)*?(?:,\s+and\s+[A-Z][a-zA-Z]+))",
        text,
    )
    if comma_list_match:
        raw = comma_list_match.group(1)
        names = [n.strip() for n in raw.replace(" and ", ", ").split(",") if n.strip()]
        return names

    # Fallback: all capitalized words that aren't sentence-starting common words
    words = re.findall(r"\b[A-Z][a-z]+\b", text)
    # Remove common sentence starters and number words
    common = {
        "The", "A", "An", "This", "That", "It", "There", "What", "How", "Who", "If", "In", "On", "At", "To", "For", "Of", "With", "By", "From", "As", "But", "Or", "Nor", "So", "Yet",
        "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    }
    names = [w for w in words if w not in common]
    # Deduplicate preserving order
    seen: set[str] = set()
    result = []
    for n in names:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


def _extract_values(text: str, names: list[str]) -> list[str]:
    """Extract possible values (pets, colors, etc.) from the text."""
    # Look for list after a colon or "different X:"
    list_match = re.search(r"different\s+\w+(?:\s+\w+)*[:\s]+([a-z,\s]+)", text, re.IGNORECASE)
    if list_match:
        raw = list_match.group(1)
        vals = [v.strip() for v in raw.replace(" and ", ", ").split(",") if v.strip()]
        return vals

    # Fallback: look for lowercase words near the end of the sentence that aren't common stop words
    # Try to find words after "each own a different" or similar
    # Extract the last few lowercase words that could be values
    words = re.findall(r"\b[a-z]+\b", text)
    # Heuristic: if names are known, values are likely in a list near them
    # Try to find a comma-separated lowercase list
    raw_lists = re.findall(r"([a-z]+(?:,\s+[a-z]+)*?(?:,\s+and\s+[a-z]+))", text)
    if raw_lists:
        raw = raw_lists[-1]
        vals = [v.strip() for v in raw.replace(" and ", ", ").split(",") if v.strip()]
        if len(vals) == len(names):
            return vals

    return []


def _parse_constraints(text: str, names: list[str], values: list[str]) -> list[dict[str, Any]]:
    """Parse simple constraints from text."""
    constraints: list[dict[str, Any]] = []
    text_lower = text.lower()

    for name in names:
        nl = name.lower()
        # "X does not own Y" / "X does not do Y"
        not_match = re.search(
            rf"{re.escape(nl)}\s+does\s+not\s+(?:own|have|like|drive|live|do)\s+(?:the\s+)?([a-z]+)",
            text_lower,
        )
        if not_match:
            val = not_match.group(1)
            if val in [v.lower() for v in values]:
                constraints.append({"type": "neq", "name": name, "value": val})

        # "X owns Y" / "X does Y" / "X has Y"
        yes_match = re.search(
            rf"{re.escape(nl)}\s+(?:owns|has|likes|drives|lives|does)\s+(?:the\s+)?([a-z]+)",
            text_lower,
        )
        if yes_match:
            val = yes_match.group(1)
            if val in [v.lower() for v in values]:
                constraints.append({"type": "eq", "name": name, "value": val})

    # Fallback: scan for name prefixes (e.g., "Sam" when full name is "Samuel")
    for name in names:
        short = name.lower()[:3]
        if len(short) < 3:
            continue
        # Only use prefix if it appears in text but full name doesn't
        if short in text_lower and name.lower() not in text_lower:
            not_match = re.search(
                rf"{re.escape(short)}\s+does\s+not\s+(?:own|have|like|drive|live|do)\s+(?:the\s+)?([a-z]+)",
                text_lower,
            )
            if not_match:
                val = not_match.group(1)
                if val in [v.lower() for v in values]:
                    constraints.append({"type": "neq", "name": name, "value": val})
            yes_match = re.search(
                rf"{re.escape(short)}\s+(?:owns|has|likes|drives|lives|does)\s+(?:the\s+)?([a-z]+)",
                text_lower,
            )
            if yes_match:
                val = yes_match.group(1)
                if val in [v.lower() for v in values]:
                    constraints.append({"type": "eq", "name": name, "value": val})

    # All different constraint
    if re.search(r"each\s+(?:own|have|like|drive|live)\s+a\s+different", text_lower):
        constraints.append({"type": "all_different"})

    return constraints


def _solve(names: list[str], values: list[str], constraints: list[dict[str, Any]]) -> dict[str, str] | None:
    """Brute-force solve by trying all permutations."""
    if len(names) != len(values):
        return None
    for perm in itertools.permutations(values):
        assignment = dict(zip(names, perm))
        ok = True
        for c in constraints:
            if c["type"] == "eq":
                if assignment.get(c["name"], "").lower() != c["value"].lower():
                    ok = False
                    break
            elif c["type"] == "neq":
                if assignment.get(c["name"], "").lower() == c["value"].lower():
                    ok = False
                    break
            elif c["type"] == "all_different":
                if len(set(v.lower() for v in assignment.values())) != len(values):
                    ok = False
                    break
        if ok:
            return assignment
    return None


def solve_logic(text: str) -> tuple[str, float] | None:
    """Attempt to solve a small logic puzzle deterministically.

    Returns (answer_string, confidence) or None.
    """
    names = _extract_names(text)
    if len(names) < 2:
        return None

    values = _extract_values(text, names)
    if len(values) != len(names):
        return None

    constraints = _parse_constraints(text, names, values)
    if not constraints:
        return None

    solution = _solve(names, values, constraints)
    if not solution:
        return None

    # Try to answer the explicit question
    # Pattern 1: "Who owns the cat?"
    question_match = re.search(r"who\s+(?:owns|has|likes|drives|lives)\s+(?:the\s+)?([a-z]+)\?", text, re.IGNORECASE)
    if question_match:
        target_value = question_match.group(1).lower()
        for name, val in solution.items():
            if val.lower() == target_value:
                return (name, 1.0)

    # Pattern 2: "What does [Name] do?" or "Who does [activity]?"
    question_match2 = re.search(r"what\s+does\s+([a-z]+)\s+do\?", text, re.IGNORECASE)
    if question_match2:
        target_name = question_match2.group(1).title()
        for name, val in solution.items():
            if name.lower() == target_name.lower():
                return (val, 1.0)

    # If no explicit question, return the full mapping
    answer = ", ".join(f"{n}: {v}" for n, v in solution.items())
    return (answer, 0.9)
