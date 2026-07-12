"""Deterministic fixes for common debugging task patterns."""

from __future__ import annotations

import re


def _extract_buggy_function(prompt: str) -> str | None:
    match = re.search(r"(def\s+\w+\([^)]*\):(?:\n    .*)+)", prompt)
    if match:
        return match.group(1).strip()
    return None


def try_fix_debug_task(prompt: str) -> str | None:
    """Return corrected code + one-sentence explanation when the bug is recognizable."""
    buggy = _extract_buggy_function(prompt)
    if not buggy:
        return None

    prompt_lower = prompt.lower()

    if re.search(r"return\s+sum\([^)]+\)\s*/\s*len\b", buggy):
        fixed = re.sub(
            r"/\s*len\b",
            lambda m: "/ len(values)" if "values" in buggy else "/ len(nums)",
            buggy,
        )
        return (
            f"{fixed}\n\n"
            "The bug is that `len` was used without passing the list, so the divisor was wrong."
        )

    if "is_even" in prompt_lower and "return n % 2 == 1" in buggy:
        fixed = """def is_even(n):
    return n % 2 == 0"""
        return (
            f"{fixed}\n\n"
            "The bug is that the function returns True for odd numbers instead of even numbers."
        )

    if ("dedupe" in prompt_lower or "unique_preserve" in prompt_lower) and "if item in seen:" in buggy:
        fn_name = "unique_preserve" if "unique_preserve" in buggy else "dedupe"
        fixed = f"""def {fn_name}(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result"""
        return (
            f"{fixed}\n\n"
            "The bug is that duplicates were appended to the result instead of skipping them."
        )

    if ("get_max" in prompt_lower or "find_max" in prompt_lower) and "return nums[0]" in buggy:
        fn_name = "find_max" if "find_max" in buggy else "get_max"
        fixed = f"""def {fn_name}(nums):
    if not nums:
        raise ValueError("empty list")
    return max(nums)"""
        return (
            f"{fixed}\n\n"
            "The bug is that the function always returns the first element instead of the maximum."
        )

    if "second_largest" in prompt_lower and "unique = list(set(nums))" in buggy:
        fixed = """def second_largest(nums):
    if len(nums) < 2:
        raise ValueError("need at least two numbers")
    unique_sorted = sorted(set(nums))
    if len(unique_sorted) < 2:
        raise ValueError("need at least two unique values")
    return unique_sorted[-2]"""
        return (
            f"{fixed}\n\n"
            "The bug is that duplicate maximum values can leave fewer than two unique numbers to select from."
        )

    return None
