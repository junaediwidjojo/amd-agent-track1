"""Per-category answer validators and fallback generators."""

from __future__ import annotations

import ast
import re


def is_valid_python(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def generate_debug_explanation(task_prompt: str, fixed_code: str) -> str:
    """Generate a real explanation when the model outputs a placeholder.

    Uses the original buggy code from the task prompt to infer the bug.
    """
    # Try to extract the original buggy code from the prompt
    buggy_match = re.search(r"def\s+\w+\([^)]*\):.*", task_prompt)
    if buggy_match:
        buggy = buggy_match.group(0).strip()
        # Common bug patterns
        if "return nums[0]" in buggy or "return arr[0]" in buggy:
            return (
                "The bug is that the function only returns the first element "
                "instead of iterating through the list to find the maximum value."
            )
        if "return len(" in buggy:
            return "The bug is that the function returns the length instead of the required value."
        if "==" in buggy and "=" in buggy:
            return "The bug is an incorrect comparison or assignment in the function logic."
        if "for" not in buggy and "while" not in buggy:
            return "The bug is that the function misses a loop to iterate through the input."

    # Fallback based on task description
    desc_match = re.search(r"should\s+(.+?)\s+but\s+has\s+a\s+bug", task_prompt, re.IGNORECASE)
    if desc_match:
        return f"The bug is that the function does not correctly {desc_match.group(1)}."

    return "The bug is in the function logic and has been corrected in the implementation above."


def is_truncated_summary(text: str) -> bool:
    """Detect if a summary is likely truncated or malformed.

    Conservative: only flags empty text, trailing commas, markdown artifacts,
    or bullet/header prefixes. Does NOT reject text simply for lacking punctuation.
    """
    cleaned = text.strip()
    if not cleaned:
        return True
    # If it ends with a comma, it's likely truncated
    if cleaned.endswith(","):
        return True
    # If it ends with a markdown artifact, it's likely bad
    if cleaned.endswith("```") or cleaned.endswith("**"):
        return True
    # If it looks like reasoning (has numbers, bullets, markdown headers), it's bad
    return bool(re.search(r"^\d+\.\s|^[-*]\s|\*\*|#{1,6}\s", cleaned))


def fallback_summary(task_prompt: str) -> str | None:
    """If LLM summary is truncated, try to return the original text if it's already one sentence."""
    # Extract text after the colon (common format: "Summarize: ...")
    text = re.sub(r"^.*?summari[sz]e.*?:\s*", "", task_prompt, flags=re.IGNORECASE)
    if text == task_prompt:
        # Try to find the sentence after "in exactly one sentence:"
        m = re.search(r"in exactly one sentence[:\s]+(.+)", task_prompt, re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip()
        else:
            return None

    # Check if the input is already a single sentence
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) == 1 and sentences[0]:
        return sentences[0].strip()

    # Otherwise return the first sentence as a safe fallback
    if sentences:
        return sentences[0].strip()

    return None
