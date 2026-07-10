"""Unit tests for shared pipeline utilities: text_utils, validators, json_utils."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.fireworks.client import FireworksClient
from app.fireworks.models import TaskItem
from app.handlers.code_generation import CodeGenerationHandler
from app.handlers.debugging import DebuggingHandler
from app.handlers.factual import FactualHandler
from app.handlers.summarization import SummarizationHandler
from app.utils.json_utils import clean_answer, extract_code_block
from app.utils.text_utils import (
    extract_final_answer,
    extract_first_code_block,
    extract_last_sentence,
    is_valid_python,
    strip_cot,
)
from app.utils.validators import fallback_summary, generate_debug_explanation, is_truncated_summary


class TestStripCot:
    def test_preserves_normal_sentences(self) -> None:
        text = "Because machine learning enables computers to learn from data."
        assert strip_cot(text) == text

    def test_strips_markdown_headers(self) -> None:
        text = "# Analysis\nThe answer is 42."
        assert strip_cot(text) == "The answer is 42."

    def test_strips_numbered_bold_steps(self) -> None:
        text = "1. **Analyze:**\n2. **Draft:**\nThe answer is 42."
        assert strip_cot(text) == "The answer is 42."

    def test_strips_reasoning_bullets(self) -> None:
        text = "- Step 1: read the text\n- Note: important\nThe answer is 42."
        assert strip_cot(text) == "The answer is 42."

    def test_preserves_bullet_answers(self) -> None:
        text = "- The answer is 42."
        assert strip_cot(text) == "- The answer is 42."

    def test_strips_markdown_fences(self) -> None:
        text = "```python\ndef f(): pass\n```"
        assert "```" not in strip_cot(text)

    def test_does_not_strip_because_in_normal_text(self) -> None:
        text = "Machine learning is powerful because it learns patterns."
        assert strip_cot(text) == text


class TestExtractFirstCodeBlock:
    def test_extracts_fenced_block(self) -> None:
        text = '```python\ndef f():\n    return 1\n```'
        assert extract_first_code_block(text) == "def f():\n    return 1"

    def test_prefers_last_fenced_block(self) -> None:
        text = '```python\ndef old(): pass\n```\nSome text\n```python\ndef new(): pass\n```'
        assert "def new()" in extract_first_code_block(text)

    def test_extracts_unfenced_def(self) -> None:
        text = "def is_even(n):\n    return n % 2 == 0\n\nThe bug is fixed."
        result = extract_first_code_block(text)
        assert "def is_even" in result
        assert "The bug is fixed." not in result

    def test_stops_at_prose_after_blank_line(self) -> None:
        text = "def foo():\n    pass\n\n\nThis is explanation."
        result = extract_first_code_block(text)
        assert "def foo" in result
        assert "This is explanation." not in result

    def test_returns_text_if_no_code(self) -> None:
        text = "Just a plain sentence."
        assert extract_first_code_block(text) == text

    def test_extracts_import_blocks(self) -> None:
        text = "import os\nfrom sys import path\n\ndef foo(): pass"
        result = extract_first_code_block(text)
        # Import-only blocks are excluded; we start at the first def/class
        assert "def foo" in result
        assert "import os" not in result


class TestExtractLastSentence:
    def test_extracts_last_sentence(self) -> None:
        text = "First sentence. Second sentence. Last one!"
        assert extract_last_sentence(text) == "Last one!"

    def test_handles_no_punctuation(self) -> None:
        text = "No punctuation here"
        assert extract_last_sentence(text) == "No punctuation here"

    def test_strips_fences_first(self) -> None:
        text = "```\nThe answer is 42.\n```"
        assert extract_last_sentence(text) == "The answer is 42."


class TestExtractFinalAnswer:
    def test_extracts_short_last_line(self) -> None:
        text = "Some reasoning\nAnswer: 42"
        assert extract_final_answer(text) == "Answer: 42"

    def test_returns_cleaned_if_last_line_long(self) -> None:
        text = (
            "A very long sentence that goes on and on and on and on and on "
            "and on and on and on and on and on and on."
        )
        assert extract_final_answer(text) == text


class TestIsValidPython:
    def test_valid_function(self) -> None:
        assert is_valid_python("def f(): return 1")

    def test_invalid_syntax(self) -> None:
        assert not is_valid_python("def f(: pass")

    def test_empty_string(self) -> None:
        assert not is_valid_python("")


class TestCleanAnswer:
    def test_strips_fences(self) -> None:
        assert clean_answer("```\nhello\n```") == "hello"

    def test_strips_language_tag(self) -> None:
        assert clean_answer("```python\ndef f(): pass\n```") == "def f(): pass"

    def test_no_fences(self) -> None:
        assert clean_answer("hello world") == "hello world"


class TestExtractCodeBlock:
    def test_extracts_code(self) -> None:
        assert extract_code_block("```\ncode\n```") == "code"

    def test_fallback_to_clean_answer(self) -> None:
        assert extract_code_block("plain text") == "plain text"


class TestIsTruncatedSummary:
    def test_empty_is_truncated(self) -> None:
        assert is_truncated_summary("")

    def test_comma_ending_is_truncated(self) -> None:
        assert is_truncated_summary("This is a summary,")

    def test_no_punctuation_is_not_truncated(self) -> None:
        assert not is_truncated_summary("Machine learning enables computers to learn from data")

    def test_markdown_artifact_is_truncated(self) -> None:
        assert is_truncated_summary("Some text ```")

    def test_numbered_prefix_is_truncated(self) -> None:
        assert is_truncated_summary("1. Some text")

    def test_normal_sentence_is_not_truncated(self) -> None:
        assert not is_truncated_summary("This is a complete sentence.")


class TestFallbackSummary:
    def test_extracts_after_summarize_colon(self) -> None:
        prompt = "Summarize: Machine learning enables computers to learn."
        result = fallback_summary(prompt)
        assert result == "Machine learning enables computers to learn."

    def test_extracts_after_in_one_sentence(self) -> None:
        prompt = "Summarize in exactly one sentence: The quick brown fox jumps over the lazy dog."
        result = fallback_summary(prompt)
        assert result == "The quick brown fox jumps over the lazy dog."

    def test_returns_none_when_no_text_found(self) -> None:
        prompt = "What is the weather?"
        assert fallback_summary(prompt) is None

    def test_returns_first_sentence_if_multiple(self) -> None:
        prompt = "Summarize: First sentence. Second sentence."
        result = fallback_summary(prompt)
        assert result == "First sentence."


class TestGenerateDebugExplanation:
    def test_first_element_bug(self) -> None:
        prompt = "Fix the bug: def max(nums): return nums[0]"
        result = generate_debug_explanation(prompt, "")
        assert "first element" in result

    def test_length_bug(self) -> None:
        prompt = "Fix the bug: def count(items): return len(items)"
        result = generate_debug_explanation(prompt, "")
        assert "length" in result

    def test_missing_loop(self) -> None:
        prompt = "Fix the bug: def sum(nums): return 0"
        result = generate_debug_explanation(prompt, "")
        assert "loop" in result


class TestDebuggingHandlerPostProcess:
    def _handler(self) -> DebuggingHandler:
        client = MagicMock(spec=FireworksClient)
        return DebuggingHandler(client)

    def test_extracts_code_and_explanation(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="d1", prompt="Fix the bug: def is_even(n): return n % 2 == 1")
        raw = (
            "```python\n"
            "def is_even(n):\n"
            "    return n % 2 == 0\n"
            "```\n\n"
            "The bug is that the function returns true for odd numbers."
        )
        result = handler.post_process(raw, task)
        assert "def is_even" in result
        assert "return n % 2 == 0" in result
        assert "bug is" in result.lower()

    def test_returns_code_even_without_explanation(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="d2", prompt="Fix the bug: def is_even(n): return n % 2 == 1")
        raw = "```python\ndef is_even(n):\n    return n % 2 == 0\n```"
        result = handler.post_process(raw, task)
        assert "def is_even" in result
        assert "return n % 2 == 0" in result

    def test_generates_explanation_when_missing(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="d3", prompt="Fix the bug: def is_even(n): return n % 2 == 1")
        raw = "def is_even(n):\n    return n % 2 == 0"
        result = handler.post_process(raw, task)
        assert "def is_even" in result
        # Should generate an explanation since none was provided
        assert "\n\n" in result

    def test_fallback_line_extraction(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="d4", prompt="Fix the bug: def foo(): return 1")
        raw = "def foo():\n    return 1\n\nSome explanation here."
        result = handler.post_process(raw, task)
        assert "def foo" in result
        # Explanation does not contain bug keywords, so a generated explanation is used
        assert "bug" in result.lower() or "loop" in result.lower()


class TestCodeGenerationHandlerPostProcess:
    def _handler(self) -> CodeGenerationHandler:
        client = MagicMock(spec=FireworksClient)
        return CodeGenerationHandler(client)

    def test_extracts_fenced_code(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="c1", prompt="Write a palindrome function")
        raw = "```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```"
        result = handler.post_process(raw, task)
        assert "def is_palindrome" in result
        assert "```" not in result

    def test_extracts_unfenced_code(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="c2", prompt="Write a function")
        raw = "def hello():\n    return 'world'\n\nThis function returns hello."
        result = handler.post_process(raw, task)
        assert "def hello" in result
        assert "This function returns hello." not in result

    def test_fallback_def_match(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="c3", prompt="Write a function")
        raw = "Here is the code:\ndef add(a, b):\n    return a + b\nEnjoy!"
        result = handler.post_process(raw, task)
        assert "def add" in result
        assert "Enjoy!" not in result


class TestSummarizationHandlerPostProcess:
    def _handler(self) -> SummarizationHandler:
        client = MagicMock(spec=FireworksClient)
        return SummarizationHandler(client)

    def test_extracts_last_sentence(self) -> None:
        handler = self._handler()
        task = TaskItem(
            task_id="s1",
            prompt="Summarize: ML enables learning from data without explicit programming.",
        )
        raw = "Here is the summary. Machine learning enables computers to learn from data."
        result = handler.post_process(raw, task)
        assert result == "Machine learning enables computers to learn from data."

    def test_uses_fallback_when_truncated(self) -> None:
        handler = self._handler()
        task = TaskItem(
            task_id="s2",
            prompt="Summarize in one sentence: The quick brown fox jumps over the lazy dog.",
        )
        raw = "1. **Analyze:** read the text\n2. **Draft:** The quick brown fox,"
        result = handler.post_process(raw, task)
        # Fallback extracts from prompt
        assert result == "The quick brown fox jumps over the lazy dog."

    def test_does_not_reject_missing_punctuation(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="s3", prompt="Summarize: Hello world")
        raw = "Hello world"
        result = handler.post_process(raw, task)
        assert result == "Hello world"


class TestFactualHandlerPostProcess:
    def _handler(self) -> FactualHandler:
        client = MagicMock(spec=FireworksClient)
        return FactualHandler(client)

    def test_strips_fences(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="f1", prompt="What is the tallest mountain?")
        raw = "```\nMount Everest\n```"
        result = handler.post_process(raw, task)
        assert result == "Mount Everest"

    def test_returns_plain_text(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="f2", prompt="What is 2+2?")
        raw = "4"
        result = handler.post_process(raw, task)
        assert result == "4"
