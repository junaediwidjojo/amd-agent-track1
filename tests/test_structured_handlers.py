"""Tests for structured extraction and writing handlers."""

from unittest.mock import MagicMock

from app.fireworks.models import TaskItem
from app.handlers.factual import FactualHandler
from app.handlers.structured_extraction import StructuredExtractionHandler
from app.handlers.structured_writing import StructuredWritingHandler


def _mock_provider() -> MagicMock:
    from app.providers.base import BaseLLMProvider
    return MagicMock(spec=BaseLLMProvider)


class TestStructuredExtractionHandler:
    def _handler(self) -> StructuredExtractionHandler:
        return StructuredExtractionHandler(_mock_provider())

    def test_valid_json_preserved(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="e1", prompt="Return JSON object with keys: a, b")
        raw = '{"a": 1, "b": 2}'
        result = handler.post_process(raw, task)
        assert result == '{"a": 1, "b": 2}'

    def test_json_extracted_from_fences(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="e2", prompt="Return JSON")
        raw = '```json\n{"x": true}\n```'
        result = handler.post_process(raw, task)
        assert result == '{"x": true}'

    def test_invalid_json_returns_cleaned_text(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="e3", prompt="Return JSON")
        raw = "This is not json at all"
        result = handler.post_process(raw, task)
        assert result == "This is not json at all"


class TestStructuredWritingHandler:
    def _handler(self) -> StructuredWritingHandler:
        return StructuredWritingHandler(_mock_provider())

    def test_preserves_section_headers(self) -> None:
        handler = self._handler()
        task = TaskItem(
            task_id="w1",
            prompt="Provide four sections titled: Bugs, Performance Improvements, Corrected Code, Explanation.",
        )
        raw = "Bugs\nFound bug.\nPerformance Improvements\nFaster loop.\nCorrected Code\ndef f(): pass\nExplanation\nFixed it."
        result = handler.post_process(raw, task)
        assert "Bugs" in result
        assert "Performance Improvements" in result
        assert "Corrected Code" in result
        assert "Explanation" in result

    def test_adds_missing_sections(self) -> None:
        handler = self._handler()
        task = TaskItem(
            task_id="w2",
            prompt="Provide two sections titled: Summary, Details.",
        )
        raw = "Summary\nThis is summary."
        result = handler.post_process(raw, task)
        assert "Summary" in result
        assert "Details" in result


class TestFactualHandlerIdentity:
    def _handler(self) -> FactualHandler:
        return FactualHandler(_mock_provider())

    def test_identity_short_circuit(self) -> None:
        handler = self._handler()
        task = TaskItem(task_id="i1", prompt="As you are for general usage, please introduce who you are.")
        result = handler.complete(task)
        assert "AMD Developer Hackathon Track 1" in result.text
