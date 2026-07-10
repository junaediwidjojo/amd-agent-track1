"""Tests for handler post-processing."""

from unittest.mock import MagicMock

from app.fireworks.client import FireworksClient
from app.fireworks.models import TaskItem
from app.handlers.code_generation import CodeGenerationHandler
from app.handlers.math import MathHandler


def _mock_client() -> FireworksClient:
    client = MagicMock(spec=FireworksClient)
    return client


def test_math_post_process_extracts_number() -> None:
    handler = MathHandler(_mock_client())
    task = TaskItem(task_id="1", prompt="calculate 5 + 3")
    result = handler.post_process("The answer is 8.", task)
    assert result == "8"


def test_codegen_post_process_strips_fences() -> None:
    handler = CodeGenerationHandler(_mock_client())
    task = TaskItem(task_id="2", prompt="write a function")
    raw = "```python\ndef f():\n    return 1\n```"
    result = handler.post_process(raw, task)
    assert result.startswith("def f")
    assert "```" not in result
