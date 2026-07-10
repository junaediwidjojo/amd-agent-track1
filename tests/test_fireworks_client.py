"""Tests for Fireworks client caching and token counter."""

from unittest.mock import patch

from app.config import Settings
from app.fireworks.client import FireworksClient, TokenCounter
from app.fireworks.models import CompletionMetrics, CompletionResult


def _settings() -> Settings:
    return Settings(
        FIREWORKS_API_KEY="test-key",
        FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1",
        ALLOWED_MODELS="model-a,model-b",
        prompt_cache_enabled=True,
    )


def test_token_counter_accumulates() -> None:
    counter = TokenCounter()
    counter.record(
        CompletionMetrics(prompt_tokens=10, completion_tokens=5, total_tokens=15, model="model-a")
    )
    counter.record(
        CompletionMetrics(prompt_tokens=20, completion_tokens=10, total_tokens=30, model="model-a")
    )
    assert counter.total_tokens == 45
    assert counter.request_count == 2


def test_prompt_cache_avoids_duplicate_calls() -> None:
    settings = _settings()
    client = FireworksClient(settings=settings)

    mock_result = CompletionResult(
        text="answer",
        metrics=CompletionMetrics(
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            model="model-a",
        ),
    )

    with patch.object(client, "_call_model", return_value=mock_result) as mock_call:
        client.complete("system", "user")
        client.complete("system", "user")
        assert mock_call.call_count == 1
        assert client.token_counter.cache_hits == 1
