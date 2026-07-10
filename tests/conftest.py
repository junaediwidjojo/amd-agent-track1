"""Shared pytest fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.config import get_settings
from app.fireworks.client import FireworksClient, TokenCounter


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide required environment variables for unit tests."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    monkeypatch.setenv("ALLOWED_MODELS", "model-a,model-b")
    get_settings.cache_clear()


@pytest.fixture
def mock_fireworks_client() -> FireworksClient:
    """Return a lightweight Fireworks client mock for agent tests."""
    client = MagicMock(spec=FireworksClient)
    client.token_counter = TokenCounter()
    return client


@pytest.fixture
def mock_hybrid_provider() -> MagicMock:
    """Return a lightweight HybridProvider mock for agent tests."""
    from app.providers.hybrid import HybridProvider
    provider = MagicMock(spec=HybridProvider)
    return provider
