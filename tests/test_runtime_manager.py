"""Tests for runtime manager phases and retry policy."""

import time

import pytest

from app.runtime_manager import RuntimeManager, RuntimePhase


def test_default_max_runtime_is_540_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    monkeypatch.setenv("ALLOWED_MODELS", "model-a,model-b")
    get_settings.cache_clear()
    assert get_settings().max_runtime_seconds == 540.0


def test_runtime_phase_normal() -> None:
    runtime = RuntimeManager(max_runtime_seconds=600.0)
    runtime.start()
    assert runtime.phase == RuntimePhase.NORMAL
    assert runtime.allow_retries()
    assert runtime.allow_escalation()
    assert runtime.max_retries() == 2


def test_runtime_phase_emergency() -> None:
    runtime = RuntimeManager(max_runtime_seconds=600.0)
    runtime._start = time.monotonic() - 580
    assert runtime.remaining_seconds < 30
    assert runtime.phase == RuntimePhase.EMERGENCY
    assert not runtime.allow_retries()
    assert runtime.is_emergency()


def test_runtime_stats_recording() -> None:
    runtime = RuntimeManager(max_runtime_seconds=600.0)
    runtime.record_backend("deterministic")
    runtime.record_backend("local")
    runtime.record_retry()
    runtime.record_validation_failure()
    runtime.record_escalation()
    assert runtime.stats.deterministic_count == 1
    assert runtime.stats.local_count == 1
    assert runtime.stats.retries == 1
    assert runtime.stats.validation_failures == 1
    assert runtime.stats.escalations == 1
