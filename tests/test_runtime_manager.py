"""Tests for runtime manager phases and retry policy.

Callers: pytest suite. No data files. User: "reduce our timeout constrains to 8 minutes ... try to fix it / dont push it yet, just build the image"
"""

import time

import pytest

from app.runtime_manager import RuntimeManager, RuntimePhase


def test_default_max_runtime_is_480_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    monkeypatch.setenv("ALLOWED_MODELS", "model-a,model-b")
    monkeypatch.delenv("MAX_RUNTIME_SECONDS", raising=False)
    get_settings.cache_clear()
    assert get_settings().max_runtime_seconds == 480.0


def test_runtime_phase_normal() -> None:
    runtime = RuntimeManager(max_runtime_seconds=480.0)
    runtime.start()
    assert runtime.phase == RuntimePhase.NORMAL
    assert runtime.allow_retries()
    assert runtime.allow_escalation()
    assert runtime.allow_local()
    assert runtime.max_retries() == 1


def test_runtime_phase_emergency() -> None:
    runtime = RuntimeManager(max_runtime_seconds=480.0)
    runtime._start = time.monotonic() - 470
    assert runtime.remaining_seconds < runtime._emergency_threshold
    assert runtime.phase == RuntimePhase.EMERGENCY
    assert not runtime.allow_retries()
    assert not runtime.allow_local()
    assert runtime.is_emergency()


def test_local_only_in_normal_phase() -> None:
    runtime = RuntimeManager(max_runtime_seconds=480.0)
    runtime._start = time.monotonic() - 250  # ~230s remaining → selective
    assert runtime.phase == RuntimePhase.SELECTIVE_RETRY
    assert not runtime.allow_local()
    assert runtime.allow_escalation()


def test_runtime_stats_recording() -> None:
    runtime = RuntimeManager(max_runtime_seconds=480.0)
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
