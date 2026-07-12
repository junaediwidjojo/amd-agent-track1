"""Tests for runtime manager phases and retry policy."""

import time

from app.runtime_manager import RuntimeManager, RuntimePhase


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
