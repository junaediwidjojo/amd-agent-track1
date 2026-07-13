"""Runtime budget tracking and retry policy for the agent pipeline.

Callers: app/agent.py, app/task_executor.py, app/backend_selector.py, tests.
User: reduce timeout to 8 minutes and fix grading timeout; build image, don't push.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class RuntimePhase(str, Enum):
    """Time-budget phase controlling retry and escalation behavior."""

    NORMAL = "normal"
    SELECTIVE_RETRY = "selective_retry"
    CONSERVATIVE = "conservative"
    EMERGENCY = "emergency"


@dataclass
class RuntimeStats:
    """Aggregated runtime statistics for end-of-run logging."""

    deterministic_count: int = 0
    local_count: int = 0
    fireworks_count: int = 0
    fireworks_strong_count: int = 0
    retries: int = 0
    validation_failures: int = 0
    escalations: int = 0


class RuntimeManager:
    """Track elapsed runtime and gate retries/escalation by remaining budget.

    Phase thresholds scale with ``max_runtime_seconds`` so an 8-minute budget
    still leaves room for selective retries without burning the grading kill.
    """

    def __init__(self, max_runtime_seconds: float) -> None:
        self.max_runtime_seconds = max_runtime_seconds
        self._start: float | None = None
        self._remaining_tasks = 0
        self._completed_tasks = 0
        self._latencies_ms: list[float] = []
        self.stats = RuntimeStats()

    def start(self) -> None:
        self._start = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        if self._start is None:
            return 0.0
        return time.monotonic() - self._start

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_runtime_seconds - self.elapsed_seconds)

    @property
    def _emergency_threshold(self) -> float:
        return max(20.0, self.max_runtime_seconds * 0.06)

    @property
    def _conservative_threshold(self) -> float:
        return max(60.0, self.max_runtime_seconds * 0.25)

    @property
    def _selective_threshold(self) -> float:
        return max(120.0, self.max_runtime_seconds * 0.55)

    @property
    def phase(self) -> RuntimePhase:
        remaining = self.remaining_seconds
        if remaining < self._emergency_threshold:
            return RuntimePhase.EMERGENCY
        if remaining < self._conservative_threshold:
            return RuntimePhase.CONSERVATIVE
        if remaining < self._selective_threshold:
            return RuntimePhase.SELECTIVE_RETRY
        return RuntimePhase.NORMAL

    def set_remaining_tasks(self, count: int) -> None:
        self._remaining_tasks = max(0, count)

    def record_task_complete(self, latency_ms: float) -> None:
        self._completed_tasks += 1
        self._latencies_ms.append(latency_ms)

    @property
    def average_latency_ms(self) -> float:
        if not self._latencies_ms:
            return 0.0
        return sum(self._latencies_ms) / len(self._latencies_ms)

    def estimated_remaining_work_seconds(self) -> float:
        """Estimate time to finish remaining tasks from average latency."""
        if self._remaining_tasks <= 0:
            return 0.0
        avg_s = self.average_latency_ms / 1000.0 if self._latencies_ms else 15.0
        return avg_s * self._remaining_tasks

    def is_budget_exceeded(self) -> bool:
        return self.elapsed_seconds >= self.max_runtime_seconds

    def is_emergency(self) -> bool:
        return self.phase == RuntimePhase.EMERGENCY

    def allow_local(self) -> bool:
        """Local CPU inference only in NORMAL — otherwise it burns wall time."""
        return self.phase == RuntimePhase.NORMAL and not self.is_budget_exceeded()

    def allow_retries(self, uncertain: bool = False) -> bool:
        phase = self.phase
        if phase == RuntimePhase.EMERGENCY:
            return False
        if phase == RuntimePhase.CONSERVATIVE:
            return False
        if phase == RuntimePhase.SELECTIVE_RETRY:
            return uncertain
        return True

    def allow_expensive_retry(self) -> bool:
        return self.phase == RuntimePhase.NORMAL

    def allow_escalation(self) -> bool:
        return self.phase in (RuntimePhase.NORMAL, RuntimePhase.SELECTIVE_RETRY)

    def max_retries(self, uncertain: bool = False) -> int:
        if not self.allow_retries(uncertain=uncertain):
            return 0
        if self.phase == RuntimePhase.SELECTIVE_RETRY:
            return 1 if uncertain else 0
        if self.allow_expensive_retry():
            return 1
        return 0

    def record_retry(self) -> None:
        self.stats.retries += 1

    def record_validation_failure(self) -> None:
        self.stats.validation_failures += 1

    def record_escalation(self) -> None:
        self.stats.escalations += 1

    def record_backend(self, backend: str) -> None:
        if backend == "deterministic":
            self.stats.deterministic_count += 1
        elif backend == "local":
            self.stats.local_count += 1
        elif backend == "fireworks_strong":
            self.stats.fireworks_strong_count += 1
        elif backend == "fireworks":
            self.stats.fireworks_count += 1
