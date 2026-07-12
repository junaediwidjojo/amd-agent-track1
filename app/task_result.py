"""Per-task execution result used for two-pass scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.fireworks.models import CompletionMetrics, TaskCategory


@dataclass
class TaskResult:
    """Stored outcome for one task pass (answer, confidence, backend)."""

    task_id: str
    answer: str
    confidence: float
    backend_used: str
    pass_number: int
    category: TaskCategory
    validation_passed: bool = True
    metrics: CompletionMetrics | None = None
    backends_tried: list[str] = field(default_factory=list)

    def needs_pass2_retry(self, threshold: float = 0.7) -> bool:
        return self.confidence < threshold or not self.validation_passed
