"""Task execution pipeline with verification, retry, and backend escalation."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.backend_selector import BackendType, has_deterministic_solver, next_escalation, select_backend_order
from app.config import get_settings
from app.fireworks.models import CompletionMetrics, CompletionResult, TaskCategory, TaskItem
from app.handlers.base import BaseHandler
from app.providers.hybrid import HybridProvider
from app.refinement import build_refinement_prompt, refine_answer
from app.runtime_manager import RuntimeManager
from app.solvers.fallback import solver_fallback
from app.utils.logger import get_logger, log_event
from app.verification import VerificationResult, verify_answer

logger = get_logger(__name__)

_EMERGENCY_ANSWER = "Unable to process this task within runtime limit."


@dataclass
class TaskExecutionResult:
    """Outcome of executing one task through the verification pipeline."""

    answer: str
    metrics: CompletionMetrics
    category: TaskCategory
    verification: VerificationResult
    backend: str = "fireworks"
    retry_count: int = 0
    escalation_reason: str | None = None
    latency_ms: float = 0.0


@dataclass
class _Candidate:
    answer: str
    metrics: CompletionMetrics
    backend: str
    verification: VerificationResult


class TaskExecutor:
    """Wraps handlers with generate → validate → refine → escalate pipeline."""

    def __init__(
        self,
        handlers: dict[TaskCategory, BaseHandler],
        provider: HybridProvider,
        runtime: RuntimeManager,
    ) -> None:
        self.handlers = handlers
        self.provider = provider
        self.runtime = runtime
        self.settings = get_settings()

    def execute(
        self,
        task: TaskItem,
        category: TaskCategory,
        *,
        emergency: bool = False,
    ) -> TaskExecutionResult:
        handler = self.handlers[category]
        started = time.perf_counter()
        retry_count = 0
        escalation_reason: str | None = None

        if emergency:
            answer = self._emergency_answer(task, category)
            verification = verify_answer(answer, task, category)
            latency = (time.perf_counter() - started) * 1000
            metrics = CompletionMetrics(backend="emergency", latency_ms=latency)
            self._log_task(task, category, "emergency", verification, retry_count, escalation_reason, latency)
            return TaskExecutionResult(
                answer=answer,
                metrics=metrics,
                category=category,
                verification=verification,
                backend="emergency",
                latency_ms=latency,
            )

        backends = select_backend_order(category, task, self.runtime)
        best: _Candidate | None = None
        current_backend = backends[0] if backends else BackendType.FIREWORKS

        while True:
            candidate = self._generate(task, category, handler, current_backend)
            answer = handler.post_process(candidate.text, task).strip() or "No answer generated."
            verification = verify_answer(answer, task, category)
            candidate_metrics = candidate.metrics
            candidate_metrics.confidence = verification.confidence

            self.runtime.record_backend(current_backend.value)
            entry = _Candidate(answer, candidate_metrics, current_backend.value, verification)

            if best is None or entry.verification.confidence > best.verification.confidence:
                best = entry

            self._log_task(
                task, category, current_backend.value, verification, retry_count, escalation_reason,
                candidate_metrics.latency_ms,
            )

            if verification.passed:
                latency = (time.perf_counter() - started) * 1000
                return TaskExecutionResult(
                    answer=answer,
                    metrics=candidate_metrics,
                    category=category,
                    verification=verification,
                    backend=current_backend.value,
                    retry_count=retry_count,
                    escalation_reason=escalation_reason,
                    latency_ms=latency,
                )

            self.runtime.record_validation_failure()
            uncertain = verification.confidence < 0.5

            if self.runtime.allow_retries(uncertain=uncertain) and retry_count < self.runtime.max_retries(uncertain=uncertain):
                refined = refine_answer(answer, task, category, verification.retry_reason)
                if refined:
                    retry_count += 1
                    self.runtime.record_retry()
                    refined_answer = handler.post_process(refined, task).strip()
                    refined_ver = verify_answer(refined_answer, task, category)
                    if refined_ver.confidence > (best.verification.confidence if best else 0):
                        best = _Candidate(refined_answer, candidate_metrics, current_backend.value, refined_ver)
                    if refined_ver.passed:
                        latency = (time.perf_counter() - started) * 1000
                        return TaskExecutionResult(
                            answer=refined_answer,
                            metrics=candidate_metrics,
                            category=category,
                            verification=refined_ver,
                            backend=current_backend.value,
                            retry_count=retry_count,
                            escalation_reason=escalation_reason,
                            latency_ms=latency,
                        )

                if self.runtime.allow_expensive_retry() or uncertain:
                    retry_count += 1
                    self.runtime.record_retry()
                    prompt = build_refinement_prompt(
                        task.prompt, answer, verification.retry_reason, category,
                    )
                    retry_result = self._complete_backend(
                        handler, task, category, current_backend, user_override=prompt,
                    )
                    retry_answer = handler.post_process(retry_result.text, task).strip()
                    retry_ver = verify_answer(retry_answer, task, category)
                    if retry_ver.confidence > (best.verification.confidence if best else 0):
                        best = _Candidate(retry_answer, retry_result.metrics, current_backend.value, retry_ver)
                    if retry_ver.passed:
                        latency = (time.perf_counter() - started) * 1000
                        return TaskExecutionResult(
                            answer=retry_answer,
                            metrics=retry_result.metrics,
                            category=category,
                            verification=retry_ver,
                            backend=current_backend.value,
                            retry_count=retry_count,
                            escalation_reason=escalation_reason,
                            latency_ms=latency,
                        )

            if not self.runtime.allow_escalation():
                break

            nxt = next_escalation(current_backend, category, task, self.runtime)
            if nxt is None or nxt == current_backend:
                break

            escalation_reason = verification.retry_reason or verification.backend_recommendation
            self.runtime.record_escalation()
            current_backend = nxt
            retry_count += 1

        assert best is not None
        latency = (time.perf_counter() - started) * 1000
        best.metrics.latency_ms = latency
        return TaskExecutionResult(
            answer=best.answer,
            metrics=best.metrics,
            category=category,
            verification=best.verification,
            backend=best.backend,
            retry_count=retry_count,
            escalation_reason=escalation_reason,
            latency_ms=latency,
        )

    def execute_retry(
        self,
        task: TaskItem,
        category: TaskCategory,
        *,
        backend: BackendType,
        pass_number: int = 2,
    ) -> TaskExecutionResult:
        """Re-run one task from a forced backend (pass-2 low-confidence retries)."""
        handler = self.handlers[category]
        started = time.perf_counter()
        candidate = self._generate(task, category, handler, backend)
        answer = handler.post_process(candidate.text, task).strip() or "No answer generated."
        verification = verify_answer(answer, task, category)
        candidate.metrics.confidence = verification.confidence
        self.runtime.record_backend(backend.value)
        latency = (time.perf_counter() - started) * 1000
        candidate.metrics.latency_ms = latency
        self._log_task(
            task,
            category,
            backend.value,
            verification,
            retry_count=pass_number,
            escalation_reason=f"pass{pass_number}_retry",
            latency_ms=latency,
        )
        return TaskExecutionResult(
            answer=answer,
            metrics=candidate.metrics,
            category=category,
            verification=verification,
            backend=backend.value,
            retry_count=pass_number,
            escalation_reason=f"pass{pass_number}_retry",
            latency_ms=latency,
        )

    def _generate(
        self,
        task: TaskItem,
        category: TaskCategory,
        handler: BaseHandler,
        backend: BackendType,
    ) -> CompletionResult:
        return self._complete_backend(handler, task, category, backend)

    def _complete_backend(
        self,
        handler: BaseHandler,
        task: TaskItem,
        category: TaskCategory,
        backend: BackendType,
        *,
        user_override: str | None = None,
    ) -> CompletionResult:
        user = user_override or task.prompt
        system = handler.load_system_prompt()
        model = self.settings.pick_model(handler.preferred_model_tags)

        if backend == BackendType.DETERMINISTIC:
            text = solver_fallback(task, category) or ""
            return CompletionResult(
                text=text,
                metrics=CompletionMetrics(backend="deterministic", model="solver"),
            )

        if backend == BackendType.LOCAL:
            try:
                return self.provider.complete_local(
                    system=system,
                    user=user,
                    max_tokens=handler.max_tokens,
                    category=category.value,
                    task_id=task.task_id,
                    task=task,
                )
            except Exception as exc:
                log_event(
                    logger,
                    "backend_local_failed",
                    task_id=task.task_id,
                    category=category.value,
                    error=str(exc),
                    exc_info=True,
                )
                return self.provider.complete_fireworks(
                    system=system,
                    user=user,
                    max_tokens=handler.max_tokens,
                    model=self.settings.pick_model(handler.preferred_model_tags),
                    category=category,
                    task_id=task.task_id,
                    task=task,
                )

        if backend == BackendType.FIREWORKS_STRONG:
            return self.provider.complete_fireworks(
                system=system,
                user=user,
                max_tokens=handler.max_tokens,
                model=self.provider.pick_escalation_model(category),
                category=category,
                task_id=task.task_id,
                task=task,
            )

        return self.provider.complete_fireworks(
            system=system,
            user=user,
            max_tokens=handler.max_tokens,
            model=model,
            category=category,
            task_id=task.task_id,
            task=task,
        )

    def _emergency_answer(self, task: TaskItem, category: TaskCategory) -> str:
        if has_deterministic_solver(category, task):
            fb = solver_fallback(task, category)
            if fb:
                return fb
        return _EMERGENCY_ANSWER

    def _log_task(
        self,
        task: TaskItem,
        category: TaskCategory,
        backend: str,
        verification: VerificationResult,
        retry_count: int,
        escalation_reason: str | None,
        latency_ms: float,
    ) -> None:
        log_event(
            logger,
            "task_execution",
            task_id=task.task_id,
            category=category.value,
            backend=backend,
            validator_passed=verification.passed,
            validator_confidence=round(verification.confidence, 2),
            retry_reason=verification.retry_reason,
            retry_count=retry_count,
            escalation_reason=escalation_reason,
            latency_ms=round(latency_ms, 2),
            runtime_remaining_s=round(self.runtime.remaining_seconds, 1),
        )
