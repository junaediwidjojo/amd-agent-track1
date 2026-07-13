"""Core agent orchestrating routing, handlers, and result assembly."""

from __future__ import annotations

from app.backend_selector import BackendType, is_local_suitable
from app.config import get_settings
from app.fireworks.client import APIExhaustedError
from app.fireworks.models import (
    BenchmarkReport,
    CompletionMetrics,
    ResultItem,
    TaskCategory,
    TaskItem,
)
from app.handlers.base import BaseHandler
from app.handlers.code_generation import CodeGenerationHandler
from app.handlers.debugging import DebuggingHandler
from app.handlers.factual import FactualHandler
from app.handlers.logic import LogicHandler
from app.handlers.math import MathHandler
from app.handlers.ner import NerHandler
from app.handlers.sentiment import SentimentHandler
from app.handlers.structured_extraction import StructuredExtractionHandler
from app.handlers.structured_writing import StructuredWritingHandler
from app.handlers.summarization import SummarizationHandler
from app.providers.hybrid import HybridProvider
from app.router import classify_task
from app.runtime_manager import RuntimeManager
from app.solvers.fallback import solver_fallback
from app.task_executor import TaskExecutor
from app.task_result import TaskResult
from app.utils.logger import get_logger, log_event
from app.verification import is_usable_answer

logger = get_logger(__name__)

_RUNTIME_LIMIT_ANSWER = "Unable to process this task within runtime limit."
_PASS2_CONFIDENCE_THRESHOLD = 0.7
_PASS2_MIN_REMAINING_SECONDS = 40.0
_PASS2_MAX_RETRIES = 5
_LLM_BACKENDS = frozenset({"fireworks", "fireworks_strong"})

_LOCAL_PASS2_CATEGORIES = {
    TaskCategory.SENTIMENT,
    TaskCategory.SUMMARIZATION,
    TaskCategory.NER,
    TaskCategory.FACTUAL,
}
_HARD_PASS2_CATEGORIES = {
    TaskCategory.CODE_GENERATION,
    TaskCategory.DEBUGGING,
    TaskCategory.MATH,
    TaskCategory.LOGIC,
    TaskCategory.STRUCTURED_EXTRACTION,
    TaskCategory.STRUCTURED_WRITING,
}


class Agent:
    """General-purpose task agent with verification-gated execution pipeline."""

    def __init__(
        self,
        provider: HybridProvider | None = None,
        max_runtime_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        self.provider = provider or HybridProvider()
        self._max_runtime_seconds = (
            max_runtime_seconds
            if max_runtime_seconds is not None
            else settings.max_runtime_seconds
        )
        self.runtime_budget_exceeded = False
        self._handlers: dict[TaskCategory, BaseHandler] = {
            TaskCategory.FACTUAL: FactualHandler(self.provider),
            TaskCategory.SUMMARIZATION: SummarizationHandler(self.provider),
            TaskCategory.MATH: MathHandler(self.provider),
            TaskCategory.SENTIMENT: SentimentHandler(self.provider),
            TaskCategory.NER: NerHandler(self.provider),
            TaskCategory.LOGIC: LogicHandler(self.provider),
            TaskCategory.DEBUGGING: DebuggingHandler(self.provider),
            TaskCategory.CODE_GENERATION: CodeGenerationHandler(self.provider),
            TaskCategory.STRUCTURED_EXTRACTION: StructuredExtractionHandler(self.provider),
            TaskCategory.STRUCTURED_WRITING: StructuredWritingHandler(self.provider),
        }
        self._runtime = RuntimeManager(self._max_runtime_seconds)
        self._executor = TaskExecutor(self._handlers, self.provider, self._runtime)
        self._task_results: dict[str, TaskResult] = {}
        self._pass2_retry_count = 0

    def get_handler(self, category: TaskCategory) -> BaseHandler:
        return self._handlers[category]

    def _start_run(self) -> None:
        self._runtime.start()
        self.runtime_budget_exceeded = False
        self._task_results = {}
        self._pass2_retry_count = 0

    def _runtime_budget_exceeded(self) -> bool:
        return self._runtime.is_budget_exceeded()

    def _runtime_limit_result(self, task: TaskItem) -> ResultItem:
        return ResultItem(task_id=task.task_id, answer=_RUNTIME_LIMIT_ANSWER)

    def process_task(self, task: TaskItem) -> tuple[ResultItem, CompletionMetrics, TaskCategory]:
        """Process a single task through the verification pipeline."""
        task_result = self._run_single_task(task, pass_number=1)
        return (
            ResultItem(task_id=task.task_id, answer=task_result.answer),
            task_result.metrics or CompletionMetrics(),
            task_result.category,
        )

    def _run_single_task(self, task: TaskItem, *, pass_number: int) -> TaskResult:
        category = classify_task(task)

        log_event(
            logger,
            "task_routed",
            task_id=task.task_id,
            category=category.value,
            pass_number=pass_number,
        )

        try:
            if self._runtime.is_emergency() and pass_number == 1:
                log_event(
                    logger,
                    "runtime_emergency_mode",
                    task_id=task.task_id,
                    remaining_seconds=round(self._runtime.remaining_seconds, 1),
                    pass_number=pass_number,
                )
                execution = self._executor.execute(task, category, emergency=True)
            else:
                execution = self._executor.execute(task, category, emergency=False)

            answer = execution.answer.strip() or "No answer generated."
            metrics = execution.metrics
            log_event(
                logger,
                "task_complete",
                task_id=task.task_id,
                category=category.value,
                backend=execution.backend,
                pass_number=pass_number,
                validator_passed=execution.verification.passed,
                final_confidence=round(execution.verification.confidence, 2),
                retry_count=execution.retry_count,
                escalation_reason=execution.escalation_reason,
                latency_ms=round(execution.latency_ms, 2),
            )
            return TaskResult(
                task_id=task.task_id,
                answer=answer,
                confidence=round(execution.verification.confidence, 3),
                backend_used=execution.backend,
                pass_number=pass_number,
                category=category,
                validation_passed=execution.verification.passed,
                metrics=metrics,
                backends_tried=[execution.backend],
            )
        except APIExhaustedError as exc:
            log_event(
                logger,
                "task_error",
                task_id=task.task_id,
                category=category.value,
                pass_number=pass_number,
                exc_info=True,
                primary_model=exc.primary_model,
                last_error=str(exc.last_error) if exc.last_error else None,
            )
            fallback = solver_fallback(task, category)
            answer = fallback or "Unable to process this task."
            return TaskResult(
                task_id=task.task_id,
                answer=answer,
                confidence=0.2,
                backend_used="fallback",
                pass_number=pass_number,
                category=category,
                validation_passed=False,
                metrics=CompletionMetrics(),
            )
        except Exception as exc:
            log_event(
                logger,
                "task_error",
                task_id=task.task_id,
                category=category.value,
                pass_number=pass_number,
                error=str(exc),
                exc_info=True,
            )
            fallback = solver_fallback(task, category)
            answer = fallback or "Unable to process this task."
            return TaskResult(
                task_id=task.task_id,
                answer=answer,
                confidence=0.2,
                backend_used="fallback",
                pass_number=pass_number,
                category=category,
                validation_passed=False,
                metrics=CompletionMetrics(),
            )

    def _select_pass2_backend(self, prior: TaskResult) -> BackendType:
        tried = set(prior.backends_tried)
        category = prior.category

        if prior.backend_used == "deterministic":
            if is_local_suitable(category) and BackendType.LOCAL.value not in tried:
                return BackendType.LOCAL
            return BackendType.FIREWORKS_STRONG if category in _HARD_PASS2_CATEGORIES else BackendType.FIREWORKS

        if prior.backend_used == "local":
            return BackendType.FIREWORKS_STRONG if category in _HARD_PASS2_CATEGORIES else BackendType.FIREWORKS

        if prior.backend_used == "fireworks":
            if category in _LOCAL_PASS2_CATEGORIES and BackendType.LOCAL.value not in tried:
                return BackendType.LOCAL
            if category in _HARD_PASS2_CATEGORIES and BackendType.FIREWORKS_STRONG.value not in tried:
                return BackendType.FIREWORKS_STRONG
            return BackendType.FIREWORKS

        if prior.backend_used in ("fireworks_strong", "emergency", "fallback"):
            if category in _LOCAL_PASS2_CATEGORIES and BackendType.LOCAL.value not in tried:
                return BackendType.LOCAL
            return BackendType.FIREWORKS

        return BackendType.FIREWORKS

    def _should_upgrade_pass2(self, prior: TaskResult, candidate: TaskResult) -> bool:
        """Decide whether Pass-2 should replace Pass-1.

        Critical rule: once a remote LLM (fireworks / fireworks_strong) was called
        and returned a usable answer, prefer that over a prior local/deterministic
        answer. Do not require a +0.05 confidence margin for LLM upgrades.
        """
        candidate_usable = is_usable_answer(candidate.answer)

        if candidate.backend_used in _LLM_BACKENDS and candidate_usable:
            # Reject clearly broken LLM output only when Pass-1 is already valid.
            if not candidate.validation_passed and prior.validation_passed and prior.confidence >= 0.9:
                # Still prefer LLM if prior was local/deterministic — user override.
                if prior.backend_used not in _LLM_BACKENDS:
                    return True
                return False
            return True

        if not candidate_usable:
            return False

        if candidate.validation_passed and not prior.validation_passed:
            return True
        if candidate.confidence > prior.confidence + 0.05:
            return True
        if prior.confidence < _PASS2_CONFIDENCE_THRESHOLD <= candidate.confidence:
            return True
        return False

    def _should_skip_pass2(self, prior: TaskResult) -> bool:
        """Skip Pass-2 when Pass-1 already used a remote LLM successfully."""
        return (
            prior.backend_used in _LLM_BACKENDS
            and prior.validation_passed
            and prior.confidence >= _PASS2_CONFIDENCE_THRESHOLD
        )

    def _run_pass2_retries(self, tasks: list[TaskItem]) -> None:
        if self._runtime.remaining_seconds < _PASS2_MIN_REMAINING_SECONDS:
            return

        retry_candidates = [
            self._task_results[task.task_id]
            for task in tasks
            if task.task_id in self._task_results
            and self._task_results[task.task_id].needs_pass2_retry(_PASS2_CONFIDENCE_THRESHOLD)
            and not self._should_skip_pass2(self._task_results[task.task_id])
        ]
        retry_candidates.sort(key=lambda tr: tr.confidence)
        retry_candidates = retry_candidates[:_PASS2_MAX_RETRIES]

        for prior in retry_candidates:
            if self._runtime_budget_exceeded():
                self.runtime_budget_exceeded = True
                break
            if self._runtime.remaining_seconds < _PASS2_MIN_REMAINING_SECONDS:
                break

            task = next(t for t in tasks if t.task_id == prior.task_id)
            backend = self._select_pass2_backend(prior)
            log_event(
                logger,
                "pass2_retry_scheduled",
                task_id=task.task_id,
                category=prior.category.value,
                prior_confidence=prior.confidence,
                prior_backend=prior.backend_used,
                retry_backend=backend.value,
            )

            try:
                execution = self._executor.execute_retry(
                    task,
                    prior.category,
                    backend=backend,
                    pass_number=2,
                )
                candidate = TaskResult(
                    task_id=task.task_id,
                    answer=execution.answer.strip() or prior.answer,
                    confidence=round(execution.verification.confidence, 3),
                    backend_used=execution.backend,
                    pass_number=2,
                    category=prior.category,
                    validation_passed=execution.verification.passed,
                    metrics=execution.metrics,
                    backends_tried=[*prior.backends_tried, execution.backend],
                )
            except Exception as exc:
                log_event(
                    logger,
                    "pass2_retry_failed",
                    task_id=task.task_id,
                    error=str(exc),
                    exc_info=True,
                )
                continue

            self._pass2_retry_count += 1
            self._runtime.record_task_complete(execution.metrics.latency_ms)

            if self._should_upgrade_pass2(prior, candidate):
                self._task_results[task.task_id] = candidate
                log_event(
                    logger,
                    "pass2_retry_upgraded",
                    task_id=task.task_id,
                    old_confidence=prior.confidence,
                    new_confidence=candidate.confidence,
                    backend=candidate.backend_used,
                )

    def process_tasks(self, tasks: list[TaskItem]) -> list[ResultItem]:
        """Process all tasks: pass 1 full batch, pass 2 low-confidence retries."""
        self._start_run()
        total = len(tasks)

        for idx, task in enumerate(tasks):
            self._runtime.set_remaining_tasks(total - idx)

            if self._runtime_budget_exceeded():
                self.runtime_budget_exceeded = True
                log_event(
                    logger,
                    "runtime_budget_exceeded",
                    task_id=task.task_id,
                    max_runtime_seconds=self._max_runtime_seconds,
                )
                self._task_results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    answer=_RUNTIME_LIMIT_ANSWER,
                    confidence=0.0,
                    backend_used="skipped",
                    pass_number=1,
                    category=classify_task(task),
                    validation_passed=False,
                )
                continue

            task_result = self._run_single_task(task, pass_number=1)
            self._task_results[task.task_id] = task_result
            self._runtime.record_task_complete(
                task_result.metrics.latency_ms if task_result.metrics else 0.0,
            )

        self._run_pass2_retries(tasks)
        self._log_two_pass_summary(len(tasks))
        return [
            ResultItem(task_id=task.task_id, answer=self._task_results[task.task_id].answer)
            for task in tasks
            if task.task_id in self._task_results
        ]

    def benchmark(self, tasks: list[TaskItem]) -> BenchmarkReport:
        """Run all tasks and return detailed metrics for tuning."""
        self._start_run()
        per_task_metrics: dict[str, CompletionMetrics] = {}
        total = len(tasks)

        for idx, task in enumerate(tasks):
            self._runtime.set_remaining_tasks(total - idx)

            if self._runtime_budget_exceeded():
                self.runtime_budget_exceeded = True
                log_event(
                    logger,
                    "runtime_budget_exceeded",
                    task_id=task.task_id,
                    max_runtime_seconds=self._max_runtime_seconds,
                )
                self._task_results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    answer=_RUNTIME_LIMIT_ANSWER,
                    confidence=0.0,
                    backend_used="skipped",
                    pass_number=1,
                    category=classify_task(task),
                    validation_passed=False,
                )
                per_task_metrics[task.task_id] = CompletionMetrics()
                continue

            task_result = self._run_single_task(task, pass_number=1)
            self._task_results[task.task_id] = task_result
            per_task_metrics[task.task_id] = task_result.metrics or CompletionMetrics()
            self._runtime.record_task_complete(
                task_result.metrics.latency_ms if task_result.metrics else 0.0,
            )
            log_event(
                logger,
                "benchmark_task",
                task_id=task.task_id,
                category=task_result.category.value,
                tokens=task_result.metrics.total_tokens if task_result.metrics else 0,
                latency_ms=round(
                    task_result.metrics.latency_ms if task_result.metrics else 0.0,
                    2,
                ),
                backend=task_result.backend_used,
                model=task_result.metrics.model if task_result.metrics else "",
                confidence=task_result.confidence,
                pass_number=1,
            )

        self._run_pass2_retries(tasks)

        for task in tasks:
            tr = self._task_results.get(task.task_id)
            if tr and tr.pass_number == 2 and tr.metrics:
                per_task_metrics[task.task_id] = tr.metrics

        self._log_two_pass_summary(len(tasks))

        results = [
            ResultItem(task_id=task.task_id, answer=self._task_results[task.task_id].answer)
            for task in tasks
            if task.task_id in self._task_results
        ]
        total_tokens = sum(m.total_tokens for m in per_task_metrics.values())
        total_latency_ms = sum(m.latency_ms for m in per_task_metrics.values())
        estimated_cost_usd = sum(m.estimated_cost_usd for m in per_task_metrics.values())
        return BenchmarkReport(
            total_tasks=len(tasks),
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            estimated_cost_usd=estimated_cost_usd,
            results=results,
            per_task_metrics=per_task_metrics,
        )

    def _log_two_pass_summary(self, pass1_count: int) -> None:
        confidences = [tr.confidence for tr in self._task_results.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        stats = self._runtime.stats
        log_event(
            logger,
            "run_summary",
            pass1_count=pass1_count,
            pass2_retries=self._pass2_retry_count,
            avg_confidence=round(avg_confidence, 3),
            deterministic_count=stats.deterministic_count,
            local_count=stats.local_count,
            fireworks_count=stats.fireworks_count,
            fireworks_strong_count=stats.fireworks_strong_count,
            retries=stats.retries,
            validation_failures=stats.validation_failures,
            escalations=stats.escalations,
            runtime_remaining_s=round(self._runtime.remaining_seconds, 1),
            runtime_elapsed_s=round(self._runtime.elapsed_seconds, 1),
            max_runtime_seconds=self._max_runtime_seconds,
        )

    def _log_run_summary(self) -> None:
        self._log_two_pass_summary(len(self._task_results))
