"""Core agent orchestrating routing, handlers, and result assembly."""

from __future__ import annotations

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
from app.utils.logger import get_logger, log_event

logger = get_logger(__name__)

_RUNTIME_LIMIT_ANSWER = "Unable to process this task within runtime limit."


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

    def get_handler(self, category: TaskCategory) -> BaseHandler:
        return self._handlers[category]

    def _start_run(self) -> None:
        self._runtime.start()
        self.runtime_budget_exceeded = False

    def _runtime_budget_exceeded(self) -> bool:
        return self._runtime.is_budget_exceeded()

    def _runtime_limit_result(self, task: TaskItem) -> ResultItem:
        return ResultItem(task_id=task.task_id, answer=_RUNTIME_LIMIT_ANSWER)

    def process_task(self, task: TaskItem) -> tuple[ResultItem, CompletionMetrics, TaskCategory]:
        """Process a single task through the verification pipeline."""
        category = classify_task(task)

        log_event(
            logger,
            "task_routed",
            task_id=task.task_id,
            category=category.value,
        )

        try:
            emergency = self._runtime.is_emergency()
            execution = self._executor.execute(task, category, emergency=emergency)
            answer = execution.answer.strip() or "No answer generated."
            metrics = execution.metrics
            log_event(
                logger,
                "task_complete",
                task_id=task.task_id,
                category=category.value,
                backend=execution.backend,
                validator_passed=execution.verification.passed,
                final_confidence=round(execution.verification.confidence, 2),
                retry_count=execution.retry_count,
                escalation_reason=execution.escalation_reason,
                latency_ms=round(execution.latency_ms, 2),
            )
        except APIExhaustedError as exc:
            log_event(
                logger,
                "task_error",
                task_id=task.task_id,
                category=category.value,
                exc_info=True,
                primary_model=exc.primary_model,
                last_error=str(exc.last_error) if exc.last_error else None,
            )
            fallback = solver_fallback(task, category)
            answer = fallback or "Unable to process this task."
            metrics = CompletionMetrics()
        except Exception as exc:
            log_event(
                logger,
                "task_error",
                task_id=task.task_id,
                category=category.value,
                error=str(exc),
                exc_info=True,
            )
            fallback = solver_fallback(task, category)
            answer = fallback or "Unable to process this task."
            metrics = CompletionMetrics()

        return ResultItem(task_id=task.task_id, answer=answer), metrics, category

    def process_tasks(self, tasks: list[TaskItem]) -> list[ResultItem]:
        """Process all tasks sequentially, never failing the entire batch."""
        self._start_run()
        results: list[ResultItem] = []
        total = len(tasks)

        for idx, task in enumerate(tasks):
            self._runtime.set_remaining_tasks(total - idx)

            if self._runtime.is_emergency():
                log_event(
                    logger,
                    "runtime_emergency_mode",
                    task_id=task.task_id,
                    remaining_seconds=round(self._runtime.remaining_seconds, 1),
                    tasks_remaining=total - idx,
                )
                category = classify_task(task)
                execution = self._executor.execute(task, category, emergency=True)
                results.append(ResultItem(task_id=task.task_id, answer=execution.answer))
                self._runtime.record_task_complete(execution.latency_ms)
                continue

            if self._runtime_budget_exceeded():
                self.runtime_budget_exceeded = True
                log_event(
                    logger,
                    "runtime_budget_exceeded",
                    task_id=task.task_id,
                    max_runtime_seconds=self._max_runtime_seconds,
                )
                results.append(self._runtime_limit_result(task))
                continue

            result, metrics, _ = self.process_task(task)
            results.append(result)
            self._runtime.record_task_complete(metrics.latency_ms)

        self._log_run_summary()
        return results

    def benchmark(self, tasks: list[TaskItem]) -> BenchmarkReport:
        """Run all tasks and return detailed metrics for tuning."""
        self._start_run()
        results: list[ResultItem] = []
        per_task_metrics: dict[str, CompletionMetrics] = {}
        total = len(tasks)

        for idx, task in enumerate(tasks):
            self._runtime.set_remaining_tasks(total - idx)

            if self._runtime.is_emergency():
                category = classify_task(task)
                execution = self._executor.execute(task, category, emergency=True)
                result = ResultItem(task_id=task.task_id, answer=execution.answer)
                results.append(result)
                per_task_metrics[task.task_id] = execution.metrics
                self._runtime.record_task_complete(execution.metrics.latency_ms)
                continue

            if self._runtime_budget_exceeded():
                self.runtime_budget_exceeded = True
                log_event(
                    logger,
                    "runtime_budget_exceeded",
                    task_id=task.task_id,
                    max_runtime_seconds=self._max_runtime_seconds,
                )
                result = self._runtime_limit_result(task)
                metrics = CompletionMetrics()
                category = classify_task(task)
                results.append(result)
                per_task_metrics[task.task_id] = metrics
                continue

            result, metrics, category = self.process_task(task)
            results.append(result)
            per_task_metrics[task.task_id] = metrics
            self._runtime.record_task_complete(metrics.latency_ms)
            log_event(
                logger,
                "benchmark_task",
                task_id=task.task_id,
                category=category.value,
                tokens=metrics.total_tokens,
                latency_ms=round(metrics.latency_ms, 2),
                backend=metrics.backend,
                model=metrics.model,
                confidence=round(metrics.confidence, 2) if metrics.confidence is not None else None,
            )

        self._log_run_summary()

        total_tokens = sum(m.total_tokens for m in per_task_metrics.values())
        total_latency_ms = sum(m.latency_ms for m in per_task_metrics.values())
        estimated_cost_usd = sum(
            m.estimated_cost_usd for m in per_task_metrics.values()
        )
        return BenchmarkReport(
            total_tasks=len(tasks),
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            estimated_cost_usd=estimated_cost_usd,
            results=results,
            per_task_metrics=per_task_metrics,
        )

    def _log_run_summary(self) -> None:
        stats = self._runtime.stats
        log_event(
            logger,
            "run_summary",
            deterministic_count=stats.deterministic_count,
            local_count=stats.local_count,
            fireworks_count=stats.fireworks_count,
            fireworks_strong_count=stats.fireworks_strong_count,
            retries=stats.retries,
            validation_failures=stats.validation_failures,
            escalations=stats.escalations,
            runtime_remaining_s=round(self._runtime.remaining_seconds, 1),
            runtime_elapsed_s=round(self._runtime.elapsed_seconds, 1),
        )
