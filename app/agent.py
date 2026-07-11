"""Core agent orchestrating routing, handlers, and result assembly."""

from __future__ import annotations

import time

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
from app.solvers.fallback import solver_fallback
from app.utils.logger import get_logger, log_event

logger = get_logger(__name__)

_RUNTIME_LIMIT_ANSWER = "Unable to process this task within runtime limit."


class Agent:
    """General-purpose task agent with rule-based routing."""

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
        self._run_start: float | None = None
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

    def get_handler(self, category: TaskCategory) -> BaseHandler:
        return self._handlers[category]

    def _start_run(self) -> None:
        self._run_start = time.monotonic()
        self.runtime_budget_exceeded = False

    def _runtime_budget_exceeded(self) -> bool:
        if self._run_start is None:
            return False
        return (time.monotonic() - self._run_start) >= self._max_runtime_seconds

    def _runtime_limit_result(self, task: TaskItem) -> ResultItem:
        return ResultItem(task_id=task.task_id, answer=_RUNTIME_LIMIT_ANSWER)

    def process_task(self, task: TaskItem) -> tuple[ResultItem, CompletionMetrics, TaskCategory]:
        """Process a single task and return result with metrics."""
        category = classify_task(task)
        handler = self.get_handler(category)

        log_event(
            logger,
            "task_routed",
            task_id=task.task_id,
            category=category.value,
        )

        try:
            completion = handler.complete(task)
            answer = completion.text.strip() or "No answer generated."
            metrics = completion.metrics
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
        for task in tasks:
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
            result, _, _ = self.process_task(task)
            results.append(result)
        return results

    def benchmark(self, tasks: list[TaskItem]) -> BenchmarkReport:
        """Run all tasks and return detailed metrics for tuning."""
        self._start_run()
        results: list[ResultItem] = []
        per_task_metrics: dict[str, CompletionMetrics] = {}

        for task in tasks:
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
