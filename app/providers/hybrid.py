"""Hybrid provider that routes to local LLM first, with Fireworks fallback."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings, get_settings
from app.fireworks.models import CompletionMetrics, CompletionResult, TaskCategory, TaskItem
from app.providers.base import BaseLLMProvider
from app.providers.fireworks import FireworksProvider
from app.providers.local import LocalProvider
from app.utils.logger import get_logger, log_event
from app.verification import is_usable_answer, verify_answer

logger = get_logger(__name__)


class HybridProvider(BaseLLMProvider):
    """Routes tasks between local CPU inference and Fireworks API.

    Accepts local responses only when verification passes AND confidence meets
    threshold AND output format is valid. Otherwise escalates to Fireworks.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.fireworks = FireworksProvider()
        self.local: LocalProvider | None = None
        if self.settings.enable_local_model:
            model_path = self.settings.local_model_path.strip()
            if model_path and Path(model_path).is_file():
                self.local = LocalProvider(
                    model_path=model_path,
                    n_ctx=self.settings.local_n_ctx,
                    n_threads=self.settings.local_n_threads,
                )

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        model: str | None = None,
        category: str | None = None,
        task_id: str = "",
    ) -> CompletionResult:
        """Legacy entry point — local-first with verification-gated fallback."""
        cat_enum = self._parse_category(category)
        if cat_enum and cat_enum.value in self.settings.local_category_set and self.local is not None:
            task = TaskItem(task_id=task_id or "legacy", prompt=user)
            return self._try_local_then_fireworks(
                system, user, max_tokens, cat_enum, task_id, task,
            )
        return self.fireworks.complete(system, user, max_tokens, model)

    def complete_local(
        self,
        system: str,
        user: str,
        max_tokens: int,
        category: str,
        task_id: str,
        task: TaskItem,
    ) -> CompletionResult:
        """Run local inference with timeout handling."""
        cat_enum = self._parse_category(category)
        if self.local is None or cat_enum is None:
            return self.complete_fireworks(
                system, user, max_tokens,
                model=self.settings.pick_model(["fast", "small"]),
                category=cat_enum or TaskCategory.FACTUAL,
                task_id=task_id,
                task=task,
            )
        try:
            return self._run_local(system, user, max_tokens, cat_enum, task_id, task)
        except Exception as exc:
            log_event(
                logger,
                "hybrid_local_failed",
                task_id=task_id,
                category=cat_enum.value,
                error=str(exc),
                exc_info=True,
            )
            return self.complete_fireworks(
                system, user, max_tokens,
                model=self.settings.pick_model(self._primary_tags(cat_enum)),
                category=cat_enum,
                task_id=task_id,
                task=task,
            )

    def complete_fireworks(
        self,
        system: str,
        user: str,
        max_tokens: int,
        model: str,
        category: TaskCategory,
        task_id: str,
        task: TaskItem,
    ) -> CompletionResult:
        """Run Fireworks with verification-based escalation."""
        result = self.fireworks.complete(system, user, max_tokens, model=model)
        verification = verify_answer(result.text, task, category)
        result.metrics.confidence = verification.confidence
        result.metrics.backend = "fireworks"
        result.metrics.model = model

        if verification.passed and verification.confidence >= 0.6:
            return result

        escalation = self.pick_escalation_model(category)
        if escalation and escalation != model and verification.backend_recommendation == "fireworks_strong":
            log_event(
                logger,
                "hybrid_fireworks_escalation",
                task_id=task_id,
                category=category.value,
                primary_model=model,
                escalation_model=escalation,
                confidence=round(verification.confidence, 2),
                retry_reason=verification.retry_reason,
            )
            strong = self.fireworks.complete(system, user, max_tokens, model=escalation)
            strong_ver = verify_answer(strong.text, task, category)
            strong.metrics.confidence = strong_ver.confidence
            strong.metrics.backend = "fireworks_strong"
            strong.metrics.model = escalation
            # Once a stronger remote LLM returns a usable answer, keep it.
            if is_usable_answer(strong.text) and (
                strong_ver.confidence >= verification.confidence or not is_usable_answer(result.text)
            ):
                return strong
        return result

    def pick_escalation_model(self, category: TaskCategory) -> str | None:
        models = self.settings.model_list
        if len(models) < 2:
            return None
        if category in (
            TaskCategory.CODE_GENERATION,
            TaskCategory.DEBUGGING,
            TaskCategory.MATH,
            TaskCategory.LOGIC,
        ):
            for tag in ("kimi", "minimax", "moonshot", "reason", "code"):
                for model in models:
                    if tag in model.lower():
                        return model
        return models[-1]

    def _parse_category(self, category: str | None) -> TaskCategory | None:
        if category is None:
            return None
        try:
            return TaskCategory(category)
        except ValueError:
            return None

    def _try_local_then_fireworks(
        self,
        system: str,
        user: str,
        max_tokens: int,
        category: TaskCategory,
        task_id: str,
        task: TaskItem,
    ) -> CompletionResult:
        if self.local is None:
            return self.complete_fireworks(
                system, user, max_tokens,
                model=self.settings.pick_model(self._primary_tags(category)),
                category=category,
                task_id=task_id,
                task=task,
            )

        try:
            local_result = self._run_local(system, user, max_tokens, category, task_id, task)
        except Exception as exc:
            log_event(
                logger,
                "hybrid_local_failed",
                task_id=task_id,
                category=category.value,
                error=str(exc),
                exc_info=True,
            )
            return self.complete_fireworks(
                system, user, max_tokens,
                model=self.settings.pick_model(self._primary_tags(category)),
                category=category,
                task_id=task_id,
                task=task,
            )

        verification = verify_answer(local_result.text, task, category)
        local_result.metrics.confidence = verification.confidence

        log_event(
            logger,
            "hybrid_local_result",
            task_id=task_id,
            category=category.value,
            confidence=round(verification.confidence, 2),
            validator_passed=verification.passed,
            model=local_result.metrics.model,
            latency_ms=round(local_result.metrics.latency_ms, 2),
        )

        if (
            verification.passed
            and verification.confidence >= max(0.6, self.settings.local_confidence_threshold - 0.1)
        ):
            return local_result

        log_event(
            logger,
            "hybrid_fallback_to_fireworks",
            task_id=task_id,
            category=category.value,
            confidence=round(verification.confidence, 2),
            threshold=self.settings.local_confidence_threshold,
            retry_reason=verification.retry_reason,
        )
        return self.complete_fireworks(
            system, user, max_tokens,
            model=self.settings.pick_model(self._primary_tags(category)),
            category=category,
            task_id=task_id,
            task=task,
        )

    def _run_local(
        self,
        system: str,
        user: str,
        max_tokens: int,
        category: TaskCategory,
        task_id: str,
        task: TaskItem,
    ) -> CompletionResult:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

        assert self.local is not None
        timeout = self.settings.local_call_timeout_seconds
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.local.complete, system, user, max_tokens)
            try:
                local_result = future.result(timeout=timeout)
            except FutureTimeoutError:
                log_event(
                    logger,
                    "hybrid_local_timeout",
                    task_id=task_id,
                    category=category.value,
                    timeout_seconds=timeout,
                )
                raise
        local_result.metrics.backend = "local"
        verification = verify_answer(local_result.text, task, category)
        local_result.metrics.confidence = verification.confidence
        return local_result

    def _primary_tags(self, category: TaskCategory) -> list[str]:
        if category in (TaskCategory.CODE_GENERATION, TaskCategory.DEBUGGING):
            return ["code", "glm", "kimi", "minimax"]
        if category in (TaskCategory.MATH, TaskCategory.LOGIC):
            return ["reason", "math", "kimi", "glm"]
        return ["fast", "small", "glm", "kimi", "minimax"]
