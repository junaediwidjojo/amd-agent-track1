"""Hybrid provider that routes to local LLM first, with Fireworks fallback."""

from __future__ import annotations

import ast
import json
import re

from app.config import Settings, get_settings
from app.fireworks.models import CompletionMetrics, CompletionResult, TaskCategory
from app.providers.base import BaseLLMProvider
from app.providers.fireworks import FireworksProvider
from app.providers.local import LocalProvider
from app.utils.logger import get_logger, log_event

logger = get_logger(__name__)


class HybridProvider(BaseLLMProvider):
    """Routes tasks between local CPU inference and Fireworks API.

    Categories configured in ``local_categories`` are attempted locally first.
    If the local response fails validation (confidence < threshold),
    the provider silently falls back to Fireworks.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.fireworks = FireworksProvider()
        self.local: LocalProvider | None = None
        if self.settings.local_model_path:
            self.local = LocalProvider(
                model_path=self.settings.local_model_path,
                n_ctx=self.settings.local_n_ctx,
                n_threads=self.settings.local_n_threads,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        model: str | None = None,
        category: str | None = None,
        task_id: str = "",
    ) -> CompletionResult:
        cat_enum = self._parse_category(category)
        if cat_enum and cat_enum.value in self.settings.local_category_set:
            if self.local is not None:
                return self._try_local_then_fireworks(
                    system, user, max_tokens, cat_enum, task_id
                )
            log_event(
                logger,
                "hybrid_local_disabled",
                task_id=task_id,
                category=cat_enum.value,
                reason="local_model_path not set",
            )
        return self.fireworks.complete(system, user, max_tokens, model)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    ) -> CompletionResult:
        assert self.local is not None
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.local.complete, system, user, max_tokens)
            try:
                local_result = future.result(timeout=20)
            except FutureTimeoutError:
                log_event(
                    logger,
                    "hybrid_local_timeout",
                    task_id=task_id,
                    category=category.value,
                    timeout_seconds=20,
                )
                return self.fireworks.complete(system, user, max_tokens)

        confidence = self._compute_confidence(local_result.text, category)
        local_result.metrics.confidence = confidence

        log_event(
            logger,
            "hybrid_local_result",
            task_id=task_id,
            category=category.value,
            confidence=round(confidence, 2),
            model=local_result.metrics.model,
            latency_ms=round(local_result.metrics.latency_ms, 2),
        )

        if confidence >= self.settings.local_confidence_threshold:
            return local_result

        log_event(
            logger,
            "hybrid_fallback_to_fireworks",
            task_id=task_id,
            category=category.value,
            confidence=round(confidence, 2),
            threshold=self.settings.local_confidence_threshold,
        )
        return self.fireworks.complete(system, user, max_tokens)

    # ------------------------------------------------------------------
    # Confidence validators (lightweight, CPU-only)
    # ------------------------------------------------------------------

    def _compute_confidence(self, text: str, category: TaskCategory) -> float:
        validator = _VALIDATORS.get(category)
        if validator is None:
            return 0.0
        return validator(text)


# ------------------------------------------------------------------
# Category-specific validators
# ------------------------------------------------------------------

def _sentiment_confidence(text: str) -> float:
    lower = text.lower().strip()
    labels = ("positive", "negative", "neutral", "mixed")
    if lower in labels:
        return 1.0
    if any(l in lower for l in labels):
        return 0.7
    return 0.0


def _summary_confidence(text: str) -> float:
    t = text.strip()
    if not t:
        return 0.0
    if len(t) < 10:
        return 0.4
    if len(t) > 800:
        return 0.5
    return 0.85


def _ner_confidence(text: str) -> float:
    t = text.strip()
    try:
        json.loads(t)
        return 0.9
    except json.JSONDecodeError:
        pass
    if re.search(r"\[.*\]", t, re.DOTALL):
        return 0.4
    return 0.0


def _factual_confidence(text: str) -> float:
    t = text.strip()
    if not t or len(t) < 2:
        return 0.0
    if re.fullmatch(r"-?\d+(?:\.\d+)?", t):
        return 0.0
    if len(t.split()) < 4:
        return 0.4
    if len(t) > 1200:
        return 0.5
    return 0.85


def _math_confidence(text: str) -> float:
    t = text.strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", t)
    if numbers and len(t.split()) <= 5:
        return 0.9
    return 0.0


def _logic_confidence(text: str) -> float:
    t = text.strip()
    if not t or len(t) < 2:
        return 0.0
    if len(t) > 200:
        return 0.5
    return 0.8


def _code_confidence(text: str) -> float:
    t = text.strip()
    try:
        ast.parse(t)
        return 0.9
    except SyntaxError:
        return 0.0


def _debug_confidence(text: str) -> float:
    lines = text.strip().split("\n")
    if len(lines) >= 2 and any("def " in line for line in lines[:3]):
        return 0.8
    return 0.0


_VALIDATORS: dict[TaskCategory, object] = {
    TaskCategory.SENTIMENT: _sentiment_confidence,
    TaskCategory.SUMMARIZATION: _summary_confidence,
    TaskCategory.NER: _ner_confidence,
    TaskCategory.FACTUAL: _factual_confidence,
    TaskCategory.MATH: _math_confidence,
    TaskCategory.LOGIC: _logic_confidence,
    TaskCategory.CODE_GENERATION: _code_confidence,
    TaskCategory.DEBUGGING: _debug_confidence,
}
