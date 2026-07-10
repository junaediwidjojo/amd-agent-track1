"""Fireworks API client with retries, fallback, caching, and metrics."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from app.config import Settings, get_settings
from app.fireworks.models import CompletionMetrics, CompletionResult
from app.utils.logger import get_logger, log_event

logger = get_logger(__name__)


class APIExhaustedError(Exception):
    """Raised when all available models have been exhausted after retries."""

    def __init__(
        self,
        message: str,
        primary_model: str,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.primary_model = primary_model
        self.last_error = last_error


@dataclass
class TokenCounter:
    """Accumulates token usage across all API calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    cache_hits: int = 0
    total_latency_ms: float = 0.0
    per_model: dict[str, int] = field(default_factory=dict)

    def record(self, metrics: CompletionMetrics) -> None:
        if metrics.cached:
            self.cache_hits += 1
            return
        self.prompt_tokens += metrics.prompt_tokens
        self.completion_tokens += metrics.completion_tokens
        self.total_tokens += metrics.total_tokens
        self.total_latency_ms += metrics.latency_ms
        self.request_count += 1
        self.per_model[metrics.model] = self.per_model.get(metrics.model, 0) + metrics.total_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return (self.prompt_tokens * 0.20 + self.completion_tokens * 0.20) / 1_000_000

    def summary(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "cache_hits": self.cache_hits,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "per_model": self.per_model,
        }


class FireworksClient:
    """Thin wrapper around the OpenAI-compatible Fireworks API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.token_counter = TokenCounter()
        self._prompt_cache: dict[str, CompletionResult] = {}
        self._client = OpenAI(
            api_key=self.settings.fireworks_api_key,
            base_url=self.settings.fireworks_base_url,
            timeout=self.settings.request_timeout_seconds,
        )

    def _cache_key(self, model: str, system: str, user: str) -> str:
        raw = f"{model}|{system}|{user}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _backoff_sleep(self, attempt: int) -> None:
        delay = self.settings.retry_backoff_base * (2**attempt)
        time.sleep(delay)

    def _call_model(
        self, model: str, system: str, user: str, max_tokens: int = 512
    ) -> CompletionResult:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage

        metrics = CompletionMetrics(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=latency_ms,
            model=model,
            cached=False,
        )

        log_event(
            logger,
            "completion",
            model=model,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            latency_ms=round(latency_ms, 2),
        )
        log_event(
            logger,
            "api_response_raw",
            model=model,
            raw_response=text[:2000] if len(text) > 2000 else text,
        )

        return CompletionResult(text=text, metrics=metrics)

    def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 512,
    ) -> CompletionResult:
        """Execute a chat completion with cache, retry, and model fallback."""
        primary = model or self.settings.primary_model
        models_to_try = [primary, *self.settings.fallback_models]
        # Deduplicate while preserving order
        seen: set[str] = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        if self.settings.prompt_cache_enabled:
            cache_key = self._cache_key(primary, system, user)
            cached = self._prompt_cache.get(cache_key)
            if cached is not None:
                cached.metrics.cached = True
                self.token_counter.record(cached.metrics)
                log_event(logger, "cache_hit", model=primary)
                return cached

        last_error: Exception | None = None

        for model_id in models_to_try:
            for attempt in range(self.settings.max_retries + 1):
                try:
                    result = self._call_model(model_id, system, user, max_tokens=max_tokens)
                    self.token_counter.record(result.metrics)

                    if self.settings.prompt_cache_enabled:
                        cache_key = self._cache_key(primary, system, user)
                        self._prompt_cache[cache_key] = result

                    return result

                except (APITimeoutError, RateLimitError) as exc:
                    last_error = exc
                    log_event(
                        logger,
                        "completion_error",
                        model=model_id,
                        attempt=attempt,
                        error=str(exc),
                        exc_info=True,
                    )
                    if attempt < self.settings.max_retries:
                        self._backoff_sleep(attempt)
                    continue
                except APIError as exc:
                    last_error = exc
                    # Fatal errors (4xx except 429) should not be retried
                    status = getattr(exc, "status_code", None)
                    if status and 400 <= status < 500 and status != 429:
                        log_event(
                            logger,
                            "completion_fatal",
                            model=model_id,
                            status=status,
                            error=str(exc),
                            exc_info=True,
                        )
                        # Break out of retry loop to try next model or fail fast
                        break
                    log_event(
                        logger,
                        "completion_error",
                        model=model_id,
                        attempt=attempt,
                        error=str(exc),
                        exc_info=True,
                    )
                    if attempt < self.settings.max_retries:
                        self._backoff_sleep(attempt)
                    continue
                except Exception as exc:
                    last_error = exc
                    log_event(
                        logger,
                        "completion_unexpected",
                        model=model_id,
                        error=str(exc),
                        exc_info=True,
                    )
                    # Do not retry unexpected exceptions; they are likely code bugs
                    break

        error_msg = f"All models failed. Last error: {last_error}"
        log_event(logger, "completion_exhausted", error=error_msg)
        raise APIExhaustedError(
            message=error_msg,
            primary_model=primary,
            last_error=last_error,
        )
