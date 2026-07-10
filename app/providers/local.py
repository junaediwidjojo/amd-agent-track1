"""Local llama.cpp provider for zero-token inference."""

from __future__ import annotations

import os
import time

from app.fireworks.models import CompletionMetrics, CompletionResult
from app.providers.base import BaseLLMProvider


class LocalProvider(BaseLLMProvider):
    """CPU-based local inference using llama-cpp-python.

    Designed for the AMD hackathon 4 GB RAM / 2 vCPU constraint.
    Loads a small (2 B–3 B) 4-bit quantized model on first use.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int | None = None,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads or min(os.cpu_count() or 2, 2)
        self._llm: object | None = None

    def _load(self) -> object:
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                msg = (
                    "llama-cpp-python is not installed. "
                    "Install it with: pip install llama-cpp-python"
                )
                raise RuntimeError(msg) from exc
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
        return self._llm

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        model: str | None = None,
        category: str | None = None,
        task_id: str = "",
    ) -> CompletionResult:
        llm = self._load()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        start = time.perf_counter()
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        text = output["choices"][0]["message"]["content"] or ""
        usage = output.get("usage", {})
        metrics = CompletionMetrics(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            model=os.path.basename(self.model_path),
            backend="local",
        )
        return CompletionResult(text=text, metrics=metrics)
