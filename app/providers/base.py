"""Base provider interface for all LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.fireworks.models import CompletionResult


class BaseLLMProvider(ABC):
    """Abstract interface for local or remote LLM inference."""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        model: str | None = None,
        category: str | None = None,
        task_id: str = "",
    ) -> CompletionResult:
        """Execute a chat completion and return a standard result."""
        ...
