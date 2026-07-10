"""Fireworks API provider adapter."""

from __future__ import annotations

from app.fireworks.client import FireworksClient
from app.fireworks.models import CompletionResult
from app.providers.base import BaseLLMProvider


class FireworksProvider(BaseLLMProvider):
    """Thin adapter that exposes the existing FireworksClient through the provider interface."""

    def __init__(self, client: FireworksClient | None = None) -> None:
        self.client = client or FireworksClient()

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        model: str | None = None,
        category: str | None = None,
        task_id: str = "",
    ) -> CompletionResult:
        return self.client.complete(
            system=system,
            user=user,
            max_tokens=max_tokens,
            model=model,
        )
