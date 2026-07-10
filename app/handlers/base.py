"""Base handler with shared prompt loading and completion logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.fireworks.models import CompletionResult, TaskItem
from app.providers.base import BaseLLMProvider
from app.utils.json_utils import clean_answer


class BaseHandler(ABC):
    """Abstract handler for a single task category."""

    prompt_file: str

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider
        self._prompts_dir = Path(__file__).resolve().parent.parent / "prompts"

    def load_system_prompt(self) -> str:
        path = self._prompts_dir / self.prompt_file
        return path.read_text(encoding="utf-8").strip()

    @property
    def max_tokens(self) -> int:
        return 512

    @property
    def preferred_model_tags(self) -> list[str]:
        """Tags used to pick the best model from ALLOWED_MODELS for this category."""
        return []

    def complete(self, task: TaskItem) -> CompletionResult:
        from app.config import get_settings
        from app.utils.logger import get_logger, log_event
        logger = get_logger(__name__)
        system = self.load_system_prompt()
        settings = get_settings()
        model = settings.pick_model(self.preferred_model_tags)
        log_event(
            logger,
            "task_start",
            task_id=task.task_id,
            category=self.category_name(),
            model=model,
            prompt_file=self.prompt_file,
            system_prompt_preview=system[:200],
        )
        result = self.provider.complete(
            system=system,
            user=task.prompt,
            max_tokens=self.max_tokens,
            model=model,
            category=self.category_name(),
            task_id=task.task_id,
        )
        log_event(
            logger,
            "api_response_raw",
            task_id=task.task_id,
            model=model,
            raw_response=result.text[:2000] if len(result.text) > 2000 else result.text,
        )
        result.text = self.post_process(result.text, task)
        log_event(
            logger,
            "post_process_result",
            task_id=task.task_id,
            category=self.category_name(),
            post_processed=result.text[:1000] if len(result.text) > 1000 else result.text,
        )
        return result

    def post_process(self, text: str, task: TaskItem) -> str:
        return clean_answer(text)

    @abstractmethod
    def category_name(self) -> str:
        """Return the handler category identifier."""
