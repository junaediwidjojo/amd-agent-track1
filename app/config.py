"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings injected by the hackathon harness or local .env."""

    model_config = SettingsConfigDict(
        env_file=(".env" if __import__("pathlib").Path(".env").is_file() else None),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fireworks_api_key: str = Field(alias="FIREWORKS_API_KEY")
    fireworks_base_url: str = Field(alias="FIREWORKS_BASE_URL")
    allowed_models: str = Field(alias="ALLOWED_MODELS")

    input_path: str = Field(default="/input/tasks.json", alias="INPUT_PATH")
    output_path: str = Field(default="/output/results.json", alias="OUTPUT_PATH")

    request_timeout_seconds: float = Field(default=25.0)
    max_retries: int = Field(default=2)
    retry_backoff_base: float = Field(default=0.5)
    prompt_cache_enabled: bool = Field(default=False)
    max_runtime_seconds: float = Field(default=540.0, alias="MAX_RUNTIME_SECONDS")

    # Local inference settings
    enable_local_model: bool = Field(default=True, alias="ENABLE_LOCAL_MODEL")
    local_model_path: str = Field(default="", alias="LOCAL_MODEL_PATH")
    local_n_ctx: int = Field(default=2048, alias="LOCAL_N_CTX")
    local_n_threads: int = Field(default=2, alias="LOCAL_N_THREADS")
    local_call_timeout_seconds: float = Field(
        default=18.0, alias="LOCAL_CALL_TIMEOUT_SECONDS"
    )
    local_categories: str = Field(
        default="sentiment,summarization,ner,factual",
        alias="LOCAL_CATEGORIES",
    )
    local_confidence_threshold: float = Field(
        default=0.7, alias="LOCAL_CONFIDENCE_THRESHOLD"
    )

    @property
    def model_list(self) -> list[str]:
        return [m.strip() for m in self.allowed_models.split(",") if m.strip()]

    @property
    def primary_model(self) -> str:
        models = self.model_list
        if not models:
            msg = "ALLOWED_MODELS is empty"
            raise ValueError(msg)
        return models[0]

    @property
    def fallback_models(self) -> list[str]:
        models = self.model_list
        return models[1:] if len(models) > 1 else []

    @property
    def local_category_set(self) -> set[str]:
        return {
            c.strip()
            for c in self.local_categories.split(",")
            if c.strip()
        }

    def pick_model(self, preferred_tags: list[str]) -> str:
        """Select the best model from ALLOWED_MODELS based on name heuristics.

        Tags are matched as substrings (case-insensitive). Falls back to primary_model.
        """
        models = self.model_list
        if not models:
            raise ValueError("ALLOWED_MODELS is empty")
        for tag in preferred_tags:
            tag_lower = tag.lower()
            for model in models:
                if tag_lower in model.lower():
                    return model
        for tag in ("minimax", "kimi", "moonshot", "glm"):
            for model in models:
                if tag in model.lower():
                    return model
        return self.primary_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
