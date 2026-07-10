"""Pydantic data models for tasks, results, and API metrics."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskCategory(str, Enum):
    FACTUAL = "factual"
    SUMMARIZATION = "summarization"
    MATH = "math"
    SENTIMENT = "sentiment"
    NER = "ner"
    LOGIC = "logic"
    DEBUGGING = "debugging"
    CODE_GENERATION = "code_generation"
    STRUCTURED_EXTRACTION = "structured_extraction"
    STRUCTURED_WRITING = "structured_writing"


class OutputShape(str, Enum):
    JSON_OBJECT = "json_object"
    JSON_ARRAY = "json_array"
    CODE_ONLY = "code_only"
    EXACT_BULLETS = "exact_bullets"
    EXACT_SECTIONS = "exact_sections"
    NUMERIC_ONLY = "numeric_only"
    FREE_TEXT = "free_text"


class TaskItem(BaseModel):
    task_id: str
    prompt: str


class ResultItem(BaseModel):
    task_id: str
    answer: str


class CompletionMetrics(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    cached: bool = False
    backend: str = "fireworks"
    confidence: Optional[float] = None

    @property
    def estimated_cost_usd(self) -> float:
        # Fireworks pricing varies; use conservative blended estimate for observability.
        # Local inference counts as zero cost.
        if self.backend == "local":
            return 0.0
        return (self.prompt_tokens * 0.20 + self.completion_tokens * 0.20) / 1_000_000


class CompletionResult(BaseModel):
    text: str
    metrics: CompletionMetrics = Field(default_factory=CompletionMetrics)


class BenchmarkReport(BaseModel):
    total_tasks: int
    total_tokens: int
    total_latency_ms: float
    estimated_cost_usd: float
    results: list[ResultItem]
    per_task_metrics: dict[str, CompletionMetrics]
