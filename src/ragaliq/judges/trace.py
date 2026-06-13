"""
Judge trace observability for RagaliQ.

This module provides structured logging of LLM judge API calls for debugging,
cost tracking, and performance analysis.
"""

import threading
from datetime import datetime

from pydantic import BaseModel, Field

from ragaliq.judges.models import DEFAULT_JUDGE_MODEL, GOLD_STANDARD_JUDGE_MODEL


class JudgeTrace(BaseModel):
    """Structured record of one LLM API call.

    Captures timing, token usage, and success/failure — deliberately not the raw
    chain-of-thought reasoning, which may contain PII.
    """

    timestamp: datetime = Field(..., description="Call timestamp (UTC)")
    operation: str = Field(..., description="Judge method name (e.g., 'evaluate_faithfulness')")
    model: str = Field(..., description="LLM model identifier")
    input_tokens: int = Field(..., ge=0, description="Prompt token count")
    output_tokens: int = Field(..., ge=0, description="Response token count")
    latency_ms: int = Field(..., ge=0, description="Call latency in milliseconds")
    success: bool = Field(..., description="Whether call succeeded")
    error: str | None = Field(default=None, description="Error message if failed")

    model_config = {"frozen": True, "extra": "forbid"}


# Per-model pricing: (input_cost_per_million, output_cost_per_million) in USD.
# Source: Anthropic pricing page. Override via TraceCollector(model_pricing=...).
_DEFAULT_MODEL_PRICING: dict[str, tuple[float, float]] = {
    DEFAULT_JUDGE_MODEL: (3.0, 15.0),
    GOLD_STANDARD_JUDGE_MODEL: (5.0, 25.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}

# Fallback for unknown models (uses Sonnet 4 pricing as reasonable middle ground).
_FALLBACK_PRICING: tuple[float, float] = (3.0, 15.0)


class TraceCollector:
    """Session-scoped collector of judge traces with aggregate stats.

    Accumulates traces across API calls and exposes token, latency, and rough
    cost totals for debugging and cost estimation.
    """

    def __init__(
        self,
        model_pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize an empty collector.

        Args:
            model_pricing: Optional per-model (input, output) USD-per-million
                overrides, merged over the built-in defaults (user wins).
        """
        self.traces: list[JudgeTrace] = []
        self._lock = threading.Lock()
        self._pricing = {**_DEFAULT_MODEL_PRICING, **(model_pricing or {})}

    def add(self, trace: JudgeTrace) -> None:
        """Record a trace (thread-safe).

        The lock guards concurrent emitters, e.g. pytest-xdist workers sharing a
        session-scoped collector.
        """
        with self._lock:
            self.traces.append(trace)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all calls (input + output)."""
        return sum(t.input_tokens + t.output_tokens for t in self.traces)

    @property
    def total_input_tokens(self) -> int:
        """Total input (prompt) tokens across all calls."""
        return sum(t.input_tokens for t in self.traces)

    @property
    def total_output_tokens(self) -> int:
        """Total output (response) tokens across all calls."""
        return sum(t.output_tokens for t in self.traces)

    @property
    def total_latency_ms(self) -> int:
        """Total latency across all calls in milliseconds."""
        return sum(t.latency_ms for t in self.traces)

    @property
    def success_count(self) -> int:
        """Number of successful calls."""
        return sum(1 for t in self.traces if t.success)

    @property
    def failure_count(self) -> int:
        """Number of failed calls."""
        return sum(1 for t in self.traces if not t.success)

    @property
    def total_cost_estimate(self) -> float:
        """Rough USD cost estimate from per-model token pricing.

        Unknown models fall back to Sonnet pricing. Ignores pricing tiers,
        caching discounts, and batch API usage, so treat it as approximate.
        """
        total = 0.0
        for trace in self.traces:
            input_rate, output_rate = self._pricing.get(trace.model, _FALLBACK_PRICING)
            total += (trace.input_tokens / 1_000_000) * input_rate
            total += (trace.output_tokens / 1_000_000) * output_rate
        return total

    def get_by_operation(self, operation: str) -> list[JudgeTrace]:
        """Return all traces for the given operation."""
        return [t for t in self.traces if t.operation == operation]

    def get_failures(self) -> list[JudgeTrace]:
        """Return all traces where `success` is False."""
        return [t for t in self.traces if not t.success]

    def clear(self) -> None:
        """Clear all collected traces."""
        with self._lock:
            self.traces.clear()

    def __repr__(self) -> str:
        return (
            f"TraceCollector("
            f"calls={len(self.traces)}, "
            f"tokens={self.total_tokens}, "
            f"latency={self.total_latency_ms}ms, "
            f"cost_estimate=${self.total_cost_estimate:.4f})"
        )
