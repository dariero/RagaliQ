"""
Judge trace observability for RagaliQ.

This module provides structured logging of LLM judge API calls for debugging,
cost tracking, and performance analysis.
"""

import threading
from datetime import datetime

from pydantic import BaseModel, Field


class JudgeTrace(BaseModel):
    """
    Structured record of a single LLM API call.

    Captures timing, token usage, and success/failure status without
    leaking raw chain-of-thought reasoning (which may contain PII).

    Attributes:
        timestamp: When the call was made (UTC).
        operation: Name of the judge method called.
        model: LLM model identifier used.
        input_tokens: Tokens in the prompt.
        output_tokens: Tokens in the response.
        latency_ms: Time from request to response in milliseconds.
        success: Whether the call succeeded.
        error: Error message if call failed, None otherwise.
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


class TraceCollector:
    """
    Session-scoped collector for judge traces.

    Accumulates traces from multiple API calls and provides aggregate
    statistics for debugging and cost estimation.

    Example:
        collector = TraceCollector()

        # During evaluation, traces are added
        trace = JudgeTrace(
            timestamp=datetime.now(timezone.utc),
            operation="evaluate_faithfulness",
            model="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
            latency_ms=2100,
            success=True,
        )
        collector.add(trace)

        # After evaluation, inspect statistics
        print(f"Total cost: ${collector.total_cost_estimate:.4f}")
        print(f"Total latency: {collector.total_latency_ms}ms")
    """

    def __init__(self) -> None:
        """Initialize empty trace collector."""
        self.traces: list[JudgeTrace] = []
        self._lock = threading.Lock()

    def add(self, trace: JudgeTrace) -> None:
        """
        Add a trace to the collection.

        Thread-safe: uses a lock to prevent corruption when multiple
        threads emit traces concurrently (e.g. pytest-xdist workers
        sharing a session-scoped collector).

        Args:
            trace: The JudgeTrace to record.
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
        """
        Rough cost estimate in USD based on token usage.

        Uses approximate pricing for Claude Sonnet 4:
        - Input: $3 per 1M tokens
        - Output: $15 per 1M tokens

        Note: This is a rough estimate. Actual costs may vary by model
        and provider pricing. Check your provider's pricing page for
        accurate rates.
        """
        # Rough pricing (per 1M tokens)
        input_cost_per_million = 3.0
        output_cost_per_million = 15.0

        input_cost = (self.total_input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (self.total_output_tokens / 1_000_000) * output_cost_per_million

        return input_cost + output_cost

    def get_by_operation(self, operation: str) -> list[JudgeTrace]:
        """
        Get all traces for a specific operation.

        Args:
            operation: Operation name to filter by.

        Returns:
            List of traces matching the operation.
        """
        return [t for t in self.traces if t.operation == operation]

    def get_failures(self) -> list[JudgeTrace]:
        """
        Get all failed traces.

        Returns:
            List of traces where success=False.
        """
        return [t for t in self.traces if not t.success]

    def clear(self) -> None:
        """Clear all collected traces."""
        with self._lock:
            self.traces.clear()

    def __repr__(self) -> str:
        """String representation with summary statistics."""
        return (
            f"TraceCollector("
            f"calls={len(self.traces)}, "
            f"tokens={self.total_tokens}, "
            f"latency={self.total_latency_ms}ms, "
            f"cost_estimate=${self.total_cost_estimate:.4f})"
        )
