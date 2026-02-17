"""Unit tests for JudgeTrace and TraceCollector."""

import contextlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ragaliq.judges.trace import JudgeTrace, TraceCollector


class TestJudgeTrace:
    """Tests for JudgeTrace model."""

    def test_create_successful_trace(self) -> None:
        """Test creating a successful trace."""
        trace = JudgeTrace(
            timestamp=datetime.now(UTC),
            operation="evaluate_faithfulness",
            model="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
            latency_ms=2100,
            success=True,
        )

        assert trace.operation == "evaluate_faithfulness"
        assert trace.model == "claude-sonnet-4-20250514"
        assert trace.input_tokens == 100
        assert trace.output_tokens == 50
        assert trace.latency_ms == 2100
        assert trace.success is True
        assert trace.error is None

    def test_create_failed_trace(self) -> None:
        """Test creating a failed trace with error message."""
        trace = JudgeTrace(
            timestamp=datetime.now(UTC),
            operation="verify_claim",
            model="claude-sonnet-4-20250514",
            input_tokens=80,
            output_tokens=0,
            latency_ms=150,
            success=False,
            error="JudgeAPIError: Rate limit exceeded",
        )

        assert trace.success is False
        assert trace.error == "JudgeAPIError: Rate limit exceeded"
        assert trace.output_tokens == 0

    def test_trace_is_immutable(self) -> None:
        """Test that traces are frozen (immutable)."""
        trace = JudgeTrace(
            timestamp=datetime.now(UTC),
            operation="extract_claims",
            model="claude-sonnet-4-20250514",
            input_tokens=50,
            output_tokens=30,
            latency_ms=1800,
            success=True,
        )

        with pytest.raises(ValidationError):  # Pydantic raises validation error on frozen model
            trace.success = False  # type: ignore[misc]


class TestTraceModelAccuracy:
    """Tests that traces record actual model from response, not config."""

    @pytest.mark.asyncio
    async def test_trace_records_actual_model_from_response(self) -> None:
        """Trace should record model from response, not config."""
        from unittest.mock import AsyncMock, MagicMock

        from ragaliq.judges.base_judge import BaseJudge
        from ragaliq.judges.transport import TransportResponse

        # Mock transport that returns different model than requested
        mock_transport = MagicMock()
        mock_transport.send = AsyncMock(
            return_value=TransportResponse(
                text='{"score": 0.9}',
                input_tokens=100,
                output_tokens=50,
                model="claude-opus-4-20250514",  # Different from config!
            )
        )

        collector = TraceCollector()

        # Create judge with config requesting sonnet, but transport returns opus
        from ragaliq.judges.base import JudgeConfig

        config = JudgeConfig(model="claude-sonnet-4-20250514")
        judge = BaseJudge(transport=mock_transport, config=config, trace_collector=collector)

        # Make a call
        await judge._call_llm("system", "user", operation="test_op")

        # Trace should record opus (what we got), not sonnet (what we asked for)
        assert len(collector.traces) == 1
        assert collector.traces[0].model == "claude-opus-4-20250514"
        assert collector.traces[0].model != config.model

    @pytest.mark.asyncio
    async def test_trace_uses_config_model_on_failure(self) -> None:
        """On failure, trace should fall back to config model."""
        from unittest.mock import AsyncMock, MagicMock

        from ragaliq.judges.base import JudgeConfig
        from ragaliq.judges.base_judge import BaseJudge

        # Mock transport that raises exception
        mock_transport = MagicMock()
        mock_transport.send = AsyncMock(side_effect=RuntimeError("API failed"))

        collector = TraceCollector()
        config = JudgeConfig(model="claude-sonnet-4-20250514")
        judge = BaseJudge(transport=mock_transport, config=config, trace_collector=collector)

        # Make a call that fails
        with contextlib.suppress(RuntimeError):
            await judge._call_llm("system", "user", operation="test_op")

        # Trace should record config model (no response to get actual model from)
        assert len(collector.traces) == 1
        assert collector.traces[0].model == "claude-sonnet-4-20250514"
        assert collector.traces[0].success is False


class TestTraceCollector:
    """Tests for TraceCollector."""

    def test_collector_starts_empty(self) -> None:
        """New collector has no traces."""
        collector = TraceCollector()

        assert len(collector.traces) == 0
        assert collector.total_tokens == 0
        assert collector.total_latency_ms == 0
        assert collector.success_count == 0
        assert collector.failure_count == 0

    def test_add_trace(self) -> None:
        """Test adding a trace to the collector."""
        collector = TraceCollector()

        trace = JudgeTrace(
            timestamp=datetime.now(UTC),
            operation="evaluate_relevance",
            model="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
            latency_ms=2000,
            success=True,
        )
        collector.add(trace)

        assert len(collector.traces) == 1
        assert collector.traces[0] == trace

    def test_total_tokens(self) -> None:
        """Test total token calculation."""
        collector = TraceCollector()

        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="op1",
                model="model",
                input_tokens=100,
                output_tokens=50,
                latency_ms=1000,
                success=True,
            )
        )
        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="op2",
                model="model",
                input_tokens=200,
                output_tokens=75,
                latency_ms=1500,
                success=True,
            )
        )

        # Total = (100+50) + (200+75) = 425
        assert collector.total_tokens == 425
        assert collector.total_input_tokens == 300
        assert collector.total_output_tokens == 125

    def test_total_latency(self) -> None:
        """Test total latency calculation."""
        collector = TraceCollector()

        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="op1",
                model="model",
                input_tokens=50,
                output_tokens=25,
                latency_ms=2100,
                success=True,
            )
        )
        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="op2",
                model="model",
                input_tokens=60,
                output_tokens=30,
                latency_ms=1800,
                success=True,
            )
        )

        assert collector.total_latency_ms == 3900

    def test_success_and_failure_counts(self) -> None:
        """Test counting successful and failed calls."""
        collector = TraceCollector()

        # Add 2 successful
        for _ in range(2):
            collector.add(
                JudgeTrace(
                    timestamp=datetime.now(UTC),
                    operation="op",
                    model="model",
                    input_tokens=50,
                    output_tokens=25,
                    latency_ms=1000,
                    success=True,
                )
            )

        # Add 1 failed
        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="op",
                model="model",
                input_tokens=50,
                output_tokens=0,
                latency_ms=200,
                success=False,
                error="Error",
            )
        )

        assert collector.success_count == 2
        assert collector.failure_count == 1
        assert len(collector.traces) == 3

    def test_cost_estimate(self) -> None:
        """Test rough cost estimation."""
        collector = TraceCollector()

        # Add trace with 1M input tokens and 1M output tokens
        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="op",
                model="claude-sonnet-4-20250514",
                input_tokens=1_000_000,
                output_tokens=1_000_000,
                latency_ms=10000,
                success=True,
            )
        )

        # Rough estimate: $3/M input + $15/M output = $18
        cost = collector.total_cost_estimate
        assert cost == pytest.approx(18.0, abs=0.01)

    def test_get_by_operation(self) -> None:
        """Test filtering traces by operation."""
        collector = TraceCollector()

        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="evaluate_faithfulness",
                model="model",
                input_tokens=100,
                output_tokens=50,
                latency_ms=2000,
                success=True,
            )
        )
        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="verify_claim",
                model="model",
                input_tokens=80,
                output_tokens=40,
                latency_ms=1500,
                success=True,
            )
        )
        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="verify_claim",
                model="model",
                input_tokens=85,
                output_tokens=42,
                latency_ms=1600,
                success=True,
            )
        )

        faithfulness_traces = collector.get_by_operation("evaluate_faithfulness")
        verify_traces = collector.get_by_operation("verify_claim")

        assert len(faithfulness_traces) == 1
        assert len(verify_traces) == 2
        assert all(t.operation == "verify_claim" for t in verify_traces)

    def test_get_failures(self) -> None:
        """Test filtering failed traces."""
        collector = TraceCollector()

        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="op1",
                model="model",
                input_tokens=100,
                output_tokens=50,
                latency_ms=2000,
                success=True,
            )
        )
        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="op2",
                model="model",
                input_tokens=80,
                output_tokens=0,
                latency_ms=200,
                success=False,
                error="Error 1",
            )
        )
        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="op3",
                model="model",
                input_tokens=90,
                output_tokens=0,
                latency_ms=150,
                success=False,
                error="Error 2",
            )
        )

        failures = collector.get_failures()

        assert len(failures) == 2
        assert all(not t.success for t in failures)
        assert failures[0].error == "Error 1"
        assert failures[1].error == "Error 2"

    def test_clear(self) -> None:
        """Test clearing all traces."""
        collector = TraceCollector()

        # Add some traces
        for _ in range(3):
            collector.add(
                JudgeTrace(
                    timestamp=datetime.now(UTC),
                    operation="op",
                    model="model",
                    input_tokens=50,
                    output_tokens=25,
                    latency_ms=1000,
                    success=True,
                )
            )

        assert len(collector.traces) == 3

        collector.clear()

        assert len(collector.traces) == 0
        assert collector.total_tokens == 0
        assert collector.total_latency_ms == 0

    def test_repr(self) -> None:
        """Test string representation."""
        collector = TraceCollector()

        collector.add(
            JudgeTrace(
                timestamp=datetime.now(UTC),
                operation="op",
                model="model",
                input_tokens=100,
                output_tokens=50,
                latency_ms=2000,
                success=True,
            )
        )

        repr_str = repr(collector)

        assert "TraceCollector" in repr_str
        assert "calls=1" in repr_str
        assert "tokens=150" in repr_str
        assert "latency=2000ms" in repr_str
        assert "cost_estimate=$" in repr_str
