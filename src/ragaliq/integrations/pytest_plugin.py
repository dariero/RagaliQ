"""
Pytest plugin for RagaliQ.

This plugin integrates RagaliQ with pytest, providing:
- Command-line options for judge configuration
- Fixtures for easy test writing
- Cost tracking and limits
- Terminal summary with statistics

Bootstrap safety: All ragaliq imports are deferred to avoid import
errors if dependencies are missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from ragaliq.core.runner import RagaliQ
    from ragaliq.judges.base import LLMJudge
    from ragaliq.judges.trace import TraceCollector


def pytest_addoption(parser: Any) -> None:
    """
    Add RagaliQ command-line options to pytest.

    Options:
        --ragaliq-judge: LLM provider (claude, openai). Default: claude.
        --ragaliq-model: Model override. Default: uses judge's default.
        --ragaliq-api-key: API key override. Default: uses environment variable.
        --ragaliq-cost-limit: Max USD to spend. Default: no limit.
        --ragaliq-latency-ms: Artificial latency in ms. Default: 0 (no delay).
    """
    group = parser.getgroup("ragaliq", "RagaliQ LLM-as-Judge testing")

    group.addoption(
        "--ragaliq-judge",
        action="store",
        default="claude",
        help="LLM judge provider: claude or openai (default: claude)",
    )
    group.addoption(
        "--ragaliq-model",
        action="store",
        default=None,
        help="Model identifier override (default: judge's default model)",
    )
    group.addoption(
        "--ragaliq-api-key",
        action="store",
        default=None,
        help="API key override (default: environment variable)",
    )
    group.addoption(
        "--ragaliq-cost-limit",
        action="store",
        type=float,
        default=None,
        help="Maximum cost in USD before aborting (default: no limit)",
    )
    group.addoption(
        "--ragaliq-latency-ms",
        action="store",
        type=int,
        default=0,
        help="Artificial latency in milliseconds added to each judge call (default: 0)",
    )


def pytest_configure(config: Any) -> None:
    """
    Configure RagaliQ pytest plugin.

    Registers the @pytest.mark.ragaliq marker and initializes
    the trace collector for cost tracking.

    If ragaliq is not installed (e.g., running pytest without editable install),
    this hook gracefully skips initialization. Fixtures will fail with clear
    error messages if actually used.
    """
    config.addinivalue_line(
        "markers",
        "ragaliq: Mark test as a RagaliQ evaluation test",
    )

    # Initialize trace collector on config for session-wide tracking
    # Try to import - if ragaliq not installed, skip (plugin loads but is inactive)
    try:
        from ragaliq.judges.trace import TraceCollector

        config._ragaliq_trace_collector = TraceCollector()
    except ImportError, ModuleNotFoundError:
        # ragaliq not installed - plugin entry point loaded but can't initialize
        # This is expected in non-editable installs or when running pytest --collect-only
        config._ragaliq_trace_collector = None


@pytest.fixture(scope="session")
def ragaliq_trace_collector(request: Any) -> TraceCollector:
    """
    Session-scoped trace collector for cost tracking.

    Returns:
        TraceCollector instance shared across all tests.

    Raises:
        RuntimeError: If ragaliq is not installed (no editable install).
    """
    collector = request.config._ragaliq_trace_collector
    if collector is None:
        raise RuntimeError(
            "RagaliQ is not installed. Please install in editable mode: "
            "pip install -e . or pip install -e '.[dev]'"
        )
    return collector


@pytest.fixture(scope="session")
def ragaliq_judge(request: Any, ragaliq_trace_collector: TraceCollector) -> LLMJudge:
    """
    Session-scoped LLM judge instance.

    Configured from pytest command-line options:
    - --ragaliq-judge: Provider selection
    - --ragaliq-model: Model override
    - --ragaliq-api-key: API key override
    - --ragaliq-latency-ms: Artificial latency injection

    Returns:
        Configured LLMJudge instance.
    """
    import asyncio

    from ragaliq.judges import ClaudeJudge, JudgeConfig
    from ragaliq.judges.transport import JudgeTransport, TransportResponse

    judge_type = request.config.getoption("--ragaliq-judge")
    model = request.config.getoption("--ragaliq-model")
    api_key = request.config.getoption("--ragaliq-api-key")
    latency_ms = request.config.getoption("--ragaliq-latency-ms")

    # Build config if model override provided
    config = None
    if model:
        config = JudgeConfig(model=model)

    if judge_type == "claude":
        judge = ClaudeJudge(
            config=config,
            api_key=api_key,
            trace_collector=ragaliq_trace_collector,
        )

        # Wrap transport with latency injection if configured
        if latency_ms > 0:

            class LatencyInjectionTransport:
                """Transport wrapper that adds artificial delay."""

                def __init__(self, inner: JudgeTransport, delay_ms: int) -> None:
                    self._inner = inner
                    self._delay_ms = delay_ms

                async def send(
                    self,
                    system_prompt: str,
                    user_prompt: str,
                    model: str = "claude-sonnet-4-20250514",
                    temperature: float = 0.0,
                    max_tokens: int = 1024,
                ) -> TransportResponse:
                    """Add delay before delegating to inner transport."""
                    await asyncio.sleep(self._delay_ms / 1000.0)
                    return await self._inner.send(
                        system_prompt, user_prompt, model, temperature, max_tokens
                    )

            # Use public API to wrap transport (not private _transport mutation)
            judge.wrap_transport(LatencyInjectionTransport(judge.transport, latency_ms))

        return judge
    elif judge_type == "openai":
        raise NotImplementedError("OpenAI judge not yet implemented")
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")


@pytest.fixture(scope="session")
def judge_factory(ragaliq_judge: LLMJudge) -> Any:
    """
    Factory that returns the session judge.

    This is a convenience fixture for compatibility with tests
    that expect a judge factory pattern.

    Returns:
        Callable that returns the session judge.
    """

    def _factory() -> LLMJudge:
        return ragaliq_judge

    return _factory


@pytest.fixture
def ragaliq_runner(ragaliq_judge: LLMJudge) -> RagaliQ:
    """
    Pre-configured RagaliQ runner with session judge.

    Returns:
        RagaliQ instance ready for test evaluation.
    """
    from ragaliq.core.runner import RagaliQ

    return RagaliQ(judge=ragaliq_judge)


def pytest_runtest_makereport(item: Any, call: Any) -> None:
    """
    Hook called after each test phase (setup, call, teardown).

    Checks if cost limit has been exceeded after each test call.
    """
    if call.when != "call":
        return

    # Check cost limit
    cost_limit = item.config.getoption("--ragaliq-cost-limit")
    if cost_limit is None:
        return

    collector: TraceCollector | None = item.config._ragaliq_trace_collector
    if collector is None:
        return
    current_cost = collector.total_cost_estimate

    if current_cost > cost_limit:
        pytest.exit(
            f"RagaliQ cost limit exceeded: ${current_cost:.4f} > ${cost_limit:.2f}",
            returncode=1,
        )


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    """
    Add RagaliQ summary to pytest terminal output.

    Shows:
    - Total LLM calls made
    - Total tokens used
    - Estimated cost
    - Total latency
    - Number of failures
    """
    collector: TraceCollector | None = config._ragaliq_trace_collector
    if collector is None or len(collector.traces) == 0:
        return

    # Write summary header
    terminalreporter.write_sep("=", "RagaliQ Summary", bold=True)

    # Calculate statistics
    total_calls = len(collector.traces)
    total_tokens = collector.total_tokens
    total_cost = collector.total_cost_estimate
    total_latency_sec = collector.total_latency_ms / 1000
    failures = collector.failure_count

    # Write statistics
    terminalreporter.write_line(f"Total LLM calls: {total_calls}")
    terminalreporter.write_line(f"Total tokens: {total_tokens:,}")
    terminalreporter.write_line(f"Total cost estimate: ${total_cost:.4f}")
    terminalreporter.write_line(f"Total latency: {total_latency_sec:.1f}s")
    terminalreporter.write_line(f"Failures: {failures}")

    # Warn if approaching common budget limits
    if total_cost > 10.0:
        terminalreporter.write_line(f"WARNING: High cost detected (${total_cost:.2f})", red=True)
