"""ClaudeJudge: an LLM-as-Judge backed by Anthropic's Claude API."""

import os
from typing import TYPE_CHECKING, Self

from ragaliq.judges.base import JudgeConfig
from ragaliq.judges.base_judge import BaseJudge
from ragaliq.judges.transport import ClaudeTransport

if TYPE_CHECKING:
    from types import TracebackType

    from ragaliq.judges.trace import TraceCollector


class ClaudeJudge(BaseJudge):
    """LLM-as-Judge using Anthropic's Claude API over a `ClaudeTransport`.

    Example:
        judge = ClaudeJudge()
        result = await judge.evaluate_faithfulness(response, context)
    """

    def __init__(
        self,
        config: JudgeConfig | None = None,
        *,
        api_key: str | None = None,
        trace_collector: TraceCollector | None = None,
        max_concurrency: int = 20,
    ) -> None:
        """Initialize with an API key (explicit, else `ANTHROPIC_API_KEY`).

        Raises:
            ValueError: If no API key is provided or found in the environment.
        """
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key required. Provide via api_key parameter "
                "or set ANTHROPIC_API_KEY environment variable."
            )

        transport = ClaudeTransport(api_key=resolved_key)
        # Kept apart from BaseJudge's transport, which wrap_transport() may replace
        # with a send-only wrapper; the judge closes only the client it created.
        self._owned_transport = transport

        super().__init__(
            transport=transport,
            config=config,
            trace_collector=trace_collector,
            max_concurrency=max_concurrency,
        )

    async def aclose(self) -> None:
        """Close the HTTP client this judge created. The judge cannot call the API after this."""
        await self._owned_transport.aclose()

    async def __aenter__(self) -> Self:
        """Return the judge."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the HTTP client when leaving an `async with` block."""
        await self.aclose()
