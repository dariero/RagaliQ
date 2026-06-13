"""ClaudeJudge: an LLM-as-Judge backed by Anthropic's Claude API."""

import os
from typing import TYPE_CHECKING

from ragaliq.judges.base import JudgeConfig
from ragaliq.judges.base_judge import BaseJudge
from ragaliq.judges.transport import ClaudeTransport

if TYPE_CHECKING:
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

        super().__init__(
            transport=ClaudeTransport(api_key=resolved_key),
            config=config,
            trace_collector=trace_collector,
            max_concurrency=max_concurrency,
        )
