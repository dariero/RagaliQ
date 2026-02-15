"""
ClaudeJudge implementation for RagaliQ.

This module provides an LLM-as-Judge implementation using Anthropic's Claude API.
It evaluates RAG responses for faithfulness and relevance using structured prompts.
"""

from __future__ import annotations

import os

from ragaliq.judges.base import JudgeConfig
from ragaliq.judges.base_judge import BaseJudge
from ragaliq.judges.transport import ClaudeTransport


class ClaudeJudge(BaseJudge):
    """
    LLM-as-Judge implementation using Anthropic's Claude API.

    ClaudeJudge evaluates RAG responses by sending structured prompts to Claude
    and parsing JSON responses containing scores and reasoning.

    Example:
        judge = ClaudeJudge()
        result = await judge.evaluate_faithfulness(
            response="Paris is the capital of France.",
            context=["France is a country in Europe. Its capital is Paris."]
        )
        print(f"Faithfulness: {result.score}")

    Attributes:
        config: Judge configuration (model, temperature, max_tokens).
    """

    def __init__(
        self,
        config: JudgeConfig | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize ClaudeJudge with optional configuration.

        Args:
            config: Judge configuration. Uses defaults if not provided.
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        # Resolve API key: explicit > environment
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key required. Provide via api_key parameter "
                "or set ANTHROPIC_API_KEY environment variable."
            )

        # Create transport and initialize base
        transport = ClaudeTransport(api_key=resolved_key)
        super().__init__(transport=transport, config=config)
