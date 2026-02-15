"""
Base judge implementation with shared logic.

This module provides BaseJudge, which implements the LLMJudge interface
using a transport layer for API calls. It handles:
- Prompt building (using prompt templates)
- JSON response parsing
- Score clamping to [0.0, 1.0]

Concrete judge classes (ClaudeJudge, OpenAIJudge) provide the transport.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ragaliq.judges.base import (
    ClaimsResult,
    ClaimVerdict,
    JudgeConfig,
    JudgeResponseError,
    JudgeResult,
    LLMJudge,
)
from ragaliq.judges.prompts.loader import get_prompt

if TYPE_CHECKING:
    from ragaliq.judges.trace import TraceCollector
    from ragaliq.judges.transport import JudgeTransport


class BaseJudge(LLMJudge):
    """
    Base LLM judge implementation with transport abstraction.

    BaseJudge implements all the prompt building, JSON parsing, and
    score clamping logic. It delegates actual API calls to a transport
    layer, making it easy to support multiple LLM providers.

    Subclasses only need to provide a transport instance.

    Example:
        class ClaudeJudge(BaseJudge):
            def __init__(self, api_key: str, config: JudgeConfig | None = None):
                transport = ClaudeTransport(api_key)
                super().__init__(transport, config)
    """

    def __init__(
        self,
        transport: JudgeTransport,
        config: JudgeConfig | None = None,
        *,
        trace_collector: TraceCollector | None = None,
    ) -> None:
        """
        Initialize base judge with transport.

        Args:
            transport: Transport layer for API calls.
            config: Judge configuration. Uses defaults if not provided.
            trace_collector: Optional trace collector for observability.
        """
        super().__init__(config)
        self._transport = transport
        self._trace_collector = trace_collector

    async def _call_llm(
        self, system_prompt: str, user_prompt: str, operation: str = "llm_call"
    ) -> tuple[str, int]:
        """
        Make an LLM call via the transport layer with tracing.

        Args:
            system_prompt: System message defining the LLM's role.
            user_prompt: User message with the task.
            operation: Name of the operation for trace logging.

        Returns:
            Tuple of (response_text, tokens_used).

        Raises:
            JudgeAPIError: If the API call fails.
            JudgeResponseError: If the response cannot be processed.
        """
        import time

        start_time = time.perf_counter()
        success = False
        error_msg = None

        try:
            response = await self._transport.send(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            tokens_used = response.input_tokens + response.output_tokens
            success = True

            return response.text, tokens_used

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            raise

        finally:
            # Emit trace if collector is configured
            if self._trace_collector is not None:
                from ragaliq.judges.trace import JudgeTrace

                latency_ms = int((time.perf_counter() - start_time) * 1000)

                # Use response token counts if available, else 0
                input_tokens = response.input_tokens if success else 0
                output_tokens = response.output_tokens if success else 0

                trace = JudgeTrace(
                    timestamp=datetime.now(timezone.utc),
                    operation=operation,
                    model=self.config.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    success=success,
                    error=error_msg,
                )
                self._trace_collector.add(trace)

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """
        Parse JSON from LLM response.

        Handles cases where the LLM wraps JSON in markdown code blocks.

        Args:
            text: Raw response text from LLM.

        Returns:
            Parsed JSON as dictionary.

        Raises:
            JudgeResponseError: If JSON parsing fails.
        """
        # Strip markdown code blocks if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag)
            lines = cleaned.split("\n")
            lines = lines[1:]  # Remove ```json or ```
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            result: dict[str, Any] = json.loads(cleaned)
            return result
        except json.JSONDecodeError as e:
            raise JudgeResponseError(
                f"Failed to parse JSON response: {e}. Raw text: {text[:200]}"
            ) from e

    def _build_faithfulness_prompt(
        self,
        response: str,
        context: list[str],
    ) -> tuple[str, str]:
        """
        Build prompts for faithfulness evaluation.

        Args:
            response: The RAG response to evaluate.
            context: Context documents used for generation.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        template = get_prompt("faithfulness")
        formatted_context = template.format_context(context)
        user_prompt = template.format_user_prompt(context=formatted_context, response=response)
        return template.system_prompt, user_prompt

    def _build_relevance_prompt(
        self,
        query: str,
        response: str,
    ) -> tuple[str, str]:
        """
        Build prompts for relevance evaluation.

        Args:
            query: The user's original query.
            response: The RAG response to evaluate.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        template = get_prompt("relevance")
        user_prompt = template.format_user_prompt(query=query, response=response)
        return template.system_prompt, user_prompt

    async def evaluate_faithfulness(
        self,
        response: str,
        context: list[str],
    ) -> JudgeResult:
        """
        Evaluate if the response is faithful to the provided context.

        Args:
            response: The RAG system's generated response.
            context: List of context documents used for generation.

        Returns:
            JudgeResult with faithfulness score (0.0-1.0) and reasoning.

        Raises:
            JudgeAPIError: If API call fails.
            JudgeResponseError: If response parsing fails.
        """
        if not context:
            # Edge case: no context means any claim is unsupported
            return JudgeResult(
                score=0.0,
                reasoning="No context provided; faithfulness cannot be assessed.",
                tokens_used=0,
            )

        system_prompt, user_prompt = self._build_faithfulness_prompt(response, context)
        raw_response, tokens_used = await self._call_llm(
            system_prompt, user_prompt, operation="evaluate_faithfulness"
        )
        parsed = self._parse_json_response(raw_response)

        # Validate required fields
        if "score" not in parsed:
            raise JudgeResponseError(f"Response missing 'score' field: {parsed}")

        try:
            score = float(parsed["score"])
        except (ValueError, TypeError) as e:
            raise JudgeResponseError(f"Invalid score value: {parsed['score']!r}") from e

        # Clamp score to valid range (defensive)
        score = max(0.0, min(1.0, score))

        return JudgeResult(
            score=score,
            reasoning=parsed.get("reasoning", ""),
            tokens_used=tokens_used,
        )

    async def evaluate_relevance(
        self,
        query: str,
        response: str,
    ) -> JudgeResult:
        """
        Evaluate if the response is relevant to the query.

        Args:
            query: The user's original query.
            response: The RAG system's generated response.

        Returns:
            JudgeResult with relevance score (0.0-1.0) and reasoning.

        Raises:
            JudgeAPIError: If API call fails.
            JudgeResponseError: If response parsing fails.
        """
        system_prompt, user_prompt = self._build_relevance_prompt(query, response)
        raw_response, tokens_used = await self._call_llm(
            system_prompt, user_prompt, operation="evaluate_relevance"
        )
        parsed = self._parse_json_response(raw_response)

        # Validate required fields
        if "score" not in parsed:
            raise JudgeResponseError(f"Response missing 'score' field: {parsed}")

        try:
            score = float(parsed["score"])
        except (ValueError, TypeError) as e:
            raise JudgeResponseError(f"Invalid score value: {parsed['score']!r}") from e

        # Clamp score to valid range (defensive)
        score = max(0.0, min(1.0, score))

        return JudgeResult(
            score=score,
            reasoning=parsed.get("reasoning", ""),
            tokens_used=tokens_used,
        )

    async def extract_claims(self, response: str) -> ClaimsResult:
        """
        Extract atomic claims from a response for verification.

        Args:
            response: The RAG system's generated response.

        Returns:
            ClaimsResult containing list of extracted claim strings.

        Raises:
            JudgeAPIError: If API call fails.
            JudgeResponseError: If response parsing fails.
        """
        # Handle empty response edge case
        if not response or not response.strip():
            return ClaimsResult(claims=[], tokens_used=0)

        template = get_prompt("extract_claims")
        user_prompt = template.format_user_prompt(response=response)
        raw_response, tokens_used = await self._call_llm(
            template.system_prompt, user_prompt, operation="extract_claims"
        )
        parsed = self._parse_json_response(raw_response)

        # Validate and extract claims list
        claims = parsed.get("claims", [])
        if not isinstance(claims, list):
            raise JudgeResponseError(
                f"Expected 'claims' to be a list, got {type(claims).__name__}: {parsed}"
            )

        # Filter out any non-string items (defensive)
        claims = [str(c) for c in claims if c]

        return ClaimsResult(claims=claims, tokens_used=tokens_used)

    async def verify_claim(
        self,
        claim: str,
        context: list[str],
    ) -> ClaimVerdict:
        """
        Verify if a single claim is supported by the context.

        Args:
            claim: The atomic claim to verify.
            context: List of context documents to check against.

        Returns:
            ClaimVerdict with verdict and supporting evidence.

        Raises:
            JudgeAPIError: If API call fails.
            JudgeResponseError: If response parsing fails.
        """
        # Handle empty context edge case
        if not context:
            return ClaimVerdict(
                verdict="NOT_ENOUGH_INFO",
                evidence="No context provided for verification.",
            )

        template = get_prompt("verify_claim")
        formatted_context = template.format_context(context)
        user_prompt = template.format_user_prompt(
            claim=claim,
            context=formatted_context,
        )
        raw_response, tokens_used = await self._call_llm(
            template.system_prompt, user_prompt, operation="verify_claim"
        )
        parsed = self._parse_json_response(raw_response)

        # Validate verdict field
        verdict = parsed.get("verdict", "").upper()
        valid_verdicts = {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}
        if verdict not in valid_verdicts:
            raise JudgeResponseError(
                f"Invalid verdict '{verdict}'. Expected one of {valid_verdicts}: {parsed}"
            )

        return ClaimVerdict(
            verdict=verdict,  # type: ignore[arg-type]
            evidence=parsed.get("evidence", ""),
            tokens_used=tokens_used,
        )
