"""
ClaudeJudge implementation for RagaliQ.

This module provides an LLM-as-Judge implementation using Anthropic's Claude API.
It evaluates RAG responses for faithfulness and relevance using structured prompts.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ragaliq.judges.base import (
    ClaimsResult,
    ClaimVerdict,
    JudgeAPIError,
    JudgeConfig,
    JudgeResponseError,
    JudgeResult,
    LLMJudge,
)
from ragaliq.judges.prompts.loader import get_prompt

if TYPE_CHECKING:
    from anthropic.types import Message


class ClaudeJudge(LLMJudge):
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
        client: Anthropic async client instance.
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
        super().__init__(config)

        # Resolve API key: explicit > environment
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key required. Provide via api_key parameter "
                "or set ANTHROPIC_API_KEY environment variable."
            )

        self._client = AsyncAnthropic(api_key=resolved_key)

    @retry(
        retry=retry_if_exception_type(APIConnectionError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_claude(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int]:
        """
        Make a retry-enabled call to Claude API.

        Uses tenacity for automatic retries on connection errors with
        exponential backoff (1s, 2s, 4s max 10s between attempts).

        Args:
            system_prompt: System message defining Claude's role.
            user_prompt: User message with the evaluation task.

        Returns:
            Tuple of (response_text, tokens_used).

        Raises:
            JudgeAPIError: If API call fails after retries.
        """
        try:
            response: Message = await self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
            )

            # Extract text content from response
            content = response.content[0]
            if content.type != "text":
                raise JudgeResponseError(f"Expected text response, got {content.type}")

            # Calculate total tokens used
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            return content.text, tokens_used

        except APIStatusError as e:
            # Map Anthropic status errors to our exception hierarchy
            raise JudgeAPIError(
                f"Claude API error: {e.message}",
                status_code=e.status_code,
            ) from e
        except APIConnectionError:
            # Let tenacity handle retries; if exhausted, reraise
            raise

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """
        Parse JSON from Claude's response.

        Handles cases where Claude wraps JSON in markdown code blocks.

        Args:
            text: Raw response text from Claude.

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
        system_prompt = """You are an expert evaluator assessing whether a response is faithful to the provided context.

Faithfulness means:
- Every claim in the response is supported by the context
- No information is fabricated or hallucinated
- No claims go beyond what the context states

Score criteria:
- 1.0: All claims are fully supported by context
- 0.7-0.9: Most claims supported, minor unsupported details
- 0.4-0.6: Mixed - some claims supported, others not
- 0.1-0.3: Most claims unsupported by context
- 0.0: Response contradicts or is completely unrelated to context

Respond ONLY with valid JSON in this exact format:
{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}"""

        # Format context documents with clear separation
        context_text = "\n\n---\n\n".join(
            f"Document {i + 1}:\n{doc}" for i, doc in enumerate(context)
        )

        user_prompt = f"""Evaluate the faithfulness of this response to the context.

CONTEXT:
{context_text}

RESPONSE TO EVALUATE:
{response}

Return JSON with score (0.0-1.0) and reasoning."""

        return system_prompt, user_prompt

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
        system_prompt = """You are an expert evaluator assessing whether a response is relevant to the user's query.

Relevance means:
- The response directly addresses what the user asked
- The information provided is useful for answering the query
- The response stays on topic and doesn't include irrelevant information

Score criteria:
- 1.0: Response perfectly addresses the query
- 0.7-0.9: Response mostly addresses query with minor gaps
- 0.4-0.6: Response partially addresses query or includes irrelevant info
- 0.1-0.3: Response barely addresses the query
- 0.0: Response is completely irrelevant to the query

Respond ONLY with valid JSON in this exact format:
{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}"""

        user_prompt = f"""Evaluate the relevance of this response to the query.

QUERY:
{query}

RESPONSE TO EVALUATE:
{response}

Return JSON with score (0.0-1.0) and reasoning."""

        return system_prompt, user_prompt

    async def evaluate_faithfulness(
        self,
        response: str,
        context: list[str],
    ) -> JudgeResult:
        """
        Evaluate if the response is faithful to the provided context.

        Uses Claude to analyze whether claims in the response are
        supported by the context documents.

        Args:
            response: The RAG system's generated response.
            context: List of context documents used for generation.

        Returns:
            JudgeResult with faithfulness score (0.0-1.0) and reasoning.

        Raises:
            JudgeAPIError: If Claude API call fails.
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

        try:
            raw_response, tokens_used = await self._call_claude(system_prompt, user_prompt)
        except APIConnectionError as e:
            raise JudgeAPIError(f"Connection to Claude API failed: {e}") from e

        parsed = self._parse_json_response(raw_response)

        # Validate required fields
        if "score" not in parsed:
            raise JudgeResponseError(f"Response missing 'score' field: {parsed}")

        score = float(parsed["score"])
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

        Uses Claude to analyze whether the response addresses
        the user's question effectively.

        Args:
            query: The user's original query.
            response: The RAG system's generated response.

        Returns:
            JudgeResult with relevance score (0.0-1.0) and reasoning.

        Raises:
            JudgeAPIError: If Claude API call fails.
            JudgeResponseError: If response parsing fails.
        """
        system_prompt, user_prompt = self._build_relevance_prompt(query, response)

        try:
            raw_response, tokens_used = await self._call_claude(system_prompt, user_prompt)
        except APIConnectionError as e:
            raise JudgeAPIError(f"Connection to Claude API failed: {e}") from e

        parsed = self._parse_json_response(raw_response)

        # Validate required fields
        if "score" not in parsed:
            raise JudgeResponseError(f"Response missing 'score' field: {parsed}")

        score = float(parsed["score"])
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

        Uses the extract_claims prompt template to break down the response
        into individual, verifiable claims.

        Args:
            response: The RAG system's generated response.

        Returns:
            ClaimsResult containing list of extracted claim strings.

        Raises:
            JudgeAPIError: If Claude API call fails.
            JudgeResponseError: If response parsing fails.
        """
        # Handle empty response edge case
        if not response or not response.strip():
            return ClaimsResult(claims=[], tokens_used=0)

        template = get_prompt("extract_claims")
        user_prompt = template.format_user_prompt(response=response)

        try:
            raw_response, tokens_used = await self._call_claude(template.system_prompt, user_prompt)
        except APIConnectionError as e:
            raise JudgeAPIError(f"Connection to Claude API failed: {e}") from e

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

        Uses the verify_claim prompt template with three-way classification:
        SUPPORTED, CONTRADICTED, or NOT_ENOUGH_INFO.

        Args:
            claim: The atomic claim to verify.
            context: List of context documents to check against.

        Returns:
            ClaimVerdict with verdict and supporting evidence.

        Raises:
            JudgeAPIError: If Claude API call fails.
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

        try:
            raw_response, _ = await self._call_claude(template.system_prompt, user_prompt)
        except APIConnectionError as e:
            raise JudgeAPIError(f"Connection to Claude API failed: {e}") from e

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
        )
