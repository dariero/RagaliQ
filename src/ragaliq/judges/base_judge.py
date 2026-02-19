"""
Base judge implementation with shared logic.

This module provides BaseJudge, which implements the LLMJudge interface
using a transport layer for API calls. It handles:
- Prompt building (using prompt templates)
- JSON response parsing
- Score clamping to [0.0, 1.0]
- Concurrency limiting to prevent API rate limit bursts

Concrete judge classes (ClaudeJudge, OpenAIJudge) provide the transport.
"""

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ragaliq.judges.base import (
    ClaimsResult,
    ClaimVerdict,
    GeneratedAnswerResult,
    GeneratedQuestionsResult,
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
        max_concurrency: int = 20,
    ) -> None:
        """
        Initialize base judge with transport.

        Args:
            transport: Transport layer for API calls.
            config: Judge configuration. Uses defaults if not provided.
            trace_collector: Optional trace collector for observability.
            max_concurrency: Maximum concurrent API calls allowed. Prevents
                rate limit bursts when evaluators have many claims/docs.
                Default: 20.
        """
        super().__init__(config)
        self._transport = transport
        self._trace_collector = trace_collector
        self._concurrency_limit = asyncio.Semaphore(max_concurrency)

    @property
    def transport(self) -> JudgeTransport:
        """
        Get the current transport layer.

        Returns:
            The transport instance used for API calls.
        """
        return self._transport

    def wrap_transport(self, wrapper: JudgeTransport) -> None:
        """
        Replace the transport layer with a wrapper.

        This is the official API for wrapping the transport with middleware
        (e.g., latency injection, retry logic, caching).

        Args:
            wrapper: New transport instance that wraps or replaces the current one.

        Example:
            # Wrap with latency injection
            class LatencyWrapper:
                def __init__(self, inner, delay_ms):
                    self._inner = inner
                    self._delay_ms = delay_ms

                async def send(self, ...):
                    await asyncio.sleep(self._delay_ms / 1000)
                    return await self._inner.send(...)

            judge.wrap_transport(LatencyWrapper(judge.transport, 100))
        """
        self._transport = wrapper

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
            # Limit concurrent API calls to prevent rate limit bursts
            async with self._concurrency_limit:
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

                # Use response data if available, else defaults
                if success:
                    # Record actual model from response (may differ from config)
                    input_tokens = response.input_tokens
                    output_tokens = response.output_tokens
                    actual_model = response.model
                else:
                    # No response on failure, use config model and zero tokens
                    input_tokens = 0
                    output_tokens = 0
                    actual_model = self.config.model

                trace = JudgeTrace(
                    timestamp=datetime.now(UTC),
                    operation=operation,
                    model=actual_model,
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

    def _build_generate_questions_prompt(
        self,
        documents: list[str],
        n: int,
    ) -> tuple[str, str]:
        """
        Build prompts for question generation.

        Args:
            documents: Source documents to generate questions from.
            n: Number of questions to generate.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        template = get_prompt("generate_questions")
        formatted_docs = template.format_context(documents)
        user_prompt = template.format_user_prompt(n=n, documents=formatted_docs)
        return template.system_prompt, user_prompt

    def _build_generate_answer_prompt(
        self,
        question: str,
        context: list[str],
    ) -> tuple[str, str]:
        """
        Build prompts for answer generation.

        Args:
            question: The question to answer.
            context: Context documents to answer from.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        template = get_prompt("generate_answer")
        formatted_context = template.format_context(context)
        user_prompt = template.format_user_prompt(context=formatted_context, question=question)
        return template.system_prompt, user_prompt

    async def generate_questions(
        self,
        documents: list[str],
        n: int,
    ) -> GeneratedQuestionsResult:
        """
        Generate questions grounded in the provided documents.

        Args:
            documents: Source documents to generate questions from.
            n: Number of questions to generate.

        Returns:
            GeneratedQuestionsResult containing the list of questions.

        Raises:
            JudgeAPIError: If API call fails.
            JudgeResponseError: If response parsing fails.
        """
        if not documents:
            return GeneratedQuestionsResult(questions=[], tokens_used=0)

        system_prompt, user_prompt = self._build_generate_questions_prompt(documents, n)
        raw_response, tokens_used = await self._call_llm(
            system_prompt, user_prompt, operation="generate_questions"
        )
        parsed = self._parse_json_response(raw_response)

        questions = parsed.get("questions", [])
        if not isinstance(questions, list):
            raise JudgeResponseError(
                f"Expected 'questions' to be a list, got {type(questions).__name__}: {parsed}"
            )

        questions = [str(q) for q in questions if q]
        return GeneratedQuestionsResult(questions=questions, tokens_used=tokens_used)

    async def generate_answer(
        self,
        question: str,
        context: list[str],
    ) -> GeneratedAnswerResult:
        """
        Generate an answer to a question using only the provided context.

        Args:
            question: The question to answer.
            context: List of context documents to answer from.

        Returns:
            GeneratedAnswerResult containing the generated answer.

        Raises:
            JudgeAPIError: If API call fails.
            JudgeResponseError: If response parsing fails.
        """
        if not context:
            return GeneratedAnswerResult(answer="", tokens_used=0)

        system_prompt, user_prompt = self._build_generate_answer_prompt(question, context)
        raw_response, tokens_used = await self._call_llm(
            system_prompt, user_prompt, operation="generate_answer"
        )
        parsed = self._parse_json_response(raw_response)

        answer = parsed.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        return GeneratedAnswerResult(answer=answer, tokens_used=tokens_used)

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
            verdict=verdict,
            evidence=parsed.get("evidence", ""),
            tokens_used=tokens_used,
        )
