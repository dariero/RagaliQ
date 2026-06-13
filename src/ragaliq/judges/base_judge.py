"""Base judge implementation shared by concrete providers.

`BaseJudge` implements the `LLMJudge` interface — prompt building, JSON parsing,
score clamping, and concurrency limiting — and delegates the actual API call to a
pluggable transport. Concrete classes (ClaudeJudge, OpenAIJudge) supply transport.
"""

import asyncio
import json
import logging
import time
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

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragaliq.judges.trace import TraceCollector
    from ragaliq.judges.transport import JudgeTransport


_CHARS_PER_TOKEN_ESTIMATE = 4
_DEFAULT_INPUT_TOKEN_WARN_THRESHOLD = 100_000
_ERROR_PREVIEW_LENGTH = 200


class BaseJudge(LLMJudge):
    """LLM judge that implements all shared logic over a pluggable transport.

    Subclasses only need to supply a transport instance.

    Example:
        class ClaudeJudge(BaseJudge):
            def __init__(self, api_key, config=None):
                super().__init__(ClaudeTransport(api_key), config)
    """

    def __init__(
        self,
        transport: JudgeTransport,
        config: JudgeConfig | None = None,
        *,
        trace_collector: TraceCollector | None = None,
        max_concurrency: int = 20,
    ) -> None:
        """Initialize with a transport, optional config, and optional trace collector.

        Args:
            transport: Transport layer for API calls.
            config: Judge configuration. Uses defaults if not provided.
            trace_collector: Optional trace collector for observability.
            max_concurrency: Cap on concurrent API calls, to avoid rate-limit
                bursts when evaluators fan out over many claims/docs.
        """
        super().__init__(config)
        self._transport = transport
        self._trace_collector = trace_collector
        self._concurrency_limit = asyncio.Semaphore(max_concurrency)

    @property
    def transport(self) -> JudgeTransport:
        """The transport instance used for API calls."""
        return self._transport

    def wrap_transport(self, wrapper: JudgeTransport) -> None:
        """Replace the transport, e.g. to add middleware (latency, retry, caching).

        Args:
            wrapper: New transport that wraps or replaces the current one.
        """
        self._transport = wrapper

    async def _call_llm(
        self, system_prompt: str, user_prompt: str, operation: str = "llm_call"
    ) -> tuple[str, int]:
        """Call the transport once, emitting a trace (on success or failure) if configured.

        Returns:
            Tuple of (response_text, tokens_used).
        """
        start_time = time.perf_counter()
        success = False
        error_msg = None

        estimated_tokens = (len(system_prompt) + len(user_prompt)) // _CHARS_PER_TOKEN_ESTIMATE
        if estimated_tokens > _DEFAULT_INPUT_TOKEN_WARN_THRESHOLD:
            logger.warning(
                "Large input detected for '%s': ~%d estimated tokens "
                "(threshold: %d). This may exceed model context limits "
                "or incur high costs.",
                operation,
                estimated_tokens,
                _DEFAULT_INPUT_TOKEN_WARN_THRESHOLD,
            )

        try:
            # Bound concurrent API calls to avoid rate-limit bursts.
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
            if self._trace_collector is not None:
                from ragaliq.judges.trace import JudgeTrace

                latency_ms = int((time.perf_counter() - start_time) * 1000)
                # The response's model may differ from the configured one.
                if success:
                    input_tokens = response.input_tokens
                    output_tokens = response.output_tokens
                    actual_model = response.model
                else:
                    input_tokens = 0
                    output_tokens = 0
                    actual_model = self.config.model

                self._trace_collector.add(
                    JudgeTrace(
                        timestamp=datetime.now(UTC),
                        operation=operation,
                        model=actual_model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=latency_ms,
                        success=success,
                        error=error_msg,
                    )
                )

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from an LLM response, unwrapping a markdown code fence if present.

        Raises:
            JudgeResponseError: If JSON parsing fails.
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]  # drop the opening ```json / ``` fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            result: dict[str, Any] = json.loads(cleaned)
            return result
        except json.JSONDecodeError as e:
            raise JudgeResponseError(
                f"Failed to parse JSON response: {e}. Raw text: {text[:_ERROR_PREVIEW_LENGTH]}"
            ) from e

    def _parse_score(self, parsed: dict[str, Any]) -> float:
        """Extract the 'score' field, coerce to float, and clamp to [0.0, 1.0].

        Raises:
            JudgeResponseError: If 'score' is missing or not numeric.
        """
        if "score" not in parsed:
            raise JudgeResponseError(f"Response missing 'score' field: {parsed}")
        try:
            score = float(parsed["score"])
        except (ValueError, TypeError) as e:
            raise JudgeResponseError(f"Invalid score value: {parsed['score']!r}") from e
        return max(0.0, min(1.0, score))

    def _parse_string_list(self, parsed: dict[str, Any], key: str) -> list[str]:
        """Extract `key` as a list of non-empty strings, rejecting non-list values.

        Raises:
            JudgeResponseError: If the field is present but not a list.
        """
        value = parsed.get(key, [])
        if not isinstance(value, list):
            raise JudgeResponseError(
                f"Expected {key!r} to be a list, got {type(value).__name__}: {parsed}"
            )
        return [str(item) for item in value if item]

    def _build_faithfulness_prompt(self, response: str, context: list[str]) -> tuple[str, str]:
        """Build (system, user) prompts for faithfulness evaluation."""
        template = get_prompt("faithfulness")
        formatted_context = template.format_context(context)
        user_prompt = template.format_user_prompt(context=formatted_context, response=response)
        return template.build_system_prompt(), user_prompt

    def _build_relevance_prompt(self, query: str, response: str) -> tuple[str, str]:
        """Build (system, user) prompts for relevance evaluation."""
        template = get_prompt("relevance")
        user_prompt = template.format_user_prompt(query=query, response=response)
        return template.build_system_prompt(), user_prompt

    def _build_generate_questions_prompt(self, documents: list[str], n: int) -> tuple[str, str]:
        """Build (system, user) prompts for question generation."""
        template = get_prompt("generate_questions")
        formatted_docs = template.format_context(documents)
        user_prompt = template.format_user_prompt(n=n, documents=formatted_docs)
        return template.build_system_prompt(), user_prompt

    def _build_generate_answer_prompt(self, question: str, context: list[str]) -> tuple[str, str]:
        """Build (system, user) prompts for answer generation."""
        template = get_prompt("generate_answer")
        formatted_context = template.format_context(context)
        user_prompt = template.format_user_prompt(context=formatted_context, question=question)
        return template.build_system_prompt(), user_prompt

    async def evaluate_faithfulness(self, response: str, context: list[str]) -> JudgeResult:
        """Score how faithful the response is to the context (0.0 if no context)."""
        if not context:
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
        return JudgeResult(
            score=self._parse_score(parsed),
            reasoning=parsed.get("reasoning", ""),
            tokens_used=tokens_used,
        )

    async def evaluate_relevance(self, query: str, response: str) -> JudgeResult:
        """Score how relevant the response is to the query (0.0 if either is empty)."""
        if not query or not query.strip() or not response or not response.strip():
            return JudgeResult(
                score=0.0,
                reasoning="Empty query or response; relevance cannot be assessed.",
                tokens_used=0,
            )

        system_prompt, user_prompt = self._build_relevance_prompt(query, response)
        raw_response, tokens_used = await self._call_llm(
            system_prompt, user_prompt, operation="evaluate_relevance"
        )
        parsed = self._parse_json_response(raw_response)
        return JudgeResult(
            score=self._parse_score(parsed),
            reasoning=parsed.get("reasoning", ""),
            tokens_used=tokens_used,
        )

    async def extract_claims(self, response: str) -> ClaimsResult:
        """Extract atomic claims from a response (empty for blank input)."""
        if not response or not response.strip():
            return ClaimsResult(claims=[], tokens_used=0)

        template = get_prompt("extract_claims")
        user_prompt = template.format_user_prompt(response=response)
        raw_response, tokens_used = await self._call_llm(
            template.build_system_prompt(), user_prompt, operation="extract_claims"
        )
        parsed = self._parse_json_response(raw_response)
        claims = self._parse_string_list(parsed, "claims")
        return ClaimsResult(claims=claims, tokens_used=tokens_used)

    async def generate_questions(self, documents: list[str], n: int) -> GeneratedQuestionsResult:
        """Generate questions grounded in the documents (empty if no documents)."""
        if not documents:
            return GeneratedQuestionsResult(questions=[], tokens_used=0)

        system_prompt, user_prompt = self._build_generate_questions_prompt(documents, n)
        raw_response, tokens_used = await self._call_llm(
            system_prompt, user_prompt, operation="generate_questions"
        )
        parsed = self._parse_json_response(raw_response)
        questions = self._parse_string_list(parsed, "questions")
        return GeneratedQuestionsResult(questions=questions, tokens_used=tokens_used)

    async def generate_answer(self, question: str, context: list[str]) -> GeneratedAnswerResult:
        """Generate an answer from the context only (empty if no context)."""
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

    async def verify_claim(self, claim: str, context: list[str]) -> ClaimVerdict:
        """Verify a single claim against context (NOT_ENOUGH_INFO if no context)."""
        if not context:
            return ClaimVerdict(
                verdict="NOT_ENOUGH_INFO",
                evidence="No context provided for verification.",
            )

        template = get_prompt("verify_claim")
        formatted_context = template.format_context(context)
        user_prompt = template.format_user_prompt(claim=claim, context=formatted_context)
        raw_response, tokens_used = await self._call_llm(
            template.build_system_prompt(), user_prompt, operation="verify_claim"
        )
        parsed = self._parse_json_response(raw_response)

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
