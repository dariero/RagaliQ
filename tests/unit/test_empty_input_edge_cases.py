"""Edge case tests for empty/degenerate inputs across all evaluators.

Tests that evaluators handle these degenerate cases gracefully:
- RAGTestCase rejects empty query/response (validation)
- Evaluators handle empty context (no unnecessary LLM calls)
- Context recall with empty context
- BaseJudge.evaluate_relevance empty-input guard
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from ragaliq.core.test_case import RAGTestCase
from ragaliq.evaluators.context_recall import ContextRecallEvaluator
from ragaliq.evaluators.faithfulness import FaithfulnessEvaluator
from ragaliq.evaluators.hallucination import HallucinationEvaluator
from ragaliq.judges.base import (
    ClaimVerdict,
    LLMJudge,
)

# =============================================================================
# RAGTestCase Validation (F4)
# =============================================================================


class TestRAGTestCaseValidation:
    """Tests that RAGTestCase rejects empty query/response."""

    def test_empty_query_rejected(self) -> None:
        """Empty string query should fail validation."""
        with pytest.raises(ValidationError, match="query"):
            RAGTestCase(
                id="test",
                name="test",
                query="",
                context=["doc"],
                response="some response",
            )

    def test_whitespace_only_query_rejected(self) -> None:
        """Whitespace-only query should fail validation after stripping."""
        with pytest.raises(ValidationError, match="query"):
            RAGTestCase(
                id="test",
                name="test",
                query="   \t\n  ",
                context=["doc"],
                response="some response",
            )

    def test_empty_response_rejected(self) -> None:
        """Empty string response should fail validation."""
        with pytest.raises(ValidationError, match="response"):
            RAGTestCase(
                id="test",
                name="test",
                query="What is X?",
                context=["doc"],
                response="",
            )

    def test_whitespace_only_response_rejected(self) -> None:
        """Whitespace-only response should fail validation after stripping."""
        with pytest.raises(ValidationError, match="response"):
            RAGTestCase(
                id="test",
                name="test",
                query="What is X?",
                context=["doc"],
                response="  \n  ",
            )

    def test_valid_inputs_accepted(self) -> None:
        """Normal non-empty inputs should pass validation."""
        tc = RAGTestCase(
            id="test",
            name="test",
            query="What is X?",
            context=["doc"],
            response="X is Y.",
        )
        assert tc.query == "What is X?"
        assert tc.response == "X is Y."

    def test_query_and_response_are_stripped(self) -> None:
        """Leading/trailing whitespace should be stripped."""
        tc = RAGTestCase(
            id="test",
            name="test",
            query="  What is X?  ",
            context=["doc"],
            response="  X is Y.  ",
        )
        assert tc.query == "What is X?"
        assert tc.response == "X is Y."

    def test_empty_context_list_allowed(self) -> None:
        """Empty context list should still be allowed (evaluators handle it)."""
        tc = RAGTestCase(
            id="test",
            name="test",
            query="What is X?",
            context=[],
            response="X is Y.",
        )
        assert tc.context == []


# =============================================================================
# Faithfulness with Empty Context (F5)
# =============================================================================


class TestFaithfulnessEmptyContext:
    """Tests that faithfulness handles empty context correctly."""

    @pytest.mark.asyncio
    async def test_empty_context_returns_0_score(self) -> None:
        """Empty context should return 0.0 (cannot assess faithfulness)."""
        judge = MagicMock(spec=LLMJudge)
        tc = RAGTestCase(id="test", name="test", query="Q?", context=[], response="Answer.")

        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(tc, judge)

        assert result.score == 0.0
        assert result.passed is False
        assert "context" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_empty_context_no_llm_calls(self) -> None:
        """Empty context should not trigger any LLM calls."""
        judge = MagicMock(spec=LLMJudge)
        judge.extract_claims = AsyncMock()
        judge.verify_claim = AsyncMock()
        tc = RAGTestCase(id="test", name="test", query="Q?", context=[], response="Answer.")

        evaluator = FaithfulnessEvaluator()
        await evaluator.evaluate(tc, judge)

        judge.extract_claims.assert_not_called()
        judge.verify_claim.assert_not_called()


# =============================================================================
# Hallucination with Empty Context (F5)
# =============================================================================


class TestHallucinationEmptyContext:
    """Tests that hallucination handles empty context correctly."""

    @pytest.mark.asyncio
    async def test_empty_context_returns_0_score(self) -> None:
        """Empty context should return 0.0 (cannot assess hallucination)."""
        judge = MagicMock(spec=LLMJudge)
        tc = RAGTestCase(id="test", name="test", query="Q?", context=[], response="Answer.")

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(tc, judge)

        assert result.score == 0.0
        assert result.passed is False
        assert "context" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_empty_context_no_llm_calls(self) -> None:
        """Empty context should not trigger any LLM calls."""
        judge = MagicMock(spec=LLMJudge)
        judge.extract_claims = AsyncMock()
        judge.verify_claim = AsyncMock()
        tc = RAGTestCase(id="test", name="test", query="Q?", context=[], response="Answer.")

        evaluator = HallucinationEvaluator()
        await evaluator.evaluate(tc, judge)

        judge.extract_claims.assert_not_called()
        judge.verify_claim.assert_not_called()


# =============================================================================
# Context Recall with Empty Context (F13)
# =============================================================================


class TestContextRecallEmptyContext:
    """Tests that context_recall handles empty context."""

    @pytest.mark.asyncio
    async def test_empty_context_with_facts_returns_0(self) -> None:
        """Empty context with expected facts should return 0.0 (all NOT_ENOUGH_INFO)."""
        judge = MagicMock(spec=LLMJudge)
        judge.verify_claim = AsyncMock(
            return_value=ClaimVerdict(
                verdict="NOT_ENOUGH_INFO",
                evidence="No context provided for verification.",
                tokens_used=0,
            )
        )
        tc = RAGTestCase(
            id="test",
            name="test",
            query="Q?",
            context=[],
            response="Answer.",
            expected_facts=["Fact A", "Fact B"],
        )

        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(tc, judge)

        assert result.score == 0.0
        assert result.passed is False


# =============================================================================
# BaseJudge.evaluate_relevance Empty Input Guard (F2)
# =============================================================================


class TestEvaluateRelevanceEmptyGuard:
    """Tests that evaluate_relevance guards against empty inputs."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_0(self) -> None:
        """Empty query in evaluate_relevance should return 0.0 immediately."""
        from ragaliq.judges.base_judge import BaseJudge

        transport = MagicMock()
        judge = BaseJudge.__new__(BaseJudge)
        judge.config = MagicMock()
        judge._transport = transport
        judge._trace_collector = None

        import asyncio

        judge._concurrency_limit = asyncio.Semaphore(20)

        result = await judge.evaluate_relevance(query="", response="Some response")

        assert result.score == 0.0
        assert result.tokens_used == 0
        assert "empty" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_empty_response_returns_0(self) -> None:
        """Empty response in evaluate_relevance should return 0.0 immediately."""
        from ragaliq.judges.base_judge import BaseJudge

        transport = MagicMock()
        judge = BaseJudge.__new__(BaseJudge)
        judge.config = MagicMock()
        judge._transport = transport
        judge._trace_collector = None

        import asyncio

        judge._concurrency_limit = asyncio.Semaphore(20)

        result = await judge.evaluate_relevance(query="What is X?", response="")

        assert result.score == 0.0
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_whitespace_only_treated_as_empty(self) -> None:
        """Whitespace-only query/response should trigger the empty guard."""
        from ragaliq.judges.base_judge import BaseJudge

        transport = MagicMock()
        judge = BaseJudge.__new__(BaseJudge)
        judge.config = MagicMock()
        judge._transport = transport
        judge._trace_collector = None

        import asyncio

        judge._concurrency_limit = asyncio.Semaphore(20)

        result = await judge.evaluate_relevance(query="  \n  ", response="Answer")

        assert result.score == 0.0
        assert result.tokens_used == 0
