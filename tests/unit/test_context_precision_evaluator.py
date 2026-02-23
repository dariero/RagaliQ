"""Unit tests for ContextPrecisionEvaluator.

Tests follow the acceptance criteria from Issue #9:
- All docs relevant -> ~1.0
- First doc relevant, rest not -> high score
- First doc irrelevant, last relevant -> low score
- Weighted precision formula correct
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragaliq.core.evaluator import EvaluationResult
from ragaliq.core.test_case import RAGTestCase
from ragaliq.evaluators.context_precision import ContextPrecisionEvaluator
from ragaliq.judges.base import JudgeAPIError, JudgeResponseError, JudgeResult, LLMJudge

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_judge() -> MagicMock:
    """Create a mock LLM judge with relevance evaluation."""
    judge = MagicMock(spec=LLMJudge)
    judge.evaluate_relevance = AsyncMock(
        return_value=JudgeResult(
            score=0.9,
            reasoning="Document is relevant to the query.",
            tokens_used=100,
        )
    )
    return judge


@pytest.fixture
def multi_doc_test_case() -> RAGTestCase:
    """A test case with multiple context documents."""
    return RAGTestCase(
        id="precision_001",
        name="Multi-doc Precision",
        query="What is the capital of France?",
        context=[
            "Paris is the capital and most populous city of France.",
            "France is a country in Western Europe.",
            "The Eiffel Tower is located in Paris, France.",
        ],
        response="The capital of France is Paris.",
    )


@pytest.fixture
def single_doc_test_case() -> RAGTestCase:
    """A test case with a single context document."""
    return RAGTestCase(
        id="precision_002",
        name="Single-doc Precision",
        query="What is the capital of France?",
        context=["Paris is the capital of France."],
        response="The capital of France is Paris.",
    )


@pytest.fixture
def empty_context_test_case() -> RAGTestCase:
    """A test case with no context documents."""
    return RAGTestCase(
        id="precision_003",
        name="Empty Context",
        query="What is the capital of France?",
        context=[],
        response="I don't have enough information.",
    )


# =============================================================================
# Test Class Attributes
# =============================================================================


class TestContextPrecisionEvaluatorAttributes:
    """Tests for evaluator class attributes and initialization."""

    def test_has_required_name(self) -> None:
        """Evaluator must have 'context_precision' as name."""
        evaluator = ContextPrecisionEvaluator()
        assert evaluator.name == "context_precision"

    def test_has_description(self) -> None:
        """Evaluator must have a meaningful description."""
        evaluator = ContextPrecisionEvaluator()
        assert evaluator.description
        assert len(evaluator.description) > 10

    def test_default_threshold_is_0_7(self) -> None:
        """Default threshold should be 0.7 per base class."""
        evaluator = ContextPrecisionEvaluator()
        assert evaluator.threshold == 0.7

    def test_custom_threshold(self) -> None:
        """Should accept custom threshold in constructor."""
        evaluator = ContextPrecisionEvaluator(threshold=0.9)
        assert evaluator.threshold == 0.9

    def test_repr(self) -> None:
        """Should have a meaningful string representation."""
        evaluator = ContextPrecisionEvaluator(threshold=0.8)
        assert "ContextPrecisionEvaluator" in repr(evaluator)
        assert "0.8" in repr(evaluator)


# =============================================================================
# Acceptance Criteria Tests
# =============================================================================


class TestContextPrecisionAcceptanceCriteria:
    """Tests directly from Issue #9 acceptance criteria."""

    @pytest.mark.asyncio
    async def test_all_docs_relevant_high_score(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """AC: All docs relevant -> ~1.0."""
        # All docs return high relevance
        mock_judge.evaluate_relevance = AsyncMock(
            return_value=JudgeResult(score=1.0, reasoning="Highly relevant.", tokens_used=100)
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert result.score == pytest.approx(1.0)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_first_doc_relevant_rest_not_high_score(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """AC: First doc relevant, rest not -> high score.

        With 3 docs and scores [1.0, 0.0, 0.0]:
        weighted = (1.0/1 + 0.0/2 + 0.0/3) / (1/1 + 1/2 + 1/3)
                 = 1.0 / (11/6)
                 ≈ 0.545
        """
        scores = [1.0, 0.0, 0.0]
        mock_judge.evaluate_relevance = AsyncMock(
            side_effect=[
                JudgeResult(score=s, reasoning=f"Score: {s}", tokens_used=100) for s in scores
            ]
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        # First doc weighted heavily, so score should be meaningfully > 0
        assert result.score > 0.5

    @pytest.mark.asyncio
    async def test_first_doc_irrelevant_last_relevant_low_score(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """AC: First doc irrelevant, last relevant -> low score.

        With 3 docs and scores [0.0, 0.0, 1.0]:
        weighted = (0.0/1 + 0.0/2 + 1.0/3) / (1/1 + 1/2 + 1/3)
                 = (1/3) / (11/6)
                 ≈ 0.182
        """
        scores = [0.0, 0.0, 1.0]
        mock_judge.evaluate_relevance = AsyncMock(
            side_effect=[
                JudgeResult(score=s, reasoning=f"Score: {s}", tokens_used=100) for s in scores
            ]
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        # Last doc has lowest weight, so score should be low
        assert result.score < 0.2

    @pytest.mark.asyncio
    async def test_ranking_matters(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """Same docs, different order should produce different scores.

        [1.0, 0.0, 0.0] should score higher than [0.0, 0.0, 1.0]
        because higher-ranked docs are weighted more.
        """
        # Score with relevant doc first
        mock_judge.evaluate_relevance = AsyncMock(
            side_effect=[
                JudgeResult(score=s, reasoning="", tokens_used=100) for s in [1.0, 0.0, 0.0]
            ]
        )
        evaluator = ContextPrecisionEvaluator()
        result_first = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        # Score with relevant doc last
        mock_judge.evaluate_relevance = AsyncMock(
            side_effect=[
                JudgeResult(score=s, reasoning="", tokens_used=100) for s in [0.0, 0.0, 1.0]
            ]
        )
        result_last = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert result_first.score > result_last.score


# =============================================================================
# Weighted Precision Formula Tests
# =============================================================================


class TestWeightedPrecisionFormula:
    """Tests verifying the weighted precision calculation."""

    @pytest.mark.asyncio
    async def test_single_doc_score_equals_relevance(
        self,
        mock_judge: MagicMock,
        single_doc_test_case: RAGTestCase,
    ) -> None:
        """Single document: score should equal its relevance score.

        With 1 doc: sum(r_1 / 1) / sum(1/1) = r_1
        """
        mock_judge.evaluate_relevance = AsyncMock(
            return_value=JudgeResult(score=0.85, reasoning="Mostly relevant.", tokens_used=100)
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(single_doc_test_case, mock_judge)

        assert result.score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_two_docs_weighted_formula(
        self,
        mock_judge: MagicMock,
    ) -> None:
        """Two documents: verify formula exactly.

        Docs [0.8, 0.4]:
        weighted = (0.8/1 + 0.4/2) / (1/1 + 1/2)
                 = (0.8 + 0.2) / 1.5
                 = 1.0 / 1.5
                 ≈ 0.6667
        """
        test_case = RAGTestCase(
            id="formula_test",
            name="Formula Verification",
            query="Test query",
            context=["Doc A", "Doc B"],
            response="Test response",
        )
        mock_judge.evaluate_relevance = AsyncMock(
            side_effect=[
                JudgeResult(score=0.8, reasoning="", tokens_used=100),
                JudgeResult(score=0.4, reasoning="", tokens_used=100),
            ]
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(test_case, mock_judge)

        expected = (0.8 / 1 + 0.4 / 2) / (1 / 1 + 1 / 2)
        assert result.score == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_three_docs_weighted_formula(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """Three documents: verify formula exactly.

        Docs [0.9, 0.5, 0.3]:
        weighted = (0.9/1 + 0.5/2 + 0.3/3) / (1/1 + 1/2 + 1/3)
                 = (0.9 + 0.25 + 0.1) / (1 + 0.5 + 0.333...)
                 = 1.25 / 1.833...
                 ≈ 0.6818
        """
        scores = [0.9, 0.5, 0.3]
        mock_judge.evaluate_relevance = AsyncMock(
            side_effect=[JudgeResult(score=s, reasoning="", tokens_used=100) for s in scores]
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        expected = (0.9 / 1 + 0.5 / 2 + 0.3 / 3) / (1 / 1 + 1 / 2 + 1 / 3)
        assert result.score == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_all_zero_scores(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """All documents irrelevant should produce score of 0.0."""
        mock_judge.evaluate_relevance = AsyncMock(
            return_value=JudgeResult(score=0.0, reasoning="Not relevant.", tokens_used=100)
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert result.score == pytest.approx(0.0)
        assert result.passed is False


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestContextPrecisionEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_context_returns_perfect_score(
        self,
        mock_judge: MagicMock,
        empty_context_test_case: RAGTestCase,
    ) -> None:
        """Empty context should return 1.0 (vacuously precise)."""
        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(empty_context_test_case, mock_judge)

        assert result.score == 1.0
        assert result.passed is True
        assert "vacuously" in result.reasoning.lower()
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_empty_context_does_not_call_judge(
        self,
        mock_judge: MagicMock,
        empty_context_test_case: RAGTestCase,
    ) -> None:
        """Empty context should not make any judge calls."""
        evaluator = ContextPrecisionEvaluator()
        await evaluator.evaluate(empty_context_test_case, mock_judge)

        mock_judge.evaluate_relevance.assert_not_called()

    @pytest.mark.asyncio
    async def test_threshold_boundary_at_score(
        self,
        mock_judge: MagicMock,
        single_doc_test_case: RAGTestCase,
    ) -> None:
        """Score exactly at threshold should pass."""
        mock_judge.evaluate_relevance = AsyncMock(
            return_value=JudgeResult(score=0.7, reasoning="Reasonably relevant.", tokens_used=100)
        )

        evaluator = ContextPrecisionEvaluator()  # default threshold=0.7
        result = await evaluator.evaluate(single_doc_test_case, mock_judge)

        assert result.score == pytest.approx(0.7)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_threshold_boundary_below(
        self,
        mock_judge: MagicMock,
        single_doc_test_case: RAGTestCase,
    ) -> None:
        """Score below threshold should fail."""
        mock_judge.evaluate_relevance = AsyncMock(
            return_value=JudgeResult(
                score=0.69, reasoning="Slightly below threshold.", tokens_used=100
            )
        )

        evaluator = ContextPrecisionEvaluator()  # default threshold=0.7
        result = await evaluator.evaluate(single_doc_test_case, mock_judge)

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold(
        self,
        mock_judge: MagicMock,
        single_doc_test_case: RAGTestCase,
    ) -> None:
        """Custom threshold should be respected."""
        mock_judge.evaluate_relevance = AsyncMock(
            return_value=JudgeResult(score=0.85, reasoning="Mostly relevant.", tokens_used=100)
        )

        # Passes with default 0.7
        evaluator_default = ContextPrecisionEvaluator()
        result_default = await evaluator_default.evaluate(single_doc_test_case, mock_judge)
        assert result_default.passed is True

        # Fails with strict 0.9
        evaluator_strict = ContextPrecisionEvaluator(threshold=0.9)
        result_strict = await evaluator_strict.evaluate(single_doc_test_case, mock_judge)
        assert result_strict.passed is False


# =============================================================================
# Raw Response / Metadata Tests
# =============================================================================


class TestContextPrecisionRawResponse:
    """Tests for raw_response metadata structure."""

    @pytest.mark.asyncio
    async def test_raw_response_contains_doc_scores(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain per-document scores."""
        scores = [0.9, 0.5, 0.3]
        mock_judge.evaluate_relevance = AsyncMock(
            side_effect=[
                JudgeResult(score=s, reasoning=f"Score: {s}", tokens_used=100) for s in scores
            ]
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert "doc_scores" in result.raw_response
        doc_scores = result.raw_response["doc_scores"]
        assert len(doc_scores) == 3

        for i, doc in enumerate(doc_scores):
            assert doc["rank"] == i + 1
            assert doc["score"] == scores[i]
            assert "document" in doc
            assert "reasoning" in doc

    @pytest.mark.asyncio
    async def test_raw_response_contains_total_docs(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain total_docs count."""
        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert result.raw_response["total_docs"] == 3

    @pytest.mark.asyncio
    async def test_raw_response_contains_weighted_precision(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain the weighted_precision score."""
        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert "weighted_precision" in result.raw_response
        assert result.raw_response["weighted_precision"] == result.score

    @pytest.mark.asyncio
    async def test_long_documents_truncated_in_metadata(
        self,
        mock_judge: MagicMock,
    ) -> None:
        """Long documents should be truncated in raw_response metadata."""
        long_doc = "A" * 500  # 500 chars
        test_case = RAGTestCase(
            id="long_doc",
            name="Long Doc Test",
            query="Test query",
            context=[long_doc],
            response="Test response",
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(test_case, mock_judge)

        from ragaliq.evaluators.context_precision import _DOC_PREVIEW_LENGTH

        stored_doc = result.raw_response["doc_scores"][0]["document"]
        assert len(stored_doc) <= _DOC_PREVIEW_LENGTH


# =============================================================================
# Token Tracking Tests
# =============================================================================


class TestContextPrecisionTokenTracking:
    """Tests for token usage tracking."""

    @pytest.mark.asyncio
    async def test_tokens_accumulated_across_docs(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """Total tokens should be sum across all document evaluations."""
        mock_judge.evaluate_relevance = AsyncMock(
            side_effect=[
                JudgeResult(score=0.9, reasoning="", tokens_used=100),
                JudgeResult(score=0.8, reasoning="", tokens_used=150),
                JudgeResult(score=0.7, reasoning="", tokens_used=120),
            ]
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert result.tokens_used == 370  # 100 + 150 + 120

    @pytest.mark.asyncio
    async def test_empty_context_zero_tokens(
        self,
        mock_judge: MagicMock,
        empty_context_test_case: RAGTestCase,
    ) -> None:
        """Empty context should use zero tokens."""
        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(empty_context_test_case, mock_judge)

        assert result.tokens_used == 0


# =============================================================================
# Result Structure Tests
# =============================================================================


class TestContextPrecisionResultStructure:
    """Tests for EvaluationResult structure compliance."""

    @pytest.mark.asyncio
    async def test_returns_evaluation_result(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should return EvaluationResult instance."""
        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert isinstance(result, EvaluationResult)

    @pytest.mark.asyncio
    async def test_evaluator_name_in_result(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """Result should contain evaluator name 'context_precision'."""
        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert result.evaluator_name == "context_precision"


# =============================================================================
# Judge Interaction Tests
# =============================================================================


class TestContextPrecisionJudgeInteraction:
    """Tests that the evaluator calls the judge correctly."""

    @pytest.mark.asyncio
    async def test_calls_judge_per_document(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should call judge.evaluate_relevance once per document."""
        evaluator = ContextPrecisionEvaluator()
        await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert mock_judge.evaluate_relevance.call_count == 3

    @pytest.mark.asyncio
    async def test_calls_judge_with_query_and_doc(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should call judge with query and each doc as response."""
        evaluator = ContextPrecisionEvaluator()
        await evaluator.evaluate(multi_doc_test_case, mock_judge)

        calls = mock_judge.evaluate_relevance.call_args_list
        for i, call in enumerate(calls):
            assert call.kwargs["query"] == multi_doc_test_case.query
            assert call.kwargs["response"] == multi_doc_test_case.context[i]

    @pytest.mark.asyncio
    async def test_does_not_call_other_judge_methods(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should only call evaluate_relevance."""
        evaluator = ContextPrecisionEvaluator()
        await evaluator.evaluate(multi_doc_test_case, mock_judge)

        mock_judge.evaluate_faithfulness.assert_not_called()
        mock_judge.extract_claims.assert_not_called()
        mock_judge.verify_claim.assert_not_called()


# =============================================================================
# Reasoning Tests
# =============================================================================


class TestContextPrecisionReasoning:
    """Tests for human-readable reasoning output."""

    @pytest.mark.asyncio
    async def test_all_relevant_reasoning(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """All relevant docs should produce appropriate reasoning."""
        mock_judge.evaluate_relevance = AsyncMock(
            return_value=JudgeResult(score=0.9, reasoning="Relevant.", tokens_used=100)
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert "all" in result.reasoning.lower()
        assert "relevant" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_none_relevant_reasoning(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """No relevant docs should produce appropriate reasoning."""
        mock_judge.evaluate_relevance = AsyncMock(
            return_value=JudgeResult(score=0.1, reasoning="Not relevant.", tokens_used=100)
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert "none" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_mixed_relevance_reasoning(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """Mixed relevance should mention count and weighting."""
        scores = [0.9, 0.2, 0.8]
        mock_judge.evaluate_relevance = AsyncMock(
            side_effect=[JudgeResult(score=s, reasoning="", tokens_used=100) for s in scores]
        )

        evaluator = ContextPrecisionEvaluator()
        result = await evaluator.evaluate(multi_doc_test_case, mock_judge)

        assert "2 of 3" in result.reasoning
        assert "weighted" in result.reasoning.lower()


# =============================================================================
# Error Propagation Tests
# =============================================================================


class TestContextPrecisionErrorPropagation:
    """Tests that judge errors propagate correctly through the evaluator."""

    @pytest.mark.asyncio
    async def test_judge_api_error_propagates(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """JudgeAPIError from judge should propagate through evaluate()."""
        mock_judge.evaluate_relevance = AsyncMock(
            side_effect=JudgeAPIError("Claude API error: Rate limit exceeded", status_code=429)
        )

        evaluator = ContextPrecisionEvaluator()
        with pytest.raises(JudgeAPIError, match="Rate limit exceeded"):
            await evaluator.evaluate(multi_doc_test_case, mock_judge)

    @pytest.mark.asyncio
    async def test_judge_response_error_propagates(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """JudgeResponseError from judge should propagate through evaluate()."""
        mock_judge.evaluate_relevance = AsyncMock(
            side_effect=JudgeResponseError("Failed to parse JSON response")
        )

        evaluator = ContextPrecisionEvaluator()
        with pytest.raises(JudgeResponseError, match="Failed to parse JSON"):
            await evaluator.evaluate(multi_doc_test_case, mock_judge)

    @pytest.mark.asyncio
    async def test_error_on_second_doc_propagates(
        self,
        mock_judge: MagicMock,
        multi_doc_test_case: RAGTestCase,
    ) -> None:
        """Error on second document should propagate (not swallowed)."""
        mock_judge.evaluate_relevance = AsyncMock(
            side_effect=[
                JudgeResult(score=0.9, reasoning="", tokens_used=100),
                JudgeAPIError("API error on second doc"),
            ]
        )

        evaluator = ContextPrecisionEvaluator()
        with pytest.raises(JudgeAPIError, match="second doc"):
            await evaluator.evaluate(multi_doc_test_case, mock_judge)


# =============================================================================
# Registry Integration Tests
# =============================================================================


class TestContextPrecisionRegistration:
    """Tests that ContextPrecisionEvaluator is properly registered."""

    def test_registered_in_registry(self) -> None:
        """Should be registered as 'context_precision' in the evaluator registry."""
        from ragaliq.evaluators import get_evaluator

        assert get_evaluator("context_precision") is ContextPrecisionEvaluator

    def test_retrievable_via_get_evaluator(self) -> None:
        """Should be retrievable via get_evaluator()."""
        from ragaliq.evaluators import get_evaluator

        evaluator_class = get_evaluator("context_precision")
        assert evaluator_class is ContextPrecisionEvaluator

    def test_appears_in_list_evaluators(self) -> None:
        """Should appear in list_evaluators()."""
        from ragaliq.evaluators import list_evaluators

        assert "context_precision" in list_evaluators()
