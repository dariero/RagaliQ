"""Unit tests for RelevanceEvaluator.

Tests follow the acceptance criteria from Issue #7:
- Score passes through from judge
- Reasoning included in result
- Threshold-based pass/fail works correctly
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragaliq.core.evaluator import EvaluationResult
from ragaliq.core.test_case import RAGTestCase
from ragaliq.evaluators.relevance import RelevanceEvaluator
from ragaliq.judges.base import JudgeResult, LLMJudge

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
            reasoning="Response directly answers the query.",
            tokens_used=150,
        )
    )
    return judge


@pytest.fixture
def relevant_test_case() -> RAGTestCase:
    """A test case where response is relevant to the query."""
    return RAGTestCase(
        id="relevant_001",
        name="Relevant Response",
        query="What is the capital of France?",
        context=["The capital city of France is Paris."],
        response="The capital of France is Paris.",
    )


@pytest.fixture
def irrelevant_test_case() -> RAGTestCase:
    """A test case where response is off-topic from the query."""
    return RAGTestCase(
        id="irrelevant_001",
        name="Irrelevant Response",
        query="What is the capital of France?",
        context=["The capital city of France is Paris."],
        response="Python is a popular programming language used for data science.",
    )


# =============================================================================
# Test Class Attributes
# =============================================================================


class TestRelevanceEvaluatorAttributes:
    """Tests for evaluator class attributes and initialization."""

    def test_has_required_name(self) -> None:
        """Evaluator must have 'relevance' as name."""
        evaluator = RelevanceEvaluator()
        assert evaluator.name == "relevance"

    def test_has_description(self) -> None:
        """Evaluator must have a meaningful description."""
        evaluator = RelevanceEvaluator()
        assert evaluator.description
        assert len(evaluator.description) > 10

    def test_default_threshold_is_0_7(self) -> None:
        """Default threshold should be 0.7 per base class."""
        evaluator = RelevanceEvaluator()
        assert evaluator.threshold == 0.7

    def test_custom_threshold(self) -> None:
        """Should accept custom threshold in constructor."""
        evaluator = RelevanceEvaluator(threshold=0.9)
        assert evaluator.threshold == 0.9

    def test_repr(self) -> None:
        """Should have a meaningful string representation."""
        evaluator = RelevanceEvaluator(threshold=0.8)
        assert "RelevanceEvaluator" in repr(evaluator)
        assert "0.8" in repr(evaluator)


# =============================================================================
# Acceptance Criteria Tests
# =============================================================================


class TestRelevanceEvaluatorAcceptanceCriteria:
    """Tests directly from Issue #7 acceptance criteria."""

    @pytest.mark.asyncio
    async def test_score_passes_through_from_judge(
        self,
        mock_judge: MagicMock,
        relevant_test_case: RAGTestCase,
    ) -> None:
        """AC: Score passes through from judge."""
        mock_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.85,
            reasoning="Mostly relevant.",
            tokens_used=120,
        )

        evaluator = RelevanceEvaluator()
        result = await evaluator.evaluate(relevant_test_case, mock_judge)

        assert result.score == 0.85

    @pytest.mark.asyncio
    async def test_reasoning_included_in_result(
        self,
        mock_judge: MagicMock,
        relevant_test_case: RAGTestCase,
    ) -> None:
        """AC: Reasoning included in result."""
        expected_reasoning = (
            "Response directly addresses the user's question about France's capital."
        )
        mock_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.95,
            reasoning=expected_reasoning,
            tokens_used=100,
        )

        evaluator = RelevanceEvaluator()
        result = await evaluator.evaluate(relevant_test_case, mock_judge)

        assert result.reasoning == expected_reasoning


# =============================================================================
# Score Passthrough Tests
# =============================================================================


class TestRelevanceEvaluatorScorePassthrough:
    """Tests that scores from the judge pass through correctly."""

    @pytest.mark.asyncio
    async def test_high_score_passes_through(
        self,
        mock_judge: MagicMock,
        relevant_test_case: RAGTestCase,
    ) -> None:
        """High relevance score (1.0) passes through unchanged."""
        mock_judge.evaluate_relevance.return_value = JudgeResult(
            score=1.0, reasoning="Perfectly relevant.", tokens_used=100
        )

        evaluator = RelevanceEvaluator()
        result = await evaluator.evaluate(relevant_test_case, mock_judge)

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_low_score_passes_through(
        self,
        mock_judge: MagicMock,
        irrelevant_test_case: RAGTestCase,
    ) -> None:
        """Low relevance score (0.1) passes through unchanged."""
        mock_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.1, reasoning="Mostly irrelevant.", tokens_used=100
        )

        evaluator = RelevanceEvaluator()
        result = await evaluator.evaluate(irrelevant_test_case, mock_judge)

        assert result.score == 0.1

    @pytest.mark.asyncio
    async def test_zero_score_passes_through(
        self,
        mock_judge: MagicMock,
        irrelevant_test_case: RAGTestCase,
    ) -> None:
        """Zero relevance score (0.0) passes through unchanged."""
        mock_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.0, reasoning="Completely off-topic.", tokens_used=80
        )

        evaluator = RelevanceEvaluator()
        result = await evaluator.evaluate(irrelevant_test_case, mock_judge)

        assert result.score == 0.0


# =============================================================================
# Threshold Logic Tests
# =============================================================================


class TestRelevanceEvaluatorThresholdLogic:
    """Tests for pass/fail based on configurable threshold."""

    @pytest.mark.asyncio
    async def test_score_at_threshold_passes(
        self,
        mock_judge: MagicMock,
        relevant_test_case: RAGTestCase,
    ) -> None:
        """Score exactly at threshold (0.7) should pass."""
        mock_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.7, reasoning="Reasonably relevant.", tokens_used=100
        )

        evaluator = RelevanceEvaluator()  # default threshold=0.7
        result = await evaluator.evaluate(relevant_test_case, mock_judge)

        assert result.score == 0.7
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_score_below_threshold_fails(
        self,
        mock_judge: MagicMock,
        irrelevant_test_case: RAGTestCase,
    ) -> None:
        """Score below threshold should fail."""
        mock_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.4, reasoning="Partially relevant.", tokens_used=100
        )

        evaluator = RelevanceEvaluator()  # default threshold=0.7
        result = await evaluator.evaluate(irrelevant_test_case, mock_judge)

        assert result.score == 0.4
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold_applied(
        self,
        mock_judge: MagicMock,
        relevant_test_case: RAGTestCase,
    ) -> None:
        """Custom threshold should be used for pass/fail."""
        mock_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.85, reasoning="Mostly relevant.", tokens_used=100
        )

        # 0.85 passes with default 0.7, but fails with 0.9
        evaluator_default = RelevanceEvaluator()
        result_default = await evaluator_default.evaluate(relevant_test_case, mock_judge)
        assert result_default.passed is True

        evaluator_strict = RelevanceEvaluator(threshold=0.9)
        result_strict = await evaluator_strict.evaluate(relevant_test_case, mock_judge)
        assert result_strict.passed is False


# =============================================================================
# Raw Response / Metadata Tests
# =============================================================================


class TestRelevanceEvaluatorRawResponse:
    """Tests for judge metadata preserved in raw_response."""

    @pytest.mark.asyncio
    async def test_raw_response_contains_score(
        self,
        mock_judge: MagicMock,
        relevant_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain the judge's score."""
        mock_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.92, reasoning="Very relevant.", tokens_used=130
        )

        evaluator = RelevanceEvaluator()
        result = await evaluator.evaluate(relevant_test_case, mock_judge)

        assert result.raw_response["score"] == 0.92

    @pytest.mark.asyncio
    async def test_raw_response_contains_reasoning(
        self,
        mock_judge: MagicMock,
        relevant_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain the judge's reasoning."""
        mock_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.92, reasoning="Very relevant.", tokens_used=130
        )

        evaluator = RelevanceEvaluator()
        result = await evaluator.evaluate(relevant_test_case, mock_judge)

        assert result.raw_response["reasoning"] == "Very relevant."

    @pytest.mark.asyncio
    async def test_raw_response_contains_tokens_used(
        self,
        mock_judge: MagicMock,
        relevant_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain tokens_used for cost tracking."""
        mock_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.92, reasoning="Very relevant.", tokens_used=130
        )

        evaluator = RelevanceEvaluator()
        result = await evaluator.evaluate(relevant_test_case, mock_judge)

        assert result.raw_response["tokens_used"] == 130


# =============================================================================
# Result Structure Tests
# =============================================================================


class TestRelevanceEvaluatorResultStructure:
    """Tests for EvaluationResult structure compliance."""

    @pytest.mark.asyncio
    async def test_returns_evaluation_result(
        self,
        mock_judge: MagicMock,
        relevant_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should return EvaluationResult instance."""
        evaluator = RelevanceEvaluator()
        result = await evaluator.evaluate(relevant_test_case, mock_judge)

        assert isinstance(result, EvaluationResult)

    @pytest.mark.asyncio
    async def test_evaluator_name_in_result(
        self,
        mock_judge: MagicMock,
        relevant_test_case: RAGTestCase,
    ) -> None:
        """Result should contain evaluator name 'relevance'."""
        evaluator = RelevanceEvaluator()
        result = await evaluator.evaluate(relevant_test_case, mock_judge)

        assert result.evaluator_name == "relevance"


# =============================================================================
# Judge Interaction Tests
# =============================================================================


class TestRelevanceEvaluatorJudgeInteraction:
    """Tests that the evaluator calls the judge correctly."""

    @pytest.mark.asyncio
    async def test_calls_judge_with_query_and_response(
        self,
        mock_judge: MagicMock,
        relevant_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should call judge.evaluate_relevance with correct args."""
        evaluator = RelevanceEvaluator()
        await evaluator.evaluate(relevant_test_case, mock_judge)

        mock_judge.evaluate_relevance.assert_called_once_with(
            query=relevant_test_case.query,
            response=relevant_test_case.response,
        )

    @pytest.mark.asyncio
    async def test_does_not_call_other_judge_methods(
        self,
        mock_judge: MagicMock,
        relevant_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should only call evaluate_relevance, not other methods."""
        evaluator = RelevanceEvaluator()
        await evaluator.evaluate(relevant_test_case, mock_judge)

        # Should NOT call faithfulness-related methods
        mock_judge.evaluate_faithfulness.assert_not_called()
        mock_judge.extract_claims.assert_not_called()
        mock_judge.verify_claim.assert_not_called()
