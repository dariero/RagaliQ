"""Unit tests for ContextRecallEvaluator.

Tests follow the acceptance criteria from Issue #10:
- All facts covered -> 1.0
- Half covered -> 0.5
- ValueError when expected_facts missing
- Proper metadata structure with fact coverage
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragaliq.core.evaluator import EvaluationResult
from ragaliq.core.test_case import RAGTestCase
from ragaliq.evaluators.context_recall import ContextRecallEvaluator
from ragaliq.judges.base import ClaimVerdict, JudgeAPIError, JudgeResponseError, LLMJudge

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_judge() -> MagicMock:
    """Create a mock LLM judge with verify_claim support."""
    judge = MagicMock(spec=LLMJudge)
    judge.verify_claim = AsyncMock(
        return_value=ClaimVerdict(
            verdict="SUPPORTED",
            evidence="The context supports this fact.",
            tokens_used=100,
        )
    )
    return judge


@pytest.fixture
def complete_recall_test_case() -> RAGTestCase:
    """Test case where context covers all facts."""
    return RAGTestCase(
        id="recall_001",
        name="Complete Recall",
        query="What is the capital of France?",
        context=[
            "Paris is the capital and most populous city of France.",
            "France is located in Western Europe.",
        ],
        response="The capital of France is Paris.",
        expected_facts=[
            "Paris is the capital of France",
            "France is in Western Europe",
        ],
    )


@pytest.fixture
def partial_recall_test_case() -> RAGTestCase:
    """Test case where context covers some facts."""
    return RAGTestCase(
        id="recall_002",
        name="Partial Recall",
        query="Tell me about Paris",
        context=["Paris is the capital of France."],
        response="Paris is the capital of France and home to the Eiffel Tower.",
        expected_facts=[
            "Paris is the capital of France",
            "The Eiffel Tower is in Paris",
        ],
    )


@pytest.fixture
def no_expected_facts_test_case() -> RAGTestCase:
    """Test case without expected_facts field."""
    return RAGTestCase(
        id="recall_003",
        name="No Expected Facts",
        query="What is the capital of France?",
        context=["Paris is the capital of France."],
        response="The capital of France is Paris.",
        expected_facts=None,
    )


@pytest.fixture
def empty_expected_facts_test_case() -> RAGTestCase:
    """Test case with empty expected_facts list."""
    return RAGTestCase(
        id="recall_004",
        name="Empty Expected Facts",
        query="What is the capital of France?",
        context=["Paris is the capital of France."],
        response="The capital of France is Paris.",
        expected_facts=[],
    )


# =============================================================================
# Test Class Attributes
# =============================================================================


class TestContextRecallEvaluatorAttributes:
    """Tests for evaluator class attributes and initialization."""

    def test_has_required_name(self) -> None:
        """Evaluator must have 'context_recall' as name."""
        evaluator = ContextRecallEvaluator()
        assert evaluator.name == "context_recall"

    def test_has_description(self) -> None:
        """Evaluator must have a meaningful description."""
        evaluator = ContextRecallEvaluator()
        assert evaluator.description
        assert len(evaluator.description) > 10

    def test_default_threshold_is_0_7(self) -> None:
        """Default threshold should be 0.7 per base class."""
        evaluator = ContextRecallEvaluator()
        assert evaluator.threshold == 0.7

    def test_custom_threshold(self) -> None:
        """Should accept custom threshold in constructor."""
        evaluator = ContextRecallEvaluator(threshold=0.9)
        assert evaluator.threshold == 0.9

    def test_repr(self) -> None:
        """Should have a meaningful string representation."""
        evaluator = ContextRecallEvaluator(threshold=0.8)
        assert "ContextRecallEvaluator" in repr(evaluator)
        assert "0.8" in repr(evaluator)


# =============================================================================
# Acceptance Criteria Tests
# =============================================================================


class TestContextRecallAcceptanceCriteria:
    """Tests directly from Issue #10 acceptance criteria."""

    @pytest.mark.asyncio
    async def test_all_facts_covered_returns_1_0(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """AC: All facts covered -> 1.0."""
        # All facts return SUPPORTED
        mock_judge.verify_claim = AsyncMock(
            return_value=ClaimVerdict(
                verdict="SUPPORTED", evidence="Found in context.", tokens_used=100
            )
        )

        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(complete_recall_test_case, mock_judge)

        assert result.score == pytest.approx(1.0)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_half_covered_returns_0_5(
        self,
        mock_judge: MagicMock,
        partial_recall_test_case: RAGTestCase,
    ) -> None:
        """AC: Half covered -> 0.5."""
        # First fact SUPPORTED, second NOT_ENOUGH_INFO
        mock_judge.verify_claim = AsyncMock(
            side_effect=[
                ClaimVerdict(verdict="SUPPORTED", evidence="Found.", tokens_used=100),
                ClaimVerdict(
                    verdict="NOT_ENOUGH_INFO", evidence="Not in context.", tokens_used=100
                ),
            ]
        )

        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(partial_recall_test_case, mock_judge)

        assert result.score == pytest.approx(0.5)
        assert result.passed is False  # default threshold 0.7

    @pytest.mark.asyncio
    async def test_value_error_when_expected_facts_missing(
        self,
        mock_judge: MagicMock,
        no_expected_facts_test_case: RAGTestCase,
    ) -> None:
        """AC: ValueError when expected_facts missing."""
        evaluator = ContextRecallEvaluator()

        with pytest.raises(ValueError, match="requires test_case.expected_facts"):
            await evaluator.evaluate(no_expected_facts_test_case, mock_judge)

    @pytest.mark.asyncio
    async def test_stores_fact_coverage_in_metadata(
        self,
        mock_judge: MagicMock,
        partial_recall_test_case: RAGTestCase,
    ) -> None:
        """AC: Store fact coverage in metadata."""
        mock_judge.verify_claim = AsyncMock(
            side_effect=[
                ClaimVerdict(verdict="SUPPORTED", evidence="Found.", tokens_used=100),
                ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="Missing.", tokens_used=100),
            ]
        )

        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(partial_recall_test_case, mock_judge)

        assert "fact_coverage" in result.raw_response
        fact_coverage = result.raw_response["fact_coverage"]
        assert len(fact_coverage) == 2
        assert fact_coverage[0]["verdict"] == "SUPPORTED"
        assert fact_coverage[1]["verdict"] == "NOT_ENOUGH_INFO"


# =============================================================================
# Score Calculation Tests
# =============================================================================


class TestContextRecallScoreCalculation:
    """Tests verifying the recall score calculation."""

    @pytest.mark.asyncio
    async def test_no_facts_covered_returns_0_0(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """No facts covered should return 0.0."""
        mock_judge.verify_claim = AsyncMock(
            return_value=ClaimVerdict(
                verdict="NOT_ENOUGH_INFO", evidence="Missing.", tokens_used=100
            )
        )

        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(complete_recall_test_case, mock_judge)

        assert result.score == pytest.approx(0.0)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_one_of_three_covered_returns_0_33(
        self,
        mock_judge: MagicMock,
    ) -> None:
        """1 of 3 facts covered should return ~0.33."""
        test_case = RAGTestCase(
            id="recall_test",
            name="One Third Recall",
            query="Test query",
            context=["Context A", "Context B"],
            response="Test response",
            expected_facts=["Fact A", "Fact B", "Fact C"],
        )

        mock_judge.verify_claim = AsyncMock(
            side_effect=[
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=100),
                ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="", tokens_used=100),
                ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="", tokens_used=100),
            ]
        )

        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(test_case, mock_judge)

        assert result.score == pytest.approx(1 / 3, abs=0.01)

    @pytest.mark.asyncio
    async def test_contradicted_facts_not_counted_as_covered(
        self,
        mock_judge: MagicMock,
        partial_recall_test_case: RAGTestCase,
    ) -> None:
        """CONTRADICTED verdict should not count as covered."""
        mock_judge.verify_claim = AsyncMock(
            side_effect=[
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=100),
                ClaimVerdict(
                    verdict="CONTRADICTED", evidence="Context contradicts this.", tokens_used=100
                ),
            ]
        )

        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(partial_recall_test_case, mock_judge)

        # Only 1 of 2 facts supported (CONTRADICTED doesn't count)
        assert result.score == pytest.approx(0.5)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestContextRecallEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_expected_facts_returns_perfect_score(
        self,
        mock_judge: MagicMock,
        empty_expected_facts_test_case: RAGTestCase,
    ) -> None:
        """Empty expected_facts list should return 1.0 (vacuously complete)."""
        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(empty_expected_facts_test_case, mock_judge)

        assert result.score == 1.0
        assert result.passed is True
        assert "vacuously" in result.reasoning.lower()
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_empty_expected_facts_does_not_call_judge(
        self,
        mock_judge: MagicMock,
        empty_expected_facts_test_case: RAGTestCase,
    ) -> None:
        """Empty expected_facts should not make any judge calls."""
        evaluator = ContextRecallEvaluator()
        await evaluator.evaluate(empty_expected_facts_test_case, mock_judge)

        mock_judge.verify_claim.assert_not_called()

    @pytest.mark.asyncio
    async def test_threshold_boundary_at_score(
        self,
        mock_judge: MagicMock,
    ) -> None:
        """Score exactly at threshold should pass."""
        test_case = RAGTestCase(
            id="threshold_test",
            name="Threshold Test",
            query="Test",
            context=["Context"],
            response="Response",
            expected_facts=["Fact A", "Fact B", "Fact C", "Fact D", "Fact E", "Fact F", "Fact G"],
        )

        # 5 of 7 facts supported = 0.714... (just above 0.7)
        mock_judge.verify_claim = AsyncMock(
            side_effect=[
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=100),
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=100),
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=100),
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=100),
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=100),
                ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="", tokens_used=100),
                ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="", tokens_used=100),
            ]
        )

        evaluator = ContextRecallEvaluator()  # default threshold=0.7
        result = await evaluator.evaluate(test_case, mock_judge)

        assert result.score == pytest.approx(5 / 7)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_custom_threshold(
        self,
        mock_judge: MagicMock,
        partial_recall_test_case: RAGTestCase,
    ) -> None:
        """Custom threshold should be respected."""
        # Score is 0.5
        # Passes with threshold 0.4
        mock_judge.verify_claim = AsyncMock(
            side_effect=[
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=100),
                ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="", tokens_used=100),
            ]
        )
        evaluator_lenient = ContextRecallEvaluator(threshold=0.4)
        result_lenient = await evaluator_lenient.evaluate(partial_recall_test_case, mock_judge)
        assert result_lenient.passed is True

        # Reset mock for second evaluation
        # Fails with threshold 0.7
        mock_judge.verify_claim = AsyncMock(
            side_effect=[
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=100),
                ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="", tokens_used=100),
            ]
        )
        evaluator_default = ContextRecallEvaluator()
        result_default = await evaluator_default.evaluate(partial_recall_test_case, mock_judge)
        assert result_default.passed is False


# =============================================================================
# Raw Response / Metadata Tests
# =============================================================================


class TestContextRecallRawResponse:
    """Tests for raw_response metadata structure."""

    @pytest.mark.asyncio
    async def test_raw_response_contains_fact_coverage(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain fact_coverage list."""
        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(complete_recall_test_case, mock_judge)

        assert "fact_coverage" in result.raw_response
        fact_coverage = result.raw_response["fact_coverage"]
        assert isinstance(fact_coverage, list)
        assert len(fact_coverage) == 2

    @pytest.mark.asyncio
    async def test_fact_coverage_structure(
        self,
        mock_judge: MagicMock,
        partial_recall_test_case: RAGTestCase,
    ) -> None:
        """Each fact_coverage entry should have fact, verdict, evidence."""
        mock_judge.verify_claim = AsyncMock(
            side_effect=[
                ClaimVerdict(verdict="SUPPORTED", evidence="Evidence A", tokens_used=100),
                ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="Evidence B", tokens_used=100),
            ]
        )

        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(partial_recall_test_case, mock_judge)

        fact_coverage = result.raw_response["fact_coverage"]
        for i, entry in enumerate(fact_coverage):
            assert "fact" in entry
            assert "verdict" in entry
            assert "evidence" in entry
            assert entry["fact"] == partial_recall_test_case.expected_facts[i]

    @pytest.mark.asyncio
    async def test_raw_response_contains_total_facts(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain total_facts count."""
        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(complete_recall_test_case, mock_judge)

        assert result.raw_response["total_facts"] == 2

    @pytest.mark.asyncio
    async def test_raw_response_contains_covered_facts(
        self,
        mock_judge: MagicMock,
        partial_recall_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain covered_facts count."""
        mock_judge.verify_claim = AsyncMock(
            side_effect=[
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=100),
                ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="", tokens_used=100),
            ]
        )

        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(partial_recall_test_case, mock_judge)

        assert result.raw_response["covered_facts"] == 1
        assert result.raw_response["total_facts"] == 2


# =============================================================================
# Token Tracking Tests
# =============================================================================


class TestContextRecallTokenTracking:
    """Tests for token usage tracking."""

    @pytest.mark.asyncio
    async def test_tokens_accumulated_across_facts(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """Total tokens should be sum across all fact verifications."""
        mock_judge.verify_claim = AsyncMock(
            side_effect=[
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=120),
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=150),
            ]
        )

        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(complete_recall_test_case, mock_judge)

        assert result.tokens_used == 270  # 120 + 150

    @pytest.mark.asyncio
    async def test_empty_facts_zero_tokens(
        self,
        mock_judge: MagicMock,
        empty_expected_facts_test_case: RAGTestCase,
    ) -> None:
        """Empty expected_facts should use zero tokens."""
        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(empty_expected_facts_test_case, mock_judge)

        assert result.tokens_used == 0


# =============================================================================
# Result Structure Tests
# =============================================================================


class TestContextRecallResultStructure:
    """Tests for EvaluationResult structure compliance."""

    @pytest.mark.asyncio
    async def test_returns_evaluation_result(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should return EvaluationResult instance."""
        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(complete_recall_test_case, mock_judge)

        assert isinstance(result, EvaluationResult)

    @pytest.mark.asyncio
    async def test_evaluator_name_in_result(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """Result should contain evaluator name 'context_recall'."""
        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(complete_recall_test_case, mock_judge)

        assert result.evaluator_name == "context_recall"


# =============================================================================
# Judge Interaction Tests
# =============================================================================


class TestContextRecallJudgeInteraction:
    """Tests that the evaluator calls the judge correctly."""

    @pytest.mark.asyncio
    async def test_calls_verify_claim_per_fact(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should call judge.verify_claim once per expected fact."""
        evaluator = ContextRecallEvaluator()
        await evaluator.evaluate(complete_recall_test_case, mock_judge)

        assert mock_judge.verify_claim.call_count == 2

    @pytest.mark.asyncio
    async def test_calls_verify_claim_with_fact_and_context(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should call verify_claim with each fact and context."""
        evaluator = ContextRecallEvaluator()
        await evaluator.evaluate(complete_recall_test_case, mock_judge)

        calls = mock_judge.verify_claim.call_args_list
        for i, call in enumerate(calls):
            # First arg is the fact (claim)
            assert call.args[0] == complete_recall_test_case.expected_facts[i]
            # Second arg is the context
            assert call.args[1] == complete_recall_test_case.context

    @pytest.mark.asyncio
    async def test_does_not_call_other_judge_methods(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should only call verify_claim."""
        evaluator = ContextRecallEvaluator()
        await evaluator.evaluate(complete_recall_test_case, mock_judge)

        mock_judge.evaluate_faithfulness.assert_not_called()
        mock_judge.evaluate_relevance.assert_not_called()
        mock_judge.extract_claims.assert_not_called()


# =============================================================================
# Reasoning Tests
# =============================================================================


class TestContextRecallReasoning:
    """Tests for human-readable reasoning output."""

    @pytest.mark.asyncio
    async def test_all_covered_reasoning(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """All facts covered should produce appropriate reasoning."""
        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(complete_recall_test_case, mock_judge)

        assert "all" in result.reasoning.lower()
        assert "covered" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_none_covered_reasoning(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """No facts covered should produce appropriate reasoning."""
        mock_judge.verify_claim = AsyncMock(
            return_value=ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="", tokens_used=100)
        )

        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(complete_recall_test_case, mock_judge)

        assert "none" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_partial_coverage_reasoning(
        self,
        mock_judge: MagicMock,
        partial_recall_test_case: RAGTestCase,
    ) -> None:
        """Partial coverage should mention counts and missing facts."""
        mock_judge.verify_claim = AsyncMock(
            side_effect=[
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=100),
                ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="", tokens_used=100),
            ]
        )

        evaluator = ContextRecallEvaluator()
        result = await evaluator.evaluate(partial_recall_test_case, mock_judge)

        assert "1 of 2" in result.reasoning
        assert "missing" in result.reasoning.lower()


# =============================================================================
# Error Propagation Tests
# =============================================================================


class TestContextRecallErrorPropagation:
    """Tests that judge errors propagate correctly through the evaluator."""

    @pytest.mark.asyncio
    async def test_judge_api_error_propagates(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """JudgeAPIError from judge should propagate through evaluate()."""
        mock_judge.verify_claim = AsyncMock(
            side_effect=JudgeAPIError("Claude API error: Rate limit exceeded", status_code=429)
        )

        evaluator = ContextRecallEvaluator()
        with pytest.raises(JudgeAPIError, match="Rate limit exceeded"):
            await evaluator.evaluate(complete_recall_test_case, mock_judge)

    @pytest.mark.asyncio
    async def test_judge_response_error_propagates(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """JudgeResponseError from judge should propagate through evaluate()."""
        mock_judge.verify_claim = AsyncMock(
            side_effect=JudgeResponseError("Failed to parse JSON response")
        )

        evaluator = ContextRecallEvaluator()
        with pytest.raises(JudgeResponseError, match="Failed to parse JSON"):
            await evaluator.evaluate(complete_recall_test_case, mock_judge)

    @pytest.mark.asyncio
    async def test_error_on_second_fact_propagates(
        self,
        mock_judge: MagicMock,
        complete_recall_test_case: RAGTestCase,
    ) -> None:
        """Error on second fact should propagate (not swallowed)."""
        mock_judge.verify_claim = AsyncMock(
            side_effect=[
                ClaimVerdict(verdict="SUPPORTED", evidence="", tokens_used=100),
                JudgeAPIError("API error on second fact"),
            ]
        )

        evaluator = ContextRecallEvaluator()
        with pytest.raises(JudgeAPIError, match="second fact"):
            await evaluator.evaluate(complete_recall_test_case, mock_judge)


# =============================================================================
# Registry Integration Tests
# =============================================================================


class TestContextRecallRegistration:
    """Tests that ContextRecallEvaluator is properly registered."""

    def test_registered_in_registry(self) -> None:
        """Should be registered as 'context_recall' in the evaluator registry."""
        from ragaliq.evaluators import get_evaluator

        assert get_evaluator("context_recall") is ContextRecallEvaluator

    def test_retrievable_via_get_evaluator(self) -> None:
        """Should be retrievable via get_evaluator()."""
        from ragaliq.evaluators import get_evaluator

        evaluator_class = get_evaluator("context_recall")
        assert evaluator_class is ContextRecallEvaluator

    def test_appears_in_list_evaluators(self) -> None:
        """Should appear in list_evaluators()."""
        from ragaliq.evaluators import list_evaluators

        assert "context_recall" in list_evaluators()
