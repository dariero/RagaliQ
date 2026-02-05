"""Unit tests for FaithfulnessEvaluator.

Tests follow the acceptance criteria from Issue #6:
- All claims supported -> 1.0
- Half supported -> 0.5
- No claims -> 1.0 (vacuously faithful)
- All unsupported -> 0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragaliq.core.evaluator import EvaluationResult
from ragaliq.core.test_case import RAGTestCase
from ragaliq.evaluators.faithfulness import FaithfulnessEvaluator
from ragaliq.judges.base import ClaimsResult, ClaimVerdict, LLMJudge

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_judge() -> MagicMock:
    """Create a mock LLM judge with claim extraction and verification."""
    judge = MagicMock(spec=LLMJudge)
    # Default: no claims extracted
    judge.extract_claims = AsyncMock(return_value=ClaimsResult(claims=[]))
    judge.verify_claim = AsyncMock()
    return judge


@pytest.fixture
def faithful_test_case() -> RAGTestCase:
    """A test case where response should be faithful to context."""
    return RAGTestCase(
        id="faithful_001",
        name="Faithful Response",
        query="What is the capital of France?",
        context=[
            "France is a country in Western Europe.",
            "The capital city of France is Paris.",
        ],
        response="The capital of France is Paris.",
    )


@pytest.fixture
def hallucinating_test_case() -> RAGTestCase:
    """A test case where response contains hallucinations."""
    return RAGTestCase(
        id="hallucinate_001",
        name="Hallucinating Response",
        query="What is the capital of France?",
        context=[
            "France is a country in Western Europe.",
            "The capital city of France is Paris.",
        ],
        response="Paris is the capital of France and was founded by Romans in 250 BC.",
    )


# =============================================================================
# Test Class Attributes
# =============================================================================


class TestFaithfulnessEvaluatorAttributes:
    """Tests for evaluator class attributes and initialization."""

    def test_has_required_name(self) -> None:
        """Evaluator must have 'faithfulness' as name."""
        evaluator = FaithfulnessEvaluator()
        assert evaluator.name == "faithfulness"

    def test_has_description(self) -> None:
        """Evaluator must have a meaningful description."""
        evaluator = FaithfulnessEvaluator()
        assert evaluator.description
        assert len(evaluator.description) > 10

    def test_default_threshold_is_0_7(self) -> None:
        """Default threshold should be 0.7 per base class."""
        evaluator = FaithfulnessEvaluator()
        assert evaluator.threshold == 0.7

    def test_custom_threshold(self) -> None:
        """Should accept custom threshold in constructor."""
        evaluator = FaithfulnessEvaluator(threshold=0.9)
        assert evaluator.threshold == 0.9

    def test_repr(self) -> None:
        """Should have a meaningful string representation."""
        evaluator = FaithfulnessEvaluator(threshold=0.8)
        assert "FaithfulnessEvaluator" in repr(evaluator)
        assert "0.8" in repr(evaluator)


# =============================================================================
# Acceptance Criteria Tests
# =============================================================================


class TestFaithfulnessEvaluatorAcceptanceCriteria:
    """Tests directly from Issue #6 acceptance criteria."""

    @pytest.mark.asyncio
    async def test_all_claims_supported_returns_1_0(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """AC: All claims supported -> 1.0"""
        # Arrange: 2 claims, both supported
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=[
                "France has a capital city",
                "The capital of France is Paris",
            ]
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence="Context states this"),
            ClaimVerdict(verdict="SUPPORTED", evidence="Context confirms Paris"),
        ]

        # Act
        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        # Assert
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_half_supported_returns_0_5(
        self,
        mock_judge: MagicMock,
        hallucinating_test_case: RAGTestCase,
    ) -> None:
        """AC: Half supported -> 0.5"""
        # Arrange: 2 claims, 1 supported, 1 not
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=[
                "Paris is the capital of France",
                "Paris was founded by Romans in 250 BC",
            ]
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence="Context confirms"),
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="No founding date in context"),
        ]

        # Act
        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(hallucinating_test_case, mock_judge)

        # Assert
        assert result.score == 0.5
        assert result.passed is False  # 0.5 < 0.7 threshold

    @pytest.mark.asyncio
    async def test_no_claims_returns_1_0(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """AC: No claims -> 1.0 (vacuously faithful)"""
        # Arrange: empty response extracts no claims
        mock_judge.extract_claims.return_value = ClaimsResult(claims=[])

        # Act
        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        # Assert
        assert result.score == 1.0
        assert result.passed is True
        # verify_claim should never be called with no claims
        mock_judge.verify_claim.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_unsupported_returns_0_0(
        self,
        mock_judge: MagicMock,
        hallucinating_test_case: RAGTestCase,
    ) -> None:
        """AC: All unsupported -> 0.0"""
        # Arrange: 2 claims, both unsupported
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=[
                "Paris was founded in 250 BC",
                "Romans established Paris",
            ]
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="No date in context"),
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="No Romans in context"),
        ]

        # Act
        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(hallucinating_test_case, mock_judge)

        # Assert
        assert result.score == 0.0
        assert result.passed is False


# =============================================================================
# Verdict Handling Tests
# =============================================================================


class TestFaithfulnessEvaluatorVerdictHandling:
    """Tests for how different verdicts affect the score."""

    @pytest.mark.asyncio
    async def test_contradicted_counts_as_unsupported(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """CONTRADICTED claims should count as unsupported (0)."""
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["Paris is in Germany"]
        )
        mock_judge.verify_claim.return_value = ClaimVerdict(
            verdict="CONTRADICTED",
            evidence="Context says France, not Germany",
        )

        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_not_enough_info_counts_as_unsupported(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """NOT_ENOUGH_INFO claims should count as unsupported (0)."""
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["The Eiffel Tower was built in 1889"]
        )
        mock_judge.verify_claim.return_value = ClaimVerdict(
            verdict="NOT_ENOUGH_INFO",
            evidence="No construction date in context",
        )

        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_mixed_verdicts_calculation(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """Test score with mix of SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO."""
        # 4 claims: 2 supported, 1 contradicted, 1 not enough info
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["Claim 1", "Claim 2", "Claim 3", "Claim 4"]
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence="Found in context"),
            ClaimVerdict(verdict="SUPPORTED", evidence="Also in context"),
            ClaimVerdict(verdict="CONTRADICTED", evidence="Context says opposite"),
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="Not mentioned"),
        ]

        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        # 2 supported / 4 total = 0.5
        assert result.score == 0.5


# =============================================================================
# Metadata Tests
# =============================================================================


class TestFaithfulnessEvaluatorMetadata:
    """Tests for claim details in raw_response metadata."""

    @pytest.mark.asyncio
    async def test_raw_response_contains_claim_details(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain claim-level details for debugging."""
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["Paris is the capital"]
        )
        mock_judge.verify_claim.return_value = ClaimVerdict(
            verdict="SUPPORTED",
            evidence="Context confirms this",
        )

        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        # Should have claims list in raw_response
        assert "claims" in result.raw_response
        claims = result.raw_response["claims"]
        assert len(claims) == 1
        assert claims[0]["claim"] == "Paris is the capital"
        assert claims[0]["verdict"] == "SUPPORTED"
        assert claims[0]["evidence"] == "Context confirms this"

    @pytest.mark.asyncio
    async def test_raw_response_contains_summary_stats(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain summary statistics."""
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["Claim 1", "Claim 2", "Claim 3"]
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence=""),
            ClaimVerdict(verdict="SUPPORTED", evidence=""),
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence=""),
        ]

        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        assert result.raw_response["total_claims"] == 3
        assert result.raw_response["supported_claims"] == 2


# =============================================================================
# Result Structure Tests
# =============================================================================


class TestFaithfulnessEvaluatorResultStructure:
    """Tests for EvaluationResult structure compliance."""

    @pytest.mark.asyncio
    async def test_returns_evaluation_result(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should return EvaluationResult instance."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=[])

        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        assert isinstance(result, EvaluationResult)

    @pytest.mark.asyncio
    async def test_evaluator_name_in_result(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """Result should contain evaluator name."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=[])

        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        assert result.evaluator_name == "faithfulness"

    @pytest.mark.asyncio
    async def test_reasoning_explains_score(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """Result reasoning should explain how score was calculated."""
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["Claim 1", "Claim 2"]
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence=""),
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence=""),
        ]

        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        # Reasoning should mention claim counts
        assert "1" in result.reasoning  # 1 supported
        assert "2" in result.reasoning  # 2 total


# =============================================================================
# Edge Cases
# =============================================================================


class TestFaithfulnessEvaluatorEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_claim_supported(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """Single supported claim should give score 1.0."""
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["Only claim"]
        )
        mock_judge.verify_claim.return_value = ClaimVerdict(
            verdict="SUPPORTED", evidence=""
        )

        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_single_claim_unsupported(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """Single unsupported claim should give score 0.0."""
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["Only claim"]
        )
        mock_judge.verify_claim.return_value = ClaimVerdict(
            verdict="NOT_ENOUGH_INFO", evidence=""
        )

        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_many_claims_precision(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """Score should handle many claims with proper precision."""
        # 7 claims, 5 supported = 5/7 ≈ 0.714...
        claims = [f"Claim {i}" for i in range(7)]
        mock_judge.extract_claims.return_value = ClaimsResult(claims=claims)
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence="") for _ in range(5)
        ] + [
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="") for _ in range(2)
        ]

        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        expected = 5 / 7
        assert abs(result.score - expected) < 0.001

    @pytest.mark.asyncio
    async def test_passed_uses_threshold_correctly(
        self,
        mock_judge: MagicMock,
        faithful_test_case: RAGTestCase,
    ) -> None:
        """passed field should correctly use is_passing() with threshold."""
        # Score will be 0.7 exactly (7/10)
        claims = [f"Claim {i}" for i in range(10)]
        mock_judge.extract_claims.return_value = ClaimsResult(claims=claims)

        def create_verdicts() -> list[ClaimVerdict]:
            """Create fresh verdicts list for each evaluation."""
            return [
                ClaimVerdict(verdict="SUPPORTED", evidence="") for _ in range(7)
            ] + [
                ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="") for _ in range(3)
            ]

        # Default threshold is 0.7
        mock_judge.verify_claim.side_effect = create_verdicts()
        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(faithful_test_case, mock_judge)

        assert result.score == 0.7
        assert result.passed is True  # 0.7 >= 0.7

        # With higher threshold - reset side_effect for fresh iteration
        mock_judge.verify_claim.side_effect = create_verdicts()
        evaluator_strict = FaithfulnessEvaluator(threshold=0.8)
        result_strict = await evaluator_strict.evaluate(faithful_test_case, mock_judge)

        assert result_strict.score == 0.7
        assert result_strict.passed is False  # 0.7 < 0.8
