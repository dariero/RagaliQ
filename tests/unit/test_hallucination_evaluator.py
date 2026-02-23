"""Unit tests for HallucinationEvaluator.

Tests follow the acceptance criteria from Issue #8:
- No hallucinations -> 1.0
- Low confidence flagged as potential hallucination
- Clear distinction from faithfulness documented
- All quality gates pass
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragaliq.core.evaluator import EvaluationResult
from ragaliq.core.test_case import RAGTestCase
from ragaliq.evaluators.hallucination import HallucinationEvaluator
from ragaliq.judges.base import (
    ClaimsResult,
    ClaimVerdict,
    JudgeAPIError,
    JudgeResponseError,
    LLMJudge,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_judge() -> MagicMock:
    """Create a mock LLM judge with claim extraction and verification."""
    judge = MagicMock(spec=LLMJudge)
    # Default: no claims extracted
    judge.extract_claims = AsyncMock(return_value=ClaimsResult(claims=[], tokens_used=10))
    judge.verify_claim = AsyncMock()
    return judge


@pytest.fixture
def grounded_test_case() -> RAGTestCase:
    """A test case where response is fully grounded in context."""
    return RAGTestCase(
        id="grounded_001",
        name="Grounded Response",
        query="What is the capital of France?",
        context=[
            "France is a country in Western Europe.",
            "The capital city of France is Paris.",
        ],
        response="The capital of France is Paris.",
    )


@pytest.fixture
def hallucinating_test_case() -> RAGTestCase:
    """A test case where response contains hallucinated facts."""
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


class TestHallucinationEvaluatorAttributes:
    """Tests for evaluator class attributes and initialization."""

    def test_has_required_name(self) -> None:
        """Evaluator must have 'hallucination' as name."""
        evaluator = HallucinationEvaluator()
        assert evaluator.name == "hallucination"

    def test_has_description(self) -> None:
        """Evaluator must have a meaningful description."""
        evaluator = HallucinationEvaluator()
        assert evaluator.description
        assert len(evaluator.description) > 10

    def test_default_threshold_is_0_8(self) -> None:
        """Default threshold should be 0.8 (stricter than faithfulness)."""
        evaluator = HallucinationEvaluator()
        assert evaluator.threshold == 0.8

    def test_custom_threshold(self) -> None:
        """Should accept custom threshold in constructor."""
        evaluator = HallucinationEvaluator(threshold=0.95)
        assert evaluator.threshold == 0.95

    def test_repr(self) -> None:
        """Should have a meaningful string representation."""
        evaluator = HallucinationEvaluator(threshold=0.85)
        assert "HallucinationEvaluator" in repr(evaluator)
        assert "0.85" in repr(evaluator)

    def test_stricter_than_faithfulness_default(self) -> None:
        """Default threshold must be stricter than FaithfulnessEvaluator (0.7)."""
        from ragaliq.evaluators.faithfulness import FaithfulnessEvaluator

        hallucination = HallucinationEvaluator()
        faithfulness = FaithfulnessEvaluator()
        assert hallucination.threshold > faithfulness.threshold


# =============================================================================
# Acceptance Criteria Tests (Issue #8)
# =============================================================================


class TestHallucinationEvaluatorAcceptanceCriteria:
    """Tests directly from Issue #8 acceptance criteria."""

    @pytest.mark.asyncio
    async def test_no_hallucinations_returns_1_0(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """AC: No hallucinations -> 1.0"""
        # Arrange: 2 claims, both supported
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=[
                "France has a capital city",
                "The capital of France is Paris",
            ],
            tokens_used=45,
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence="Context states this", tokens_used=22),
            ClaimVerdict(verdict="SUPPORTED", evidence="Context confirms Paris", tokens_used=18),
        ]

        # Act
        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        # Assert
        assert result.score == 1.0
        assert result.passed is True
        assert result.tokens_used == 85  # 45 + 22 + 18

    @pytest.mark.asyncio
    async def test_not_enough_info_flagged_as_hallucination(
        self,
        mock_judge: MagicMock,
        hallucinating_test_case: RAGTestCase,
    ) -> None:
        """AC: Low confidence (NOT_ENOUGH_INFO) flagged as potential hallucination."""
        # Arrange: 2 claims, 1 supported, 1 not enough info
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
        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(hallucinating_test_case, mock_judge)

        # Assert: 1 hallucinated out of 2 → score = 0.5
        assert result.score == 0.5
        assert result.passed is False  # 0.5 < 0.8 threshold

    @pytest.mark.asyncio
    async def test_contradicted_flagged_as_hallucination(
        self,
        mock_judge: MagicMock,
        hallucinating_test_case: RAGTestCase,
    ) -> None:
        """AC: CONTRADICTED claims are hallucinated."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["Paris is in Germany"])
        mock_judge.verify_claim.return_value = ClaimVerdict(
            verdict="CONTRADICTED",
            evidence="Context says France, not Germany",
        )

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(hallucinating_test_case, mock_judge)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_hallucinated_claims_stored_in_metadata(
        self,
        mock_judge: MagicMock,
        hallucinating_test_case: RAGTestCase,
    ) -> None:
        """AC: hallucinated_claims stored in metadata."""
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=[
                "Paris is the capital of France",
                "Paris was founded in 250 BC",
                "Romans established Paris",
            ]
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence="Confirmed"),
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="No date in context"),
            ClaimVerdict(verdict="CONTRADICTED", evidence="Not mentioned"),
        ]

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(hallucinating_test_case, mock_judge)

        # Should have exactly 2 hallucinated claims
        assert "hallucinated_claims" in result.raw_response
        hallucinated = result.raw_response["hallucinated_claims"]
        assert len(hallucinated) == 2
        assert hallucinated[0]["claim"] == "Paris was founded in 250 BC"
        assert hallucinated[1]["claim"] == "Romans established Paris"


# =============================================================================
# Verdict Classification Tests
# =============================================================================


class TestHallucinationEvaluatorVerdictHandling:
    """Tests for how different verdicts affect hallucination classification."""

    @pytest.mark.asyncio
    async def test_only_supported_is_not_hallucinated(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """Only SUPPORTED claims count as non-hallucinated."""
        # 3 claims: 1 each of SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["Claim A", "Claim B", "Claim C"]
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence="Found in context"),
            ClaimVerdict(verdict="CONTRADICTED", evidence="Context says opposite"),
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="Not mentioned"),
        ]

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        # 2 hallucinated out of 3 → score = 1.0 - (2/3) ≈ 0.333
        expected = 1.0 - (2 / 3)
        assert abs(result.score - expected) < 0.001

    @pytest.mark.asyncio
    async def test_all_hallucinated_returns_0_0(
        self,
        mock_judge: MagicMock,
        hallucinating_test_case: RAGTestCase,
    ) -> None:
        """All claims hallucinated should give score 0.0."""
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["Fake claim 1", "Fake claim 2"]
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="Not in context"),
            ClaimVerdict(verdict="CONTRADICTED", evidence="Contradicts context"),
        ]

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(hallucinating_test_case, mock_judge)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_mixed_verdicts_calculation(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """Test score with mix of verdicts: 3 supported, 1 contradicted, 1 NEI."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["C1", "C2", "C3", "C4", "C5"])
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence=""),
            ClaimVerdict(verdict="SUPPORTED", evidence=""),
            ClaimVerdict(verdict="SUPPORTED", evidence=""),
            ClaimVerdict(verdict="CONTRADICTED", evidence=""),
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence=""),
        ]

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        # 2 hallucinated / 5 total → score = 1.0 - 0.4 = 0.6
        assert result.score == pytest.approx(0.6)


# =============================================================================
# Metadata Tests
# =============================================================================


class TestHallucinationEvaluatorMetadata:
    """Tests for hallucination details in raw_response metadata."""

    @pytest.mark.asyncio
    async def test_raw_response_contains_all_claim_details(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain full claim-level details."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["Paris is the capital"])
        mock_judge.verify_claim.return_value = ClaimVerdict(
            verdict="SUPPORTED",
            evidence="Context confirms this",
        )

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        assert "claims" in result.raw_response
        claims = result.raw_response["claims"]
        assert len(claims) == 1
        assert claims[0]["claim"] == "Paris is the capital"
        assert claims[0]["verdict"] == "SUPPORTED"
        assert claims[0]["evidence"] == "Context confirms this"

    @pytest.mark.asyncio
    async def test_raw_response_contains_hallucination_summary(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """raw_response should contain hallucination count and list."""
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["Claim 1", "Claim 2", "Claim 3"]
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence=""),
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="Missing"),
            ClaimVerdict(verdict="SUPPORTED", evidence=""),
        ]

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        assert result.raw_response["total_claims"] == 3
        assert result.raw_response["hallucination_count"] == 1
        assert len(result.raw_response["hallucinated_claims"]) == 1
        assert result.raw_response["hallucinated_claims"][0]["claim"] == "Claim 2"

    @pytest.mark.asyncio
    async def test_empty_claims_metadata(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """Empty claims should produce clean metadata with score 0.0."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=[])

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        assert result.score == 0.0
        assert result.passed is False
        assert result.raw_response["total_claims"] == 0
        assert result.raw_response["hallucination_count"] == 0
        assert result.raw_response["hallucinated_claims"] == []
        assert result.raw_response["claims"] == []


# =============================================================================
# Result Structure Tests
# =============================================================================


class TestHallucinationEvaluatorResultStructure:
    """Tests for EvaluationResult structure compliance."""

    @pytest.mark.asyncio
    async def test_returns_evaluation_result(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """evaluate() should return EvaluationResult instance."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=[])

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        assert isinstance(result, EvaluationResult)

    @pytest.mark.asyncio
    async def test_evaluator_name_in_result(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """Result should contain evaluator name."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=[])

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        assert result.evaluator_name == "hallucination"

    @pytest.mark.asyncio
    async def test_reasoning_mentions_hallucination(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """Result reasoning should use hallucination framing."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["Claim 1", "Claim 2"])
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence=""),
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence=""),
        ]

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        # Reasoning should mention hallucination, not just "supported"
        assert "hallucinated" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_score_bounded_0_to_1(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """Score must always be between 0.0 and 1.0."""
        # All hallucinated → 0.0
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["Bad claim"])
        mock_judge.verify_claim.return_value = ClaimVerdict(verdict="CONTRADICTED", evidence="")

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        assert 0.0 <= result.score <= 1.0


# =============================================================================
# Edge Cases
# =============================================================================


class TestHallucinationEvaluatorEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_no_claims_returns_0_0(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """No claims extracted should give score 0.0 (cannot assess)."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=[])

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        assert result.score == 0.0
        assert result.passed is False
        mock_judge.verify_claim.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_claim_supported(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """Single supported claim should give score 1.0."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["Only claim"])
        mock_judge.verify_claim.return_value = ClaimVerdict(verdict="SUPPORTED", evidence="")

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_single_claim_hallucinated(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """Single hallucinated claim should give score 0.0."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["Made up fact"])
        mock_judge.verify_claim.return_value = ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="")

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_many_claims_precision(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """Score should handle many claims with proper precision."""
        # 7 claims, 2 hallucinated = score 1.0 - (2/7) ≈ 0.714...
        claims = [f"Claim {i}" for i in range(7)]
        mock_judge.extract_claims.return_value = ClaimsResult(claims=claims)
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence="") for _ in range(5)
        ] + [ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="") for _ in range(2)]

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        expected = 1.0 - (2 / 7)
        assert abs(result.score - expected) < 0.001

    @pytest.mark.asyncio
    async def test_passed_uses_threshold_correctly(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """passed field should correctly use is_passing() with threshold."""
        # Score will be 0.8 exactly (8/10 supported, 2/10 hallucinated)
        claims = [f"Claim {i}" for i in range(10)]
        mock_judge.extract_claims.return_value = ClaimsResult(claims=claims)

        def create_verdicts() -> list[ClaimVerdict]:
            """Create fresh verdicts list for each evaluation."""
            return [ClaimVerdict(verdict="SUPPORTED", evidence="") for _ in range(8)] + [
                ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="") for _ in range(2)
            ]

        # Default threshold is 0.8 → score 0.8 passes
        mock_judge.verify_claim.side_effect = create_verdicts()
        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        assert result.score == 0.8
        assert result.passed is True  # 0.8 >= 0.8

        # With stricter threshold → score 0.8 fails
        mock_judge.verify_claim.side_effect = create_verdicts()
        evaluator_strict = HallucinationEvaluator(threshold=0.9)
        result_strict = await evaluator_strict.evaluate(grounded_test_case, mock_judge)

        assert result_strict.score == 0.8
        assert result_strict.passed is False  # 0.8 < 0.9

    @pytest.mark.asyncio
    async def test_no_hallucinations_reasoning(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """Reasoning for perfect score mentions no hallucinations."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["Good claim"])
        mock_judge.verify_claim.return_value = ClaimVerdict(verdict="SUPPORTED", evidence="")

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        assert "no hallucination" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_all_hallucinated_reasoning(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """Reasoning for zero score mentions all claims hallucinated."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["Bad 1", "Bad 2"])
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="CONTRADICTED", evidence=""),
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence=""),
        ]

        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(grounded_test_case, mock_judge)

        assert "all" in result.reasoning.lower()
        assert "hallucinated" in result.reasoning.lower()


# =============================================================================
# Error Propagation Tests
# =============================================================================


class TestHallucinationEvaluatorErrorPropagation:
    """Tests that judge errors propagate correctly through the evaluator."""

    @pytest.mark.asyncio
    async def test_extract_claims_api_error_propagates(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """JudgeAPIError from extract_claims should propagate through evaluate()."""
        mock_judge.extract_claims = AsyncMock(
            side_effect=JudgeAPIError("Claude API error: Rate limit exceeded", status_code=429)
        )

        evaluator = HallucinationEvaluator()
        with pytest.raises(JudgeAPIError, match="Rate limit exceeded"):
            await evaluator.evaluate(grounded_test_case, mock_judge)

    @pytest.mark.asyncio
    async def test_verify_claim_response_error_propagates(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """JudgeResponseError from verify_claim should propagate through evaluate()."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["A claim"])
        mock_judge.verify_claim = AsyncMock(
            side_effect=JudgeResponseError("Failed to parse JSON response")
        )

        evaluator = HallucinationEvaluator()
        with pytest.raises(JudgeResponseError, match="Failed to parse JSON"):
            await evaluator.evaluate(grounded_test_case, mock_judge)

    @pytest.mark.asyncio
    async def test_verify_claim_api_error_on_second_claim(
        self,
        mock_judge: MagicMock,
        grounded_test_case: RAGTestCase,
    ) -> None:
        """API error on a subsequent claim should propagate (no partial results)."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["Claim 1", "Claim 2"])
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence=""),
            JudgeAPIError("Server error", status_code=500),
        ]

        evaluator = HallucinationEvaluator()
        with pytest.raises(JudgeAPIError, match="Server error"):
            await evaluator.evaluate(grounded_test_case, mock_judge)


# =============================================================================
# Package Export Tests
# =============================================================================


class TestHallucinationEvaluatorExport:
    """Tests that HallucinationEvaluator is properly exported."""

    def test_importable_from_evaluators_package(self) -> None:
        """HallucinationEvaluator should be importable from the evaluators package."""
        from ragaliq.evaluators import HallucinationEvaluator as Imported

        assert Imported is HallucinationEvaluator
