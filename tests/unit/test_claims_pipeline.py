"""Unit tests for the shared claim verification pipeline (_claims.py).

Tests the verify_all_claims() function directly, covering:
- Token accumulation across extraction + verification
- claims_empty flag behavior
- context_empty short-circuit (no LLM calls on empty context)
- Error propagation from extract_claims vs verify_claim
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragaliq.evaluators._claims import ClaimVerificationResult, verify_all_claims
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
    """Create a mock LLM judge."""
    judge = MagicMock(spec=LLMJudge)
    judge.extract_claims = AsyncMock(return_value=ClaimsResult(claims=[], tokens_used=0))
    judge.verify_claim = AsyncMock()
    return judge


# =============================================================================
# Empty Context Short-Circuit (F5)
# =============================================================================


class TestEmptyContextShortCircuit:
    """Tests that empty context skips all LLM calls."""

    @pytest.mark.asyncio
    async def test_empty_context_returns_context_empty_flag(self, mock_judge: MagicMock) -> None:
        """Empty context should set context_empty=True."""
        result = await verify_all_claims("Some response", [], mock_judge)

        assert result.context_empty is True
        assert result.claims_empty is False
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_empty_context_skips_extract_claims(self, mock_judge: MagicMock) -> None:
        """Empty context should NOT call extract_claims (saves tokens)."""
        await verify_all_claims("Some response", [], mock_judge)

        mock_judge.extract_claims.assert_not_called()
        mock_judge.verify_claim.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_context_returns_empty_details(self, mock_judge: MagicMock) -> None:
        """Empty context result should have empty claim details."""
        result = await verify_all_claims("Some response", [], mock_judge)

        assert result.claim_details == []
        assert result.verdicts == []


# =============================================================================
# Empty Claims Behavior
# =============================================================================


class TestEmptyClaimsBehavior:
    """Tests for when extract_claims returns no claims."""

    @pytest.mark.asyncio
    async def test_empty_claims_sets_flag(self, mock_judge: MagicMock) -> None:
        """When extract_claims returns [], claims_empty should be True."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=[], tokens_used=20)

        result = await verify_all_claims("Short response", ["context doc"], mock_judge)

        assert result.claims_empty is True
        assert result.context_empty is False
        assert result.total_tokens == 20

    @pytest.mark.asyncio
    async def test_empty_claims_skips_verification(self, mock_judge: MagicMock) -> None:
        """When no claims extracted, verify_claim should NOT be called."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=[], tokens_used=10)

        await verify_all_claims("Short response", ["context doc"], mock_judge)

        mock_judge.verify_claim.assert_not_called()


# =============================================================================
# Token Accumulation
# =============================================================================


class TestTokenAccumulation:
    """Tests that tokens are correctly tracked across extraction + verification."""

    @pytest.mark.asyncio
    async def test_tokens_sum_extraction_and_verification(self, mock_judge: MagicMock) -> None:
        """Total tokens should include extraction + all verification calls."""
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["Claim A", "Claim B"], tokens_used=50
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence="Found", tokens_used=30),
            ClaimVerdict(verdict="NOT_ENOUGH_INFO", evidence="Missing", tokens_used=25),
        ]

        result = await verify_all_claims("Response text", ["context"], mock_judge)

        assert result.total_tokens == 105  # 50 + 30 + 25

    @pytest.mark.asyncio
    async def test_tokens_with_single_claim(self, mock_judge: MagicMock) -> None:
        """Token tracking works for single claim."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["Only claim"], tokens_used=40)
        mock_judge.verify_claim.return_value = ClaimVerdict(
            verdict="SUPPORTED", evidence="Yes", tokens_used=20
        )

        result = await verify_all_claims("Response", ["context"], mock_judge)

        assert result.total_tokens == 60  # 40 + 20


# =============================================================================
# Claim Details
# =============================================================================


class TestClaimDetails:
    """Tests for claim detail construction."""

    @pytest.mark.asyncio
    async def test_claim_details_match_verdicts(self, mock_judge: MagicMock) -> None:
        """Each ClaimDetail should pair the claim text with its verdict."""
        mock_judge.extract_claims.return_value = ClaimsResult(
            claims=["Claim X", "Claim Y"], tokens_used=10
        )
        mock_judge.verify_claim.side_effect = [
            ClaimVerdict(verdict="SUPPORTED", evidence="Evidence X", tokens_used=5),
            ClaimVerdict(verdict="CONTRADICTED", evidence="Evidence Y", tokens_used=5),
        ]

        result = await verify_all_claims("Response", ["context"], mock_judge)

        assert len(result.claim_details) == 2
        assert result.claim_details[0].claim == "Claim X"
        assert result.claim_details[0].verdict == "SUPPORTED"
        assert result.claim_details[0].evidence == "Evidence X"
        assert result.claim_details[1].claim == "Claim Y"
        assert result.claim_details[1].verdict == "CONTRADICTED"

    @pytest.mark.asyncio
    async def test_verdicts_list_preserved(self, mock_judge: MagicMock) -> None:
        """Raw ClaimVerdict objects should be available in verdicts list."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["Claim Z"], tokens_used=10)
        verdict = ClaimVerdict(verdict="SUPPORTED", evidence="Found", tokens_used=5)
        mock_judge.verify_claim.return_value = verdict

        result = await verify_all_claims("Response", ["context"], mock_judge)

        assert len(result.verdicts) == 1
        assert result.verdicts[0].verdict == "SUPPORTED"


# =============================================================================
# Error Propagation
# =============================================================================


class TestErrorPropagation:
    """Tests that errors from judge methods propagate correctly."""

    @pytest.mark.asyncio
    async def test_extract_claims_error_propagates(self, mock_judge: MagicMock) -> None:
        """Error in extract_claims should propagate immediately."""
        mock_judge.extract_claims = AsyncMock(
            side_effect=JudgeAPIError("API failure", status_code=500)
        )

        with pytest.raises(JudgeAPIError, match="API failure"):
            await verify_all_claims("Response", ["context"], mock_judge)

    @pytest.mark.asyncio
    async def test_verify_claim_error_propagates(self, mock_judge: MagicMock) -> None:
        """Error in verify_claim should propagate."""
        mock_judge.extract_claims.return_value = ClaimsResult(claims=["A claim"], tokens_used=10)
        mock_judge.verify_claim = AsyncMock(side_effect=JudgeResponseError("Parse failure"))

        with pytest.raises(JudgeResponseError, match="Parse failure"):
            await verify_all_claims("Response", ["context"], mock_judge)


# =============================================================================
# Result Model
# =============================================================================


class TestClaimVerificationResultModel:
    """Tests for ClaimVerificationResult defaults and immutability."""

    def test_default_values(self) -> None:
        """Default result should be empty with no flags set."""
        result = ClaimVerificationResult()

        assert result.claim_details == []
        assert result.verdicts == []
        assert result.total_tokens == 0
        assert result.claims_empty is False
        assert result.context_empty is False

    def test_frozen_model(self) -> None:
        """Result should be immutable (frozen=True)."""
        result = ClaimVerificationResult(claims_empty=True, total_tokens=50)

        with pytest.raises(Exception):  # noqa: B017
            result.claims_empty = False  # type: ignore[misc]
