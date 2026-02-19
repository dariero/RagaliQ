"""
Shared claim verification pipeline for claim-based evaluators.

This module extracts the common extract→verify→aggregate pattern used by
FaithfulnessEvaluator and HallucinationEvaluator, eliminating duplication
and providing a single place to maintain the claim verification flow.
"""

import asyncio
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ragaliq.judges.base import ClaimVerdict

if TYPE_CHECKING:
    from ragaliq.judges.base import LLMJudge


class ClaimDetail(BaseModel):
    """A single claim with its verification verdict and evidence."""

    claim: str
    verdict: str
    evidence: str

    model_config = {"frozen": True, "extra": "forbid"}


class ClaimVerificationResult(BaseModel):
    """Result of verifying all claims extracted from a response.

    Attributes:
        claim_details: Per-claim verification results.
        verdicts: Raw ClaimVerdict objects for further analysis.
        total_tokens: Total tokens consumed across extraction and verification.
        claims_empty: True if no claims were extracted (vacuous case).
    """

    claim_details: list[ClaimDetail] = Field(default_factory=list)
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    total_tokens: int = Field(default=0, ge=0)
    claims_empty: bool = False

    model_config = {"frozen": True, "extra": "forbid"}


async def verify_all_claims(
    response: str,
    context: list[str],
    judge: LLMJudge,
) -> ClaimVerificationResult:
    """Extract claims from a response and verify each against the context.

    Implements the shared pipeline:
    1. Extract atomic claims via judge.extract_claims()
    2. Verify each claim in parallel via judge.verify_claim()
    3. Collect results with token tracking

    Args:
        response: The RAG system's generated response.
        context: List of context documents to verify against.
        judge: The LLM judge instance.

    Returns:
        ClaimVerificationResult with all claim details and verdicts.
    """
    # Step 1: Extract atomic claims
    claims_result = await judge.extract_claims(response)
    claims = claims_result.claims
    total_tokens = claims_result.tokens_used

    # Handle empty claims (vacuous case)
    if not claims:
        return ClaimVerificationResult(
            claims_empty=True,
            total_tokens=total_tokens,
        )

    # Step 2: Verify each claim in parallel
    verification_tasks = [judge.verify_claim(claim, context) for claim in claims]
    verdicts = await asyncio.gather(*verification_tasks)

    # Step 3: Build detailed results
    claim_details: list[ClaimDetail] = []
    for i, verdict in enumerate(verdicts):
        total_tokens += verdict.tokens_used
        claim_details.append(
            ClaimDetail(
                claim=claims[i],
                verdict=verdict.verdict,
                evidence=verdict.evidence,
            )
        )

    return ClaimVerificationResult(
        claim_details=claim_details,
        verdicts=list(verdicts),
        total_tokens=total_tokens,
    )
