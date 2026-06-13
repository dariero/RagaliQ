"""Shared claim verification pipeline (extract → verify → aggregate).

Factored out of FaithfulnessEvaluator and HallucinationEvaluator so the
claim-based flow lives in one place.
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
    """Aggregated result of extracting and verifying a response's claims."""

    claim_details: list[ClaimDetail] = Field(default_factory=list)
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    total_tokens: int = Field(default=0, ge=0)
    claims_empty: bool = False
    context_empty: bool = False

    model_config = {"frozen": True, "extra": "forbid"}


async def verify_all_claims(
    response: str,
    context: list[str],
    judge: LLMJudge,
) -> ClaimVerificationResult:
    """Extract atomic claims from `response` and verify each against `context`.

    Returns early (without an LLM call) when context is empty, and after
    extraction when no claims are found.

    Returns:
        ClaimVerificationResult with per-claim details, verdicts, and token total.
    """
    # No context ⇒ every claim is NOT_ENOUGH_INFO; skip the extract call to save tokens.
    if not context:
        return ClaimVerificationResult(context_empty=True)

    claims_result = await judge.extract_claims(response)
    claims = claims_result.claims
    total_tokens = claims_result.tokens_used

    if not claims:
        return ClaimVerificationResult(claims_empty=True, total_tokens=total_tokens)

    verification_tasks = [judge.verify_claim(claim, context) for claim in claims]
    verdicts = await asyncio.gather(*verification_tasks)

    claim_details = [
        ClaimDetail(claim=claims[i], verdict=verdict.verdict, evidence=verdict.evidence)
        for i, verdict in enumerate(verdicts)
    ]
    total_tokens += sum(verdict.tokens_used for verdict in verdicts)

    return ClaimVerificationResult(
        claim_details=claim_details,
        verdicts=list(verdicts),
        total_tokens=total_tokens,
    )
