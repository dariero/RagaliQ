"""
Hallucination evaluator for RagaliQ.

This module implements claim-level hallucination detection, identifying
fabricated facts in an LLM response that are not grounded in the provided
context.

Algorithm:
    1. Extract atomic claims from the response
    2. Verify each claim against the context (SUPPORTED/CONTRADICTED/NOT_ENOUGH_INFO)
    3. Classify non-SUPPORTED claims as hallucinated
    4. Score = 1.0 - (hallucinated_claims / total_claims)
    5. Empty claims = 1.0 (no hallucinations possible)

Distinction from FaithfulnessEvaluator:
    - Stricter default threshold (0.8 vs 0.7)
    - Metadata focuses on hallucinated claims (which claims were fabricated)
    - Reasoning framed as hallucination detection, not faithfulness assessment
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragaliq.core.evaluator import EvaluationResult, Evaluator

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestCase
    from ragaliq.judges.base import LLMJudge


class HallucinationEvaluator(Evaluator):
    """
    Evaluator that detects hallucinated claims in a response.

    Hallucination means the response contains claims that are not grounded
    in the provided context — the model fabricated information. This evaluator
    uses claim-level decomposition to identify exactly which statements are
    hallucinated.

    This is stricter than FaithfulnessEvaluator: default threshold is 0.8
    (vs 0.7), and metadata explicitly lists hallucinated claims for debugging.

    Attributes:
        name: "hallucination" - unique identifier for this evaluator.
        description: Human-readable description of what is evaluated.
        threshold: Minimum score to pass (default 0.8, stricter than faithfulness).

    Example:
        evaluator = HallucinationEvaluator(threshold=0.9)
        result = await evaluator.evaluate(test_case, judge)

        if not result.passed:
            # Inspect which claims are hallucinated
            for claim in result.raw_response["hallucinated_claims"]:
                print(f"Hallucinated: {claim['claim']} ({claim['verdict']})")
    """

    name: str = "hallucination"
    description: str = "Detects hallucinated claims not grounded in context"
    threshold: float = 0.8

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge,
    ) -> EvaluationResult:
        """
        Evaluate the response for hallucinated claims.

        Implements the claim-based hallucination detection algorithm:
        1. Extract claims from response via judge.extract_claims()
        2. Verify each claim via judge.verify_claim()
        3. Classify non-SUPPORTED verdicts as hallucinated
        4. Score = 1.0 - (hallucinated / total)
        5. Empty claims = 1.0 (no hallucinations)

        Args:
            test_case: The RAG test case containing response and context.
            judge: The LLM judge instance for claim extraction and verification.

        Returns:
            EvaluationResult with:
                - score: 1.0 minus ratio of hallucinated claims (0.0 to 1.0)
                - passed: Whether score meets threshold
                - reasoning: Human-readable explanation
                - raw_response: Detailed hallucination metadata
        """
        # Step 1: Extract atomic claims from the response
        claims_result = await judge.extract_claims(test_case.response)
        claims = claims_result.claims

        # Handle empty claims edge case: no claims means no hallucinations
        if not claims:
            return EvaluationResult(
                evaluator_name=self.name,
                score=1.0,
                passed=True,
                reasoning="No claims to verify; no hallucinations detected.",
                raw_response={
                    "claims": [],
                    "total_claims": 0,
                    "hallucinated_claims": [],
                    "hallucination_count": 0,
                },
            )

        # Step 2: Verify each claim and classify hallucinations
        claim_details: list[dict[str, Any]] = []
        hallucinated: list[dict[str, Any]] = []

        for claim in claims:
            verdict = await judge.verify_claim(claim, test_case.context)

            detail = {
                "claim": claim,
                "verdict": verdict.verdict,
                "evidence": verdict.evidence,
            }
            claim_details.append(detail)

            # Non-SUPPORTED claims are hallucinated
            if verdict.verdict != "SUPPORTED":
                hallucinated.append(detail)

        # Step 3: Calculate score = 1.0 - (hallucinated / total)
        total_claims = len(claims)
        hallucination_count = len(hallucinated)
        score = 1.0 - (hallucination_count / total_claims)

        # Step 4: Build reasoning
        reasoning = self._build_reasoning(hallucination_count, total_claims)

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=self.is_passing(score),
            reasoning=reasoning,
            raw_response={
                "claims": claim_details,
                "total_claims": total_claims,
                "hallucinated_claims": hallucinated,
                "hallucination_count": hallucination_count,
            },
        )

    def _build_reasoning(self, hallucinated: int, total: int) -> str:
        """
        Build human-readable reasoning for the hallucination score.

        Args:
            hallucinated: Number of hallucinated claims.
            total: Total number of claims.

        Returns:
            Reasoning string explaining the hallucination detection result.
        """
        if total == 0:
            return "No claims to verify."

        score_pct = (1.0 - hallucinated / total) * 100

        if hallucinated == 0:
            return f"All {total} claims are grounded in the context. No hallucinations detected."
        elif hallucinated == total:
            return f"All {total} claims are hallucinated — none are supported by the context."
        else:
            grounded = total - hallucinated
            return (
                f"Found {hallucinated} hallucinated claim(s) out of {total} "
                f"({score_pct:.0f}% grounded). "
                f"{grounded} claim(s) are supported by the context."
            )
