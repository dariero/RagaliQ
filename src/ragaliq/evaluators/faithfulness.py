"""
Faithfulness evaluator for RagaliQ.

This module implements claim-level faithfulness evaluation, measuring whether
an LLM response is grounded only in the provided context without hallucinations.

Algorithm:
    1. Extract atomic claims from the response
    2. Verify each claim against the context (SUPPORTED/CONTRADICTED/NOT_ENOUGH_INFO)
    3. Score = supported_claims / total_claims
    4. Empty claims = 1.0 (vacuously faithful)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.evaluators.registry import register_evaluator

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestCase
    from ragaliq.judges.base import LLMJudge


@register_evaluator("faithfulness")
class FaithfulnessEvaluator(Evaluator):
    """
    Evaluator that measures response faithfulness to context.

    Faithfulness means every claim in the response is supported by the context.
    A faithful response does not hallucinate or add information beyond what
    the context provides.

    This evaluator uses claim-level decomposition:
    1. Extract atomic claims from the response
    2. Verify each claim against the context
    3. Calculate score as ratio of supported claims

    Attributes:
        name: "faithfulness" - unique identifier for this evaluator.
        description: Human-readable description of what is evaluated.
        threshold: Minimum score to pass (default 0.7).

    Example:
        evaluator = FaithfulnessEvaluator(threshold=0.8)
        result = await evaluator.evaluate(test_case, judge)

        if result.passed:
            print(f"Response is faithful: {result.score:.2f}")
        else:
            # Inspect which claims failed
            for claim in result.raw_response["claims"]:
                if claim["verdict"] != "SUPPORTED":
                    print(f"Unsupported: {claim['claim']}")
    """

    name: str = "faithfulness"
    description: str = "Measures if response is grounded only in context"

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge,
    ) -> EvaluationResult:
        """
        Evaluate faithfulness of the response to the context.

        Implements the claim-based faithfulness algorithm:
        1. Extract claims from response via judge.extract_claims()
        2. Verify each claim via judge.verify_claim()
        3. Score = supported_claims / total_claims
        4. Empty claims = 1.0 (vacuously faithful)

        Args:
            test_case: The RAG test case containing response and context.
            judge: The LLM judge instance for claim extraction and verification.

        Returns:
            EvaluationResult with:
                - score: Ratio of supported claims (0.0 to 1.0)
                - passed: Whether score meets threshold
                - reasoning: Human-readable explanation
                - raw_response: Detailed claim-level metadata
        """
        # Step 1: Extract atomic claims from the response
        claims_result = await judge.extract_claims(test_case.response)
        claims = claims_result.claims
        total_tokens = claims_result.tokens_used

        # Handle empty claims edge case: vacuously faithful
        if not claims:
            return EvaluationResult(
                evaluator_name=self.name,
                score=1.0,
                passed=True,
                reasoning="No claims to verify; response is vacuously faithful.",
                raw_response={
                    "claims": [],
                    "total_claims": 0,
                    "supported_claims": 0,
                },
                tokens_used=total_tokens,
            )

        # Step 2: Verify each claim against the context (in parallel)
        import asyncio

        # Verify all claims concurrently (errors propagate)
        verification_tasks = [judge.verify_claim(claim, test_case.context) for claim in claims]
        verdicts = await asyncio.gather(*verification_tasks)

        # Process results
        claim_details: list[dict[str, Any]] = []
        supported_count = 0

        for i, verdict in enumerate(verdicts):
            total_tokens += verdict.tokens_used

            claim_details.append(
                {
                    "claim": claims[i],
                    "verdict": verdict.verdict,
                    "evidence": verdict.evidence,
                }
            )

            if verdict.verdict == "SUPPORTED":
                supported_count += 1

        # Step 3: Calculate score as ratio
        total_claims = len(claims)
        score = supported_count / total_claims

        # Step 4: Build reasoning
        reasoning = self._build_reasoning(supported_count, total_claims)

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=self.is_passing(score),
            reasoning=reasoning,
            raw_response={
                "claims": claim_details,
                "total_claims": total_claims,
                "supported_claims": supported_count,
            },
            tokens_used=total_tokens,
        )

    def _build_reasoning(self, supported: int, total: int) -> str:
        """
        Build human-readable reasoning for the faithfulness score.

        Args:
            supported: Number of supported claims.
            total: Total number of claims.

        Returns:
            Reasoning string explaining the score calculation.
        """
        if total == 0:
            return "No claims to verify."

        unsupported = total - supported
        score_pct = (supported / total) * 100

        if supported == total:
            return f"All {total} claims are supported by the context."
        elif supported == 0:
            return f"None of the {total} claims are supported by the context."
        else:
            return (
                f"{supported} of {total} claims are supported ({score_pct:.0f}%). "
                f"{unsupported} claim(s) not grounded in context."
            )
