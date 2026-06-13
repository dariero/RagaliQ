"""Hallucination evaluator: which response claims are NOT grounded in context?

Algorithm:
    1. Extract atomic claims from the response.
    2. Verify each against the context.
    3. Treat every non-SUPPORTED verdict as a hallucination.
    4. Score = 1.0 - hallucinated / total.
    5. Empty context or no claims = 0.0 (cannot assess).

Stricter than FaithfulnessEvaluator: default threshold 0.8, and the metadata
lists the specific hallucinated claims.
"""

from typing import TYPE_CHECKING

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.evaluators._claims import verify_all_claims
from ragaliq.evaluators.registry import register_evaluator

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestCase
    from ragaliq.judges.base import LLMJudge


@register_evaluator("hallucination")
class HallucinationEvaluator(Evaluator):
    """Detects claims in the response that are not grounded in the context.

    Uses claim-level decomposition to flag exactly which statements were
    fabricated. Stricter than faithfulness (default threshold 0.8).
    """

    name: str = "hallucination"
    description: str = "Detects hallucinated claims not grounded in context"
    threshold: float = 0.8

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge,
    ) -> EvaluationResult:
        """Score as 1.0 minus the fraction of response claims not supported by context."""
        verification = await verify_all_claims(test_case.response, test_case.context, judge)

        if verification.context_empty:
            return self._empty_result(
                "No context provided; hallucination detection cannot be performed.",
                tokens_used=0,
            )
        if verification.claims_empty:
            return self._empty_result(
                "No claims could be extracted from the response; "
                "hallucination detection cannot be performed.",
                tokens_used=verification.total_tokens,
            )

        all_details = [d.model_dump() for d in verification.claim_details]
        hallucinated = [d for d in all_details if d["verdict"] != "SUPPORTED"]
        total_claims = len(verification.claim_details)
        hallucination_count = len(hallucinated)
        score = 1.0 - (hallucination_count / total_claims)

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=self.is_passing(score),
            reasoning=self._build_reasoning(hallucination_count, total_claims),
            raw_response={
                "claims": all_details,
                "total_claims": total_claims,
                "hallucinated_claims": hallucinated,
                "hallucination_count": hallucination_count,
            },
            tokens_used=verification.total_tokens,
        )

    def _empty_result(self, reasoning: str, tokens_used: int) -> EvaluationResult:
        """Build a failing 0.0 result for cases where hallucination can't be assessed."""
        return EvaluationResult(
            evaluator_name=self.name,
            score=0.0,
            passed=False,
            reasoning=reasoning,
            raw_response={
                "claims": [],
                "total_claims": 0,
                "hallucinated_claims": [],
                "hallucination_count": 0,
            },
            tokens_used=tokens_used,
        )

    def _build_reasoning(self, hallucinated: int, total: int) -> str:
        """Summarize how many claims were hallucinated vs grounded."""
        if total == 0:
            return "No claims to verify."

        if hallucinated == 0:
            return f"All {total} claims are grounded in the context. No hallucinations detected."
        if hallucinated == total:
            return f"All {total} claims are hallucinated — none are supported by the context."

        grounded = total - hallucinated
        score_pct = (1.0 - hallucinated / total) * 100
        return (
            f"Found {hallucinated} hallucinated claim(s) out of {total} "
            f"({score_pct:.0f}% grounded). "
            f"{grounded} claim(s) are supported by the context."
        )
