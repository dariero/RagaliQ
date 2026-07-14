"""Faithfulness evaluator: is the response grounded only in the context?

Algorithm:
    1. Extract atomic claims from the response.
    2. Verify each claim against the context (SUPPORTED/CONTRADICTED/NOT_ENOUGH_INFO).
    3. Score = supported_claims / total_claims.
    4. Empty context or no claims = 0.0 (cannot assess faithfulness).
"""

from typing import TYPE_CHECKING

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.evaluators._claims import verify_all_claims
from ragaliq.evaluators.registry import register_evaluator

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestCase
    from ragaliq.judges.base import LLMJudge


@register_evaluator("faithfulness")
class FaithfulnessEvaluator(Evaluator):
    """Measures whether every claim in the response is supported by the context.

    A faithful response adds no information beyond what the context provides. The
    score is the ratio of supported claims; see the module docstring for the
    claim-level algorithm.
    """

    name: str = "faithfulness"
    description: str = "Measures if response is grounded only in context"

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge,
    ) -> EvaluationResult:
        """Score faithfulness as the fraction of response claims supported by context."""
        verification = await verify_all_claims(test_case.response, test_case.context, judge)

        if verification.context_empty:
            return self._empty_result(
                "No context provided; faithfulness cannot be assessed.", tokens_used=0
            )
        if verification.claims_empty:
            return self._empty_result(
                "No claims could be extracted from the response; faithfulness cannot be assessed.",
                tokens_used=verification.total_tokens,
            )

        supported_count = sum(1 for d in verification.claim_details if d.verdict == "SUPPORTED")
        total_claims = len(verification.claim_details)
        score = supported_count / total_claims

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=self.is_passing(score),
            reasoning=self._build_reasoning(supported_count, total_claims),
            raw_response={
                "claims": [d.model_dump() for d in verification.claim_details],
                "total_claims": total_claims,
                "supported_claims": supported_count,
            },
            tokens_used=verification.total_tokens,
        )

    def _empty_result(self, reasoning: str, tokens_used: int) -> EvaluationResult:
        """Build a failing 0.0 result for cases where faithfulness can't be assessed."""
        # Intentionally per-evaluator, not shared: the Evaluator pattern keeps each
        # metric a self-contained class, and the raw_response shape is metric-specific.
        return EvaluationResult(
            evaluator_name=self.name,
            score=0.0,
            passed=False,
            reasoning=reasoning,
            raw_response={"claims": [], "total_claims": 0, "supported_claims": 0},
            tokens_used=tokens_used,
        )

    def _build_reasoning(self, supported: int, total: int) -> str:
        """Summarize the supported/total claim ratio in human-readable form."""
        if total == 0:
            return "No claims to verify."

        if supported == total:
            return f"All {total} claims are supported by the context."
        if supported == 0:
            return f"None of the {total} claims are supported by the context."

        unsupported = total - supported
        score_pct = (supported / total) * 100
        return (
            f"{supported} of {total} claims are supported ({score_pct:.0f}%). "
            f"{unsupported} claim(s) not grounded in context."
        )
