"""Context recall evaluator: does the retrieved context contain the needed facts?

Uses `test_case.expected_facts` as ground truth, verifying each against the
context:
    Score = covered_facts / total_expected_facts.
Empty facts list = 1.0 (vacuously complete); a missing `expected_facts` raises.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.evaluators.registry import register_evaluator

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestCase
    from ragaliq.judges.base import LLMJudge


@register_evaluator("context_recall")
class ContextRecallEvaluator(Evaluator):
    """Measures whether the retrieved context covers all expected facts.

    A retrieval-quality metric: each ground-truth fact in `expected_facts` is
    verified against the context with the judge's `verify_claim()`.
    """

    name: str = "context_recall"
    description: str = "Measures if context covers all needed information"

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge,
    ) -> EvaluationResult:
        """Score the fraction of expected facts the context supports.

        Raises:
            ValueError: If `test_case.expected_facts` is None.
        """
        if test_case.expected_facts is None:
            raise ValueError(
                f"context_recall requires 'expected_facts' in test case '{test_case.id}'. "
                "Add a list of ground-truth facts that should be present in the context, e.g.:\n"
                "  RAGTestCase(..., expected_facts=['Paris is the capital of France'])\n"
                "If you don't have ground-truth facts, use 'context_precision' instead — "
                "it evaluates retrieval quality without requiring expected answers."
            )

        if not test_case.expected_facts:
            return EvaluationResult(
                evaluator_name=self.name,
                score=1.0,
                passed=True,
                reasoning="No expected facts to verify; context is vacuously complete.",
                raw_response={
                    "fact_coverage": [],
                    "total_facts": 0,
                    "covered_facts": 0,
                },
                tokens_used=0,
            )

        verification_tasks = [
            judge.verify_claim(fact, test_case.context) for fact in test_case.expected_facts
        ]
        verdicts = await asyncio.gather(*verification_tasks)

        fact_coverage: list[dict[str, Any]] = []
        covered_count = 0
        total_tokens = 0
        for i, verdict in enumerate(verdicts):
            total_tokens += verdict.tokens_used
            fact_coverage.append(
                {
                    "fact": test_case.expected_facts[i],
                    "verdict": verdict.verdict,
                    "evidence": verdict.evidence,
                }
            )
            if verdict.verdict == "SUPPORTED":
                covered_count += 1

        total_facts = len(test_case.expected_facts)
        score = covered_count / total_facts

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=self.is_passing(score),
            reasoning=self._build_reasoning(covered_count, total_facts),
            raw_response={
                "fact_coverage": fact_coverage,
                "total_facts": total_facts,
                "covered_facts": covered_count,
            },
            tokens_used=total_tokens,
        )

    def _build_reasoning(self, covered: int, total: int) -> str:
        """Summarize how many expected facts the context covered."""
        if total == 0:
            return "No expected facts to verify."

        if covered == total:
            return f"All {total} expected facts are covered by the context."
        if covered == 0:
            return f"None of the {total} expected facts are covered by the context."

        missing = total - covered
        coverage_pct = (covered / total) * 100
        return (
            f"{covered} of {total} expected facts are covered ({coverage_pct:.0f}%). "
            f"{missing} fact(s) missing from context."
        )
