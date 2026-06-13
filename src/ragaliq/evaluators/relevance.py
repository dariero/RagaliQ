"""Relevance evaluator: does the response address the user's query?

A thin adapter over the judge's `evaluate_relevance()`, mapping its `JudgeResult`
onto the `Evaluator` interface with threshold-based pass/fail.
"""

from typing import TYPE_CHECKING

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.evaluators.registry import register_evaluator

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestCase
    from ragaliq.judges.base import LLMJudge


@register_evaluator("relevance")
class RelevanceEvaluator(Evaluator):
    """Measures whether the response stays on topic and answers the query.

    Delegates scoring entirely to the judge's `evaluate_relevance()`.
    """

    name: str = "relevance"
    description: str = "Measures if response answers the query"

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge,
    ) -> EvaluationResult:
        """Score query relevance via the judge and adapt it to an EvaluationResult."""
        judge_result = await judge.evaluate_relevance(
            query=test_case.query,
            response=test_case.response,
        )

        return EvaluationResult(
            evaluator_name=self.name,
            score=judge_result.score,
            passed=self.is_passing(judge_result.score),
            reasoning=judge_result.reasoning,
            raw_response={
                "score": judge_result.score,
                "reasoning": judge_result.reasoning,
                "tokens_used": judge_result.tokens_used,
            },
            tokens_used=judge_result.tokens_used,
        )
