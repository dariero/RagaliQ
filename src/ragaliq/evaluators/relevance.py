"""
Relevance evaluator for RagaliQ.

This module implements relevance evaluation, measuring whether an LLM response
addresses the user's query. It delegates scoring entirely to the judge's
evaluate_relevance() method and adapts the result to the Evaluator interface.

Algorithm:
    1. Call judge.evaluate_relevance(query, response)
    2. Pass through score directly (already 0-1)
    3. Include judge's reasoning and metadata in result
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragaliq.core.evaluator import EvaluationResult, Evaluator

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestCase
    from ragaliq.judges.base import LLMJudge


class RelevanceEvaluator(Evaluator):
    """
    Evaluator that measures response relevance to the query.

    Relevance means the response addresses what the user actually asked.
    A relevant response stays on topic and provides information that
    answers or directly relates to the query.

    This evaluator delegates entirely to the judge's evaluate_relevance()
    method, adapting the JudgeResult into an EvaluationResult with
    threshold-based pass/fail semantics.

    Attributes:
        name: "relevance" - unique identifier for this evaluator.
        description: Human-readable description of what is evaluated.
        threshold: Minimum score to pass (default 0.7).

    Example:
        evaluator = RelevanceEvaluator(threshold=0.8)
        result = await evaluator.evaluate(test_case, judge)

        if result.passed:
            print(f"Response is relevant: {result.score:.2f}")
        else:
            print(f"Low relevance ({result.score:.2f}): {result.reasoning}")
    """

    name: str = "relevance"
    description: str = "Measures if response answers the query"

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge,
    ) -> EvaluationResult:
        """
        Evaluate relevance of the response to the query.

        Delegates to judge.evaluate_relevance() and adapts the result.

        Args:
            test_case: The RAG test case containing query and response.
            judge: The LLM judge instance for relevance scoring.

        Returns:
            EvaluationResult with:
                - score: Relevance score from judge (0.0 to 1.0)
                - passed: Whether score meets threshold
                - reasoning: Judge's explanation of the score
                - raw_response: Full judge metadata for debugging
        """
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
