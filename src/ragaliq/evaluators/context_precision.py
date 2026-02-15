"""
Context precision evaluator for RagaliQ.

This module implements context precision evaluation, measuring whether the
retrieved documents are relevant to the user's query. It tests retrieval
quality rather than response quality.

Algorithm:
    1. Score each context document's relevance to the query via judge
    2. Apply weighted precision: higher-ranked documents weighted more
    3. Score = sum(relevance_i / rank_i) / sum(1 / rank_i)
    4. Empty context = 1.0 (vacuously precise — no irrelevant docs retrieved)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.evaluators.registry import register_evaluator

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestCase
    from ragaliq.judges.base import LLMJudge


@register_evaluator("context_precision")
class ContextPrecisionEvaluator(Evaluator):
    """
    Evaluator that measures the precision of retrieved context documents.

    Context precision assesses whether the documents retrieved by the RAG
    system are actually relevant to the user's query. Higher-ranked documents
    are weighted more heavily, reflecting the importance of retrieval ordering.

    This evaluator tests retrieval quality, not response quality. It uses
    the judge's evaluate_relevance method on each document individually,
    treating each document as the "response" to assess its relevance to
    the query.

    The weighted precision formula:
        Score = sum(relevance_i / rank_i) / sum(1 / rank_i)

    where rank_i is the 1-based position of document i, and relevance_i
    is the judge's relevance score for that document.

    Attributes:
        name: "context_precision" - unique identifier for this evaluator.
        description: Human-readable description of what is evaluated.
        threshold: Minimum score to pass (default 0.7).

    Example:
        evaluator = ContextPrecisionEvaluator(threshold=0.8)
        result = await evaluator.evaluate(test_case, judge)

        if result.passed:
            print(f"Context is precise: {result.score:.2f}")
        else:
            # Inspect per-document scores
            for doc in result.raw_response["doc_scores"]:
                print(f"Doc {doc['rank']}: {doc['score']:.2f}")
    """

    name: str = "context_precision"
    description: str = "Measures if retrieved documents are relevant to the query"

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge,
    ) -> EvaluationResult:
        """
        Evaluate the precision of retrieved context documents.

        Implements the weighted precision algorithm:
        1. Score each document's relevance via judge.evaluate_relevance()
        2. Apply rank-based weighting (1/rank)
        3. Compute weighted average

        Args:
            test_case: The RAG test case containing query and context.
            judge: The LLM judge instance for relevance scoring.

        Returns:
            EvaluationResult with:
                - score: Weighted precision score (0.0 to 1.0)
                - passed: Whether score meets threshold
                - reasoning: Human-readable explanation
                - raw_response: Per-document scores and metadata
        """
        # Handle empty context: no docs means vacuously precise
        if not test_case.context:
            return EvaluationResult(
                evaluator_name=self.name,
                score=1.0,
                passed=True,
                reasoning="No context documents to evaluate; vacuously precise.",
                raw_response={
                    "doc_scores": [],
                    "total_docs": 0,
                    "weighted_precision": 1.0,
                },
                tokens_used=0,
            )

        # Step 1: Score each document's relevance to the query (in parallel)
        import asyncio

        # Score all documents concurrently (errors propagate)
        scoring_tasks = [
            judge.evaluate_relevance(query=test_case.query, response=doc)
            for doc in test_case.context
        ]
        results = await asyncio.gather(*scoring_tasks)

        # Process results
        doc_scores: list[dict[str, Any]] = []
        total_tokens = 0

        for i, result in enumerate(results):
            total_tokens += result.tokens_used

            rank = i + 1  # 1-based rank
            doc_scores.append(
                {
                    "rank": rank,
                    "document": test_case.context[i][:200],  # Truncate for metadata readability
                    "score": result.score,
                    "reasoning": result.reasoning,
                }
            )

        # Step 2: Calculate weighted precision
        # Score = sum(relevance_i / rank_i) / sum(1 / rank_i)
        weighted_sum = sum(d["score"] / d["rank"] for d in doc_scores)
        weight_total = sum(1.0 / d["rank"] for d in doc_scores)
        score = weighted_sum / weight_total

        # Step 3: Build reasoning
        reasoning = self._build_reasoning(doc_scores, score)

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=self.is_passing(score),
            reasoning=reasoning,
            raw_response={
                "doc_scores": doc_scores,
                "total_docs": len(doc_scores),
                "weighted_precision": score,
            },
            tokens_used=total_tokens,
        )

    def _build_reasoning(self, doc_scores: list[dict[str, Any]], score: float) -> str:
        """
        Build human-readable reasoning for the context precision score.

        Args:
            doc_scores: Per-document score details.
            score: The final weighted precision score.

        Returns:
            Reasoning string explaining the context precision result.
        """
        total = len(doc_scores)
        if total == 0:
            return "No context documents to evaluate."

        score_pct = score * 100
        high_relevance = sum(1 for d in doc_scores if d["score"] >= 0.7)

        if high_relevance == total:
            return (
                f"All {total} retrieved documents are relevant to the query "
                f"({score_pct:.0f}% weighted precision)."
            )
        elif high_relevance == 0:
            return (
                f"None of the {total} retrieved documents are relevant to the query "
                f"({score_pct:.0f}% weighted precision)."
            )
        else:
            return (
                f"{high_relevance} of {total} retrieved documents are relevant "
                f"({score_pct:.0f}% weighted precision). "
                f"Higher-ranked documents are weighted more heavily."
            )
