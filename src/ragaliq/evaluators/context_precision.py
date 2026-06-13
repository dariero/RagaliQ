"""Context precision evaluator: are the retrieved documents relevant to the query?

Tests retrieval quality (not response quality) with rank-weighted precision:
    Score = sum(relevance_i / rank_i) / sum(1 / rank_i)
where rank_i is the 1-based position of document i. Empty context = 1.0
(vacuously precise).
"""

import asyncio
from typing import TYPE_CHECKING, Any

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.evaluators.registry import register_evaluator

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestCase
    from ragaliq.judges.base import LLMJudge

_DOC_PREVIEW_LENGTH = 200


@register_evaluator("context_precision")
class ContextPrecisionEvaluator(Evaluator):
    """Measures whether retrieved documents are relevant to the query.

    Scores each document with the judge's `evaluate_relevance()` (treating the
    document as the "response") and combines the scores with rank-based
    weighting, so irrelevant high-ranked documents are penalized more.
    """

    name: str = "context_precision"
    description: str = "Measures if retrieved documents are relevant to the query"

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge,
    ) -> EvaluationResult:
        """Score retrieved-document relevance as a rank-weighted precision average."""
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

        scoring_tasks = [
            judge.evaluate_relevance(query=test_case.query, response=doc)
            for doc in test_case.context
        ]
        results = await asyncio.gather(*scoring_tasks)

        doc_scores: list[dict[str, Any]] = []
        total_tokens = 0
        for i, result in enumerate(results):
            total_tokens += result.tokens_used
            doc_scores.append(
                {
                    "rank": i + 1,
                    "document": test_case.context[i][:_DOC_PREVIEW_LENGTH],
                    "score": result.score,
                    "reasoning": result.reasoning,
                }
            )

        # Rank-weighted precision: sum(score / rank) / sum(1 / rank).
        weighted_sum = sum(d["score"] / d["rank"] for d in doc_scores)
        weight_total = sum(1.0 / d["rank"] for d in doc_scores)
        score = weighted_sum / weight_total

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=self.is_passing(score),
            reasoning=self._build_reasoning(doc_scores, score),
            raw_response={
                "doc_scores": doc_scores,
                "total_docs": len(doc_scores),
                "weighted_precision": score,
            },
            tokens_used=total_tokens,
        )

    def _build_reasoning(self, doc_scores: list[dict[str, Any]], score: float) -> str:
        """Summarize how many retrieved documents were relevant to the query."""
        total = len(doc_scores)
        if total == 0:
            return "No context documents to evaluate."

        score_pct = score * 100
        high_relevance = sum(1 for d in doc_scores if d["score"] >= self.threshold)

        if high_relevance == total:
            return (
                f"All {total} retrieved documents are relevant to the query "
                f"({score_pct:.0f}% weighted precision)."
            )
        if high_relevance == 0:
            return (
                f"None of the {total} retrieved documents are relevant to the query "
                f"({score_pct:.0f}% weighted precision)."
            )
        return (
            f"{high_relevance} of {total} retrieved documents are relevant "
            f"({score_pct:.0f}% weighted precision). "
            f"Higher-ranked documents are weighted more heavily."
        )
