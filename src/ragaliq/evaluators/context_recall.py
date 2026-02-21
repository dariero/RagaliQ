"""
Context recall evaluator for RagaliQ.

This module implements context recall evaluation, measuring whether the
retrieved context contains all the necessary information to answer the query.
It uses the expected facts as ground truth.

Algorithm:
    1. Requires test_case.expected_facts (raises ValueError if missing)
    2. Verify each fact against the context via judge.verify_claim()
    3. Score = covered_facts / total_expected_facts
    4. Empty facts list = 1.0 (vacuously complete - no facts required)
"""

from typing import TYPE_CHECKING, Any

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.evaluators.registry import register_evaluator

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestCase
    from ragaliq.judges.base import LLMJudge


@register_evaluator("context_recall")
class ContextRecallEvaluator(Evaluator):
    """
    Evaluator that measures whether context covers all needed information.

    Context recall assesses whether the retrieved context contains all the
    necessary facts to answer the user's query. This is a retrieval quality
    metric that tests if the retrieval system found all relevant information,
    not whether the response used it.

    This evaluator requires expected_facts in the test case - a list of
    ground-truth facts that should be present in the retrieved context.
    Each fact is verified against the context using the judge's verify_claim
    method.

    Score calculation:
        Score = number_of_covered_facts / total_expected_facts

    Attributes:
        name: "context_recall" - unique identifier for this evaluator.
        description: Human-readable description of what is evaluated.
        threshold: Minimum score to pass (default 0.7).

    Example:
        test_case = RAGTestCase(
            query="What is the capital of France?",
            context=["Paris is the capital of France."],
            response="The capital of France is Paris.",
            expected_facts=["Paris is the capital of France"]
        )

        evaluator = ContextRecallEvaluator(threshold=0.8)
        result = await evaluator.evaluate(test_case, judge)

        if result.passed:
            print(f"Context has good recall: {result.score:.2f}")
        else:
            # Inspect which facts were missing
            for fact in result.raw_response["fact_coverage"]:
                if fact["verdict"] != "SUPPORTED":
                    print(f"Missing: {fact['fact']}")
    """

    name: str = "context_recall"
    description: str = "Measures if context covers all needed information"

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge,
    ) -> EvaluationResult:
        """
        Evaluate whether context covers all expected facts.

        Implements the context recall algorithm:
        1. Validate that expected_facts is present (raise ValueError if not)
        2. Verify each fact against context via judge.verify_claim()
        3. Score = supported_facts / total_facts
        4. Empty facts = 1.0 (vacuously complete)

        Args:
            test_case: The RAG test case containing context and expected_facts.
            judge: The LLM judge instance for fact verification.

        Returns:
            EvaluationResult with:
                - score: Ratio of covered facts (0.0 to 1.0)
                - passed: Whether score meets threshold
                - reasoning: Human-readable explanation
                - raw_response: Per-fact coverage details

        Raises:
            ValueError: If test_case.expected_facts is None.
        """
        # Step 0: Validate required field
        if test_case.expected_facts is None:
            raise ValueError(
                f"context_recall requires 'expected_facts' in test case '{test_case.id}'. "
                "Add a list of ground-truth facts that should be present in the context, e.g.:\n"
                "  RAGTestCase(..., expected_facts=['Paris is the capital of France'])\n"
                "If you don't have ground-truth facts, use 'context_precision' instead — "
                "it evaluates retrieval quality without requiring expected answers."
            )

        # Handle empty facts list: vacuously complete
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

        # Step 1: Verify each fact against the context (in parallel)
        import asyncio

        # Verify all facts concurrently (errors propagate)
        verification_tasks = [
            judge.verify_claim(fact, test_case.context) for fact in test_case.expected_facts
        ]
        verdicts = await asyncio.gather(*verification_tasks)

        # Process results
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

        # Step 2: Calculate score as ratio
        total_facts = len(test_case.expected_facts)
        score = covered_count / total_facts

        # Step 3: Build reasoning
        reasoning = self._build_reasoning(covered_count, total_facts)

        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=self.is_passing(score),
            reasoning=reasoning,
            raw_response={
                "fact_coverage": fact_coverage,
                "total_facts": total_facts,
                "covered_facts": covered_count,
            },
            tokens_used=total_tokens,
        )

    def _build_reasoning(self, covered: int, total: int) -> str:
        """
        Build human-readable reasoning for the context recall score.

        Args:
            covered: Number of facts covered by context.
            total: Total number of expected facts.

        Returns:
            Reasoning string explaining the score calculation.
        """
        if total == 0:
            return "No expected facts to verify."

        missing = total - covered
        coverage_pct = (covered / total) * 100

        if covered == total:
            return f"All {total} expected facts are covered by the context."
        elif covered == 0:
            return f"None of the {total} expected facts are covered by the context."
        else:
            return (
                f"{covered} of {total} expected facts are covered ({coverage_pct:.0f}%). "
                f"{missing} fact(s) missing from context."
            )
