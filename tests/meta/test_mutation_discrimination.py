"""Live mutation (adversarial) discrimination test.

A faithfulness evaluator that returns high scores for everything is useless even
if it never crashes. This test takes a known-faithful answer, injects a
fabricated claim, and asserts the score actually DROPS. It verifies the
evaluator discriminates — the property unit tests with mocked judges cannot
prove, because the mock decides the verdict, not the model.

Gated behind `meta` and `RAGALIQ_RUN_META=1`; skipped without ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import pytest

from ragaliq.core.test_case import RAGTestCase
from ragaliq.evaluators.faithfulness import FaithfulnessEvaluator
from ragaliq.judges.base import LLMJudge

pytestmark = pytest.mark.meta

# A mutated answer must drop at least this far below the clean answer's score.
MIN_SCORE_DROP = 0.20


async def test_injected_hallucination_lowers_faithfulness(live_judge: LLMJudge) -> None:
    context = [
        "France is a country in Western Europe.",
        "The capital city of France is Paris.",
    ]
    evaluator = FaithfulnessEvaluator(threshold=0.7)

    clean = RAGTestCase(
        id="clean",
        name="Faithful answer",
        query="What is the capital of France?",
        context=context,
        response="The capital of France is Paris.",
    )
    mutated = RAGTestCase(
        id="mutated",
        name="Answer with injected fabrication",
        query="What is the capital of France?",
        context=context,
        response=(
            "The capital of France is Paris, which has a population of 80 billion "
            "people and was founded by Napoleon in 1620."
        ),
    )

    clean_result = await evaluator.evaluate(clean, live_judge)
    mutated_result = await evaluator.evaluate(mutated, live_judge)

    drop = clean_result.score - mutated_result.score
    print(
        f"\nMutation discrimination: clean={clean_result.score:.3f} "
        f"mutated={mutated_result.score:.3f} drop={drop:.3f}"
    )
    print(f"  clean reasoning:   {clean_result.reasoning}")
    print(f"  mutated reasoning: {mutated_result.reasoning}")

    assert clean_result.score >= 0.9, (
        f"clean answer should score high, got {clean_result.score:.3f}"
    )
    assert drop >= MIN_SCORE_DROP, (
        f"injected fabrications only dropped the score by {drop:.3f}; "
        f"evaluator is not discriminating (expected >= {MIN_SCORE_DROP})"
    )
    assert not mutated_result.passed, "mutated answer should fail the 0.7 threshold"
