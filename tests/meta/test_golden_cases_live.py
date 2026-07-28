"""Tier-0 live smoke: the five golden cases through the real runner.

Every other paid test bypasses `RagaliQ`. `test_judge_agreement` calls
`judge.verify_claim` directly and `test_mutation_discrimination` instantiates
`FaithfulnessEvaluator` itself, so the runner path — evaluator registry,
concurrency semaphores, result aggregation, status derivation — has no live
coverage at all. That is the path an SDK bump actually threatens.

This test is cheap and end to end: it evaluates the five `GoldenCase` triples
through `RagaliQ` and asserts each faithfulness score lands inside its
human-labelled band.

Gated behind `meta` and `RAGALIQ_RUN_META=1`; fails (does not skip) if the key
is missing, per the `live_judge` fixture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ragaliq.core.runner import RagaliQ
from ragaliq.core.test_case import EvalStatus, RAGTestCase

if TYPE_CHECKING:
    from ragaliq.judges.base import LLMJudge
    from tests.meta.meta_metrics import GoldenCase

pytestmark = pytest.mark.meta


async def test_golden_cases_land_in_expected_bands(
    live_judge: LLMJudge, golden_cases: list[GoldenCase]
) -> None:
    """Every golden case scores inside its band when run through `RagaliQ`."""
    runner = RagaliQ(judge=live_judge)

    test_cases = [
        RAGTestCase(
            id=case.id,
            name=case.note or case.id,
            query=case.query,
            context=case.context,
            response=case.response,
        )
        for case in golden_cases
    ]

    # evaluate_batch_async, not the sync evaluate(): we are already inside an
    # event loop (asyncio_mode=auto), and the sync wrapper cannot nest.
    results = await runner.evaluate_batch_async(test_cases)

    assert len(results) == len(golden_cases)

    out_of_band: list[str] = []

    for case, result in zip(golden_cases, results, strict=True):
        # Status first. An evaluator that raises is recorded with a fabricated
        # score of 0.0 (runner.py:171-180) and status ERROR, which would slide
        # past a bare score check for the low-band cases.
        assert result.status is not EvalStatus.ERROR, (
            f"{case.id}: evaluator errored, score is fabricated. details={result.details}"
        )

        score = result.get_score("faithfulness")
        assert score is not None, f"{case.id}: no faithfulness score in {result.scores}"

        low, high = case.expected_band
        print(
            f"  {case.id:<28} score={score:.3f} band=[{low:.2f}, {high:.2f}] "
            f"status={result.status.value} tokens={result.judge_tokens_used}"
        )
        if not low <= score <= high:
            out_of_band.append(
                f"{case.id}: {score:.3f} outside [{low:.2f}, {high:.2f}] — "
                f"{result.details.get('faithfulness', {}).get('reasoning', '')}"
            )

    # Report every failure at once; one run costs real money, so a single
    # assertion per case would hide the rest behind the first failure.
    assert not out_of_band, "faithfulness scores outside their bands:\n" + "\n".join(out_of_band)
