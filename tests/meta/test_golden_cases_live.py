"""Tier-0 live smoke: the five golden cases through the real runner.

Every other paid test bypasses `RagaliQ`. `test_judge_agreement` calls
`judge.verify_claim` directly and `test_mutation_discrimination` instantiates
`FaithfulnessEvaluator` itself, so the runner path — evaluator registry,
concurrency semaphores, result aggregation, status derivation — has no live
coverage at all. That is the path an SDK bump actually threatens.

The gate is `expected_class`, not `expected_band`. A band assumes a claim
count: issue #102 saw K=3 where the fixture assumed 2, putting a correct
judgement at 0.333 and outside [0.40, 0.70]. The class assertions hold for
every K, because 0/K and K/K are exact for all K. The band is still recorded
in the snapshot as a drift diagnostic.

Gated behind `meta` and `RAGALIQ_RUN_META=1`; fails (does not skip) if the key
is missing, per the `live_judge` fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ragaliq.core.runner import RagaliQ
from ragaliq.core.test_case import EvalStatus, RAGTestCase

if TYPE_CHECKING:
    from ragaliq.judges.base import LLMJudge
    from tests.meta.meta_metrics import GoldenCase

pytestmark = pytest.mark.meta

_SNAPSHOT = Path(__file__).parent / "baselines" / "golden_cases_latest.json"


def _claim_stats(details: dict[str, Any]) -> tuple[int | None, int | None]:
    """Pull (K, supported) out of the faithfulness evaluator's raw payload.

    `FaithfulnessEvaluator` records `total_claims` and `supported_claims` in
    `raw_response`, which the runner surfaces under details[...]["raw"]. Read
    the structured values — the reasoning string carries the same numbers in
    prose, but parsing prose to recover a number the evaluator already
    published is a defect waiting to happen.
    """
    raw = details.get("faithfulness", {}).get("raw") or {}
    return raw.get("total_claims"), raw.get("supported_claims")


async def test_golden_cases_match_expected_class(
    live_judge: LLMJudge, golden_cases: list[GoldenCase]
) -> None:
    """Every golden case lands in its expected faithfulness class."""
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

    failures: list[str] = []
    snapshot: list[dict[str, Any]] = []

    for case, result in zip(golden_cases, results, strict=True):
        # Status first. An evaluator that raises is recorded with a fabricated
        # score of 0.0 and status ERROR, which would slide past a
        # FULLY_HALLUCINATED assertion as a false pass.
        assert result.status is not EvalStatus.ERROR, (
            f"{case.id}: evaluator errored, score is fabricated. details={result.details}"
        )

        score = result.get_score("faithfulness")
        assert score is not None, f"{case.id}: no faithfulness score in {result.scores}"

        k, supported = _claim_stats(result.details)
        low, high = case.expected_band

        snapshot.append(
            {
                "id": case.id,
                "expected_class": case.expected_class,
                "score": round(score, 4),
                "k": k,
                "supported": supported,
                "recorded_band": [low, high],
                "in_recorded_band": low <= score <= high,
            }
        )

        print(
            f"  {case.id:<28} score={score:.3f} K={k} supported={supported} "
            f"class={case.expected_class} band=[{low:.2f}, {high:.2f}]"
        )

        # Exact equality at the extremes: 0/K and K/K are exact for every K, so
        # a tolerance here would only mask a verifier defect.
        if case.expected_class == "FULLY_FAITHFUL":
            ok = score == 1.0
        elif case.expected_class == "FULLY_HALLUCINATED":
            ok = score == 0.0
        else:
            ok = 0.0 < score < 1.0

        if not ok:
            failures.append(
                f"{case.id}: expected_class={case.expected_class} but score={score:.3f} "
                f"(K={k}, supported={supported}, recorded band=[{low:.2f}, {high:.2f}]) — "
                f"{result.details.get('faithfulness', {}).get('reasoning', '')}"
            )

    _SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"  snapshot -> {_SNAPSHOT}")

    # Report every failure at once; one run costs real money, so a per-case
    # assert would hide the rest behind the first.
    assert not failures, "golden cases outside their expected class:\n" + "\n".join(failures)
