"""Live judge-agreement benchmark (the actual meta-evaluation).

Runs the real ClaudeJudge over the human-labelled golden set and measures how
often it agrees. This is the only test in the repo that answers 'is the judge
good?' rather than 'does the plumbing work?'.

Gated behind the `meta` marker and an explicit paid-evaluation opt-in:

    RAGALIQ_RUN_META=1 pytest -m meta

On each run it writes baselines/judge_agreement_latest.json. To pin a baseline
for drift detection, copy it once:

    cp tests/meta/baselines/judge_agreement_latest.json \
        tests/meta/baselines/judge_agreement_baseline.json

Thereafter the test fails if macro-F1 regresses past MACRO_F1_DRIFT_TOLERANCE —
your early-warning for a model-version change quietly degrading judge quality.
"""

from __future__ import annotations

import pytest

from ragaliq.judges.base import LLMJudge
from tests.meta.meta_metrics import (
    GoldenClaim,
    load_baseline,
    run_claim_agreement,
    write_snapshot,
)

# Absolute floors. Measured 2026-07-28 against claude-sonnet-4-6 over three
# consecutive runs (see issue #89): accuracy, kappa and macro-F1 were all
# exactly 1.000 every run, with zero disagreements across all 14 claims. The
# floors below are therefore cleared with full headroom and are kept
# deliberately loose — they catch a judge that breaks, not one that drifts.
MIN_ACCURACY = 0.70
MIN_KAPPA = 0.50  # 'moderate' agreement beyond chance
MIN_MACRO_F1 = 0.65

# How much macro-F1 may drop versus a pinned baseline before it's a regression.
# Derived, not chosen: observed_spread over the three runs was 0.000, so
# round(max(2 * 0.000, 0.05), 2) = 0.05.
#
# Caveat: with n=14 a single misclassification moves accuracy by 1/14 = 0.071,
# so this tolerance sits below the metric's own resolution. The fix is more
# golden claims, not a tighter constant — tracked in issue #90.
MACRO_F1_DRIFT_TOLERANCE = 0.05

pytestmark = pytest.mark.meta


async def test_judge_agreement_against_golden(
    live_judge: LLMJudge, golden_claims: list[GoldenClaim]
) -> None:
    report = await run_claim_agreement(live_judge, golden_claims)
    snapshot_path = write_snapshot(report)

    # Human-readable diagnostics on failure.
    print(f"\nJudge agreement ({report.model}, n={report.n}):")
    print(f"  accuracy   = {report.accuracy:.3f}")
    print(f"  cohen_kappa= {report.cohen_kappa:.3f}")
    print(f"  macro_f1   = {report.macro_f1:.3f}")
    for label, m in report.per_class.items():
        print(f"  {label:<16} P={m.precision:.2f} R={m.recall:.2f} F1={m.f1:.2f} n={m.support}")
    if report.disagreements:
        print("  disagreements:")
        for d in report.disagreements:
            print(f"    [{d['id']}] expected {d['expected']} got {d['predicted']}: {d['claim']}")
    print(f"  snapshot -> {snapshot_path}")

    # Absolute floors.
    assert report.accuracy >= MIN_ACCURACY, (
        f"judge accuracy {report.accuracy:.3f} below floor {MIN_ACCURACY}"
    )
    assert report.cohen_kappa >= MIN_KAPPA, (
        f"judge kappa {report.cohen_kappa:.3f} below floor {MIN_KAPPA}"
    )
    assert report.macro_f1 >= MIN_MACRO_F1, (
        f"judge macro-F1 {report.macro_f1:.3f} below floor {MIN_MACRO_F1}"
    )

    # Drift regression against a pinned baseline, if one exists.
    baseline = load_baseline()
    if baseline is not None:
        delta = baseline["macro_f1"] - report.macro_f1
        assert delta <= MACRO_F1_DRIFT_TOLERANCE, (
            f"macro-F1 regressed {delta:.3f} vs baseline "
            f"({baseline['macro_f1']:.3f} -> {report.macro_f1:.3f}, "
            f"model {baseline.get('model')} -> {report.model}); "
            f"exceeds tolerance {MACRO_F1_DRIFT_TOLERANCE}"
        )
