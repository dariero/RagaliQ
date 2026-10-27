"""CI-safe tests for the meta-evaluation math and harness.

No network. These prove the *meta-eval itself* is correct: that the confusion
matrix, F1, Cohen's kappa, and the agreement harness compute what we claim. A
StubJudge with hard-coded verdicts lets us assert exact numbers.

This mirrors RagaliQ's own philosophy: test the tester before trusting it.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from ragaliq.judges.base import (
    ClaimsResult,
    ClaimVerdict,
    GeneratedAnswerResult,
    GeneratedQuestionsResult,
    JudgeResult,
    LLMJudge,
)
from ragaliq.judges.models import DEFAULT_JUDGE_MODEL
from tests.meta import meta_metrics
from tests.meta.meta_metrics import (
    FAITHFULNESS_CLASSES,
    INCOMPARABLE_INPUTS,
    AgreementSnapshot,
    BaselineFingerprint,
    GoldenClaim,
    accuracy,
    band_mae,
    build_confusion,
    claim_count_delta,
    class_metrics,
    cohen_kappa,
    current_fingerprint,
    fingerprint_drift,
    in_band,
    load_baseline,
    load_golden_cases,
    load_golden_claims,
    macro_f1,
    resolve_meta_model,
    run_claim_agreement,
    write_snapshot,
)

# ---------------------------------------------------------------------------
# StubJudge: returns canned verdicts keyed by claim and context. Network-free.
# ---------------------------------------------------------------------------


class StubJudge(LLMJudge):
    """LLMJudge whose verify_claim returns a verdict from a lookup table.

    Only verify_claim is exercised by the claim-agreement harness; the other
    abstract methods are implemented as inert stubs to satisfy the interface.
    """

    def __init__(self, verdict_by_item: dict[tuple[str, tuple[str, ...]], str]) -> None:
        super().__init__()
        self._verdicts = verdict_by_item

    async def verify_claim(self, claim: str, context: list[str]) -> ClaimVerdict:
        # Keyed on claim and context together: contrast groups reuse one claim with several contexts.
        verdict = self._verdicts.get((claim, tuple(context)), "NOT_ENOUGH_INFO")
        return ClaimVerdict(verdict=verdict, evidence="stub", tokens_used=0)

    async def evaluate_faithfulness(self, response: str, context: list[str]) -> JudgeResult:
        del response, context
        return JudgeResult(score=0.0, reasoning="stub", tokens_used=0)

    async def evaluate_relevance(self, query: str, response: str) -> JudgeResult:
        del query, response
        return JudgeResult(score=0.0, reasoning="stub", tokens_used=0)

    async def extract_claims(self, response: str) -> ClaimsResult:
        del response
        return ClaimsResult(claims=[], tokens_used=0)

    async def generate_questions(self, documents: list[str], n: int) -> GeneratedQuestionsResult:
        del documents, n
        return GeneratedQuestionsResult(questions=[], tokens_used=0)

    async def generate_answer(self, question: str, context: list[str]) -> GeneratedAnswerResult:
        del question, context
        return GeneratedAnswerResult(answer="", tokens_used=0)


def verdicts_of(golden: list[GoldenClaim]) -> dict[tuple[str, tuple[str, ...]], str]:
    return {(c.claim, tuple(c.context)): c.expected for c in golden}


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def test_confusion_perfect_agreement() -> None:
    expected = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
    matrix = build_confusion(expected, expected)
    assert matrix["SUPPORTED"]["SUPPORTED"] == 1
    assert matrix["CONTRADICTED"]["CONTRADICTED"] == 1
    assert matrix["NOT_ENOUGH_INFO"]["NOT_ENOUGH_INFO"] == 1


def test_confusion_off_schema_label_folds_to_other() -> None:
    matrix = build_confusion(["SUPPORTED"], ["GARBAGE"])
    assert matrix["SUPPORTED"]["OTHER"] == 1
    assert matrix["SUPPORTED"]["SUPPORTED"] == 0


def test_confusion_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        build_confusion(["SUPPORTED"], [])


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def test_accuracy_half() -> None:
    expected = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "CONTRADICTED"]
    predicted = ["SUPPORTED", "CONTRADICTED", "CONTRADICTED", "SUPPORTED"]
    assert accuracy(expected, predicted) == 0.5


def test_accuracy_empty() -> None:
    assert accuracy([], []) == 0.0


# ---------------------------------------------------------------------------
# Precision / recall / F1
# ---------------------------------------------------------------------------


def test_class_metrics_hand_computed() -> None:
    # SUPPORTED: 2 true. Judge predicts SUPPORTED 3x: 2 correct, 1 false positive.
    expected = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
    predicted = ["SUPPORTED", "SUPPORTED", "SUPPORTED", "NOT_ENOUGH_INFO"]
    matrix = build_confusion(expected, predicted)
    m = class_metrics(matrix, "SUPPORTED")
    # TP=2, FP=1 (the CONTRADICTED item predicted SUPPORTED), FN=0
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0))
    assert m.support == 2


def test_macro_f1_perfect() -> None:
    expected = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
    matrix = build_confusion(expected, expected)
    assert macro_f1(matrix) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def test_kappa_perfect() -> None:
    labels = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO", "SUPPORTED"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_kappa_chance_level_is_near_zero() -> None:
    # Both raters use the same marginal split but agree only at chance.
    expected = ["SUPPORTED", "CONTRADICTED"] * 10
    predicted = (["SUPPORTED", "CONTRADICTED"] * 5) + (["CONTRADICTED", "SUPPORTED"] * 5)
    k = cohen_kappa(expected, predicted)
    assert abs(k) < 0.25  # near chance


def test_kappa_hand_computed() -> None:
    # 4 items: 3 agree, 1 disagrees.
    expected = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
    predicted = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "SUPPORTED"]
    # p_o = 3/4 = 0.75
    # marginals expected: SUP=2, CON=1, NEI=1 ; predicted: SUP=3, CON=1, NEI=0
    # p_e = (2/4*3/4) + (1/4*1/4) + (1/4*0/4) = 0.375 + 0.0625 + 0 = 0.4375
    # kappa = (0.75 - 0.4375) / (1 - 0.4375) = 0.3125 / 0.5625
    assert cohen_kappa(expected, predicted) == pytest.approx(0.3125 / 0.5625)


def test_kappa_empty() -> None:
    assert cohen_kappa([], []) == 0.0


# ---------------------------------------------------------------------------
# Band checks
# ---------------------------------------------------------------------------


def test_in_band() -> None:
    assert in_band(0.95, (0.9, 1.0))
    assert in_band(0.9, (0.9, 1.0))  # inclusive
    assert not in_band(0.8, (0.9, 1.0))


def test_band_mae() -> None:
    # score 0.6 vs midpoint 0.5 -> err 0.1 ; score 1.0 vs midpoint 0.95 -> err 0.05
    mae = band_mae([0.6, 1.0], [(0.4, 0.6), (0.9, 1.0)])
    assert mae == pytest.approx((0.1 + 0.05) / 2)


# ---------------------------------------------------------------------------
# Golden set integrity (catches typos / bad labels at author time)
# ---------------------------------------------------------------------------


def test_golden_claims_are_well_formed() -> None:
    claims = load_golden_claims()
    assert len(claims) >= 10, "golden claim set too small to be meaningful"
    ids = [c.id for c in claims]
    assert len(ids) == len(set(ids)), "duplicate golden claim ids"
    valid = {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}
    for c in claims:
        assert c.expected in valid, f"{c.id} has invalid label {c.expected!r}"
        assert c.claim.strip(), f"{c.id} has empty claim"
        assert c.context and all(ctx.strip() for ctx in c.context), f"{c.id} bad context"


def test_golden_claims_cover_all_three_verdicts() -> None:
    claims = load_golden_claims()
    labels = {c.expected for c in claims}
    assert labels == {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}, (
        "golden set must exercise every verdict class or per-class F1 is undefined"
    )


def test_golden_cases_bands_are_valid() -> None:
    cases = load_golden_cases()
    assert len(cases) >= 3
    for case in cases:
        low, high = case.expected_band
        assert 0.0 <= low <= high <= 1.0, f"{case.id} has an invalid band"


def test_golden_cases_classes_are_valid() -> None:
    """Every case carries a known granularity-invariant class.

    `expected_class` is what the live tier-0 test gates on, so an unrecognised
    value must fail here — cheaply, offline — rather than during a paid run.
    """
    cases = load_golden_cases()
    for case in cases:
        assert case.expected_class in FAITHFULNESS_CLASSES, (
            f"{case.id} has unknown expected_class {case.expected_class!r}; "
            f"expected one of {sorted(FAITHFULNESS_CLASSES)}"
        )


def test_golden_cases_bands_agree_with_classes() -> None:
    """The recorded band must not contradict the class it accompanies.

    The band is a diagnostic, not a gate, but a band that disagrees with its
    class means one of the two is wrong and the fixture is lying either way.
    """
    for case in load_golden_cases():
        low, high = case.expected_band
        if case.expected_class == "FULLY_FAITHFUL":
            assert high == 1.0, f"{case.id}: FULLY_FAITHFUL band must reach 1.0"
        elif case.expected_class == "FULLY_HALLUCINATED":
            assert low == 0.0, f"{case.id}: FULLY_HALLUCINATED band must reach 0.0"
        else:
            assert low > 0.0 and high < 1.0, (
                f"{case.id}: PARTIALLY_FAITHFUL band must exclude both extremes"
            )


def test_golden_cases_record_a_claim_count() -> None:
    """Every case records the K it was observed with, so a live run can report drift (#108)."""
    for case in load_golden_cases():
        assert type(case.recorded_k) is int and case.recorded_k >= 1, (
            f"{case.id}: recorded_k must be a positive int, got {case.recorded_k!r}"
        )


def test_golden_case_without_recorded_k_does_not_load(tmp_path: Path) -> None:
    """A missing recorded_k fails the load; a default would switch the K report off unseen."""
    path = tmp_path / "cases.yaml"
    path.write_text(
        "cases:\n"
        "  - id: no_k\n"
        "    query: q\n"
        "    context: [c]\n"
        "    response: r\n"
        "    expected_band: [0.9, 1.0]\n"
        "    expected_class: FULLY_FAITHFUL\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="recorded_k"):
        load_golden_cases(path)


@pytest.mark.parametrize(
    ("k", "recorded_k", "delta"),
    [(3, 3, 0), (4, 3, 1), (2, 3, -1), (None, 3, None)],
)
def test_claim_count_delta(k: int | None, recorded_k: int, delta: int | None) -> None:
    assert claim_count_delta(k, recorded_k) == delta


# ---------------------------------------------------------------------------
# Live judge tier (RAGALIQ_META_MODEL)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["claude-sonnet-5-5", "claude-opus-5-5"])
def test_resolve_meta_model_accepts_each_tier(value: str) -> None:
    assert resolve_meta_model(value) == value


def test_resolve_meta_model_unset_keeps_the_default() -> None:
    assert resolve_meta_model(None) is None


@pytest.mark.parametrize("value", ["claude-opus-5.5", ""], ids=["typo", "empty-string"])
def test_resolve_meta_model_rejects_any_other_value(value: str) -> None:
    with pytest.raises(ValueError, match=re.escape(repr(value))):
        resolve_meta_model(value)


# ---------------------------------------------------------------------------
# The harness itself, driven by a deterministic StubJudge
# ---------------------------------------------------------------------------


async def test_harness_perfect_judge() -> None:
    golden = load_golden_claims()
    perfect = StubJudge(verdicts_of(golden))
    report = await run_claim_agreement(perfect, golden)
    assert report.n == len(golden)
    assert report.accuracy == pytest.approx(1.0)
    assert report.cohen_kappa == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.disagreements == []


async def test_harness_records_disagreements() -> None:
    golden = load_golden_claims()
    # Flip exactly one label so the judge disagrees on one item.
    target = golden[0]
    wrong = "CONTRADICTED" if target.expected != "CONTRADICTED" else "SUPPORTED"
    lookup = verdicts_of(golden)
    lookup[(target.claim, tuple(target.context))] = wrong

    report = await run_claim_agreement(StubJudge(lookup), golden)
    assert len(report.disagreements) == 1
    d = report.disagreements[0]
    assert d["id"] == target.id
    assert d["expected"] == target.expected
    assert d["predicted"] == wrong
    assert report.accuracy == pytest.approx((len(golden) - 1) / len(golden))


async def test_harness_keeps_items_that_share_a_claim_apart() -> None:
    # A contrast group can reuse one claim with different contexts and labels.
    claim = "The press is based in Ghent."
    golden = [
        GoldenClaim(
            id="ctx_ghent", claim=claim, context=["The press is in Ghent."], expected="SUPPORTED"
        ),
        GoldenClaim(
            id="ctx_bruges",
            claim=claim,
            context=["The press is only in Bruges."],
            expected="CONTRADICTED",
        ),
    ]
    report = await run_claim_agreement(StubJudge(verdicts_of(golden)), golden)
    assert report.accuracy == pytest.approx(1.0)
    assert report.disagreements == []


async def test_harness_snapshot_is_serialisable() -> None:
    golden = load_golden_claims()
    report = await run_claim_agreement(StubJudge(verdicts_of(golden)), golden)
    snap = report.to_snapshot()
    assert snap["accuracy"] == 1.0
    assert set(snap["per_class"]) == {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}
    assert not math.isnan(snap["cohen_kappa"])


# ---------------------------------------------------------------------------
# Baseline fingerprint
# ---------------------------------------------------------------------------


def _snapshot_with(fingerprint: BaselineFingerprint | None) -> AgreementSnapshot:
    snap: AgreementSnapshot = {
        "model": DEFAULT_JUDGE_MODEL,
        "n": 0,
        "accuracy": 1.0,
        "cohen_kappa": 1.0,
        "macro_f1": 1.0,
        "per_class": {},
    }
    if fingerprint is not None:
        snap["fingerprint"] = fingerprint
    return snap


_GOLDEN_YAML = """\
# header comment
items:
  - id: a
    source: v1
    claim: "The press is based in Ghent."
    context:
      - "The press is in Ghent."
    expected: SUPPORTED
    note: "Stated."
"""


def test_pinned_baseline_matches_current_inputs() -> None:
    # Free in CI: changing the default model, the golden set, the verify_claim
    # prompt or the SDK without a paid re-baseline and a re-pin fails here,
    # not in the weekly run.
    baseline = load_baseline()
    if baseline is None:
        pytest.skip("no baseline pinned")
    assert fingerprint_drift(baseline, DEFAULT_JUDGE_MODEL) == {}


def test_fingerprint_drift_is_empty_when_inputs_match() -> None:
    assert fingerprint_drift(_snapshot_with(current_fingerprint()), DEFAULT_JUDGE_MODEL) == {}


def test_fingerprint_drift_names_each_changed_input() -> None:
    pinned = current_fingerprint()
    pinned["golden_sha256"] = "0" * 64
    pinned["anthropic"] = "0.0.0"
    drift = fingerprint_drift(_snapshot_with(pinned), "claude-other")
    assert list(drift) == ["model", "golden_sha256", "anthropic"]
    assert set(drift) & INCOMPARABLE_INPUTS == {"golden_sha256"}


def test_fingerprint_drift_rejects_a_baseline_without_one() -> None:
    drift = fingerprint_drift(_snapshot_with(None), DEFAULT_JUDGE_MODEL)
    assert drift == {"fingerprint": "fingerprint: missing (the baseline was pinned before #90)"}
    assert set(drift) <= INCOMPARABLE_INPUTS


def test_fingerprint_ignores_what_the_judge_never_sees(tmp_path: Path) -> None:
    golden = tmp_path / "claim_verdicts.yaml"
    golden.write_text(_GOLDEN_YAML, encoding="utf-8")
    before = current_fingerprint(golden)["golden_sha256"]
    edited = _GOLDEN_YAML.replace("# header comment", "# another comment")
    edited = edited.replace("source: v1", "source: darie").replace('"Stated."', '"Reworded."')
    golden.write_bytes(edited.replace("\n", "\r\n").encode("utf-8"))
    assert current_fingerprint(golden)["golden_sha256"] == before


def test_fingerprint_changes_with_what_the_judge_sees(tmp_path: Path) -> None:
    golden = tmp_path / "claim_verdicts.yaml"
    golden.write_text(_GOLDEN_YAML, encoding="utf-8")
    before = current_fingerprint(golden)["golden_sha256"]
    golden.write_text(_GOLDEN_YAML.replace("is in Ghent", "is in Bruges"), encoding="utf-8")
    assert current_fingerprint(golden)["golden_sha256"] != before


async def test_write_snapshot_records_the_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(meta_metrics, "_BASELINE_DIR", tmp_path)
    golden = load_golden_claims()
    report = await run_claim_agreement(StubJudge(verdicts_of(golden)), golden)
    written = json.loads(write_snapshot(report).read_text(encoding="utf-8"))
    assert written["fingerprint"] == current_fingerprint()
