"""CI-safe tests for the meta-evaluation math and harness.

No network. These prove the *meta-eval itself* is correct: that the confusion
matrix, F1, Cohen's kappa, and the agreement harness compute what we claim. A
StubJudge with hard-coded verdicts lets us assert exact numbers.

This mirrors RagaliQ's own philosophy: test the tester before trusting it.
"""

from __future__ import annotations

import math

import pytest

from ragaliq.judges.base import (
    ClaimsResult,
    ClaimVerdict,
    GeneratedAnswerResult,
    GeneratedQuestionsResult,
    JudgeResult,
    LLMJudge,
)
from tests.meta.meta_metrics import (
    accuracy,
    band_mae,
    build_confusion,
    class_metrics,
    cohen_kappa,
    in_band,
    load_golden_cases,
    load_golden_claims,
    macro_f1,
    run_claim_agreement,
)

# ---------------------------------------------------------------------------
# StubJudge: returns canned verdicts keyed by claim text. Network-free.
# ---------------------------------------------------------------------------


class StubJudge(LLMJudge):
    """LLMJudge whose verify_claim returns a verdict from a lookup table.

    Only verify_claim is exercised by the claim-agreement harness; the other
    abstract methods are implemented as inert stubs to satisfy the interface.
    """

    def __init__(self, verdict_by_claim: dict[str, str]) -> None:
        super().__init__()
        self._verdicts = verdict_by_claim

    async def verify_claim(self, claim: str, context: list[str]) -> ClaimVerdict:
        del context
        verdict = self._verdicts.get(claim, "NOT_ENOUGH_INFO")
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


# ---------------------------------------------------------------------------
# The harness itself, driven by a deterministic StubJudge
# ---------------------------------------------------------------------------


async def test_harness_perfect_judge() -> None:
    golden = load_golden_claims()
    perfect = StubJudge({c.claim: c.expected for c in golden})
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
    lookup = {c.claim: c.expected for c in golden}
    lookup[target.claim] = wrong

    report = await run_claim_agreement(StubJudge(lookup), golden)
    assert len(report.disagreements) == 1
    d = report.disagreements[0]
    assert d["id"] == target.id
    assert d["expected"] == target.expected
    assert d["predicted"] == wrong
    assert report.accuracy == pytest.approx((len(golden) - 1) / len(golden))


async def test_harness_snapshot_is_serialisable() -> None:
    golden = load_golden_claims()
    report = await run_claim_agreement(StubJudge({c.claim: c.expected for c in golden}), golden)
    snap = report.to_snapshot()
    assert snap["accuracy"] == 1.0
    assert set(snap["per_class"]) == {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}
    assert not math.isnan(snap["cohen_kappa"])
