"""
Meta-evaluation metrics and harness for RagaliQ.

This module answers the question the rest of the test suite does not:
*is the judge itself any good?* Everything in tests/unit and tests/integration
assumes the judge's verdict is ground truth. Here we compare the live judge's
verdicts against a human-labelled golden set and quantify agreement.

The module is deliberately dependency-light (stdlib + pyyaml, both already
required) and split into two halves:

1. PURE FUNCTIONS — confusion matrix, precision/recall/F1, accuracy, Cohen's
   kappa, score-band checks. No I/O, no judge, fully unit-testable.
2. ASYNC HARNESS — run_claim_agreement(): loops a golden set through any
   LLMJudge's verify_claim() and returns an AgreementReport. Depends only on the
   LLMJudge protocol, so it works with the real ClaudeJudge or a stub.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import yaml

if TYPE_CHECKING:
    from ragaliq.judges.base import LLMJudge

VERDICTS: tuple[str, ...] = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")

_GOLDEN_DIR = Path(__file__).parent / "golden"
_BASELINE_DIR = Path(__file__).parent / "baselines"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenClaim:
    """A single human-labelled (claim, context) -> verdict triple."""

    id: str
    claim: str
    context: list[str]
    expected: str
    note: str = ""


@dataclass(frozen=True)
class GoldenCase:
    """A human-labelled full RAG case with an expected faithfulness score band."""

    id: str
    query: str
    context: list[str]
    response: str
    expected_band: tuple[float, float]
    note: str = ""


@dataclass(frozen=True)
class ClassMetrics:
    """Precision/recall/F1 for one verdict class."""

    precision: float
    recall: float
    f1: float
    support: int  # number of golden items whose true label is this class


class ClassMetricsSnapshot(TypedDict):
    """JSON-safe form of one verdict class's metrics."""

    precision: float
    recall: float
    f1: float
    support: int


class AgreementSnapshot(TypedDict):
    """JSON-safe subset of an agreement report used for drift checks."""

    model: str
    n: int
    accuracy: float
    cohen_kappa: float
    macro_f1: float
    per_class: dict[str, ClassMetricsSnapshot]


@dataclass
class AgreementReport:
    """Result of comparing judge verdicts against the golden labels."""

    model: str
    n: int
    accuracy: float
    cohen_kappa: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion: dict[str, dict[str, int]]  # confusion[true][pred] = count
    disagreements: list[dict[str, str]] = field(default_factory=list)

    def to_snapshot(self) -> AgreementSnapshot:
        """Flatten to a JSON-serialisable snapshot for drift regression."""
        return {
            "model": self.model,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "cohen_kappa": round(self.cohen_kappa, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_class": {
                label: {
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "support": metrics.support,
                }
                for label, metrics in self.per_class.items()
            },
        }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_golden_claims(path: Path | None = None) -> list[GoldenClaim]:
    """Load the human-labelled claim-verdict golden set."""
    p = path or (_GOLDEN_DIR / "claim_verdicts.yaml")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return [
        GoldenClaim(
            id=item["id"],
            claim=item["claim"],
            context=list(item["context"]),
            expected=item["expected"],
            note=item.get("note", ""),
        )
        for item in raw["items"]
    ]


def load_golden_cases(path: Path | None = None) -> list[GoldenCase]:
    """Load the case-level faithfulness golden set."""
    p = path or (_GOLDEN_DIR / "faithfulness_cases.yaml")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return [
        GoldenCase(
            id=item["id"],
            query=item["query"],
            context=list(item["context"]),
            response=item["response"],
            expected_band=(float(item["expected_band"][0]), float(item["expected_band"][1])),
            note=item.get("note", ""),
        )
        for item in raw["cases"]
    ]


# ---------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------


def build_confusion(expected: list[str], predicted: list[str]) -> dict[str, dict[str, int]]:
    """Build a confusion matrix as confusion[true_label][pred_label] = count.

    Unknown predicted labels (a judge returning something off-schema) are folded
    into a synthetic 'OTHER' column so they count as errors rather than crashing.
    """
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must be the same length")

    cols = (*VERDICTS, "OTHER")
    matrix = {true_label: dict.fromkeys(cols, 0) for true_label in VERDICTS}
    for true_label, pred_label in zip(expected, predicted, strict=True):
        if true_label not in matrix:
            raise ValueError(f"Unknown true label: {true_label!r}")
        col = pred_label if pred_label in VERDICTS else "OTHER"
        matrix[true_label][col] += 1
    return matrix


def accuracy(expected: list[str], predicted: list[str]) -> float:
    """Fraction of items where judge agrees with the human label."""
    if not expected:
        return 0.0
    correct = sum(1 for e, p in zip(expected, predicted, strict=True) if e == p)
    return correct / len(expected)


def class_metrics(confusion: dict[str, dict[str, int]], label: str) -> ClassMetrics:
    """Precision/recall/F1 for one class from the confusion matrix."""
    tp = confusion[label][label]
    # Predicted == label across all true rows, minus the true positives.
    predicted_positives = sum(confusion[t].get(label, 0) for t in confusion)
    fp = predicted_positives - tp
    # Actual == label (the row total), minus true positives.
    actual_positives = sum(confusion[label].values())
    fn = actual_positives - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return ClassMetrics(precision=precision, recall=recall, f1=f1, support=actual_positives)


def macro_f1(confusion: dict[str, dict[str, int]]) -> float:
    """Unweighted mean F1 across the three verdict classes."""
    f1s = [class_metrics(confusion, label).f1 for label in VERDICTS]
    return sum(f1s) / len(f1s) if f1s else 0.0


def cohen_kappa(expected: list[str], predicted: list[str]) -> float:
    """Cohen's kappa: agreement corrected for chance.

    kappa = (p_o - p_e) / (1 - p_e)
      p_o = observed agreement (= accuracy)
      p_e = agreement expected by chance from the marginal label distributions

    Range: 1.0 perfect, 0.0 chance-level, negative worse than chance. For a
    three-way judge, >0.6 is 'substantial', >0.8 is 'near-perfect'.
    """
    n = len(expected)
    if n == 0:
        return 0.0

    labels = sorted(set(expected) | set(predicted))
    exp_counts = {label: expected.count(label) for label in labels}
    pred_counts = {label: predicted.count(label) for label in labels}

    p_o = accuracy(expected, predicted)
    p_e = sum((exp_counts[label] / n) * (pred_counts[label] / n) for label in labels)

    if p_e == 1.0:
        # Degenerate: everything is one label and both agree on it.
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def in_band(score: float, band: tuple[float, float]) -> bool:
    """Whether a score falls inside the human-labelled band (inclusive)."""
    low, high = band
    return low <= score <= high


def band_mae(scores: list[float], bands: list[tuple[float, float]]) -> float:
    """Mean absolute error of scores against their band midpoints."""
    if not scores:
        return 0.0
    total = 0.0
    for score, (low, high) in zip(scores, bands, strict=True):
        midpoint = (low + high) / 2
        total += abs(score - midpoint)
    return total / len(scores)


# ---------------------------------------------------------------------------
# Async agreement harness
# ---------------------------------------------------------------------------


async def run_claim_agreement(judge: LLMJudge, golden: list[GoldenClaim]) -> AgreementReport:
    """Run the golden claim set through judge.verify_claim and score agreement.

    Works with any LLMJudge implementation: the real ClaudeJudge for a live
    benchmark, or a StubJudge for deterministic harness tests.
    """
    expected = [g.expected for g in golden]
    predicted: list[str] = []
    disagreements: list[dict[str, str]] = []
    model = getattr(getattr(judge, "config", None), "model", "unknown")

    for item in golden:
        verdict = await judge.verify_claim(item.claim, item.context)
        predicted.append(verdict.verdict)
        if verdict.verdict != item.expected:
            disagreements.append(
                {
                    "id": item.id,
                    "claim": item.claim,
                    "expected": item.expected,
                    "predicted": verdict.verdict,
                    "evidence": verdict.evidence,
                }
            )

    confusion = build_confusion(expected, predicted)
    return AgreementReport(
        model=model,
        n=len(golden),
        accuracy=accuracy(expected, predicted),
        cohen_kappa=cohen_kappa(expected, predicted),
        macro_f1=macro_f1(confusion),
        per_class={label: class_metrics(confusion, label) for label in VERDICTS},
        confusion=confusion,
        disagreements=disagreements,
    )


# ---------------------------------------------------------------------------
# Snapshot / baseline I/O for drift regression
# ---------------------------------------------------------------------------


def write_snapshot(report: AgreementReport, name: str = "judge_agreement_latest") -> Path:
    """Persist a report snapshot under baselines/ for inspection and drift checks."""
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = _BASELINE_DIR / f"{name}.json"
    path.write_text(json.dumps(report.to_snapshot(), indent=2), encoding="utf-8")
    return path


def load_baseline(name: str = "judge_agreement_baseline") -> AgreementSnapshot | None:
    """Load a pinned baseline snapshot if one exists, else None.

    Returning None on a missing baseline lets the first live run establish the
    baseline manually (copy latest -> baseline) rather than failing.
    """
    path = _BASELINE_DIR / f"{name}.json"
    if not path.exists():
        return None
    return cast(AgreementSnapshot, json.loads(path.read_text(encoding="utf-8")))
