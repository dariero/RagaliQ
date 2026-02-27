"""Shared utilities for RagaliQ reporters."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestResult


def collect_evaluator_stats(
    results: list[RAGTestResult],
    threshold: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Compute sorted evaluator names and per-evaluator aggregate statistics.

    Args:
        results: Evaluation results to aggregate.
        threshold: Score threshold for pass/fail classification.

    Returns:
        Tuple of (evaluator_names, stats_rows) where evaluator_names is a
        sorted list of metric names and stats_rows is a list of dicts with
        keys: name, passed, failed, avg_score.
    """
    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r.scores.keys())
    evaluator_names = sorted(all_keys)

    stats: list[dict[str, Any]] = []
    for name in evaluator_names:
        scores = [r.scores[name] for r in results if name in r.scores]
        ev_passed = sum(1 for s in scores if s >= threshold)
        avg = sum(scores) / len(scores) if scores else 0.0
        stats.append(
            {
                "name": name,
                "passed": ev_passed,
                "failed": len(scores) - ev_passed,
                "avg_score": avg,
            }
        )

    return evaluator_names, stats
