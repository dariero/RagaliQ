"""JSON report exporter for RagaliQ evaluation results."""

import datetime
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestResult


class JSONReporter:
    """
    JSON report exporter for evaluation results.

    Produces a structured JSON document with a summary section and
    full per-result data, suitable for CI artefact storage or
    downstream processing.

    Example:
        >>> reporter = JSONReporter(threshold=0.7)
        >>> json_str = reporter.export(results)
        >>> Path("report.json").write_text(json_str)

    Output structure::

        {
          "generated_at": "2026-01-01T00:00:00+00:00",
          "threshold": 0.7,
          "summary": {
            "total": 3,
            "passed": 2,
            "failed": 1,
            "errored": 0,
            "pass_rate": 0.6667,
            "total_tokens": 1500,
            "total_execution_ms": 3400,
            "evaluators": {
              "faithfulness": {"passed": 2, "failed": 1, "avg_score": 0.733}
            }
          },
          "results": [...]
        }
    """

    def __init__(self, threshold: float = 0.7) -> None:
        """
        Initialize JSONReporter.

        Args:
            threshold: Score threshold for pass/fail classification (0.0–1.0).
        """
        self._threshold = threshold

    def export(self, results: list[RAGTestResult]) -> str:
        """
        Serialize evaluation results to a JSON string.

        Args:
            results: List of evaluation results to export.

        Returns:
            Pretty-printed JSON string (indent=2, UTF-8 safe).
        """
        from ragaliq.core.test_case import EvalStatus

        evaluator_names = sorted(results[0].scores.keys()) if results else []

        # Per-evaluator aggregate statistics
        ev_stats: dict[str, dict[str, Any]] = {}
        for name in evaluator_names:
            scores = [r.scores[name] for r in results if name in r.scores]
            ev_passed = sum(1 for s in scores if s >= self._threshold)
            avg = sum(scores) / len(scores) if scores else 0.0
            ev_stats[name] = {
                "passed": ev_passed,
                "failed": len(scores) - ev_passed,
                "avg_score": round(avg, 4),
            }

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        errored = sum(1 for r in results if r.status == EvalStatus.ERROR)
        total_tokens = sum(r.judge_tokens_used for r in results)
        total_ms = sum(r.execution_time_ms for r in results)

        doc = {
            "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "threshold": self._threshold,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "errored": errored,
                "pass_rate": round(passed / total, 4) if total else 0.0,
                "total_tokens": total_tokens,
                "total_execution_ms": total_ms,
                "evaluators": ev_stats,
            },
            "results": [self._serialize_result(r) for r in results],
        }

        return json.dumps(doc, indent=2, ensure_ascii=False)

    def _serialize_result(self, result: RAGTestResult) -> dict[str, Any]:
        """Serialize a single RAGTestResult to a plain dict."""
        tc = result.test_case
        return {
            "id": tc.id,
            "name": tc.name,
            "status": str(result.status),
            "passed": result.passed,
            "scores": result.scores,
            "details": result.details,
            "execution_time_ms": result.execution_time_ms,
            "judge_tokens_used": result.judge_tokens_used,
            "test_case": {
                "id": tc.id,
                "name": tc.name,
                "query": tc.query,
                "context": tc.context,
                "response": tc.response,
                "expected_answer": tc.expected_answer,
                "expected_facts": tc.expected_facts,
                "tags": tc.tags,
            },
        }
