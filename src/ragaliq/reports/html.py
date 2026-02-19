"""HTML report exporter for RagaliQ evaluation results."""

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestResult


class HTMLReporter:
    """
    HTML report exporter for evaluation results.

    Renders a self-contained HTML file using a Jinja2 template with
    embedded CSS — no external dependencies, no JavaScript.

    Features:
    - Summary cards (total, passed, failed, errored, pass rate, tokens)
    - Per-evaluator statistics table
    - Results table with colour-coded scores
    - Expandable failure details via HTML5 <details>/<summary>

    Example:
        >>> reporter = HTMLReporter(threshold=0.7)
        >>> html = reporter.export(results)
        >>> Path("report.html").write_text(html, encoding="utf-8")
    """

    _TEMPLATE_DIR = Path(__file__).parent / "templates"
    _TEMPLATE_NAME = "report.html.j2"

    def __init__(self, threshold: float = 0.7) -> None:
        """
        Initialize HTMLReporter.

        Args:
            threshold: Score threshold for pass/fail coloring (0.0–1.0).
        """
        self._threshold = threshold

    def export(self, results: list[RAGTestResult]) -> str:
        """
        Render evaluation results to a self-contained HTML string.

        Args:
            results: List of evaluation results to render.

        Returns:
            Complete HTML document as a string.
        """
        from jinja2 import Environment, FileSystemLoader

        env = Environment(
            loader=FileSystemLoader(str(self._TEMPLATE_DIR)),
            autoescape=True,
        )
        template = env.get_template(self._TEMPLATE_NAME)
        context = self._build_context(results)
        return template.render(**context)

    def _build_context(self, results: list[RAGTestResult]) -> dict[str, Any]:
        """Prepare the template context from evaluation results."""
        from ragaliq.core.test_case import EvalStatus

        evaluator_names = sorted(results[0].scores.keys()) if results else []

        # Per-evaluator aggregate stats
        ev_rows = []
        for name in evaluator_names:
            scores = [r.scores[name] for r in results if name in r.scores]
            ev_passed = sum(1 for s in scores if s >= self._threshold)
            avg = sum(scores) / len(scores) if scores else 0.0
            ev_rows.append(
                {
                    "name": name,
                    "passed": ev_passed,
                    "failed": len(scores) - ev_passed,
                    "avg_score": avg,
                }
            )

        # Per-result data with pre-processed scores and failure details
        processed: list[dict[str, Any]] = []
        for result in results:
            # Score cells ordered by evaluator_names
            score_cells = []
            for name in evaluator_names:
                value = result.scores.get(name)
                score_cells.append(
                    {
                        "value": value,
                        "passes": value is not None and value >= self._threshold,
                    }
                )

            # Failure details (reasoning / errors for failing evaluators)
            failing_details: list[dict[str, Any]] = []
            if not result.passed and isinstance(result.details, dict):
                for ev_name, detail in result.details.items():
                    if not isinstance(detail, dict):
                        continue
                    score = result.scores.get(ev_name)
                    error: str = detail.get("error", "")
                    reasoning: str = detail.get("reasoning", "")
                    if error:
                        failing_details.append(
                            {"evaluator": ev_name, "text": error, "is_error": True}
                        )
                    elif score is not None and score < self._threshold and reasoning:
                        failing_details.append(
                            {"evaluator": ev_name, "text": reasoning, "is_error": False}
                        )

            processed.append(
                {
                    "name": result.test_case.name,
                    "status": str(result.status),
                    "passed": result.passed,
                    "scores": score_cells,
                    "execution_time_ms": result.execution_time_ms,
                    "failing_details": failing_details,
                }
            )

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        errored = sum(1 for r in results if r.status == EvalStatus.ERROR)
        total_tokens = sum(r.judge_tokens_used for r in results)
        generated_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")

        return {
            "generated_at": generated_at,
            "threshold": self._threshold,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "errored": errored,
                "pass_rate": passed / total if total else 0.0,
                "total_tokens": total_tokens,
            },
            "evaluators": ev_rows,
            "results": processed,
        }
