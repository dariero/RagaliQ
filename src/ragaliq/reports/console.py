"""Console reporter for RagaliQ evaluation results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestResult


class ConsoleReporter:
    """
    Rich-based console reporter for evaluation results.

    Renders a structured terminal report with:
    - Results table with per-evaluator scores
    - Detailed reasoning for failed/errored test cases
    - Summary statistics per evaluator and overall

    Example:
        >>> reporter = ConsoleReporter(threshold=0.7)
        >>> reporter.report(results)

        # With custom console (e.g., for capturing output in tests)
        >>> from rich.console import Console
        >>> console = Console(record=True, width=120)
        >>> reporter = ConsoleReporter(console=console)
        >>> reporter.report(results)
        >>> text = console.export_text()
    """

    def __init__(
        self,
        console: Console | None = None,
        threshold: float = 0.7,
        verbose: bool = False,
    ) -> None:
        """
        Initialize ConsoleReporter.

        Args:
            console: Rich Console instance. Creates a default Console if not provided.
            threshold: Score threshold for pass/fail coloring (0.0–1.0).
            verbose: If True, show reasoning details for all results, not just failures.
        """
        self._console = console or Console()
        self._threshold = threshold
        self._verbose = verbose

    def report(self, results: list[RAGTestResult]) -> None:
        """
        Render a complete evaluation report to the console.

        Sections rendered in order:
        1. Results table with per-evaluator scores
        2. Failure details (reasoning/errors) for failed and errored cases
        3. Summary statistics per evaluator and overall pass/fail count

        Args:
            results: List of evaluation results to report.
        """
        self._print_results_table(results)
        self._print_failed_details(results)
        self._print_summary(results)

    def _print_results_table(self, results: list[RAGTestResult]) -> None:
        """Print results as a Rich table with per-evaluator scores."""
        from ragaliq.core.test_case import EvalStatus

        table = Table(title="Evaluation Results", show_lines=True)
        table.add_column("Test Case", style="bold")
        table.add_column("Status", justify="center")

        evaluator_names: list[str] = []
        if results:
            evaluator_names = sorted(results[0].scores.keys())

        for name in evaluator_names:
            table.add_column(name.replace("_", " ").title(), justify="center")

        status_map = {
            EvalStatus.PASSED: "[green]PASS[/green]",
            EvalStatus.FAILED: "[red]FAIL[/red]",
            EvalStatus.ERROR: "[yellow]ERROR[/yellow]",
            EvalStatus.SKIPPED: "[dim]SKIP[/dim]",
        }

        for result in results:
            status_str = status_map.get(result.status, str(result.status))
            score_cells = []
            for name in evaluator_names:
                score = result.scores.get(name)
                if score is None:
                    score_cells.append("—")
                elif score >= self._threshold:
                    score_cells.append(f"[green]{score:.2f}[/green]")
                else:
                    score_cells.append(f"[red]{score:.2f}[/red]")
            table.add_row(result.test_case.name, status_str, *score_cells)

        self._console.print(table)

    def _print_failed_details(self, results: list[RAGTestResult]) -> None:
        """Print reasoning/error details for failed and errored results.

        Uses scores vs threshold (not details["passed"]) to identify failing
        evaluators, so this works correctly even when details are sparse.
        """
        from ragaliq.core.test_case import EvalStatus

        targets = [
            r for r in results if self._verbose or r.status in {EvalStatus.FAILED, EvalStatus.ERROR}
        ]
        if not targets:
            return

        self._console.rule("[bold red]Failed Test Details[/bold red]")

        for result in targets:
            self._console.print(f"\n[bold]{result.test_case.name}[/bold]")
            details: dict[str, Any] = result.details if isinstance(result.details, dict) else {}
            for evaluator_name, detail in details.items():
                if not isinstance(detail, dict):
                    continue
                score = result.scores.get(evaluator_name)
                error: str = detail.get("error", "")
                reasoning: str = detail.get("reasoning", "")
                if error:
                    self._console.print(
                        f"  [yellow]{evaluator_name}[/yellow] [red]ERROR[/red]: {error}"
                    )
                elif score is not None and score < self._threshold and reasoning:
                    self._console.print(f"  [yellow]{evaluator_name}[/yellow]: {reasoning}")

    def _print_summary(self, results: list[RAGTestResult]) -> None:
        """Print per-evaluator statistics and overall pass/fail count."""
        if not results:
            return

        self._console.rule("[bold]Summary[/bold]")

        evaluator_names = sorted(results[0].scores.keys())
        if evaluator_names:
            table = Table(show_header=True, show_lines=False, box=None, padding=(0, 1))
            table.add_column("Evaluator", style="bold cyan")
            table.add_column("Passed", justify="right")
            table.add_column("Failed", justify="right")
            table.add_column("Avg Score", justify="right")

            for name in evaluator_names:
                scores = [r.scores[name] for r in results if name in r.scores]
                ev_passed = sum(1 for s in scores if s >= self._threshold)
                ev_failed = len(scores) - ev_passed
                avg = sum(scores) / len(scores) if scores else 0.0
                color = "green" if ev_failed == 0 else "red"
                table.add_row(
                    name,
                    f"[green]{ev_passed}[/green]",
                    f"[red]{ev_failed}[/red]",
                    f"[{color}]{avg:.2f}[/{color}]",
                )
            self._console.print(table)

        total = len(results)
        total_tokens = sum(r.judge_tokens_used for r in results)
        passed_total = sum(1 for r in results if r.passed)
        failed_total = total - passed_total
        tokens_str = f"  ({total_tokens:,} tokens)" if total_tokens else ""

        if failed_total == 0:
            self._console.print(
                f"\n[bold green]✓ {passed_total}/{total} passed[/bold green]{tokens_str}"
            )
        else:
            self._console.print(
                f"\n[bold red]✗ {passed_total}/{total} passed, "
                f"{failed_total} failed[/bold red]{tokens_str}"
            )
