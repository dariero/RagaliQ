"""CLI entry point for RagaliQ."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

import ragaliq

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestResult

app = typer.Typer(no_args_is_help=True)


def version_callback(value: bool) -> None:
    """Display version and exit."""
    if value:
        typer.echo(f"RagaliQ version {ragaliq.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """RagaliQ - LLM Testing Framework for RAG Systems."""


@app.command("run")
def run(
    dataset: Path = typer.Argument(..., help="Path to dataset file (JSON, YAML, or CSV)."),
    evaluator: list[str] | None = typer.Option(
        None,
        "--evaluator",
        "-e",
        help="Evaluators to run. Can be specified multiple times. Defaults to faithfulness and relevance.",
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", "-t", help="Pass/fail threshold (0.0–1.0)."
    ),
    judge: str = typer.Option("claude", "--judge", "-j", help="LLM judge to use."),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Stop on first evaluator error."),
) -> None:
    """Run evaluations against a dataset."""
    import asyncio

    from ragaliq import RagaliQ
    from ragaliq.datasets import DatasetLoader, DatasetLoadError

    console = Console()

    try:
        dataset_obj = DatasetLoader.load(dataset)
    except DatasetLoadError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None

    test_cases = dataset_obj.test_cases
    total = len(test_cases)
    typer.echo(f"\nRagaliQ — {total} test case{'s' if total != 1 else ''} loaded\n")

    runner_obj = RagaliQ(
        judge=judge,  # type: ignore[arg-type]
        evaluators=evaluator if evaluator else None,
        default_threshold=threshold,
        fail_fast=fail_fast,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Evaluating...", total=None)
        results = asyncio.run(runner_obj.evaluate_batch_async(test_cases))

    _print_results_table(results, console)

    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    if failed:
        typer.echo(f"\nSummary: {passed}/{total} passed, {failed} failed")
        raise typer.Exit(code=1)
    else:
        typer.echo(f"\nSummary: {passed}/{total} passed")


def _print_results_table(results: list[RAGTestResult], console: Console) -> None:
    """Render evaluation results as a Rich table.

    Args:
        results: List of RAGTestResult objects.
        console: Rich Console to render to.
    """
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
            elif score >= 0.7:
                score_cells.append(f"[green]{score:.2f}[/green]")
            else:
                score_cells.append(f"[red]{score:.2f}[/red]")
        table.add_row(result.test_case.name, status_str, *score_cells)

    console.print(table)


@app.command("list-evaluators")
def list_evaluators_cmd() -> None:
    """List all available evaluators."""
    import ragaliq.evaluators  # noqa: F401 — triggers registration of built-ins
    from ragaliq.evaluators import get_evaluator, list_evaluators

    console = Console()
    names = list_evaluators()

    if not names:
        typer.echo("No evaluators registered.")
        return

    table = Table(title="Available Evaluators", show_lines=True)
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Default Threshold", justify="center")

    for name in names:
        cls = get_evaluator(name)
        description = getattr(cls, "description", "—")
        threshold = getattr(cls, "threshold", 0.7)
        table.add_row(name, description, str(threshold))

    console.print(table)


@app.command("validate")
def validate(
    dataset: Path = typer.Argument(..., help="Path to dataset file to validate."),
) -> None:
    """Validate dataset schema without running evaluations."""
    from ragaliq.datasets import DatasetLoader, DatasetLoadError

    try:
        dataset_obj = DatasetLoader.load(dataset)
    except DatasetLoadError as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from None

    total = len(dataset_obj.test_cases)
    typer.echo(f"Valid — {total} test case{'s' if total != 1 else ''} found in {dataset}")


@app.command("version")
def version_cmd() -> None:
    """Show version information."""
    typer.echo(f"RagaliQ version {ragaliq.__version__}")


if __name__ == "__main__":
    app()
