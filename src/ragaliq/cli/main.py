"""CLI entry point for RagaliQ."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

import ragaliq

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

    from ragaliq.reports.console import ConsoleReporter

    ConsoleReporter(console=console, threshold=threshold).report(results)

    passed = sum(1 for r in results if r.passed)
    if passed < total:
        raise typer.Exit(code=1)



@app.command("generate")
def generate(
    docs_path: Path = typer.Argument(
        ...,
        help="Path to documents: a .txt file, a directory of .txt files, or a .json/.yaml list.",
    ),
    n: int = typer.Option(10, "--num", "-n", help="Number of test cases to generate."),
    output: Path = typer.Option(
        Path("output.json"), "--output", "-o", help="Output JSON file path."
    ),
    judge: str = typer.Option("claude", "--judge", "-j", help="LLM judge to use."),
) -> None:
    """Generate test cases from documents using an LLM."""
    import asyncio
    import json

    from ragaliq.datasets.generator import TestCaseGenerator
    from ragaliq.datasets.schemas import DatasetSchema

    console = Console()

    documents = _load_documents(docs_path)
    if not documents:
        typer.echo(f"Error: no documents found at {docs_path}", err=True)
        raise typer.Exit(code=1)

    doc_word = "document" if len(documents) == 1 else "documents"
    typer.echo(
        f"\nRagaliQ Generate — {len(documents)} {doc_word} loaded, generating {n} test cases\n"
    )

    if judge == "claude":
        from ragaliq.judges import ClaudeJudge

        judge_instance = ClaudeJudge()
    else:
        typer.echo(f"Error: unsupported judge '{judge}'. Supported: claude", err=True)
        raise typer.Exit(code=1)

    generator = TestCaseGenerator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Generating test cases...", total=None)
        test_cases = asyncio.run(
            generator.generate_from_documents(documents=documents, n=n, judge=judge_instance)
        )

    dataset = DatasetSchema(
        test_cases=test_cases,
        metadata={"generator": "ragaliq", "source": str(docs_path)},
    )
    output.write_text(
        json.dumps(dataset.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    case_word = "test case" if len(test_cases) == 1 else "test cases"
    typer.echo(f"Generated {len(test_cases)} {case_word} → {output}")


def _load_documents(docs_path: Path) -> list[str]:
    """
    Load documents from a file or directory.

    Supports:
    - .txt file: read as a single document
    - directory: read all .txt files as separate documents
    - .json file: must contain a list of strings
    - .yaml / .yml file: must contain a list of strings

    Args:
        docs_path: Path to the document source.

    Returns:
        List of non-empty document strings.
    """
    if not docs_path.exists():
        return []

    if docs_path.is_dir():
        docs = [
            txt_file.read_text(encoding="utf-8").strip()
            for txt_file in sorted(docs_path.glob("*.txt"))
        ]
        return [d for d in docs if d]

    suffix = docs_path.suffix.lower()

    if suffix == ".txt":
        content = docs_path.read_text(encoding="utf-8").strip()
        return [content] if content else []

    if suffix == ".json":
        import json

        data = json.loads(docs_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(d).strip() for d in data if d]

    if suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(docs_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(d).strip() for d in data if d]

    return []


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
