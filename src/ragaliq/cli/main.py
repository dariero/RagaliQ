"""CLI entry point for RagaliQ."""


import typer

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
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """RagaliQ - LLM Testing Framework for RAG Systems."""
    pass


@app.command()
def evaluate() -> None:
    """
    Evaluate RAG test cases.

    This command is not yet implemented. For now, use RagaliQ as a library:

        from ragaliq import RagaliQ, RAGTestCase

        tester = RagaliQ(judge="claude")
        test_case = RAGTestCase(...)
        result = tester.evaluate(test_case)
    """
    typer.echo("The 'evaluate' command is not yet implemented.")
    typer.echo("Please use RagaliQ as a library for now.")
    typer.echo("\nSee: https://github.com/dariero/ragaliq for usage examples.")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
