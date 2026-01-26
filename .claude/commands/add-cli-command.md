# add-cli-command

## Purpose
Add a new CLI command to the ragaliq tool. Commands provide user-friendly interfaces for running evaluations, generating test cases, and managing configurations.

## Usage
Invoke when:
- Adding new functionality to the CLI (e.g., `ragaliq compare`, `ragaliq export`)
- Creating utility commands (e.g., `ragaliq config`, `ragaliq cache-clear`)
- Building interactive commands (e.g., `ragaliq wizard`)
- Implementing CI-focused commands (e.g., `ragaliq check --fail-under 0.8`)

## Automated Steps

1. **Analyze existing CLI structure**
   - Review `src/ragaliq/cli/main.py` for patterns
   - Check existing commands and their options
   - Understand Rich integration for output

2. **Implement new command**
   ```
   src/ragaliq/cli/main.py (add to existing)
   or
   src/ragaliq/cli/{command}.py (for complex commands)
   ```

3. **Add command options**
   - Use Typer annotations for type safety
   - Provide sensible defaults
   - Support environment variables for secrets

4. **Implement Rich output**
   - Tables for structured data
   - Progress bars for long operations
   - Colored status indicators

5. **Create tests**
   ```
   tests/unit/test_cli.py
   ```
   - Use `typer.testing.CliRunner`
   - Test success and error paths
   - Test option combinations

6. **Update documentation**
   - Add to CLI reference in README
   - Include usage examples

## Domain Expertise Applied

### Typer CLI Patterns

**1. Basic Command**
```python
@app.command()
def evaluate(
    dataset: str = typer.Argument(..., help="Path to dataset file"),
    evaluators: str = typer.Option(
        "faithfulness,relevance",
        "-e", "--evaluators",
        help="Comma-separated evaluator names"
    ),
    threshold: float = typer.Option(
        0.7, "-t", "--threshold",
        help="Minimum passing score"
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.console, "-o", "--output",
        help="Output format"
    ),
    api_key: str = typer.Option(
        None, "--api-key",
        envvar="ANTHROPIC_API_KEY",
        help="API key (or set ANTHROPIC_API_KEY)"
    ),
):
    """Evaluate RAG responses against quality metrics."""
    ...
```

**2. Subcommand Groups**
```python
# For complex commands like `ragaliq dataset load`, `ragaliq dataset generate`
dataset_app = typer.Typer(help="Dataset management commands")
app.add_typer(dataset_app, name="dataset")

@dataset_app.command("load")
def dataset_load(path: str):
    """Load and validate a dataset."""
    ...

@dataset_app.command("generate")
def dataset_generate(docs_dir: str, count: int = 10):
    """Generate test cases from documents."""
    ...
```

**3. Rich Output Integration**
```python
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

@app.command()
def run(dataset: str):
    # Progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(test_cases))
        for tc in test_cases:
            result = evaluate(tc)
            progress.advance(task)

    # Results table
    table = Table(title="Results")
    table.add_column("Test", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Status")

    for result in results:
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        table.add_row(result.name, f"{result.score:.2f}", status)

    console.print(table)
```

**4. Error Handling**
```python
@app.command()
def run(dataset: str):
    try:
        test_cases = DatasetLoader().load(dataset)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Dataset not found: {dataset}")
        raise typer.Exit(1)
    except ValidationError as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise typer.Exit(1)
```

**5. CI-Friendly Exit Codes**
```python
@app.command()
def check(
    dataset: str,
    fail_under: float = typer.Option(0.7, help="Fail if pass rate below this")
):
    """Run evaluation and exit non-zero if quality threshold not met."""
    results = run_evaluation(dataset)
    pass_rate = sum(r.passed for r in results) / len(results)

    if pass_rate < fail_under:
        console.print(f"[red]FAILED:[/red] Pass rate {pass_rate:.1%} < {fail_under:.1%}")
        raise typer.Exit(1)

    console.print(f"[green]PASSED:[/green] Pass rate {pass_rate:.1%}")
    raise typer.Exit(0)
```

### CLI Design Best Practices
- **Consistency**: Follow existing option naming patterns
- **Defaults**: Make common use cases require minimal options
- **Environment variables**: Support envvars for secrets
- **Exit codes**: 0 for success, non-zero for failures
- **Quiet mode**: Support `--quiet` for CI pipelines
- **JSON output**: Support `--output json` for scripting

### Pitfalls to Avoid
- Don't print to stdout in library code (return data, CLI prints)
- Don't hardcode paths - use Path for cross-platform
- Don't forget to handle Ctrl+C gracefully
- Don't mix async code without proper handling

## Interactive Prompts

**Ask for:**
- Command name and verb (e.g., `run`, `generate`, `export`)
- What does the command do?
- Required arguments?
- Optional flags with defaults?
- Output format (table, JSON, progress)?

**Suggest:**
- Similar existing commands to follow
- Appropriate Typer option types
- Rich components for output

**Validate:**
- Command fits CLI structure
- Options are intuitive
- Error messages are helpful

## Success Criteria
- [ ] Command added to CLI app
- [ ] Type hints on all parameters
- [ ] Help text for command and options
- [ ] Rich output for user-facing display
- [ ] Error handling with clear messages
- [ ] Exit codes for CI compatibility
- [ ] Unit tests with CliRunner
- [ ] `ragaliq {command} --help` works
- [ ] `make test` passes
- [ ] README CLI section updated
