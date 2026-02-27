# Contributing to RagaliQ

Thank you for your interest in contributing! RagaliQ is an open-source LLM/RAG evaluation framework and we welcome bug reports, feature ideas, and pull requests.

## Before You Start

- Check the [GitHub Issues](https://github.com/dariero/RagaliQ/issues) to see if your bug or feature is already tracked.
- For significant changes, open an issue first to discuss the approach before writing code.

## Development Setup

**Requirements:** Python 3.12+, [Hatch](https://hatch.pypa.io/)

```bash
git clone https://github.com/dariero/RagaliQ.git
cd RagaliQ
pip install hatch
hatch shell         # creates and activates a virtual environment
```

## Quality Gates

All contributions must pass these checks before merging:

```bash
hatch run lint       # ruff check (style and imports)
hatch run typecheck  # mypy strict mode
hatch run test       # pytest with coverage
```

Run them together:

```bash
hatch run check
```

## Code Standards

- **Type hints** are required on all public functions and methods.
- **Docstrings** must use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- **Evaluator pattern**: each new metric MUST be a separate `Evaluator` subclass with an `evaluate()` method. See `src/ragaliq/evaluators/` for examples.
- **Async-first**: all LLM calls must be `async def`.
- **Pydantic models** for all data structures (no raw dicts).

## Submitting a Pull Request

1. Fork the repository and create a branch: `feat/<short-description>`.
2. Make your changes and ensure all quality gates pass.
3. Write or update tests in `tests/` mirroring the `src/` structure.
4. Open a PR against `main` with a clear description of what changed and why.

## Reporting Bugs

Open a [GitHub Issue](https://github.com/dariero/RagaliQ/issues/new) with:
- Python version and OS
- RagaliQ version (`pip show ragaliq`)
- A minimal reproducible example
- The full error traceback

## Security Vulnerabilities

Please do **not** open a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
