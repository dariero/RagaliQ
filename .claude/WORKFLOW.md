# RagaliQ Development with Claude Code

This guide explains how to use Claude Code effectively for RagaliQ development.

## Quick Start

```bash
cd /path/to/ragaliq
claude
```

Claude Code automatically reads `CLAUDE.md` for project context.

## Available Commands

Use slash commands for common operations:

| Command | Purpose |
|---------|---------|
| `/new-evaluator` | Create new LLM evaluator |
| `/new-judge` | Add LLM provider support |
| `/add-cli-command` | Extend CLI |
| `/setup-cicd` | Configure GitHub Actions |

See [README.md](README.md) for full command catalog.

## Task Workflow

**GitHub Board:** https://github.com/users/dariero/projects/2/views/1

1. Check current task in "In Progress" column
2. Use appropriate `/command` or describe the task
3. Run `make test && make typecheck` after implementation
4. Mark task complete, move to next

## Effective Prompts

**Be specific:**
```
Implement FaithfulnessEvaluator in src/ragaliq/evaluators/faithfulness.py.
Follow the pattern from the Evaluator base class.
Include unit tests with a mocked judge.
```

**Reference existing code:**
```
Follow the same pattern as test_case.py
Use the EvaluationResult model from core/evaluator.py
```

**Ask for tests together:**
```
Implement X and add unit tests in tests/unit/test_X.py
```

## Development Commands

```bash
# Verify changes
make test           # Run all tests
make test-fast      # Quick test without coverage
make lint           # Check code style
make typecheck      # Run mypy

# Build
make build          # Build package
make clean          # Remove artifacts
```

## Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dev dependencies
pip install -e ".[dev]"

# Set API key
export ANTHROPIC_API_KEY=your-key-here
```

## Project Structure

```
src/ragaliq/
├── core/           # TestCase, Evaluator base, Runner
├── evaluators/     # Faithfulness, Relevance, etc.
├── judges/         # LLM judge implementations
├── datasets/       # Data loading and generation
├── reports/        # Console, HTML, JSON reporters
├── integrations/   # Pytest plugin, CI helpers
└── cli/            # Typer CLI commands
```
