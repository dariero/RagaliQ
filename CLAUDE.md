# RagaliQ - Project Context

## What is RagaliQ?

RagaliQ (RAG + Quality) is a Python library + CLI tool for automated testing of RAG (Retrieval-Augmented Generation) pipelines. It enables QA and AI engineers to write quality tests for LLM responses using pytest-like syntax.

## Tech Stack

- Python 3.14+
- Pydantic v2 for data validation
- Anthropic SDK (Claude as LLM-as-Judge)
- Typer for CLI
- Rich for terminal output
- Pytest plugin architecture

## Project Structure

```
src/ragaliq/
├── core/           # TestCase, Evaluator base, Runner
├── evaluators/     # Faithfulness, Relevance, Hallucination, etc.
├── judges/         # LLM judge implementations (Claude, OpenAI)
├── datasets/       # Test data loading and generation
├── reports/        # Console, HTML, JSON reporters
├── integrations/   # Pytest plugin, CI helpers
└── cli/            # Typer CLI commands
```

## Key Design Decisions

1. **Evaluator Pattern**: Each metric is a separate Evaluator class with `evaluate()` method
2. **LLM-as-Judge**: Claude API assesses response quality, not hardcoded rules
3. **Async-first**: All LLM calls are async for performance
4. **Pydantic everywhere**: Strict typing with Pydantic models
5. **Pytest-native**: Library feels natural to pytest users

## Code Style

- Use `ruff` for linting and formatting
- Type hints required for all public functions
- Docstrings in Google style
- Tests in `tests/` mirroring `src/` structure

## Current Status

**GitHub Board**: https://github.com/users/dariero/projects/2/views/1

Phase 1 in progress - building judge integration and core evaluators.

## Commands

```bash
hatch run lint       # ruff check
hatch run format     # ruff format (auto-fix)
hatch run typecheck  # mypy
hatch run test       # pytest
```

## Automation

Two slash commands for the dev workflow:
- `/start-work <issue>` - Begin work (branch + board update)
- `/ship` - Ship to main (commit + check + PR + merge + cleanup)

Implementation patterns (evaluator, judge, prompt optimization) are documented in `.claude/WORKFLOW.md`.
