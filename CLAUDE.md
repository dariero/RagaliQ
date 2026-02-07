# RagaliQ - Project Instructions

<constraints>
- ALWAYS run `hatch run lint && hatch run typecheck && hatch run test` before committing
- NEVER commit files matching: .env*, *.pem, *credentials*, *secret*, .DS_Store
- NEVER push directly to main; all work ships through `/ship`
- NEVER add or remove dependencies without explicit user approval
- NEVER change existing architectural patterns without explicit approval
- Type hints MUST be present on all public functions
- Docstrings MUST use Google style
- Use `ruff` for linting and formatting -- no manual style overrides
</constraints>

## Design Decisions (MUST follow in all new code)

1. **Evaluator Pattern**: Each metric MUST be a separate Evaluator class with `evaluate()` method. DO NOT refactor to alternative patterns without explicit approval.
2. **LLM-as-Judge**: Claude API assesses response quality, not hardcoded rules.
3. **Async-first**: All LLM calls MUST be async.
4. **Pydantic everywhere**: Strict typing with Pydantic models for all data structures.
5. **Pytest-native**: Library MUST feel natural to pytest users.

## Project Context

RagaliQ (RAG + Quality) is a Python library + CLI for automated testing of RAG pipelines. Enables QA and AI engineers to write quality tests for LLM responses using pytest-like syntax.

**GitHub Board**: https://github.com/users/dariero/projects/2/views/1 — Phase 1: judge integration and core evaluators.

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

## Code Style

- Use `ruff` for linting and formatting
- Type hints required for all public functions
- Docstrings in Google style
- Tests in `tests/` mirroring `src/` structure

## Commands

```bash
hatch run lint       # ruff check
hatch run format     # ruff format (auto-fix)
hatch run typecheck  # mypy
hatch run test       # pytest
```

## Architecture Decision Records (ADRs)

Document significant architectural decisions in `.decisions/` directory following the ADR format. See `.decisions/README.md` for structure and guidelines.

**When to write an ADR:**
- Introducing new architectural patterns or components
- Choosing between multiple valid implementation approaches
- Modifying existing contracts or interfaces
- Decisions with non-obvious trade-offs

**ADR naming:** `.decisions/ADR-NNN-short-title.md` where NNN is the GitHub issue number if applicable.

**Required sections:** Context, Proposed Solution, Principles Applied, Alternatives Considered.

## Automation

Two slash commands for the dev workflow:
- `/start-work <issue>` - Begin work (branch + board update)
- `/ship` - Ship to main (commit + check + PR + merge + cleanup)

Implementation patterns (evaluator, judge, prompt optimization) are documented in `.claude/WORKFLOW.md`.
Project constants (IDs, branch naming, commit format) are in `.claude/CONSTANTS.md`.
