# RagaliQ - Project Instructions

<constraints>
- ALWAYS run `hatch run lint && hatch run typecheck && hatch run test` before committing
- **2026-10-12 override:** the owner's decision at Checkpoint 1 of the pre-PR-B run, until revoked: agents on the owner's machine do not run hatch (its environments live under `~/Library`, outside their scope). For them the gate above is the same commands run from the repo's `.venv` (synced from `pylock.toml`): `.venv/bin/ruff check src/ tests/ scripts/ && .venv/bin/mypy src/ scripts/ && .venv/bin/pytest`, plus CI's test form `.venv/bin/coverage run --source=src/ragaliq -m pytest tests/unit/ tests/meta/test_meta_metrics.py -o addopts="" -q && .venv/bin/coverage report`.
- NEVER commit files matching: .env*, *.pem, *credentials*, *secret*, .DS_Store
- NEVER push directly to main; all work ships through `/ship`
- NEVER add or remove dependencies without explicit user approval
- NEVER change existing architectural patterns without explicit approval
- Type hints MUST be present on all public functions
- Docstrings MUST use Google style
- Use `ruff` for linting and formatting -- no manual style overrides
</constraints>

## Dependency Policy

**Floors move for a reason, never because something newer exists.**

A `>=` floor in `pyproject.toml` is a support claim: the oldest version the code
is known to work with. Raise it only when

1. a feature the code now uses requires it, or
2. support for older versions is being deliberately dropped.

"A newer release exists" is not a reason. Dependabot proposes lock and pin
updates; it is not wired to floors.

**A floor bump with no lock regeneration changes nothing about what CI runs.**
CI installs with `uv pip sync pylock.toml` then `uv pip install -e . --no-deps`,
so the floors are never resolved — the lock is the only thing exercised. Bump a
floor without regenerating, and the PR is green and inert.

Regenerate one package at a time and diff the lock before committing; see
CONTRIBUTING.md. `scripts/verify_release.py --consistency-only` asserts the
floors, the pins and the lock still agree.

**2026-10-11 override:** the owner's instruction for the `deps-latest-2026-10-11` run, approved at its Checkpoint 1, overrides the one-package-at-a-time rule and the no-adding-or-removing-dependencies rule for that branch only: `pylock.toml` was regenerated with a bare `--upgrade`, which adds and removes transitive packages, and `[project.dependencies]` floors were left unchanged.

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
