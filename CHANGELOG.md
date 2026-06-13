# Changelog

All notable changes to RagaliQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-13

### Added

- Few-shot examples from prompt templates now reach the judge: prompt builders call `PromptTemplate.build_system_prompt()` so each template's `examples:` block is sent to the model (previously authored but never wired in). (#77)
- `DEFAULT_JUDGE_MODEL` and `GOLD_STANDARD_JUDGE_MODEL` constants, re-exported from `ragaliq.judges`, as the single source of truth for judge model identifiers. (#80)

### Changed

- Judge prompts now include few-shot examples by default, which may shift evaluation scores relative to 0.1.0. (#77)
- **Raised minimum dependency versions** to their latest releases: `anthropic>=0.109`, `pydantic>=2.13`, `typer>=0.26`, and `rich>=15` (a major bump). Downstream consumers must meet these higher floors. (#84)
- `mypy>=2.1` and `pytest>=9.1` are now the development floors (mypy resolves a type-checking hang under Python 3.14); dependencies are pinned in a cross-platform `pylock.toml`. (#84)

### Fixed

- `GETTING_STARTED.md` examples corrected to match the public API: `result.scores`, `assert_rag_quality(test_case, judge=...)`, `RagaliQ(default_threshold=...)`, and the `{version, test_cases}` dataset schema. (#76)
- Replaced the invalid judge model id `claude-opus-4-6` with `claude-opus-4-8` across code, docs, and the trace pricing table. (#79)
- Documented the local `mypy` hang on macOS + Python 3.14 and corrected the stale "Python 3.12+" requirement in `CONTRIBUTING.md` to 3.14+. (#78)

## [0.1.0] - 2026-02-21

### Added

- **LLM-as-Judge architecture** — Claude and OpenAI as semantic evaluators for RAG response quality
- **5 built-in evaluators** — Faithfulness, Relevance, Hallucination, Context Precision, Context Recall
- **Pytest plugin** — `rag_tester` fixture, `@pytest.mark.rag_test` marker, `assert_rag_quality` helper
- **CLI** — `ragaliq run`, `ragaliq generate`, `ragaliq validate`, `ragaliq list-evaluators`
- **Dataset support** — JSON, YAML, and CSV dataset loading; LLM-based test case generation from documents
- **Report exporters** — Console (Rich), HTML, and JSON output formats
- **GitHub Actions integration** — step summaries, PR annotations, step outputs, CI auto-detection
- **Custom evaluator registry** — `@register_evaluator` decorator for extending with custom metrics
- **Async-first execution** — concurrent evaluations with configurable parallelism
- **Retry with backoff** — Tenacity-based retry for transient LLM API failures
- **Cost tracking** — Token usage and cost monitoring across evaluation sessions
- **Pydantic v2 models** — Strict typing for all data structures (TestCase, EvaluationResult, JudgeConfig)

### Architecture

- Evaluator pattern: each metric is a separate `Evaluator` class with `evaluate()` method
- Judge injection via method parameter (dependency injection, not constructor)
- `EvaluationResult` carries `raw_response` for debugging and `error` field for graceful failure handling
- FaithfulnessEvaluator uses multi-step pipeline: claim extraction, verification, aggregation

[Unreleased]: https://github.com/dariero/RagaliQ/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dariero/RagaliQ/releases/tag/v0.2.0
[0.1.0]: https://github.com/dariero/RagaliQ/releases/tag/v0.1.0
