# Changelog

All notable changes to RagaliQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/dariero/RagaliQ/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dariero/RagaliQ/releases/tag/v0.1.0
