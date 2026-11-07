# Changelog

All notable changes to RagaliQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The GitHub Release body's install line names the version without the tag's `v` (`pip install ragaliq==0.4.0`; pip also accepted `==v0.3.0`), as its Docker line already did. (#181)

### Removed

- Python 3.14 support: Python 3.15 is the minimum (`requires-python = ">=3.15"`), and CI, the Docker image and the release build run on 3.15. On Python 3.14, pip keeps installing 0.3.0. PyYAML 6.0.3 publishes no wheel for Python 3.15, so it installs from its source distribution. (#175)
- `scripts/verify_release.py`'s release mode (required files, TODO scan, import check, build, twine check). No workflow ran it, and it could not pass in any repo environment; the release workflow builds and checks the distributions itself. The script now runs only the four consistency checks that CI runs on every PR, and takes no option. (#181)
- `evaluate_faithfulness()` on `LLMJudge`, `BaseJudge` and `ClaudeJudge`, and the `faithfulness` prompt template that only it used: `get_prompt("faithfulness")` now raises `FileNotFoundError`. No evaluator called it, and a judge that subclasses `LLMJudge` no longer has to implement it. For a faithfulness score, run `await FaithfulnessEvaluator().evaluate(test_case, judge)` on a `RAGTestCase`. Its score is the share of the response's claims that `verify_claim()` finds `SUPPORTED`, from one `extract_claims()` call and one `verify_claim()` call per claim; the removed method made one call and returned the model's own rating. (#183)
- `TestCaseGenerator.generate_from_documents_sync()`. Call `asyncio.run(generator.generate_from_documents(...))` instead. (#183)
- `EvalStatus.SKIPPED`. Nothing set it: a result's status is `PASSED`, `FAILED` or `ERROR`. (#183)
- The pytest marker `ragaliq`. The plugin registers `rag_test` and `rag_slow`. Under `--strict-markers`, a test marked `@pytest.mark.ragaliq` now stops collection with "'ragaliq' not found in `markers` configuration option"; without the flag, pytest warns `PytestUnknownMarkWarning` and runs the test. Mark it `rag_test`, or register `ragaliq` in the suite's own configuration. (#183)
- `ragaliq.judges.prompts.list_prompts()`. Nothing in RagaliQ called it. The shipped templates are the `.yaml` files in `importlib.resources.files("ragaliq.judges.prompts")`. (#183)
- `TraceCollector.clear()`. A judge keeps the collector it was built with, so to start from zero, build a new judge with a new collector (`ClaudeJudge(trace_collector=TraceCollector())`). To measure part of a run on one collector, such as the plugin's `ragaliq_trace_collector`, note `len(collector.traces)` before and read `collector.traces[start:]` after. (#183)
- The `max_examples` parameter of `PromptTemplate.get_examples_text()` and `PromptTemplate.build_system_prompt()`. Both include every example of the template, as every judge call already did. (#183)

### Fixed

- A judge used in more than one event loop. The sync API (`RagaliQ.evaluate()`, `evaluate_batch()`, `assert_rag_quality()`, `ragaliq run`) runs each call in a new event loop, and pytest-asyncio runs each async test in its own loop, while a judge such as the plugin's session-scoped `ragaliq_judge` outlives them. `ClaudeTransport` now creates one HTTP client per event loop and closes it when that loop shuts down. Before, every later call sent its first request twice: once on a connection left from a closed loop, and again after the SDK's retry backoff. Its sockets stayed open (`ResourceWarning`), and `aclose()` from a new loop raised `RuntimeError: Event loop is closed`. (#179)
- A judge's concurrency limit is one semaphore per event loop. When calls had to wait for it in two separate sync calls, the second call raised `RuntimeError: ... is bound to a different event loop`, and `RagaliQ` reported those test cases as ERROR. (#179)
- `ClaudeTransport.send()` after `aclose()` raises `RuntimeError` at once. Before, it failed with `JudgeAPIError` after its retries. (#179)
- For code that builds on `ClaudeTransport`: each event loop's SDK client is created at that loop's first request, so `ANTHROPIC_BASE_URL` and the SDK's other environment settings are read then, not when the transport is built. A test double that replaces `AsyncAnthropic` must make `close()` awaitable (for example `AsyncMock`): it is awaited when the event loop shuts down. (#179)

## [0.3.0] - 2026-11-01

### Added

- Python 3.15 support: the package claims 3.15, and CI tests 3.14 and 3.15. Python 3.14 stays the minimum. This release is built and checked on Python 3.14 only; Python 3.15 is tested by CI on main, not by the release workflow. (#146)
- `ClaudeJudge` and `ClaudeTransport` close their HTTP client: `await judge.aclose()`, or `async with ClaudeJudge(...) as judge:`. (#141)
- Rejected responses are reported by cause: a response that stopped at `max_tokens`, was a refusal, hit the model's context window, was empty or held no text raises `JudgeResponseError` naming the cause, and its billed tokens are traced, so `total_cost_estimate` and `--ragaliq-cost-limit` count them. (#150, #155, #163)
- Meta-evaluation of the judge, in `tests/meta/` (#85): a weekly paid run in CI (#95, #128); a golden claim set of 42 claims, 14 per label, with written label rules and provenance (#161); the pinned baseline records a fingerprint of the golden set, the verify_claim prompt and the anthropic version, and a free CI test fails when one changes without a re-pin (#161); the five golden RAG cases run through the full runner (#97) and record their claim counts, which the weekly summary reports (#164).

### Changed

- **Claim verification prompt v2.0:** the numbered context documents are read in order as one passage, so a pronoun or role noun in one document resolves to a person or thing named in another. A fact about one person is not taken to contradict a claim about another, and a relation stated one way is not taken to contradict its reverse. The judge writes its evidence before its verdict. On the 42-claim golden set the judge matched every label in 3 of 3 runs (macro-F1 1.0; v1 matched 39, 40 and 40). `faithfulness`, `hallucination` and `context_recall` scores can shift relative to 0.2.0. The pinned meta-eval baseline is re-pinned. (#171)
- **Judge models:** the default tier is `claude-sonnet-5-5` and the gold tier `claude-opus-5-5` (`DEFAULT_JUDGE_MODEL`, `GOLD_STANDARD_JUDGE_MODEL`). (#150)
- **Request settings** (ADR-149): `JudgeConfig.temperature` is optional and defaults to `None`, so no temperature is sent unless you set one; `JudgeConfig.max_tokens` defaults to 16000 (was 1024) with an upper bound of 21333 (was 4096); the default tier sends `thinking: {"type": "between_tools"}`, so it runs without up-front thinking, and the gold tier runs with the API's adaptive thinking. Scores can shift relative to 0.2.0. (#150)
- **Raised minimum dependency versions:** `anthropic>=2.3` (was `>=0.109`) and `typer>=0.27` (was `>=0.26`). anthropic 2.x sends API version `2026-10-21`, under which an explicit `temperature` returns HTTP 400 on every model; the judge sends none by default. (#86, #111, #113, #141, #160)
- The faithfulness threshold's policy is documented: at the default 0.7, a response with K claims passes with up to floor(0.3 × K) unsupported ones, and K depends on the judge's claim extraction. The threshold is unchanged. (#166)
- One threshold for every evaluator is documented: `RagaliQ`, `ragaliq run` and `assert_rag_quality` judge every evaluator at one threshold, 0.7 unless you set it, so `hallucination` runs at that threshold there, not at its class default 0.8. The behaviour is unchanged; #165 tracks a change to it. (#173)
- Contributors: every GitHub Action is pinned to a full commit SHA (#160); Dependabot updates the actions and the Docker base image (#124, #152, #157); the quality gate runs from `.venv` (#139, #148); agent instructions live in `AGENTS.md` (#148); the docs build runs on pull requests (#128); the release workflow's validate job installs from `pylock.toml`, as CI does (#173).

### Removed

- The `openai` extra: it installed a package that no code used. OpenAI as a judge is tracked in #114. (#116)
- `GETTING_STARTED.md` and `docs/PROJECT_PLAN.md`; `docs/TUTORIAL.md` is the getting-started guide. (#157)
- Contributors: the hatch environments and scripts in `pyproject.toml`; hatchling stays the build backend. (#152)

### Fixed

- The gold tier sent `temperature=0.0` to `claude-opus-4-8` by default, which returns HTTP 400 on Claude 4.7 and later models. (#150)
- `evaluate_batch_async` created one task per test case up front, and its semaphore only limited how many ran at once. It now runs min(`max_concurrency`, number of test cases) worker tasks. With `fail_fast=True`, the first exception cancels the other workers, so no further evaluation starts. A `max_concurrency` below 1 raises `ValueError`; 0 used to hang the batch. (#167)
- A judge answer with text before its JSON object raised `JudgeResponseError`, so the test case ended as ERROR. The object is now parsed when it ends the answer, and a warning is logged; JSON followed by more text is still rejected. (#169)
- Text files are read and written as UTF-8 explicitly. (#140)
- The release body's `docker pull` line named an image tag with a `v` prefix that is never published. (#118)

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

[Unreleased]: https://github.com/dariero/RagaliQ/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/dariero/RagaliQ/releases/tag/v0.3.0
[0.2.0]: https://github.com/dariero/RagaliQ/releases/tag/v0.2.0
[0.1.0]: https://github.com/dariero/RagaliQ/releases/tag/v0.1.0
