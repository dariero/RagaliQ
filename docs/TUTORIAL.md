# RagaliQ Tutorial

A step-by-step guide from installation to production CI/CD integration.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Your First Evaluation](#2-your-first-evaluation)
3. [Understanding Results](#3-understanding-results)
4. [Batch Evaluation](#4-batch-evaluation)
5. [Choosing Evaluators](#5-choosing-evaluators)
6. [Dataset Files](#6-dataset-files)
7. [CLI Usage](#7-cli-usage)
8. [Pytest Integration](#8-pytest-integration)
9. [Reports](#9-reports)
10. [GitHub Actions CI/CD](#10-github-actions-cicd)
11. [Advanced: Custom Evaluators](#11-advanced-custom-evaluators)
12. [Advanced: JudgeConfig](#12-advanced-judgeconfig)
13. [Advanced: Observability](#13-advanced-observability)

---

## 1. Installation

```bash
pip install ragaliq
export ANTHROPIC_API_KEY=sk-ant-...
```

Verify the install:

```bash
ragaliq version
ragaliq list-evaluators
```

You should see RagaliQ version and the five built-in evaluators listed.

---

## 2. Your First Evaluation

The core workflow is: create a `RAGTestCase` → pass it to `RagaliQ.evaluate()` → inspect the `RAGTestResult`.

```python
from ragaliq import RagaliQ, RAGTestCase

# RagaliQ defaults to Claude as judge, and runs faithfulness + relevance
tester = RagaliQ(judge="claude")

test_case = RAGTestCase(
    id="tutorial-1",
    name="Capital query",
    query="What is the capital of France?",
    context=[
        "France is a country in Western Europe.",
        "The capital city of France is Paris, which is home to the Eiffel Tower.",
    ],
    response="The capital of France is Paris.",
)

result = tester.evaluate(test_case)

print(f"Status:       {result.status}")
print(f"Faithfulness: {result.scores['faithfulness']:.2f}")
print(f"Relevance:    {result.scores['relevance']:.2f}")
print(f"Passed:       {result.passed}")
```

**What happens internally:**

1. `RagaliQ` initialises lazily on the first call — it creates a `ClaudeJudge` and instantiates the `FaithfulnessEvaluator` and `RelevanceEvaluator`.
2. Both evaluators run **in parallel** via `asyncio.gather`.
3. `FaithfulnessEvaluator` first calls the judge to extract atomic claims from the response, then verifies each claim against the context. Score = supported claims / total claims.
4. `RelevanceEvaluator` makes a single judge call to score how well the response answers the query.
5. If all metric scores meet the threshold (default 0.7), status is `PASSED`.

---

## 3. Understanding Results

`RAGTestResult` carries everything you need for reporting and debugging:

```python
result.status          # EvalStatus.PASSED | FAILED | ERROR | SKIPPED
result.passed          # bool — True when status == PASSED
result.scores          # {"faithfulness": 0.92, "relevance": 0.88}
result.execution_time_ms   # wall-clock time in ms
result.judge_tokens_used   # total tokens consumed across all evaluators

# Per-evaluator details (reasoning, raw judge response, errors)
faith_detail = result.details["faithfulness"]
print(faith_detail["reasoning"])   # Human-readable explanation
print(faith_detail["passed"])      # bool
print(faith_detail["raw"])         # Raw judge response dict for debugging
```

**Error handling:** If an evaluator fails (network error, bad API response, etc.), the result carries `status=ERROR` and the error is recorded in `details["evaluator_name"]["error"]`. The other evaluators in the batch still run — errors are isolated, not fatal.

---

## 4. Batch Evaluation

Run many test cases efficiently with `evaluate_batch()`:

```python
import json
from ragaliq import RagaliQ, RAGTestCase

tester = RagaliQ(
    judge="claude",
    evaluators=["faithfulness", "relevance", "hallucination"],
    default_threshold=0.75,
    max_concurrency=5,        # Up to 5 test cases run in parallel
)

test_cases = [
    RAGTestCase(id=f"tc-{i}", name=f"Query {i}", query="...", context=["..."], response="...")
    for i in range(20)
]

results = tester.evaluate_batch(test_cases)

passed = sum(1 for r in results if r.passed)
print(f"{passed}/{len(results)} passed")

# Inspect failures
for r in results:
    if not r.passed:
        for metric, score in r.scores.items():
            if score < 0.75:
                print(f"  [{r.test_case.name}] {metric}: {score:.2f}")
```

**Concurrency model:** `max_concurrency=5` means up to 5 test cases evaluate simultaneously. Each test case itself runs all its evaluators in parallel (bounded by `max_judge_concurrency=20`). For faithfulness, each claim is verified in parallel. This means a batch of 20 test cases makes many concurrent API calls — adjust `max_concurrency` based on your rate limits.

---

## 5. Choosing Evaluators

### Core Evaluators

```python
# Default: faithfulness + relevance
tester = RagaliQ(judge="claude")

# Override to specific evaluators
tester = RagaliQ(
    judge="claude",
    evaluators=["faithfulness", "relevance", "hallucination"],
)

# All five built-in evaluators
tester = RagaliQ(
    judge="claude",
    evaluators=["faithfulness", "relevance", "hallucination",
                "context_precision", "context_recall"],
)
```

### Evaluator Reference

**`faithfulness`** — Decomposes the response into atomic claims, verifies each claim against the context. Score = supported claims / total claims. Use when you need to ensure the response doesn't add facts beyond what the context contains.

**`relevance`** — Single judge call asking: "Does this response answer the query?" Score 0–1. Use when you need to ensure the response is on-topic.

**`hallucination`** — Same claim pipeline as faithfulness, but scored as 1 − (hallucinated / total). Threshold defaults to 0.8 (stricter). Use when the cost of made-up information is high.

**`context_precision`** — Evaluates whether the *retrieved context* is relevant to the query. Uses weighted rank-order precision (top-ranked docs contribute more). Measures retrieval quality, not response quality.

**`context_recall`** — Checks whether the context covers all `expected_facts`. Requires `test_case.expected_facts` to be set. Measures whether retrieval missed anything important.

```python
# context_recall requires expected_facts
test_case = RAGTestCase(
    id="recall-1",
    name="Facts check",
    query="Tell me about Python.",
    context=["Python was created by Guido van Rossum and released in 1991."],
    response="Python was created by Guido van Rossum in 1991.",
    expected_facts=["created by Guido van Rossum", "released in 1991"],
)
```

---

## 6. Dataset Files

### JSON Format

```json
{
  "version": "1.0",
  "metadata": {"source": "my-docs", "date": "2026-01"},
  "test_cases": [
    {
      "id": "tc-001",
      "name": "Python creation date",
      "query": "When was Python created?",
      "context": ["Python was created by Guido van Rossum and released in 1991."],
      "response": "Python was first released in 1991 by Guido van Rossum.",
      "expected_answer": "1991",
      "expected_facts": ["released in 1991", "created by Guido van Rossum"],
      "tags": ["python", "history"]
    }
  ]
}
```

### YAML Format

```yaml
version: "1.0"
test_cases:
  - id: tc-001
    name: Python creation date
    query: When was Python created?
    context:
      - Python was created by Guido van Rossum and released in 1991.
    response: Python was first released in 1991 by Guido van Rossum.
    tags:
      - python
      - history
```

### CSV Format

Required columns: `id`, `name`, `query`, `context`, `response`

```csv
id,name,query,context,response,tags
tc-001,Python date,When was Python created?,Python was released in 1991.,Python was created in 1991.,python|history
```

List fields (`context`, `expected_facts`, `tags`) use pipe-separated values: `doc1|doc2|doc3`.

### Loading Datasets

```python
from ragaliq.datasets import DatasetLoader

dataset = DatasetLoader.load("my_dataset.json")
print(f"Loaded {len(dataset.test_cases)} test cases")

results = tester.evaluate_batch(dataset.test_cases)
```

### Generating Test Datasets

Generate test cases automatically from your documentation:

```python
import asyncio
from ragaliq import TestCaseGenerator
from ragaliq.judges import ClaudeJudge

judge = ClaudeJudge()
generator = TestCaseGenerator()

# From in-memory documents
docs = [
    "Python is a high-level programming language created in 1991.",
    "Python supports multiple programming paradigms including OOP and functional.",
]
test_cases = asyncio.run(
    generator.generate_from_documents(documents=docs, n=5, judge=judge)
)

# Or via CLI (reads .txt files from a directory)
# ragaliq generate ./docs/ --num 50 --output dataset.json
```

Generated cases are tagged with `["generated"]` and given UUID-based IDs.

---

## 7. CLI Usage

### `ragaliq run`

```bash
# Basic run (faithfulness + relevance at threshold 0.7)
ragaliq run dataset.json

# Custom evaluators and threshold
ragaliq run dataset.json \
  --evaluator faithfulness \
  --evaluator hallucination \
  --threshold 0.8

# Export to JSON (CI-friendly)
ragaliq run dataset.json --output json --output-file report.json

# Export to HTML (shareable)
ragaliq run dataset.json --output html --output-file report.html

# Stop on first evaluator error (debug mode)
ragaliq run dataset.json --fail-fast
```

Exit code is `0` when all tests pass, `1` when any fail — integrates naturally with CI.

### `ragaliq generate`

```bash
# From a directory of .txt files
ragaliq generate ./docs/ --num 20 --output dataset.json

# From a single document
ragaliq generate document.txt --num 10 --output dataset.json
```

### `ragaliq validate`

```bash
# Check dataset schema without making any LLM calls
ragaliq validate dataset.json
```

### `ragaliq list-evaluators`

```bash
ragaliq list-evaluators
```

---

## 8. Pytest Integration

The pytest plugin is registered automatically when RagaliQ is installed. No `conftest.py` setup needed.

### Basic Test

```python
# test_rag.py
import pytest
from ragaliq import RAGTestCase
from ragaliq.integrations.pytest_plugin import assert_rag_quality


@pytest.mark.rag_test
def test_faithful_answer(rag_tester):
    """Use rag_tester fixture for direct evaluation."""
    test_case = RAGTestCase(
        id="pytest-1",
        name="Capital query",
        query="What is the capital of France?",
        context=["France is a country in Western Europe. Its capital city is Paris."],
        response="The capital of France is Paris.",
    )
    result = rag_tester.evaluate(test_case)
    assert result.passed, f"Quality check failed: {result.scores}"


@pytest.mark.rag_test
def test_with_helper(ragaliq_judge):
    """Use assert_rag_quality for inline assertion."""
    test_case = RAGTestCase(
        id="pytest-2",
        name="ML definition",
        query="What is machine learning?",
        context=["Machine learning is a subset of AI that enables systems to learn from data."],
        response="Machine learning enables systems to learn from data.",
    )
    # Raises AssertionError with failing metric names+scores if any metric < threshold
    assert_rag_quality(test_case, judge=ragaliq_judge, threshold=0.8)
```

### Custom Threshold Per Test

```python
@pytest.mark.rag_test
def test_strict_quality(rag_tester):
    """Use a stricter threshold for critical content."""
    from ragaliq import RagaliQ

    strict_tester = RagaliQ(judge=rag_tester._judge, default_threshold=0.9)
    test_case = RAGTestCase(
        id="pytest-strict-1",
        name="Medical query",
        query="What is the dosage?",
        context=["The recommended dosage is 500mg twice daily."],
        response="The dosage is 500mg twice per day.",
    )
    result = strict_tester.evaluate(test_case)
    assert result.passed, f"Strict check failed: {result.scores}"
```

### Running Tests

```bash
# Run with real API key
ANTHROPIC_API_KEY=sk-ant-... pytest tests/ -v

# Skip slow tests (marked with @pytest.mark.rag_slow)
pytest tests/ -m "not rag_slow"

# Limit spend per session
pytest tests/ --ragaliq-cost-limit 2.00

# Simulate latency (useful for testing resilience)
pytest tests/ --ragaliq-latency-ms 200

# Show RagaliQ stats in terminal output
pytest tests/ -v --ragaliq-judge claude
```

After the test session, RagaliQ appends a summary showing total API calls, tokens used, estimated cost, and any failures.

---

## 9. Reports

### Console Report (default)

```python
from ragaliq.reports import ConsoleReporter
from rich.console import Console

reporter = ConsoleReporter(threshold=0.7)
reporter.report(results)

# Verbose: show reasoning for all results, not just failures
reporter = ConsoleReporter(threshold=0.7, verbose=True)
reporter.report(results)
```

Renders a table of scores (green/red relative to threshold), failure reasoning, and a per-evaluator summary.

### JSON Report

```python
from ragaliq.reports import JSONReporter
from pathlib import Path

json_str = JSONReporter(threshold=0.7).export(results)
Path("report.json").write_text(json_str)
```

The JSON includes a `summary` block (total, passed, failed, pass_rate, per-evaluator stats) and a `results` array with full test case data. Suitable for downstream processing and CI artifact storage.

### HTML Report

```python
from ragaliq.reports import HTMLReporter
from pathlib import Path

html_str = HTMLReporter(threshold=0.7).export(results)
Path("report.html").write_text(html_str)
```

Self-contained HTML — no external dependencies or JavaScript. Shareable as a single file.

---

## 10. GitHub Actions CI/CD

RagaliQ auto-detects GitHub Actions and activates CI mode:

- **Rich spinner disabled** — clean logs, no animated widgets
- **Step summary** — Markdown results table appears in the workflow run UI
- **PR annotations** — failing test cases appear as `::error::` annotations on diffs
- **Step outputs** — `total`, `passed`, `failed`, `pass_rate` available to downstream steps

### Minimal Workflow

```yaml
name: RagaliQ Evaluation

on: [pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - run: pip install ragaliq
      - name: Run evaluations
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          ragaliq run dataset.json \
            --output json \
            --output-file report.json \
            --threshold 0.7
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: ragaliq-report
          path: report.json
```

See `examples/ci_cd_example/ragaliq-ci.yml` for the full example.

### Using CI Helpers Programmatically

```python
from ragaliq.integrations.github_actions import (
    is_github_actions,
    emit_ci_summary,
    write_step_summary,
    create_annotations,
)

if is_github_actions():
    emit_ci_summary(results, threshold=0.7)
    # Equivalent to:
    # write_step_summary(format_summary_markdown(results))
    # create_annotations(results)
    # set_output("passed", str(passed_count))
```

---

## 11. Advanced: Custom Evaluators

Custom evaluators register themselves with the global registry and become available everywhere — CLI, pytest fixtures, `RagaliQ(evaluators=[...])`.

```python
from ragaliq.evaluators import register_evaluator
from ragaliq.core.evaluator import Evaluator, EvaluationResult
from ragaliq.core.test_case import RAGTestCase
from ragaliq.judges.base import LLMJudge


@register_evaluator("conciseness")
class ConcisenessEvaluator(Evaluator):
    name = "conciseness"
    description = "Measures whether the response is appropriately brief"
    threshold = 0.7

    async def evaluate(self, test_case: RAGTestCase, judge: LLMJudge) -> EvaluationResult:
        # Use an existing judge capability
        result = await judge.evaluate_relevance(
            query=f"Is this response concise and to the point for the query: {test_case.query}",
            response=test_case.response,
        )
        return EvaluationResult(
            evaluator_name=self.name,
            score=result.score,
            passed=self.is_passing(result.score),
            reasoning=result.reasoning,
            tokens_used=result.tokens_used,
            raw_response={"score": result.score, "reasoning": result.reasoning},
        )


# Now available anywhere
from ragaliq import RagaliQ
tester = RagaliQ(judge="claude", evaluators=["faithfulness", "conciseness"])
```

**Key rules for custom evaluators:**
- Must be a subclass of `Evaluator`
- Must implement `async def evaluate(self, test_case, judge) -> EvaluationResult`
- `score` must be in `[0.0, 1.0]`
- Set `error` field in `EvaluationResult` for graceful failure handling
- Register before creating a `RagaliQ` instance that uses it

---

## 12. Advanced: JudgeConfig

Control which Claude model and generation parameters the judge uses:

```python
from ragaliq import RagaliQ
from ragaliq.judges import ClaudeJudge, JudgeConfig

# Via JudgeConfig
config = JudgeConfig(
    model="claude-sonnet-4-20250514",   # default
    temperature=0.0,                     # deterministic scoring
    max_tokens=1024,                     # response length cap
)
tester = RagaliQ(judge="claude", judge_config=config)

# Via pre-configured ClaudeJudge (gives access to trace_collector)
from ragaliq.judges.trace import TraceCollector
collector = TraceCollector()
judge = ClaudeJudge(config=config, trace_collector=collector)
tester = RagaliQ(judge=judge)

# After evaluation
print(f"Total tokens: {collector.total_tokens}")
print(f"Estimated cost: ${collector.total_cost_estimate:.4f}")
```

---

## 13. Advanced: Observability

`TraceCollector` records every judge API call with timing, token counts, and success/failure:

```python
from ragaliq.judges import ClaudeJudge
from ragaliq.judges.trace import TraceCollector

collector = TraceCollector()
judge = ClaudeJudge(trace_collector=collector)

from ragaliq import RagaliQ
tester = RagaliQ(judge=judge, evaluators=["faithfulness", "hallucination"])
results = tester.evaluate_batch(test_cases)

# Aggregate stats
print(f"API calls:      {len(collector.traces)}")
print(f"Total tokens:   {collector.total_tokens}")
print(f"Input tokens:   {collector.total_input_tokens}")
print(f"Output tokens:  {collector.total_output_tokens}")
print(f"Total latency:  {collector.total_latency_ms}ms")
print(f"Est. cost:      ${collector.total_cost_estimate:.4f}")
print(f"Failures:       {collector.failure_count}")

# Per-operation breakdown
verify_traces = collector.get_by_operation("verify_claim")
print(f"Claim verifications: {len(verify_traces)}")

# Inspect failures
for trace in collector.get_failures():
    print(f"  Failed: {trace.operation} — {trace.error}")
```

---

*For a complete runnable example, see `examples/basic_usage.py`.*
