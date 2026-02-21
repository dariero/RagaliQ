# RagaliQ

**RAG + Quality** — A Testing Framework for LLM and RAG Systems

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RagaliQ brings software testing discipline to LLM and RAG systems. Write quality tests for your AI responses as naturally as you write unit tests — using pytest, the CLI, or the Python API.

---

## Why RagaliQ?

When you deploy a RAG (Retrieval-Augmented Generation) system, how do you know the answers are accurate? How do you catch hallucinations before your users do? How do you ensure your AI stays grounded in the retrieved documents?

RagaliQ answers these questions with a structured evaluation framework that uses an LLM-as-Judge to assess response quality — just like you would test any other software system.

---

## Features

- **5 Built-in Evaluators** — faithfulness, relevance, hallucination, context precision, context recall
- **LLM-as-Judge** — Claude evaluates response quality with semantic understanding, not keyword matching
- **Pytest Plugin** — native fixtures and markers for RAG tests alongside your unit tests
- **CLI** — run evaluations from the command line, generate test datasets from documents
- **Rich Reports** — console, HTML, and JSON reports built for CI/CD pipelines
- **GitHub Actions Integration** — automatic step summaries and PR annotations for failures
- **Async-First** — concurrent evaluations with configurable parallelism

---

## Installation

```bash
pip install ragaliq
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

---

## Quick Start

### Python API

```python
from ragaliq import RagaliQ, RAGTestCase

tester = RagaliQ(judge="claude")

test = RAGTestCase(
    id="test-1",
    name="Capital of France",
    query="What is the capital of France?",
    context=["France is a country in Western Europe. Its capital city is Paris."],
    response="The capital of France is Paris, known for the Eiffel Tower.",
)

result = tester.evaluate(test)
print(f"Faithfulness: {result.scores['faithfulness']:.2f}")
print(f"Relevance:    {result.scores['relevance']:.2f}")
print(f"Status:       {'PASSED' if result.passed else 'FAILED'}")
```

### Pytest Integration

The pytest plugin loads automatically when RagaliQ is installed. No imports needed for fixtures.

```python
# test_rag_quality.py
import pytest
from ragaliq import RAGTestCase
from ragaliq.integrations.pytest_plugin import assert_rag_quality


@pytest.mark.rag_test
def test_faithful_answer(rag_tester):
    test_case = RAGTestCase(
        id="t1",
        name="Capital of France",
        query="What is the capital of France?",
        context=["France is a country in Western Europe. Its capital city is Paris."],
        response="The capital of France is Paris.",
    )
    result = rag_tester.evaluate(test_case)
    assert result.passed, f"Quality check failed: {result.scores}"


@pytest.mark.rag_test
def test_with_helper(ragaliq_judge):
    test_case = RAGTestCase(
        id="t2",
        name="ML definition",
        query="What is machine learning?",
        context=["Machine learning is a subset of AI that enables systems to learn from data."],
        response="Machine learning is an AI technique that allows systems to improve from data.",
    )
    assert_rag_quality(test_case, judge=ragaliq_judge)
```

Run with:

```bash
ANTHROPIC_API_KEY=sk-ant-... pytest tests/ -v
```

### CLI

```bash
# Run evaluations against a dataset
ragaliq run dataset.json --evaluator faithfulness --evaluator relevance --threshold 0.8

# Generate a test dataset from documents
ragaliq generate ./docs/ --num 20 --output test_cases.json

# Validate a dataset file without running evaluations
ragaliq validate dataset.json

# List all available evaluators
ragaliq list-evaluators
```

---

## Evaluators

| Name | Measures | Default Threshold |
|---|---|---|
| `faithfulness` | Response grounded only in provided context | 0.7 |
| `relevance` | Response actually answers the query | 0.7 |
| `hallucination` | Response free from unsupported claims | **0.8** |
| `context_precision` | Retrieved documents are relevant to the query | 0.7 |
| `context_recall` | Context covers all expected facts (requires `expected_facts`) | 0.7 |

### Custom Evaluators

```python
from ragaliq.evaluators import register_evaluator
from ragaliq.core.evaluator import Evaluator, EvaluationResult
from ragaliq.core.test_case import RAGTestCase
from ragaliq.judges.base import LLMJudge


@register_evaluator("conciseness")
class ConcisenessEvaluator(Evaluator):
    name = "conciseness"
    description = "Measures whether the response is appropriately concise"
    threshold = 0.7

    async def evaluate(self, test_case: RAGTestCase, judge: LLMJudge) -> EvaluationResult:
        result = await judge.evaluate_relevance(
            query=test_case.query,
            response=test_case.response,
        )
        return EvaluationResult(
            evaluator_name=self.name,
            score=result.score,
            passed=self.is_passing(result.score),
            reasoning=result.reasoning,
            tokens_used=result.tokens_used,
        )
```

---

## Dataset Formats

RagaliQ accepts JSON, YAML, and CSV datasets. The JSON format is:

```json
{
  "version": "1.0",
  "test_cases": [
    {
      "id": "tc-1",
      "name": "Capital query",
      "query": "What is the capital of France?",
      "context": ["France is a country in Western Europe. Its capital is Paris."],
      "response": "The capital of France is Paris.",
      "expected_answer": "Paris",
      "expected_facts": ["capital is Paris"],
      "tags": ["geography"]
    }
  ]
}
```

Generate a dataset from your own documents:

```bash
ragaliq generate ./docs/ --num 50 --output dataset.json
```

Or programmatically:

```python
import asyncio
from ragaliq import TestCaseGenerator
from ragaliq.judges import ClaudeJudge

judge = ClaudeJudge()
generator = TestCaseGenerator()
test_cases = asyncio.run(
    generator.generate_from_documents(documents=["..."], n=10, judge=judge)
)
```

---

## Reports

### Console

```python
from ragaliq.reports import ConsoleReporter
ConsoleReporter(threshold=0.7).report(results)
```

### JSON

```python
from ragaliq.reports import JSONReporter
json_str = JSONReporter(threshold=0.7).export(results)
```

### HTML

```python
from ragaliq.reports import HTMLReporter
html_str = HTMLReporter(threshold=0.7).export(results)
```

Via CLI:

```bash
ragaliq run dataset.json --output html --output-file report.html
ragaliq run dataset.json --output json --output-file report.json
```

---

## Pytest Plugin Reference

### Fixtures

| Fixture | Scope | Description |
|---|---|---|
| `rag_tester` | function | Pre-configured `RagaliQ` runner using the session judge |
| `ragaliq_judge` | session | Shared `LLMJudge` instance configured from CLI options |
| `ragaliq_runner` | function | Alias for `rag_tester` |
| `ragaliq_trace_collector` | session | Tracks token usage and cost across the session |

### `assert_rag_quality` Helper

```python
from ragaliq.integrations.pytest_plugin import assert_rag_quality

assert_rag_quality(
    test_case,
    judge=ragaliq_judge,        # optional — creates default ClaudeJudge if omitted
    evaluators=["faithfulness"], # optional — defaults to ["faithfulness", "relevance"]
    threshold=0.8,               # optional — defaults to 0.7
)
```

Raises `AssertionError` with failing metric names and scores if any metric falls below the threshold.

### Markers

```python
@pytest.mark.rag_test     # Mark as RAG quality test
@pytest.mark.rag_slow     # Skip with: pytest -m "not rag_slow"
```

### CLI Options

```bash
pytest --ragaliq-judge claude \
       --ragaliq-model claude-sonnet-4-6 \
       --ragaliq-api-key sk-ant-... \
       --ragaliq-cost-limit 5.00 \
       --ragaliq-latency-ms 100
```

For complex multi-step or gold-standard judging flows, use
`--ragaliq-model claude-opus-4-6`.

---

## GitHub Actions Integration

RagaliQ auto-detects GitHub Actions and enables:

- **Step summaries** — Markdown results table in the Actions run UI
- **PR annotations** — `::error::` annotations on failing test cases
- **Step outputs** — `total`, `passed`, `failed`, `pass_rate` for downstream steps
- **Clean logs** — Rich spinner disabled, plain text output

```yaml
# .github/workflows/ragaliq-ci.yml
- name: Run evaluations
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: ragaliq run dataset.json --output json --output-file report.json

- name: Upload report
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: ragaliq-report
    path: report.json
```

See `examples/ci_cd_example/ragaliq-ci.yml` for a complete workflow.

---

## Architecture

```
src/ragaliq/
├── core/           # RAGTestCase, Evaluator base, RagaliQ runner
├── evaluators/     # Faithfulness, Relevance, Hallucination, ContextPrecision, ContextRecall
├── judges/         # ClaudeJudge, LLMJudge ABC, JudgeConfig, TraceCollector
├── datasets/       # DatasetLoader (JSON/YAML/CSV), TestCaseGenerator
├── reports/        # ConsoleReporter, HTMLReporter, JSONReporter
├── integrations/   # Pytest plugin, GitHub Actions helpers
└── cli/            # Typer CLI (run, generate, validate, list-evaluators)
```

---

## Development

```bash
git clone https://github.com/dariero/RagaliQ.git
cd RagaliQ

pip install hatch
hatch run test          # pytest + coverage
hatch run lint          # ruff check
hatch run format        # ruff format + auto-fix
hatch run typecheck     # mypy
```

---

## Documentation

- [Tutorial](docs/TUTORIAL.md) — Full walkthrough from install to CI/CD
- [Examples](examples/) — Runnable scripts and pytest examples
- [Architecture Decisions](.decisions/) — Design rationale

---

## Why "RagaliQ"?

**RAG** (Retrieval-Augmented Generation) + **Quality** = **RagaliQ**

Because quality matters when building AI systems that people rely on.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
