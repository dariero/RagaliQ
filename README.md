# RagaliQ: The Ultimate LLM & RAG Evaluation Testing Framework

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/ragaliq.svg)](https://pypi.org/project/ragaliq/)

**RagaliQ** (**RAG** + **Quality**) is an open-source LLM/RAG testing toolkit that brings software testing discipline to Retrieval-Augmented Generation pipelines. It provides automated **hallucination detection**, **faithfulness metrics**, **answer relevance scoring**, **context precision**, and **context recall** evaluation — all powered by an LLM-as-Judge architecture. Write quality tests for your AI responses as naturally as you write unit tests with pytest.

---

## Why RagaliQ?

When you deploy a RAG system, how do you know the answers are accurate? How do you catch hallucinations before your users do? How do you ensure your retrieval pipeline returns the right documents?

Traditional keyword-matching approaches miss semantic errors. RagaliQ solves this with **LLM-as-Judge evaluation**: Claude (or OpenAI) assesses response quality with deep semantic understanding, scoring each response across multiple evaluation metrics. This is the same approach used in academic LLM benchmarking — now available as a developer-friendly testing framework.

---

## Key Features

| Capability | What It Does | How It Helps |
|---|---|---|
| **Hallucination Detection** | Identifies claims not supported by retrieved context | Catches fabricated facts before users see them |
| **Faithfulness Metrics** | Multi-step claim extraction and verification against source documents | Ensures responses stay grounded in your data |
| **Answer Relevance Scoring** | Evaluates whether the response actually answers the user's query | Prevents off-topic or evasive answers |
| **Context Precision** | Measures whether retrieved documents are relevant to the query | Audits your vector database retrieval quality |
| **Context Recall** | Verifies that context covers all expected facts | Validates your embedding similarity and retrieval coverage |
| **Pytest Plugin** | Native fixtures, markers, and assert helpers | RAG tests run alongside your existing unit tests |
| **CLI & CI/CD** | Command-line interface with GitHub Actions integration | Automated quality gates in your deployment pipeline |
| **Async-First** | Concurrent evaluations with configurable parallelism | Fast evaluation even with large test datasets |
| **Rich Reports** | Console, HTML, and JSON output formats | Actionable results for developers and stakeholders |

---

## Installation

```bash
pip install ragaliq
```

Set your API key:

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

The pytest plugin loads automatically when RagaliQ is installed. No configuration needed.

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

## Evaluation Metrics

RagaliQ ships with five built-in evaluators for comprehensive RAG pipeline testing:

| Evaluator | Measures | Default Threshold |
|---|---|---|
| `faithfulness` | Response grounded only in provided context | 0.7 |
| `relevance` | Response actually answers the query | 0.7 |
| `hallucination` | Response free from unsupported claims | **0.8** |
| `context_precision` | Retrieved documents are relevant to the query | 0.7 |
| `context_recall` | Context covers all expected facts (requires `expected_facts`) | 0.7 |

### Custom Evaluators

Extend RagaliQ with your own evaluation metrics using the evaluator registry:

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

RagaliQ accepts JSON, YAML, and CSV test datasets:

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

Generate a test dataset from your own documents:

```bash
ragaliq generate ./docs/ --num 50 --output dataset.json
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

## GitHub Actions Integration

RagaliQ auto-detects GitHub Actions and enables:

- **Step summaries** — Markdown results table in the Actions run UI
- **PR annotations** — `::error::` annotations on failing test cases
- **Step outputs** — `total`, `passed`, `failed`, `pass_rate` for downstream steps
- **Clean logs** — Rich spinner disabled, plain text output

```yaml
# .github/workflows/ragaliq-ci.yml
- name: Run RAG evaluations
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
- [Changelog](CHANGELOG.md) — Release history and updates

---

## Comparison with Alternatives

| Feature | RagaliQ | RAGAS | DeepEval |
|---|---|---|---|
| Pytest-native integration | Yes | No | Partial |
| LLM-as-Judge (Claude) | Yes | No | Yes |
| CLI with dataset generation | Yes | No | Yes |
| GitHub Actions integration | Yes | No | No |
| Async-first architecture | Yes | Partial | No |
| Custom evaluator registry | Yes | Yes | Yes |
| HTML/JSON reporting | Yes | No | Yes |
| Open source (MIT) | Yes | Yes | Partial |

---

## Why "RagaliQ"?

**RAG** (Retrieval-Augmented Generation) + **Quality** = **RagaliQ**

Because answer correctness matters when building AI systems that people rely on. RagaliQ helps you audit your retrieval pipeline, detect hallucinations, and ship with confidence.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
