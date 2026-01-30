# RagaliQ

**RAG + Quality** - A Testing Framework for LLM and RAG Systems

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RagaliQ brings software testing discipline to LLM and RAG systems. Write quality tests for your AI responses as naturally as you write unit tests.

> **New to RagaliQ?** Check out [GETTING_STARTED.md](GETTING_STARTED.md) for a 5-minute quickstart.

---

## Why RagaliQ?

When you deploy a RAG (Retrieval-Augmented Generation) system, how do you know the answers are accurate? How do you catch hallucinations before your users do? How do you ensure your AI stays grounded in the retrieved documents?

RagaliQ answers these questions by providing a structured testing framework that evaluates LLM responses against multiple quality dimensions - just like you would test any other software system.

---

## Features

- **Multiple Evaluators**: Test for faithfulness, relevance, hallucination, context precision, and more
- **LLM-as-Judge**: Uses Claude or GPT to intelligently assess response quality (not just keyword matching)
- **Pytest Integration**: Native pytest plugin for seamless integration into existing test workflows
- **Rich Reports**: Console, HTML, and JSON reports suitable for CI/CD pipelines
- **Easy CLI**: Run tests from the command line with a single command
- **Async-First**: Designed for performance with async LLM calls

---

## Installation

```bash
pip install ragaliq
```

---

## Quick Start

### Python API

```python
from ragaliq import RagaliQ, RAGTestCase

# Initialize with Claude as judge
tester = RagaliQ(judge="claude")

# Create a test case
test = RAGTestCase(
    id="test_1",
    name="Capital of France",
    query="What is the capital of France?",
    context=["France is a country in Western Europe. Its capital city is Paris."],
    response="The capital of France is Paris, which is known for the Eiffel Tower."
)

# Run evaluation
result = tester.evaluate(test)

print(f"Faithfulness: {result.scores['faithfulness']:.2f}")
print(f"Relevance: {result.scores['relevance']:.2f}")
print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
```

### Pytest Integration

```python
# test_my_rag.py
import pytest
from ragaliq.pytest import rag_tester, assert_rag_quality

def test_factual_response(rag_tester):
    query = "When was Python created?"
    context = ["Python was created by Guido van Rossum and first released in 1991."]
    response = "Python was created in 1991 by Guido van Rossum."

    assert_rag_quality(
        rag_tester,
        query=query,
        context=context,
        response=response,
        min_faithfulness=0.9,
        min_relevance=0.8
    )
```

### CLI Usage

```bash
# Run tests from a dataset file
ragaliq run tests/qa_dataset.json --eval faithfulness --eval relevance

# Generate test cases from documents
ragaliq generate ./docs/ --count 50 --output tests.json

# Quick single test
ragaliq test \
    --query "What is RAG?" \
    --context "RAG combines retrieval with generation..." \
    --response "RAG is a technique..."
```

---

## Core Concepts

### Test Cases

A test case in RagaliQ represents a single RAG interaction to evaluate:

| Field | Description |
|-------|-------------|
| `query` | The user question or input |
| `context` | List of retrieved documents/chunks |
| `response` | The LLM-generated response to evaluate |
| `expected_answer` | Optional ground truth for comparison |
| `expected_facts` | Optional list of facts that should appear |

### Evaluators

Evaluators are the heart of RagaliQ. Each evaluator assesses a specific quality dimension:

| Evaluator | What It Measures |
|-----------|------------------|
| `faithfulness` | Is the response grounded only in the provided context? |
| `relevance` | Does the response actually answer the question? |
| `hallucination` | Does the response contain made-up information? |
| `context_precision` | Are the retrieved documents relevant to the query? |
| `context_recall` | Do retrieved documents cover all aspects of the answer? |

### LLM-as-Judge

Rather than relying on keyword matching or hardcoded rules, RagaliQ uses an LLM (Claude or OpenAI) as a judge to assess response quality. This approach:

- Understands semantic meaning, not just lexical similarity
- Can reason about whether claims are supported by context
- Provides human-readable explanations for its scores
- Handles nuanced cases that rule-based systems miss

---

## Architecture

```
src/ragaliq/
    core/           # TestCase, Evaluator base, Runner
    evaluators/     # Faithfulness, Relevance, Hallucination, etc.
    judges/         # LLM judge implementations (Claude, OpenAI)
    datasets/       # Test data loading and generation
    reports/        # Console, HTML, JSON reporters
    integrations/   # Pytest plugin, CI helpers
    cli/            # Typer CLI commands
```

---

## Configuration

### Environment Variables

Set your API key for the LLM judge:

```bash
export ANTHROPIC_API_KEY=your-key-here
# or
export OPENAI_API_KEY=your-key-here
```

### Configuration File

Create a `ragaliq.yaml` in your project root:

```yaml
judge: claude
model: claude-opus-4-5-20251101
evaluators:
  - faithfulness
  - relevance
  - hallucination
thresholds:
  faithfulness: 0.8
  relevance: 0.7
```

---

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/dariero/RagaliQ.git
cd ragaliq

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### Commands

```bash
make install        # Install in production mode
make install-dev    # Install with dev dependencies
make test           # Run tests with coverage
make test-fast      # Run tests without coverage
make lint           # Check code style with ruff
make format         # Format code with ruff
make typecheck      # Run mypy type checking
make clean          # Remove build artifacts
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`) and linting (`make lint`)
5. Commit your changes with a descriptive message
6. Push to your branch
7. Open a Pull Request

### Code Style

- We use `ruff` for linting and formatting
- Type hints are required for all public functions
- Docstrings should follow Google style
- Tests go in `tests/` mirroring the `src/` structure

---

## Roadmap

- [x] Core models (RAGTestCase, RAGTestResult, Evaluator)
- [x] Claude judge integration (LLMJudge base + ClaudeJudge implementation)
- [ ] Core evaluators (faithfulness, relevance, hallucination)
- [ ] RAG-specific evaluators (context precision, context recall)
- [ ] CLI with Typer
- [ ] Pytest plugin
- [ ] HTML/JSON reports
- [ ] Dataset generation from documents

---

## Why "RagaliQ"?

**RAG** (Retrieval-Augmented Generation) + **Quality** = **RagaliQ**

Because quality matters when building AI systems that people rely on.

---

## Documentation

- [Project Plan](docs/PROJECT_PLAN.md) - Implementation roadmap and milestones

---

## Author

Created by Darie Ro

---

## License

MIT License - see [LICENSE](LICENSE) for details.
