# Getting Started with RagaliQ

This guide walks you through your first RagaliQ evaluation in under 5 minutes.

## Prerequisites

- Python 3.14+
- Anthropic API key

## Installation

```bash
pip install ragaliq
```

Set your API key:
```bash
export ANTHROPIC_API_KEY=your-key-here
```

## Your First Test

### Step 1: Create a Test Case

```python
# my_first_test.py
from ragaliq import RagaliQ, RAGTestCase

# Define what you want to test
test_case = RAGTestCase(
    id="test-1",
    name="Capital Question",
    query="What is the capital of France?",
    context=["Paris is the capital city of France. It is known for the Eiffel Tower."],
    response="The capital of France is Paris."
)
```

### Step 2: Run Evaluation

```python
import asyncio

async def main():
    # Initialize RagaliQ
    tester = RagaliQ(
        evaluators=["faithfulness", "relevance"],
        threshold=0.7
    )

    # Evaluate
    result = await tester.evaluate_async(test_case)

    # Check results
    print(f"Status: {result.status}")
    for eval_result in result.evaluations:
        print(f"  {eval_result.evaluator}: {eval_result.score:.2f}")

asyncio.run(main())
```

### Step 3: Run It

```bash
python my_first_test.py
```

Expected output:
```
Status: passed
  faithfulness: 1.00
  relevance: 0.95
```

## Using the CLI

Create a test dataset file:

```json
// tests.json
[
  {
    "id": "1",
    "name": "Capital Question",
    "query": "What is the capital of France?",
    "context": ["Paris is the capital of France."],
    "response": "The capital of France is Paris."
  }
]
```

Run evaluation:

```bash
ragaliq run tests.json
```

## Using with Pytest

```python
# test_my_rag.py
import pytest
from ragaliq import RAGTestCase
from ragaliq.integrations.pytest_plugin import assert_rag_quality

@pytest.mark.rag_test
def test_factual_response(rag_tester):
    test_case = RAGTestCase(
        id="1",
        name="Test",
        query="What is Python?",
        context=["Python is a programming language created in 1991."],
        response="Python is a programming language."
    )
    assert_rag_quality(rag_tester, test_case)
```

Run tests:
```bash
pytest -m rag_test
```

## Understanding Scores

| Score | Meaning |
|-------|---------|
| 0.9 - 1.0 | Excellent - response fully meets criteria |
| 0.7 - 0.9 | Good - minor issues, passes threshold |
| 0.4 - 0.7 | Needs improvement - significant gaps |
| 0.0 - 0.4 | Poor - major issues detected |

Default pass threshold is **0.7**.

## Available Evaluators

| Evaluator | What It Checks |
|-----------|----------------|
| `faithfulness` | Is response grounded in context only? |
| `relevance` | Does response answer the query? |
| `hallucination` | Any made-up facts not in context? |
| `context_precision` | Are retrieved docs relevant? |
| `context_recall` | Do docs cover needed information? |

## Next Steps

- [Full README](README.md) - Complete feature documentation
- [CLI Reference](README.md#cli-usage) - All command options
- [Pytest Integration](README.md#pytest-integration) - Testing patterns
