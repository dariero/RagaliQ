# add-docs-example

## Purpose
Generate documentation with real-world usage examples for RagaliQ. Create tutorials, API documentation, and integration guides that help users understand and adopt the framework.

## Usage
Invoke when:
- Writing tutorials for new features
- Creating API reference documentation
- Building integration guides (pytest, LangChain, CI/CD)
- Adding runnable code examples
- Documenting evaluator interpretation

## Automated Steps

1. **Analyze documentation needs**
   - Check existing docs in `docs/`
   - Review README.md structure
   - Identify undocumented features

2. **Create documentation**
   ```
   docs/
   ├── TUTORIAL.md           # Getting started guide
   ├── API.md                 # API reference
   ├── EVALUATORS.md          # Evaluator deep-dive
   ├── INTEGRATIONS.md        # Integration guides
   └── EXAMPLES.md            # Example index
   ```

3. **Create runnable examples**
   ```
   examples/
   ├── basic_usage.py
   ├── custom_evaluator.py
   ├── pytest_example/
   ├── langchain_example/
   └── ci_cd_example/
   ```

4. **Validate examples**
   - Run all code snippets
   - Verify outputs match documentation
   - Test with current API

5. **Update README**
   - Link to detailed docs
   - Add quick start section
   - Include badges and status

## Domain Expertise Applied

### Documentation Patterns

**1. Tutorial Structure**
```markdown
# Testing Your First RAG System with RagaliQ

## What You'll Learn
- How to create RAG test cases
- How to run evaluations with RagaliQ
- How to interpret evaluation results
- How to integrate with pytest

## Prerequisites
- Python 3.14+
- Anthropic API key
- Basic understanding of RAG systems

## Step 1: Installation
pip install ragaliq

## Step 2: Create Your First Test Case
[Code example with explanation]

## Step 3: Run Evaluation
[Code example with expected output]

## Step 4: Interpret Results
[Explanation of scores and what they mean]

## Next Steps
- [Custom Evaluators](./custom_evaluators.md)
- [Pytest Integration](./pytest_integration.md)
- [CI/CD Setup](./cicd.md)
```

**2. API Reference Format**
```markdown
# API Reference

## Core Classes

### RagaliQ

Main entry point for running evaluations.

**Constructor**
python
RagaliQ(
    evaluators: list[str] | None = None,
    threshold: float = 0.7,
    api_key: str | None = None
)


**Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| evaluators | list[str] | ["faithfulness", "relevance", "hallucination"] | Evaluators to run |
| threshold | float | 0.7 | Minimum passing score |
| api_key | str | None | Anthropic API key (or use env var) |

**Methods**

#### evaluate_async(test_case: RAGTestCase) -> RAGTestResult
Evaluate a single test case asynchronously.

python
result = await tester.evaluate_async(test_case)
print(f"Status: {result.status}")  # "passed" or "failed"


#### evaluate_batch_async(test_cases: list[RAGTestCase]) -> list[RAGTestResult]
Evaluate multiple test cases in parallel.
```

**3. Evaluator Documentation**
```markdown
# Evaluators

## Faithfulness Evaluator

**What it measures**: Whether the response is grounded only in the provided context.

**How it works**:
1. Extracts factual claims from the response
2. Verifies each claim against the context
3. Score = (supported claims) / (total claims)

**Score interpretation**:
| Score | Meaning |
|-------|---------|
| 1.0 | All claims are supported by context |
| 0.7-0.9 | Most claims supported, minor unsupported details |
| 0.4-0.6 | Significant unsupported claims |
| 0.0-0.3 | Response largely not grounded in context |

**When to use**: Always include for RAG systems where factual accuracy is critical.

**Example**:
python
# Good: Response grounded in context
test_case = RAGTestCase(
    query="What is the capital of France?",
    context=["Paris is the capital of France."],
    response="The capital of France is Paris."
)
# Expected faithfulness score: ~1.0

# Bad: Response has unsupported claims
test_case = RAGTestCase(
    query="What is the capital of France?",
    context=["Paris is the capital of France."],
    response="Paris is the capital of France and has 10 million people."
)
# Expected faithfulness score: ~0.5 (population claim unsupported)

```

**4. Runnable Examples**
```python
# examples/basic_usage.py
"""
Basic RagaliQ Usage Example

This example demonstrates how to:
1. Create RAG test cases
2. Run evaluations
3. Interpret results

Requirements:
- pip install ragaliq
- export ANTHROPIC_API_KEY=your-key
"""

import asyncio
from ragaliq import RagaliQ, RAGTestCase

async def main():
    # Create a test case
    test_case = RAGTestCase(
        id="example-1",
        name="Capital of France",
        query="What is the capital of France?",
        context=[
            "Paris is the capital and largest city of France.",
            "France is a country in Western Europe."
        ],
        response="The capital of France is Paris."
    )

    # Initialize RagaliQ
    tester = RagaliQ(
        evaluators=["faithfulness", "relevance"],
        threshold=0.7
    )

    # Run evaluation
    result = await tester.evaluate_async(test_case)

    # Print results
    print(f"Test: {result.test_case.name}")
    print(f"Status: {result.status}")
    print("\nEvaluation Scores:")
    for eval_result in result.evaluations:
        status = "PASS" if eval_result.passed else "FAIL"
        print(f"  {eval_result.evaluator}: {eval_result.score:.2f} [{status}]")

if __name__ == "__main__":
    asyncio.run(main())

# Expected output:
# Test: Capital of France
# Status: passed
#
# Evaluation Scores:
#   faithfulness: 1.00 [PASS]
#   relevance: 0.95 [PASS]
```

**5. README Quick Start**
```markdown
## Quick Start

### 1. Install
pip install ragaliq

### 2. Create test file
# tests/my_rag_tests.json
[
  {
    "id": "1",
    "name": "Test query",
    "query": "What is X?",
    "context": ["X is Y."],
    "response": "X is Y."
  }
]

### 3. Run
ragaliq run tests/my_rag_tests.json

### 4. Review results
  RagaliQ Evaluation Report
  Tests: 1 | Passed: 1 | Failed: 0 | Pass Rate: 100%
```

### Documentation Best Practices
- **Runnable code**: Every example should be copy-paste runnable
- **Expected output**: Show what users should expect to see
- **Progressive complexity**: Start simple, build up
- **Real scenarios**: Use realistic RAG examples
- **Error cases**: Document common errors and solutions

### Pitfalls to Avoid
- Don't assume knowledge - explain RAG basics if needed
- Don't use placeholder values - use realistic examples
- Don't skip error handling - show robust patterns
- Don't let docs get stale - test examples in CI

## Interactive Prompts

**Ask for:**
- What type of documentation? (tutorial, API, example)
- Target audience? (beginner, experienced, enterprise)
- Which feature to document?
- Preferred format? (markdown, docstrings, notebooks)

**Suggest:**
- Documentation structure
- Example scenarios
- Cross-references to related docs

**Validate:**
- Code examples run successfully
- Documentation is complete
- Links are valid

## Success Criteria
- [ ] Documentation created in appropriate location
- [ ] All code examples are runnable
- [ ] Expected outputs documented
- [ ] Linked from README or index
- [ ] Reviewed for accuracy
- [ ] No broken links
- [ ] Follows project style
