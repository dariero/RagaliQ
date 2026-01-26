# add-test-pattern

## Purpose
Add pytest-style test patterns and assertion helpers for RAG/LLM testing. Creates reusable fixtures, markers, and assertion functions that make testing AI systems feel like testing regular software.

## Usage
Invoke when:
- Creating new pytest fixtures for RAG testing scenarios
- Adding custom assertion helpers (e.g., `assert_no_hallucination`)
- Implementing test markers for filtering AI tests
- Building parametrized test generators

## Automated Steps

1. **Analyze existing test patterns**
   - Review `tests/conftest.py` for current fixtures
   - Check `src/ragaliq/integrations/pytest_plugin.py`
   - Understand existing markers and helpers

2. **Implement new pattern**

   For **fixtures**:
   ```
   tests/conftest.py or src/ragaliq/integrations/pytest_plugin.py
   ```

   For **assertion helpers**:
   ```
   src/ragaliq/integrations/assertions.py
   ```

   For **markers**:
   ```
   src/ragaliq/integrations/pytest_plugin.py (pytest_configure)
   ```

3. **Create example tests demonstrating pattern**
   ```
   examples/pytest_example/test_{pattern}.py
   ```

4. **Add documentation**
   - Document in docstrings
   - Add to pytest integration section in README

5. **Run verification**
   ```bash
   make test && pytest examples/pytest_example/ --collect-only
   ```

## Domain Expertise Applied

### RAG Testing Patterns

**1. Fixture Patterns**
```python
@pytest.fixture
def rag_tester() -> RagaliQ:
    """Pre-configured RagaliQ instance."""
    return RagaliQ(evaluators=["faithfulness", "relevance"])

@pytest.fixture
def sample_context() -> list[str]:
    """Reusable context documents for testing."""
    return [...]

@pytest.fixture(params=["faithful", "hallucinating", "irrelevant"])
def test_case_type(request) -> RAGTestCase:
    """Parametrized fixture for different response types."""
    ...
```

**2. Assertion Patterns**
```python
def assert_faithful(
    tester: RagaliQ,
    response: str,
    context: list[str],
    threshold: float = 0.7
) -> None:
    """Assert response is grounded in context."""
    test_case = RAGTestCase(query="", context=context, response=response, ...)
    result = asyncio.run(tester.evaluate_async(test_case))
    faithfulness = next(e for e in result.evaluations if e.evaluator == "faithfulness")
    assert faithfulness.score >= threshold, f"Faithfulness {faithfulness.score} < {threshold}"

def assert_no_hallucination(tester: RagaliQ, test_case: RAGTestCase) -> None:
    """Assert response contains no hallucinated claims."""
    ...

def assert_answers_query(tester: RagaliQ, query: str, response: str) -> None:
    """Assert response is relevant to query."""
    ...
```

**3. Marker Patterns**
```python
# In pytest_configure:
config.addinivalue_line("markers", "rag_slow: requires LLM API calls")
config.addinivalue_line("markers", "rag_unit: fast, mocked tests")
config.addinivalue_line("markers", "evaluator(name): test specific evaluator")

# Usage:
@pytest.mark.rag_slow
@pytest.mark.evaluator("faithfulness")
def test_faithfulness_edge_case():
    ...
```

**4. Parametrized Test Patterns**
```python
@pytest.mark.parametrize("response,expected_faithful", [
    ("Paris is the capital of France.", True),
    ("Paris has 50 million people.", False),  # Hallucination
])
def test_faithfulness_detection(rag_tester, response, expected_faithful):
    ...
```

### Best Practices for AI Testing
- **Determinism**: Use mocked judges for unit tests
- **Isolation**: Each test should create its own test case
- **Speed**: Mark slow tests, run fast tests in CI
- **Clarity**: Assertion messages should explain what failed and why
- **Coverage**: Test edge cases (empty context, long responses, etc.)

### Pitfalls to Avoid
- Don't make all tests hit real APIs - use mocks
- Don't hardcode scores - they vary between runs
- Don't forget async handling in fixtures
- Don't skip negative test cases (expected failures)

## Interactive Prompts

**Ask for:**
- Pattern type: fixture / assertion / marker / parametrized?
- Pattern name and purpose
- What RAG scenarios does this address?
- Should this be in conftest.py or pytest_plugin.py?

**Suggest:**
- Similar existing patterns to extend
- Appropriate scope (function/module/session)
- Whether to include in public API

**Validate:**
- Pattern is reusable across different tests
- Edge cases are considered
- Documentation is clear

## Success Criteria
- [ ] Pattern implemented with type hints
- [ ] Example test demonstrates usage
- [ ] Works with `pytest --collect-only`
- [ ] Documented with docstrings
- [ ] `make test` passes
- [ ] Pattern is genuinely reusable
