# integrate-langchain

## Purpose
Add LangChain integration to RagaliQ. Enable seamless testing of LangChain chains, agents, and retrievers within the RagaliQ evaluation framework.

## Usage
Invoke when:
- Creating adapters for LangChain chains to RagaliQ test cases
- Building evaluators that work with LangChain memory/history
- Integrating with LangChain's retriever interface
- Adding LangChain callback handlers for evaluation

## Automated Steps

1. **Analyze integration points**
   - Review LangChain Chain/Agent/Retriever interfaces
   - Map LangChain outputs to RAGTestCase fields
   - Identify callback opportunities

2. **Create adapter module**
   ```
   src/ragaliq/integrations/langchain/
   ├── __init__.py
   ├── adapters.py      # Chain -> RAGTestCase converters
   ├── callbacks.py     # LangChain callbacks for auto-evaluation
   ├── retrievers.py    # Retriever evaluation helpers
   └── testing.py       # Test utilities
   ```

3. **Implement adapters**
   - Chain output to RAGTestCase
   - Retriever results to context
   - Memory to conversation history

4. **Create callback handler**
   - Auto-capture chain inputs/outputs
   - Evaluate on chain completion
   - Log results to RagaliQ

5. **Add tests**
   ```
   tests/unit/test_langchain_integration.py
   tests/integration/test_langchain_chains.py
   ```

6. **Document integration**
   - Add LangChain section to README
   - Create example in `examples/langchain_example/`

## Domain Expertise Applied

### LangChain Adapter Patterns

**1. Chain to RAGTestCase Adapter**
```python
# src/ragaliq/integrations/langchain/adapters.py
from langchain.chains.base import Chain
from langchain.schema import Document
from ragaliq.core import RAGTestCase

class ChainAdapter:
    """Convert LangChain chain execution to RAGTestCase."""

    @staticmethod
    def from_chain_result(
        chain: Chain,
        inputs: dict,
        outputs: dict,
        retrieved_docs: list[Document] | None = None
    ) -> RAGTestCase:
        """Create RAGTestCase from chain execution."""
        # Extract query from inputs
        query = inputs.get("question") or inputs.get("query") or inputs.get("input", "")

        # Extract response from outputs
        response = outputs.get("answer") or outputs.get("result") or outputs.get("output", "")

        # Convert Documents to context strings
        context = []
        if retrieved_docs:
            context = [doc.page_content for doc in retrieved_docs]

        return RAGTestCase(
            id=f"langchain-{hash(query)}",
            name=f"LangChain: {query[:50]}...",
            query=query,
            context=context,
            response=response,
            metadata={"chain_type": chain.__class__.__name__}
        )
```

**2. Retriever Evaluation Helper**
```python
from langchain.schema import BaseRetriever

class RetrieverEvaluator:
    """Evaluate LangChain retriever quality."""

    def __init__(self, retriever: BaseRetriever, tester: RagaliQ):
        self.retriever = retriever
        self.tester = tester

    async def evaluate_retrieval(
        self,
        query: str,
        expected_facts: list[str] | None = None
    ) -> EvaluationResult:
        """Evaluate retriever's context precision and recall."""
        docs = await self.retriever.aget_relevant_documents(query)

        test_case = RAGTestCase(
            id=f"retrieval-{hash(query)}",
            name=f"Retrieval: {query[:50]}",
            query=query,
            context=[doc.page_content for doc in docs],
            response="",  # No response for retrieval-only eval
            expected_facts=expected_facts
        )

        return await self.tester.evaluate_async(
            test_case,
            evaluators=["context_precision", "context_recall"]
        )
```

**3. LangChain Callback Handler**
```python
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any

class RagaliQCallbackHandler(BaseCallbackHandler):
    """Auto-evaluate LangChain chain outputs."""

    def __init__(self, tester: RagaliQ, threshold: float = 0.7):
        self.tester = tester
        self.threshold = threshold
        self.results: list[RAGTestResult] = []
        self._current_inputs: dict = {}
        self._retrieved_docs: list[Document] = []

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        self._current_inputs = inputs
        self._retrieved_docs = []

    def on_retriever_end(self, documents: list[Document], **kwargs):
        self._retrieved_docs.extend(documents)

    def on_chain_end(self, outputs: dict, **kwargs):
        test_case = ChainAdapter.from_chain_result(
            chain=kwargs.get("chain"),
            inputs=self._current_inputs,
            outputs=outputs,
            retrieved_docs=self._retrieved_docs
        )

        # Run evaluation (sync wrapper for callback)
        import asyncio
        result = asyncio.run(self.tester.evaluate_async(test_case))
        self.results.append(result)

        # Warn if below threshold
        if result.status == "failed":
            import warnings
            warnings.warn(f"RAG quality below threshold for: {test_case.name}")

# Usage:
from langchain.chains import RetrievalQA

callback = RagaliQCallbackHandler(RagaliQ())
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
result = chain.invoke({"query": "What is X?"}, callbacks=[callback])
print(callback.results)  # Evaluation results
```

**4. Testing Utilities**
```python
# src/ragaliq/integrations/langchain/testing.py
import pytest

@pytest.fixture
def mock_retriever():
    """Create mock retriever for testing."""
    from langchain.schema import Document
    from unittest.mock import AsyncMock

    retriever = AsyncMock()
    retriever.aget_relevant_documents.return_value = [
        Document(page_content="Test context 1"),
        Document(page_content="Test context 2"),
    ]
    return retriever

def assert_chain_quality(
    chain: Chain,
    test_inputs: list[dict],
    tester: RagaliQ,
    threshold: float = 0.7
) -> list[RAGTestResult]:
    """Assert chain outputs meet quality threshold."""
    results = []
    for inputs in test_inputs:
        outputs = chain.invoke(inputs)
        test_case = ChainAdapter.from_chain_result(chain, inputs, outputs)
        result = asyncio.run(tester.evaluate_async(test_case))
        results.append(result)

        assert result.status == "passed", (
            f"Chain failed quality check: {result.summary()}"
        )
    return results
```

### Integration Best Practices
- **Lazy imports**: Import LangChain only when integration is used
- **Version compatibility**: Support LangChain 0.1.x and 0.2.x
- **Optional dependency**: Don't require LangChain for core RagaliQ
- **Async support**: Use async methods when available

### Pitfalls to Avoid
- Don't assume chain input/output key names - make configurable
- Don't block event loop in callbacks - use sync wrappers carefully
- Don't forget memory/history in conversational chains
- Don't ignore streaming outputs - handle incrementally

## Interactive Prompts

**Ask for:**
- Which LangChain components to integrate?
- Chain types to support (QA, conversational, agent)?
- Input/output field mappings?
- Callback vs. explicit evaluation?

**Suggest:**
- Appropriate adapter patterns
- Field mapping defaults
- Testing approach

**Validate:**
- Works with target LangChain version
- Handles async correctly
- Optional dependency properly managed

## Success Criteria
- [ ] Adapters convert Chain outputs to RAGTestCase
- [ ] Callback handler auto-captures evaluations
- [ ] Retriever evaluation works
- [ ] LangChain is optional dependency
- [ ] Works with LangChain 0.1.x and 0.2.x
- [ ] Unit tests with mocked LangChain
- [ ] Example in `examples/langchain_example/`
- [ ] `make test` passes
- [ ] README documents LangChain integration
