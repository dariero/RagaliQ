"""
Example: Using the RagaliQ pytest plugin for RAG quality testing.

This file demonstrates how to write RAG quality tests using:
  - The `rag_tester` fixture for direct evaluation
  - The `ragaliq_judge` fixture with `assert_rag_quality()` helper
  - `@pytest.mark.rag_test` and `@pytest.mark.rag_slow` markers

Run with a real API key:
    ANTHROPIC_API_KEY=sk-ant-... pytest examples/pytest_example/ -v

Skip slow tests:
    pytest examples/pytest_example/ -m "not rag_slow"
"""

import pytest

from ragaliq.core.test_case import RAGTestCase
from ragaliq.integrations.pytest_plugin import assert_rag_quality


@pytest.mark.rag_test
def test_faithful_answer(rag_tester):
    """Faithful response — all facts present in the context."""
    test_case = RAGTestCase(
        id="ex-faithful-1",
        name="Faithful capital answer",
        query="What is the capital of France?",
        context=["France is a country in Western Europe. Its capital city is Paris."],
        response="The capital of France is Paris.",
    )
    result = rag_tester.evaluate(test_case)
    assert result.passed, f"Quality check failed: {result.scores}"


@pytest.mark.rag_test
def test_relevant_answer(ragaliq_judge):
    """Relevant response — uses assert_rag_quality helper."""
    test_case = RAGTestCase(
        id="ex-relevant-1",
        name="Relevant ML answer",
        query="What is machine learning?",
        context=["Machine learning is a subset of AI that enables systems to learn from data."],
        response="Machine learning is an AI technique that allows systems to improve from data.",
    )
    assert_rag_quality(test_case, judge=ragaliq_judge)


@pytest.mark.rag_test
@pytest.mark.rag_slow
def test_multi_doc_context(rag_tester):
    """Multi-document context — marked slow as it makes several judge calls."""
    test_case = RAGTestCase(
        id="ex-multi-1",
        name="Multi-doc synthesis",
        query="What are the benefits of async programming?",
        context=[
            "Async programming allows handling many tasks concurrently without blocking.",
            "In Python, asyncio enables writing non-blocking I/O-bound code.",
            "Async code improves throughput for network-bound applications.",
        ],
        response="Async programming improves concurrency and throughput for I/O-bound tasks.",
    )
    result = rag_tester.evaluate(test_case)
    assert result.passed, f"Scores: {result.scores}"


@pytest.mark.rag_test
def test_with_custom_threshold(rag_tester):
    """Custom threshold — requires stricter quality on this specific test."""
    from ragaliq.core.runner import RagaliQ

    strict_tester = RagaliQ(judge=rag_tester._judge, default_threshold=0.9)
    test_case = RAGTestCase(
        id="ex-strict-1",
        name="Strict quality check",
        query="Explain photosynthesis.",
        context=[
            "Photosynthesis is the process by which plants convert sunlight, "
            "water, and CO2 into glucose and oxygen."
        ],
        response="Plants use sunlight, water, and CO2 to produce glucose and oxygen.",
    )
    result = strict_tester.evaluate(test_case)
    assert result.passed, f"Strict quality check failed: {result.scores}"
