"""Pytest configuration and fixtures for RagaliQ tests."""

import pytest

from ragaliq.core.test_case import RAGTestCase


@pytest.fixture
def sample_test_case() -> RAGTestCase:
    """Create a sample test case for testing."""
    return RAGTestCase(
        id="test_001",
        name="Capital of France",
        query="What is the capital of France?",
        context=[
            "France is a country located in Western Europe.",
            "The capital city of France is Paris.",
            "Paris is known for the Eiffel Tower.",
        ],
        response="The capital of France is Paris.",
    )


@pytest.fixture
def hallucinating_test_case() -> RAGTestCase:
    """Create a test case with a hallucinating response."""
    return RAGTestCase(
        id="test_002",
        name="Hallucination Example",
        query="What is the capital of France?",
        context=[
            "France is a country located in Western Europe.",
            "The capital city of France is Paris.",
        ],
        response="The capital of France is Paris, which was founded in 250 BC by the Romans.",
    )


@pytest.fixture
def irrelevant_response_test_case() -> RAGTestCase:
    """Create a test case with an irrelevant response."""
    return RAGTestCase(
        id="test_003",
        name="Irrelevant Response",
        query="What is the capital of France?",
        context=[
            "France is a country located in Western Europe.",
            "The capital city of France is Paris.",
        ],
        response="The weather in France is generally mild with four distinct seasons.",
    )
