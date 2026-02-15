"""Pytest configuration and fixtures for RagaliQ tests."""

import pytest

from ragaliq.core.test_case import RAGTestCase

# Enable pytester fixture for testing pytest plugins
pytest_plugins = ["pytester"]


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
