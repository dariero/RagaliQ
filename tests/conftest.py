"""Pytest configuration and fixtures for RagaliQ tests."""

import pytest

from ragaliq.core.test_case import RAGTestCase

# Enable pytester fixture for testing pytest plugins
pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _configure_pytester_asyncio(request: pytest.FixtureRequest) -> None:
    """Inject asyncio_default_fixture_loop_scope into inner pytester sessions.

    pytester spawns isolated subprocess pytest sessions in temp directories that
    don't inherit the project pyproject.toml. Without this, pytest-asyncio warns
    about the unset option in every inner session (once per runpytest() call).
    """
    if "pytester" not in request.fixturenames:
        return
    pytester = request.getfixturevalue("pytester")
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")


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
