"""Unit tests for core data models."""

import pytest
from pydantic import ValidationError

from ragaliq.core.test_case import RAGTestCase, RAGTestResult, TestStatus
from ragaliq.core.evaluator import EvaluationResult


class TestRAGTestCase:
    """Tests for RAGTestCase model."""

    def test_create_valid_test_case(self):
        """Test creating a valid test case."""
        test_case = RAGTestCase(
            id="test_1",
            name="Test Name",
            query="What is X?",
            context=["Context 1", "Context 2"],
            response="X is something.",
        )

        assert test_case.id == "test_1"
        assert test_case.name == "Test Name"
        assert test_case.query == "What is X?"
        assert len(test_case.context) == 2
        assert test_case.response == "X is something."
        assert test_case.expected_answer is None
        assert test_case.tags == []

    def test_create_test_case_with_optional_fields(self):
        """Test creating a test case with optional fields."""
        test_case = RAGTestCase(
            id="test_2",
            name="Full Test",
            query="Query",
            context=["Context"],
            response="Response",
            expected_answer="Expected",
            expected_facts=["Fact 1", "Fact 2"],
            tags=["unit", "critical"],
        )

        assert test_case.expected_answer == "Expected"
        assert test_case.expected_facts == ["Fact 1", "Fact 2"]
        assert test_case.tags == ["unit", "critical"]

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            RAGTestCase(
                id="test_3",
                name="Missing Fields",
                # Missing: query, context, response
            )

        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "query" in field_names
        assert "context" in field_names
        assert "response" in field_names

    def test_empty_context_is_valid(self):
        """Test that empty context list is valid (might want to change this)."""
        test_case = RAGTestCase(
            id="test_4",
            name="Empty Context",
            query="Query",
            context=[],
            response="Response",
        )
        assert test_case.context == []


class TestRAGTestResult:
    """Tests for RAGTestResult model."""

    def test_create_passed_result(self, sample_test_case):
        """Test creating a passed result."""
        result = RAGTestResult(
            test_case=sample_test_case,
            status=TestStatus.PASSED,
            scores={"faithfulness": 0.95, "relevance": 0.88},
            execution_time_ms=150,
        )

        assert result.passed is True
        assert result.status == TestStatus.PASSED
        assert result.scores["faithfulness"] == 0.95

    def test_create_failed_result(self, sample_test_case):
        """Test creating a failed result."""
        result = RAGTestResult(
            test_case=sample_test_case,
            status=TestStatus.FAILED,
            scores={"faithfulness": 0.45, "relevance": 0.30},
        )

        assert result.passed is False
        assert result.status == TestStatus.FAILED

    def test_get_score(self, sample_test_case):
        """Test get_score helper method."""
        result = RAGTestResult(
            test_case=sample_test_case,
            status=TestStatus.PASSED,
            scores={"faithfulness": 0.9, "relevance": 0.8},
        )

        assert result.get_score("faithfulness") == 0.9
        assert result.get_score("relevance") == 0.8
        assert result.get_score("nonexistent") is None

    def test_details_storage(self, sample_test_case):
        """Test storing detailed evaluation info."""
        result = RAGTestResult(
            test_case=sample_test_case,
            status=TestStatus.PASSED,
            scores={"faithfulness": 0.9},
            details={
                "faithfulness": {
                    "claims": ["Claim 1", "Claim 2"],
                    "verified": [True, True],
                }
            },
        )

        assert "faithfulness" in result.details
        assert len(result.details["faithfulness"]["claims"]) == 2


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_create_evaluation_result(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            evaluator_name="faithfulness",
            score=0.85,
            passed=True,
            reasoning="All claims are supported by context.",
        )

        assert result.evaluator_name == "faithfulness"
        assert result.score == 0.85
        assert result.passed is True
        assert "supported" in result.reasoning

    def test_score_bounds(self):
        """Test that score must be between 0 and 1."""
        # Valid scores
        EvaluationResult(evaluator_name="test", score=0.0, passed=False)
        EvaluationResult(evaluator_name="test", score=1.0, passed=True)
        EvaluationResult(evaluator_name="test", score=0.5, passed=True)

        # Invalid scores
        with pytest.raises(ValidationError):
            EvaluationResult(evaluator_name="test", score=-0.1, passed=False)

        with pytest.raises(ValidationError):
            EvaluationResult(evaluator_name="test", score=1.1, passed=True)

    def test_raw_response_storage(self):
        """Test storing raw LLM response."""
        result = EvaluationResult(
            evaluator_name="test",
            score=0.8,
            passed=True,
            raw_response={
                "model": "claude-sonnet-4-20250514",
                "tokens_used": 150,
                "response_text": "...",
            },
        )

        assert result.raw_response["model"] == "claude-sonnet-4-20250514"


class TestTestStatus:
    """Tests for TestStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert TestStatus.PASSED.value == "passed"
        assert TestStatus.FAILED.value == "failed"
        assert TestStatus.SKIPPED.value == "skipped"
        assert TestStatus.ERROR.value == "error"

    def test_status_comparison(self):
        """Test status comparison."""
        assert TestStatus.PASSED == TestStatus.PASSED
        assert TestStatus.PASSED != TestStatus.FAILED
