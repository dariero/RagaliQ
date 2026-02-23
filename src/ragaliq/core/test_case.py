"""Test case models for RagaliQ."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EvalStatus(StrEnum):
    """Status of an evaluation execution."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class RAGTestCase(BaseModel):
    """
    A single test case for RAG system evaluation.

    Attributes:
        id: Unique identifier for the test case.
        name: Human-readable name/description.
        query: The user question or input.
        context: List of retrieved documents/chunks.
        response: The LLM-generated response to evaluate.
        expected_answer: Optional ground truth answer.
        expected_facts: Optional list of facts that should be in response.
        tags: Optional tags for filtering/grouping tests.
    """

    id: str = Field(..., description="Unique identifier for the test case")
    name: str = Field(..., description="Human-readable name")
    query: str = Field(..., min_length=1, description="The user question or input")
    context: list[str] = Field(..., description="Retrieved documents/chunks")
    response: str = Field(..., min_length=1, description="LLM response to evaluate")

    @field_validator("query", "response", mode="before")
    @classmethod
    def _strip_whitespace(cls, v: str) -> str:
        """Strip whitespace so whitespace-only strings fail min_length."""
        if isinstance(v, str):
            return v.strip()
        return v

    expected_answer: str | None = Field(default=None, description="Ground truth answer")
    expected_facts: list[str] | None = Field(
        default=None, description="Facts that should be present"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")

    model_config = {"frozen": False, "extra": "forbid"}


class RAGTestResult(BaseModel):
    """
    Result of evaluating a RAG test case.

    Attributes:
        test_case: The original test case.
        status: Overall pass/fail status.
        scores: Dictionary of metric scores (0-1).
        details: Detailed evaluation information.
        execution_time_ms: Time taken to evaluate.
        judge_tokens_used: Number of tokens used by LLM judge.
    """

    test_case: RAGTestCase = Field(..., description="The evaluated test case")
    status: EvalStatus = Field(..., description="Overall test status")
    scores: dict[str, float] = Field(default_factory=dict, description="Metric scores (0-1)")
    details: dict[str, Any] = Field(default_factory=dict, description="Detailed evaluation info")
    execution_time_ms: int = Field(default=0, description="Evaluation time in ms")
    judge_tokens_used: int = Field(default=0, description="Tokens used by judge")

    @property
    def passed(self) -> bool:
        """Check if the test passed."""
        return self.status == EvalStatus.PASSED

    def get_score(self, metric: str) -> float | None:
        """Get score for a specific metric."""
        return self.scores.get(metric)

    model_config = {"frozen": False, "extra": "forbid"}
