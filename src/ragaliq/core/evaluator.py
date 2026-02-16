"""Base evaluator interface for RagaliQ."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestCase
    from ragaliq.judges.base import LLMJudge


class EvaluationResult(BaseModel):
    """
    Result of a single evaluation metric.

    Attributes:
        evaluator_name: Name of the evaluator that produced this result.
        score: Numeric score from 0.0 to 1.0.
        passed: Whether the score meets the threshold.
        reasoning: Human-readable explanation of the score.
        raw_response: Raw response from the LLM judge for debugging.
        tokens_used: Total tokens consumed by this evaluation.
        error: Error message if evaluation failed. Non-None indicates failure.
    """

    evaluator_name: str = Field(..., description="Name of the evaluator")
    score: float = Field(..., ge=0.0, le=1.0, description="Score from 0 to 1")
    passed: bool = Field(..., description="Whether threshold was met")
    reasoning: str = Field(default="", description="Explanation of the score")
    raw_response: dict[str, Any] = Field(default_factory=dict, description="Raw judge response")
    tokens_used: int = Field(default=0, ge=0, description="Total tokens consumed")
    error: str | None = Field(default=None, description="Error message if evaluation failed")

    model_config = {"frozen": True, "extra": "forbid"}


class Evaluator(ABC):
    """
    Abstract base class for RAG evaluators.

    Each evaluator assesses a specific quality dimension of RAG responses.

    Attributes:
        name: Unique identifier for this evaluator.
        description: Human-readable description.
        threshold: Minimum score to pass (default 0.7).
    """

    name: str
    description: str
    threshold: float = 0.7

    def __init__(self, threshold: float | None = None) -> None:
        """
        Initialize the evaluator.

        Args:
            threshold: Optional custom threshold (overrides class default).
        """
        if threshold is not None:
            self.threshold = threshold

    @abstractmethod
    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge,
    ) -> EvaluationResult:
        """
        Evaluate a test case using the LLM judge.

        Args:
            test_case: The RAG test case to evaluate.
            judge: The LLM judge to use for assessment.

        Returns:
            EvaluationResult with score and reasoning.
        """
        pass

    def is_passing(self, score: float) -> bool:
        """Check if a score meets the threshold."""
        return score >= self.threshold

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"
