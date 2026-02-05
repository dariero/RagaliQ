"""
LLM Judge abstract base class for RagaliQ.

This module defines the core interface for LLM-as-Judge implementations.
All judge providers (Claude, OpenAI, etc.) must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field


class JudgeConfig(BaseModel):
    """
    Configuration for LLM judge instances.

    Attributes:
        model: The model identifier to use for judging.
        temperature: Sampling temperature (0.0 for deterministic).
        max_tokens: Maximum tokens in judge response.
    """

    model: str = Field(default="claude-sonnet-4-20250514", description="Model identifier")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: int = Field(default=1024, ge=1, le=4096, description="Max response tokens")

    model_config = {"frozen": True, "extra": "forbid"}


class JudgeResult(BaseModel):
    """
    Result from a judge evaluation.

    Attributes:
        score: Normalized score from 0.0 (worst) to 1.0 (best).
        reasoning: Human-readable explanation of the score.
        tokens_used: Number of tokens consumed by this judge call.
    """

    score: float = Field(..., ge=0.0, le=1.0, description="Score from 0 to 1")
    reasoning: str = Field(default="", description="Explanation of the score")
    tokens_used: int = Field(default=0, ge=0, description="Tokens used in judge call")

    model_config = {"frozen": True, "extra": "forbid"}


class ClaimVerdict(BaseModel):
    """
    Result of verifying a single claim against context.

    The verdict follows a three-way classification:
    - SUPPORTED: Context explicitly states or directly implies the claim
    - CONTRADICTED: Context explicitly contradicts the claim
    - NOT_ENOUGH_INFO: Context neither supports nor contradicts

    Attributes:
        verdict: Three-way classification of claim support.
        evidence: Quote from context or explanation supporting the verdict.
    """

    verdict: Literal["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"] = Field(
        ..., description="Three-way verdict on claim support"
    )
    evidence: str = Field(default="", description="Supporting quote or explanation")

    model_config = {"frozen": True, "extra": "forbid"}


class ClaimsResult(BaseModel):
    """
    Result of extracting atomic claims from a response.

    An atomic claim is a single, verifiable statement of fact that
    is self-contained and cannot be broken down further.

    Attributes:
        claims: List of extracted atomic claim strings.
        tokens_used: Tokens consumed by the extraction call.
    """

    claims: list[str] = Field(default_factory=list, description="Extracted claims")
    tokens_used: int = Field(default=0, ge=0, description="Tokens used")

    model_config = {"frozen": True, "extra": "forbid"}


class JudgeError(Exception):
    """Base exception for judge operations."""

    pass


class JudgeAPIError(JudgeError):
    """Raised when the LLM API call fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """
        Initialize the API error.

        Args:
            message: Error description.
            status_code: HTTP status code if applicable.
        """
        super().__init__(message)
        self.status_code = status_code


class JudgeResponseError(JudgeError):
    """Raised when the LLM response cannot be parsed."""

    pass


class LLMJudge(ABC):
    """
    Abstract base class for LLM-as-Judge implementations.

    LLM judges evaluate RAG responses by assessing qualities like
    faithfulness to context and relevance to the query. Each concrete
    implementation wraps a specific LLM provider (Claude, OpenAI, etc.).

    Example:
        class ClaudeJudge(LLMJudge):
            async def evaluate_faithfulness(self, response, context):
                # Call Claude API with faithfulness prompt
                ...

            async def evaluate_relevance(self, query, response):
                # Call Claude API with relevance prompt
                ...
    """

    def __init__(self, config: JudgeConfig | None = None) -> None:
        """
        Initialize the judge with optional configuration.

        Args:
            config: Judge configuration. Uses defaults if not provided.
        """
        self.config = config or JudgeConfig()

    @abstractmethod
    async def evaluate_faithfulness(
        self,
        response: str,
        context: list[str],
    ) -> JudgeResult:
        """
        Evaluate if the response is faithful to the provided context.

        Faithfulness measures whether claims in the response are supported
        by the context documents. A faithful response does not hallucinate
        or add information beyond what the context provides.

        Args:
            response: The RAG system's generated response.
            context: List of context documents used for generation.

        Returns:
            JudgeResult with faithfulness score and reasoning.

        Raises:
            JudgeAPIError: If the LLM API call fails.
            JudgeResponseError: If the response cannot be parsed.
        """
        ...

    @abstractmethod
    async def evaluate_relevance(
        self,
        query: str,
        response: str,
    ) -> JudgeResult:
        """
        Evaluate if the response is relevant to the query.

        Relevance measures whether the response addresses the user's
        question or request. A relevant response stays on topic and
        provides information the user actually asked for.

        Args:
            query: The user's original query.
            response: The RAG system's generated response.

        Returns:
            JudgeResult with relevance score and reasoning.

        Raises:
            JudgeAPIError: If the LLM API call fails.
            JudgeResponseError: If the response cannot be parsed.
        """
        ...

    @abstractmethod
    async def extract_claims(self, response: str) -> ClaimsResult:
        """
        Extract atomic claims from a response for verification.

        An atomic claim is a single, verifiable statement of fact that
        is self-contained and understandable without additional context.

        Args:
            response: The RAG system's generated response.

        Returns:
            ClaimsResult containing list of extracted claim strings.

        Raises:
            JudgeAPIError: If the LLM API call fails.
            JudgeResponseError: If the response cannot be parsed.
        """
        ...

    @abstractmethod
    async def verify_claim(
        self,
        claim: str,
        context: list[str],
    ) -> ClaimVerdict:
        """
        Verify if a single claim is supported by the context.

        Uses three-way classification:
        - SUPPORTED: Context explicitly supports the claim
        - CONTRADICTED: Context explicitly contradicts the claim
        - NOT_ENOUGH_INFO: Context neither supports nor contradicts

        Args:
            claim: The atomic claim to verify.
            context: List of context documents to check against.

        Returns:
            ClaimVerdict with verdict and supporting evidence.

        Raises:
            JudgeAPIError: If the LLM API call fails.
            JudgeResponseError: If the response cannot be parsed.
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of the judge."""
        return f"{self.__class__.__name__}(model={self.config.model!r})"
