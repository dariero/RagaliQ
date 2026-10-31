"""LLM-as-Judge interface and result types for RagaliQ.

Defines the `LLMJudge` ABC that every provider (Claude, OpenAI, ...) implements,
plus the frozen Pydantic result models and the judge exception hierarchy.
"""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from ragaliq.judges.models import DEFAULT_JUDGE_MODEL


class JudgeConfig(BaseModel):
    """Configuration for an LLM judge: model, sampling, and token budget."""

    model: str = Field(default=DEFAULT_JUDGE_MODEL, description="Model identifier")
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Sampling temperature; None sends none. Under API version 2026-10-21"
            " (anthropic 2.x) any value returns HTTP 400 on every model."
        ),
    )
    # Above 21333 the anthropic SDK refuses a non-streaming request on its default timeout.
    max_tokens: int = Field(
        default=16000, ge=1, le=21333, description="Max output tokens: thinking plus text"
    )

    model_config = {"frozen": True, "extra": "forbid"}


class JudgeResult(BaseModel):
    """Score, reasoning, and token usage from a single judge evaluation."""

    score: float = Field(..., ge=0.0, le=1.0, description="Score from 0 to 1")
    reasoning: str = Field(default="", description="Explanation of the score")
    tokens_used: int = Field(default=0, ge=0, description="Tokens used in judge call")

    model_config = {"frozen": True, "extra": "forbid"}


class ClaimVerdict(BaseModel):
    """A claim's verdict against context: SUPPORTED, CONTRADICTED, or NOT_ENOUGH_INFO."""

    verdict: Literal["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"] = Field(
        ..., description="Three-way verdict on claim support"
    )
    evidence: str = Field(default="", description="Supporting quote or explanation")
    tokens_used: int = Field(default=0, ge=0, description="Tokens used")

    model_config = {"frozen": True, "extra": "forbid"}


class ClaimsResult(BaseModel):
    """Atomic claims extracted from a response, with token usage.

    An atomic claim is a single, self-contained, verifiable statement of fact.
    """

    claims: list[str] = Field(default_factory=list, description="Extracted claims")
    tokens_used: int = Field(default=0, ge=0, description="Tokens used")

    model_config = {"frozen": True, "extra": "forbid"}


class GeneratedQuestionsResult(BaseModel):
    """Questions generated from documents, with token usage."""

    questions: list[str] = Field(default_factory=list, description="Generated questions")
    tokens_used: int = Field(default=0, ge=0, description="Tokens used")

    model_config = {"frozen": True, "extra": "forbid"}


class GeneratedAnswerResult(BaseModel):
    """An answer generated for a question from context, with token usage."""

    answer: str = Field(default="", description="Generated answer")
    tokens_used: int = Field(default=0, ge=0, description="Tokens used")

    model_config = {"frozen": True, "extra": "forbid"}


class JudgeError(Exception):
    """Base exception for judge operations."""


class JudgeAPIError(JudgeError):
    """Raised when the LLM API call fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class JudgeResponseError(JudgeError):
    """Raised when the LLM response cannot be parsed.

    A response the transport rejects (stopped at ``max_tokens``, empty, or with no
    text) was still billed. The transport puts its token usage here, so the judge's
    trace counts it; errors raised while parsing a returned text keep the 0 defaults.
    """

    def __init__(self, message: str, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class LLMJudge(ABC):
    """Abstract base class for LLM-as-Judge implementations.

    Judges assess RAG responses on qualities such as faithfulness to context and
    relevance to the query. Each concrete subclass wraps one LLM provider and
    supplies the actual API transport.

    Example:
        class ClaudeJudge(LLMJudge):
            async def evaluate_faithfulness(self, response, context): ...
            async def evaluate_relevance(self, query, response): ...
    """

    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()

    @abstractmethod
    async def evaluate_faithfulness(
        self,
        response: str,
        context: list[str],
    ) -> JudgeResult:
        """Evaluate whether the response is faithful to the provided context.

        Faithfulness measures whether the response's claims are supported by the
        context; a faithful response adds nothing beyond what the context states.

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
        """Evaluate whether the response is relevant to the query.

        Relevance measures whether the response stays on topic and addresses what
        the user actually asked.

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
        """Extract atomic claims from a response for verification.

        An atomic claim is a single, self-contained, verifiable statement of fact.

        Args:
            response: The RAG system's generated response.

        Returns:
            ClaimsResult containing the list of extracted claim strings.

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
        """Verify whether a single claim is supported by the context.

        Returns a three-way verdict: SUPPORTED (context supports the claim),
        CONTRADICTED (context contradicts it), or NOT_ENOUGH_INFO (neither).

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

    @abstractmethod
    async def generate_questions(
        self,
        documents: list[str],
        n: int,
    ) -> GeneratedQuestionsResult:
        """Generate `n` diverse questions answerable solely from the documents.

        Used to build RAG test datasets grounded in source content.

        Args:
            documents: Source documents to generate questions from.
            n: Number of questions to generate.

        Returns:
            GeneratedQuestionsResult containing the list of questions.

        Raises:
            JudgeAPIError: If the LLM API call fails.
            JudgeResponseError: If the response cannot be parsed.
        """
        ...

    @abstractmethod
    async def generate_answer(
        self,
        question: str,
        context: list[str],
    ) -> GeneratedAnswerResult:
        """Answer a question using only the provided context.

        Simulates a RAG response grounded in context, introducing no outside
        knowledge.

        Args:
            question: The question to answer.
            context: List of context documents to answer from.

        Returns:
            GeneratedAnswerResult containing the generated answer.

        Raises:
            JudgeAPIError: If the LLM API call fails.
            JudgeResponseError: If the response cannot be parsed.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model!r})"
