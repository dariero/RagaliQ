"""Unit tests for LLM Judge base classes."""

import pytest
from pydantic import ValidationError

from ragaliq.judges import (
    ClaimsResult,
    ClaimVerdict,
    GeneratedAnswerResult,
    GeneratedQuestionsResult,
    JudgeAPIError,
    JudgeConfig,
    JudgeError,
    JudgeResponseError,
    JudgeResult,
    LLMJudge,
)


class TestJudgeConfig:
    """Tests for JudgeConfig model."""

    def test_default_values(self) -> None:
        """Test that defaults are set correctly."""
        config = JudgeConfig()
        assert config.model == "claude-sonnet-4-6"
        assert config.temperature == 0.0
        assert config.max_tokens == 1024

    def test_custom_values(self) -> None:
        """Test setting custom configuration values."""
        config = JudgeConfig(
            model="gpt-4",
            temperature=0.5,
            max_tokens=2048,
        )
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048

    def test_temperature_bounds_lower(self) -> None:
        """Test that temperature cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeConfig(temperature=-0.1)
        assert "temperature" in str(exc_info.value)

    def test_temperature_bounds_upper(self) -> None:
        """Test that temperature cannot exceed 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeConfig(temperature=1.1)
        assert "temperature" in str(exc_info.value)

    def test_max_tokens_minimum(self) -> None:
        """Test that max_tokens must be at least 1."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeConfig(max_tokens=0)
        assert "max_tokens" in str(exc_info.value)

    def test_max_tokens_maximum(self) -> None:
        """Test that max_tokens cannot exceed 4096."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeConfig(max_tokens=5000)
        assert "max_tokens" in str(exc_info.value)

    def test_immutable(self) -> None:
        """Test that config is frozen."""
        config = JudgeConfig()
        with pytest.raises(ValidationError):
            config.temperature = 0.5  # type: ignore[misc]

    def test_no_extra_fields(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeConfig(unknown_field="value")  # type: ignore[call-arg]
        assert "extra" in str(exc_info.value).lower()


class TestJudgeResult:
    """Tests for JudgeResult model."""

    def test_minimal_result(self) -> None:
        """Test creating result with only required fields."""
        result = JudgeResult(score=0.85)
        assert result.score == 0.85
        assert result.reasoning == ""
        assert result.tokens_used == 0

    def test_full_result(self) -> None:
        """Test creating result with all fields."""
        result = JudgeResult(
            score=0.95,
            reasoning="Response is highly relevant and addresses the query.",
            tokens_used=150,
        )
        assert result.score == 0.95
        assert result.reasoning == "Response is highly relevant and addresses the query."
        assert result.tokens_used == 150

    def test_score_bounds_lower(self) -> None:
        """Test that score cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeResult(score=-0.1)
        assert "score" in str(exc_info.value)

    def test_score_bounds_upper(self) -> None:
        """Test that score cannot exceed 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeResult(score=1.1)
        assert "score" in str(exc_info.value)

    def test_score_boundary_values(self) -> None:
        """Test that boundary values 0.0 and 1.0 are valid."""
        result_zero = JudgeResult(score=0.0)
        result_one = JudgeResult(score=1.0)
        assert result_zero.score == 0.0
        assert result_one.score == 1.0

    def test_tokens_used_non_negative(self) -> None:
        """Test that tokens_used cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeResult(score=0.5, tokens_used=-1)
        assert "tokens_used" in str(exc_info.value)

    def test_immutable(self) -> None:
        """Test that result is frozen."""
        result = JudgeResult(score=0.5)
        with pytest.raises(ValidationError):
            result.score = 0.9  # type: ignore[misc]

    def test_no_extra_fields(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeResult(score=0.5, extra_field="value")  # type: ignore[call-arg]
        assert "extra" in str(exc_info.value).lower()


class TestJudgeExceptions:
    """Tests for judge exception classes."""

    def test_judge_error_is_exception(self) -> None:
        """Test that JudgeError inherits from Exception."""
        assert issubclass(JudgeError, Exception)

    def test_judge_error_message(self) -> None:
        """Test raising JudgeError with message."""
        with pytest.raises(JudgeError, match="Something went wrong"):
            raise JudgeError("Something went wrong")

    def test_judge_api_error_inherits(self) -> None:
        """Test that JudgeAPIError inherits from JudgeError."""
        assert issubclass(JudgeAPIError, JudgeError)

    def test_judge_api_error_with_status_code(self) -> None:
        """Test JudgeAPIError with status code."""
        error = JudgeAPIError("Rate limited", status_code=429)
        assert str(error) == "Rate limited"
        assert error.status_code == 429

    def test_judge_api_error_without_status_code(self) -> None:
        """Test JudgeAPIError without status code."""
        error = JudgeAPIError("Connection failed")
        assert str(error) == "Connection failed"
        assert error.status_code is None

    def test_judge_response_error_inherits(self) -> None:
        """Test that JudgeResponseError inherits from JudgeError."""
        assert issubclass(JudgeResponseError, JudgeError)

    def test_judge_response_error_message(self) -> None:
        """Test raising JudgeResponseError with message."""
        with pytest.raises(JudgeResponseError, match="Invalid JSON"):
            raise JudgeResponseError("Invalid JSON in response")


class TestLLMJudge:
    """Tests for LLMJudge abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that LLMJudge cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            LLMJudge()  # type: ignore[abstract]

    def test_concrete_implementation_default_config(self) -> None:
        """Test that concrete implementation gets default config."""

        class MockJudge(LLMJudge):
            async def evaluate_faithfulness(
                self, _response: str, _context: list[str]
            ) -> JudgeResult:
                return JudgeResult(score=1.0)

            async def evaluate_relevance(self, _query: str, _response: str) -> JudgeResult:
                return JudgeResult(score=1.0)

            async def extract_claims(self, _response: str) -> ClaimsResult:
                return ClaimsResult(claims=[])

            async def verify_claim(self, _claim: str, _context: list[str]) -> ClaimVerdict:
                return ClaimVerdict(verdict="SUPPORTED")

            async def generate_questions(
                self, _documents: list[str], _n: int
            ) -> GeneratedQuestionsResult:
                return GeneratedQuestionsResult(questions=[])

            async def generate_answer(
                self, _question: str, _context: list[str]
            ) -> GeneratedAnswerResult:
                return GeneratedAnswerResult(answer="")

        judge = MockJudge()
        assert judge.config.model == "claude-sonnet-4-6"
        assert judge.config.temperature == 0.0

    def test_concrete_implementation_custom_config(self) -> None:
        """Test that concrete implementation accepts custom config."""

        class MockJudge(LLMJudge):
            async def evaluate_faithfulness(
                self, _response: str, _context: list[str]
            ) -> JudgeResult:
                return JudgeResult(score=1.0)

            async def evaluate_relevance(self, _query: str, _response: str) -> JudgeResult:
                return JudgeResult(score=1.0)

            async def extract_claims(self, _response: str) -> ClaimsResult:
                return ClaimsResult(claims=[])

            async def verify_claim(self, _claim: str, _context: list[str]) -> ClaimVerdict:
                return ClaimVerdict(verdict="SUPPORTED")

            async def generate_questions(
                self, _documents: list[str], _n: int
            ) -> GeneratedQuestionsResult:
                return GeneratedQuestionsResult(questions=[])

            async def generate_answer(
                self, _question: str, _context: list[str]
            ) -> GeneratedAnswerResult:
                return GeneratedAnswerResult(answer="")

        config = JudgeConfig(model="gpt-4", temperature=0.3)
        judge = MockJudge(config=config)
        assert judge.config.model == "gpt-4"
        assert judge.config.temperature == 0.3

    def test_repr(self) -> None:
        """Test string representation of judge."""

        class MockJudge(LLMJudge):
            async def evaluate_faithfulness(
                self, _response: str, _context: list[str]
            ) -> JudgeResult:
                return JudgeResult(score=1.0)

            async def evaluate_relevance(self, _query: str, _response: str) -> JudgeResult:
                return JudgeResult(score=1.0)

            async def extract_claims(self, _response: str) -> ClaimsResult:
                return ClaimsResult(claims=[])

            async def verify_claim(self, _claim: str, _context: list[str]) -> ClaimVerdict:
                return ClaimVerdict(verdict="SUPPORTED")

            async def generate_questions(
                self, _documents: list[str], _n: int
            ) -> GeneratedQuestionsResult:
                return GeneratedQuestionsResult(questions=[])

            async def generate_answer(
                self, _question: str, _context: list[str]
            ) -> GeneratedAnswerResult:
                return GeneratedAnswerResult(answer="")

        judge = MockJudge()
        assert repr(judge) == "MockJudge(model='claude-sonnet-4-6')"

    @pytest.mark.asyncio
    async def test_evaluate_faithfulness_signature(self) -> None:
        """Test that evaluate_faithfulness can be called with correct args."""

        class MockJudge(LLMJudge):
            async def evaluate_faithfulness(
                self,
                response: str,  # noqa: ARG002
                context: list[str],  # noqa: ARG002
            ) -> JudgeResult:
                return JudgeResult(
                    score=0.9,
                    reasoning="Response is faithful",
                    tokens_used=100,
                )

            async def evaluate_relevance(
                self,
                query: str,  # noqa: ARG002
                response: str,  # noqa: ARG002
            ) -> JudgeResult:
                return JudgeResult(score=1.0)

            async def extract_claims(self, _response: str) -> ClaimsResult:
                return ClaimsResult(claims=[])

            async def verify_claim(self, _claim: str, _context: list[str]) -> ClaimVerdict:
                return ClaimVerdict(verdict="SUPPORTED")

            async def generate_questions(
                self, _documents: list[str], _n: int
            ) -> GeneratedQuestionsResult:
                return GeneratedQuestionsResult(questions=[])

            async def generate_answer(
                self, _question: str, _context: list[str]
            ) -> GeneratedAnswerResult:
                return GeneratedAnswerResult(answer="")

        judge = MockJudge()
        result = await judge.evaluate_faithfulness(
            response="The capital of France is Paris.",
            context=["Paris is the capital city of France."],
        )
        assert result.score == 0.9
        assert result.tokens_used == 100

    @pytest.mark.asyncio
    async def test_evaluate_relevance_signature(self) -> None:
        """Test that evaluate_relevance can be called with correct args."""

        class MockJudge(LLMJudge):
            async def evaluate_faithfulness(
                self,
                response: str,  # noqa: ARG002
                context: list[str],  # noqa: ARG002
            ) -> JudgeResult:
                return JudgeResult(score=1.0)

            async def evaluate_relevance(
                self,
                query: str,  # noqa: ARG002
                response: str,  # noqa: ARG002
            ) -> JudgeResult:
                return JudgeResult(
                    score=0.95,
                    reasoning="Response directly answers the question",
                    tokens_used=80,
                )

            async def extract_claims(self, _response: str) -> ClaimsResult:
                return ClaimsResult(claims=[])

            async def verify_claim(self, _claim: str, _context: list[str]) -> ClaimVerdict:
                return ClaimVerdict(verdict="SUPPORTED")

            async def generate_questions(
                self, _documents: list[str], _n: int
            ) -> GeneratedQuestionsResult:
                return GeneratedQuestionsResult(questions=[])

            async def generate_answer(
                self, _question: str, _context: list[str]
            ) -> GeneratedAnswerResult:
                return GeneratedAnswerResult(answer="")

        judge = MockJudge()
        result = await judge.evaluate_relevance(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
        )
        assert result.score == 0.95
        assert result.tokens_used == 80

    def test_missing_abstract_method_faithfulness(self) -> None:
        """Test that missing evaluate_faithfulness raises TypeError."""

        with pytest.raises(TypeError, match="abstract"):

            class IncompleteJudge(LLMJudge):  # type: ignore[abstract]
                async def evaluate_relevance(self, _query: str, _response: str) -> JudgeResult:
                    return JudgeResult(score=1.0)

                async def extract_claims(self, _response: str) -> ClaimsResult:
                    return ClaimsResult(claims=[])

                async def verify_claim(self, _claim: str, _context: list[str]) -> ClaimVerdict:
                    return ClaimVerdict(verdict="SUPPORTED")

            IncompleteJudge()

    def test_missing_abstract_method_relevance(self) -> None:
        """Test that missing evaluate_relevance raises TypeError."""

        with pytest.raises(TypeError, match="abstract"):

            class IncompleteJudge(LLMJudge):  # type: ignore[abstract]
                async def evaluate_faithfulness(
                    self, _response: str, _context: list[str]
                ) -> JudgeResult:
                    return JudgeResult(score=1.0)

                async def extract_claims(self, _response: str) -> ClaimsResult:
                    return ClaimsResult(claims=[])

                async def verify_claim(self, _claim: str, _context: list[str]) -> ClaimVerdict:
                    return ClaimVerdict(verdict="SUPPORTED")

            IncompleteJudge()

    def test_missing_abstract_method_extract_claims(self) -> None:
        """Test that missing extract_claims raises TypeError."""

        with pytest.raises(TypeError, match="abstract"):

            class IncompleteJudge(LLMJudge):  # type: ignore[abstract]
                async def evaluate_faithfulness(
                    self, _response: str, _context: list[str]
                ) -> JudgeResult:
                    return JudgeResult(score=1.0)

                async def evaluate_relevance(self, _query: str, _response: str) -> JudgeResult:
                    return JudgeResult(score=1.0)

                async def verify_claim(self, _claim: str, _context: list[str]) -> ClaimVerdict:
                    return ClaimVerdict(verdict="SUPPORTED")

            IncompleteJudge()

    def test_missing_abstract_method_verify_claim(self) -> None:
        """Test that missing verify_claim raises TypeError."""

        with pytest.raises(TypeError, match="abstract"):

            class IncompleteJudge(LLMJudge):  # type: ignore[abstract]
                async def evaluate_faithfulness(
                    self, _response: str, _context: list[str]
                ) -> JudgeResult:
                    return JudgeResult(score=1.0)

                async def evaluate_relevance(self, _query: str, _response: str) -> JudgeResult:
                    return JudgeResult(score=1.0)

                async def extract_claims(self, _response: str) -> ClaimsResult:
                    return ClaimsResult(claims=[])

            IncompleteJudge()
