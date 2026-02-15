"""Unit tests for ClaudeJudge implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import APIConnectionError, APIStatusError

from ragaliq.judges import (
    ClaimsResult,
    ClaimVerdict,
    ClaudeJudge,
    JudgeAPIError,
    JudgeConfig,
    JudgeResponseError,
    JudgeResult,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_anthropic_client() -> Generator[MagicMock]:
    """Create a mock Anthropic client."""
    with patch("ragaliq.judges.transport.AsyncAnthropic") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_response() -> MagicMock:
    """Create a mock Claude API response."""
    response = MagicMock()
    response.content = [MagicMock(type="text", text='{"score": 0.85, "reasoning": "Test"}')]
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    return response


@pytest.mark.usefixtures("mock_anthropic_client")
class TestClaudeJudgeInit:
    """Tests for ClaudeJudge initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        judge = ClaudeJudge(api_key="test-key")
        assert judge.config.model == "claude-sonnet-4-20250514"
        assert judge.config.temperature == 0.0

    def test_init_with_env_var(self) -> None:
        """Test initialization with environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            judge = ClaudeJudge()
            assert judge.config.model == "claude-sonnet-4-20250514"

    def test_init_no_api_key_raises(self) -> None:
        """Test that missing API key raises ValueError."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("ragaliq.judges.transport.AsyncAnthropic"),
            pytest.raises(ValueError, match="Anthropic API key required"),
        ):
            # Ensure ANTHROPIC_API_KEY is not set
            import os

            os.environ.pop("ANTHROPIC_API_KEY", None)
            ClaudeJudge()

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        config = JudgeConfig(model="claude-opus-4-20250514", temperature=0.3, max_tokens=2048)
        judge = ClaudeJudge(config=config, api_key="test-key")
        assert judge.config.model == "claude-opus-4-20250514"
        assert judge.config.temperature == 0.3
        assert judge.config.max_tokens == 2048

    def test_repr(self) -> None:
        """Test string representation."""
        judge = ClaudeJudge(api_key="test-key")
        assert repr(judge) == "ClaudeJudge(model='claude-sonnet-4-20250514')"


class TestClaudeJudgeFaithfulness:
    """Tests for evaluate_faithfulness method."""

    @pytest.mark.asyncio
    async def test_evaluate_faithfulness_success(
        self,
        mock_anthropic_client: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Test successful faithfulness evaluation."""
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.evaluate_faithfulness(
            response="Paris is the capital of France.",
            context=["France is a country in Europe. Its capital is Paris."],
        )

        assert isinstance(result, JudgeResult)
        assert result.score == 0.85
        assert result.reasoning == "Test"
        assert result.tokens_used == 150  # 100 + 50

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_anthropic_client")
    async def test_evaluate_faithfulness_empty_context(self) -> None:
        """Test faithfulness with empty context returns 0.0."""
        judge = ClaudeJudge(api_key="test-key")
        result = await judge.evaluate_faithfulness(
            response="Any response",
            context=[],
        )

        assert result.score == 0.0
        assert "No context provided" in result.reasoning
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_evaluate_faithfulness_multiple_context_docs(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test faithfulness with multiple context documents."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text='{"score": 0.95, "reasoning": "All claims supported"}')
        ]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 60
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.evaluate_faithfulness(
            response="Paris is in France. It has the Eiffel Tower.",
            context=[
                "Paris is the capital of France.",
                "The Eiffel Tower is located in Paris.",
            ],
        )

        assert result.score == 0.95
        assert result.tokens_used == 260

    @pytest.mark.asyncio
    async def test_evaluate_faithfulness_score_clamping(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test that out-of-range scores are clamped."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text='{"score": 1.5, "reasoning": "Too high"}')
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.evaluate_faithfulness(
            response="Test",
            context=["Test context"],
        )

        assert result.score == 1.0  # Clamped to max

    @pytest.mark.asyncio
    async def test_evaluate_faithfulness_negative_score_clamping(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test that negative scores are clamped to 0.0."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text='{"score": -0.5, "reasoning": "Negative"}')
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.evaluate_faithfulness(
            response="Test",
            context=["Test context"],
        )

        assert result.score == 0.0  # Clamped to min


class TestClaudeJudgeRelevance:
    """Tests for evaluate_relevance method."""

    @pytest.mark.asyncio
    async def test_evaluate_relevance_success(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test successful relevance evaluation."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="text", text='{"score": 0.92, "reasoning": "Directly answers the question"}'
            )
        ]
        mock_response.usage.input_tokens = 80
        mock_response.usage.output_tokens = 40
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.evaluate_relevance(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
        )

        assert isinstance(result, JudgeResult)
        assert result.score == 0.92
        assert "answers" in result.reasoning
        assert result.tokens_used == 120

    @pytest.mark.asyncio
    async def test_evaluate_relevance_irrelevant(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test relevance evaluation for irrelevant response."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="text",
                text='{"score": 0.1, "reasoning": "Response does not address the query"}',
            )
        ]
        mock_response.usage.input_tokens = 80
        mock_response.usage.output_tokens = 40
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.evaluate_relevance(
            query="What is the capital of France?",
            response="The weather is nice today.",
        )

        assert result.score == 0.1


class TestClaudeJudgeErrorHandling:
    """Tests for error handling in ClaudeJudge."""

    @pytest.mark.asyncio
    async def test_api_status_error_non_retryable(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test handling of non-retryable API status errors (4xx except 429)."""
        error_response = MagicMock()
        error_response.status_code = 400
        mock_anthropic_client.messages.create = AsyncMock(
            side_effect=APIStatusError(
                message="Bad request",
                response=error_response,
                body={"error": "invalid_request"},
            )
        )

        judge = ClaudeJudge(api_key="test-key")
        with pytest.raises(JudgeAPIError) as exc_info:
            await judge.evaluate_faithfulness(
                response="Test",
                context=["Context"],
            )

        assert exc_info.value.status_code == 400
        assert "Bad request" in str(exc_info.value)
        # Should not retry 400 errors
        assert mock_anthropic_client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_connection_error_after_retries(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test that connection errors are retried then raised."""
        mock_anthropic_client.messages.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )

        judge = ClaudeJudge(api_key="test-key")
        with pytest.raises(JudgeAPIError, match="Connection to Claude API failed"):
            await judge.evaluate_faithfulness(
                response="Test",
                context=["Context"],
            )

    @pytest.mark.asyncio
    async def test_connection_error_succeeds_on_retry(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test that a transient connection error recovers on retry."""
        success_response = MagicMock()
        success_response.content = [
            MagicMock(type="text", text='{"score": 0.85, "reasoning": "Recovered"}')
        ]
        success_response.usage.input_tokens = 100
        success_response.usage.output_tokens = 50

        mock_anthropic_client.messages.create = AsyncMock(
            side_effect=[
                APIConnectionError(request=MagicMock()),
                success_response,
            ]
        )

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.evaluate_faithfulness(
            response="Test",
            context=["Context"],
        )

        assert isinstance(result, JudgeResult)
        assert result.score == 0.85
        assert mock_anthropic_client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_429_error_retried_and_exhausted(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test that 429 errors are retried 3 times before raising."""
        error_response = MagicMock()
        error_response.status_code = 429
        mock_anthropic_client.messages.create = AsyncMock(
            side_effect=APIStatusError(
                message="Rate limit exceeded",
                response=error_response,
                body={"error": "rate_limited"},
            )
        )

        judge = ClaudeJudge(api_key="test-key")
        with pytest.raises(JudgeAPIError, match="Claude API error"):
            await judge.evaluate_faithfulness(
                response="Test",
                context=["Context"],
            )

        # Should attempt 3 times (initial + 2 retries)
        assert mock_anthropic_client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_500_error_retried(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test that 5xx errors are retried before raising."""
        error_response = MagicMock()
        error_response.status_code = 500
        mock_anthropic_client.messages.create = AsyncMock(
            side_effect=APIStatusError(
                message="Internal server error",
                response=error_response,
                body={"error": "server_error"},
            )
        )

        judge = ClaudeJudge(api_key="test-key")
        with pytest.raises(JudgeAPIError):
            await judge.evaluate_faithfulness(
                response="Test",
                context=["Context"],
            )

        # Should retry 5xx errors
        assert mock_anthropic_client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_transient_429_succeeds_on_retry(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test that transient 429 errors recover on retry."""
        error_response = MagicMock()
        error_response.status_code = 429

        success_response = MagicMock()
        success_response.content = [
            MagicMock(type="text", text='{"score": 0.9, "reasoning": "Success after retry"}')
        ]
        success_response.usage.input_tokens = 100
        success_response.usage.output_tokens = 50

        mock_anthropic_client.messages.create = AsyncMock(
            side_effect=[
                APIStatusError(
                    message="Rate limit exceeded",
                    response=error_response,
                    body={"error": "rate_limited"},
                ),
                success_response,
            ]
        )

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.evaluate_faithfulness(
            response="Test",
            context=["Context"],
        )

        assert isinstance(result, JudgeResult)
        assert result.score == 0.9
        assert mock_anthropic_client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_invalid_json_response(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test handling of invalid JSON in response."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Not valid JSON")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        with pytest.raises(JudgeResponseError, match="Failed to parse JSON"):
            await judge.evaluate_faithfulness(
                response="Test",
                context=["Context"],
            )

    @pytest.mark.asyncio
    async def test_missing_score_field(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test handling of response missing score field."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text='{"reasoning": "No score provided"}')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        with pytest.raises(JudgeResponseError, match="missing 'score' field"):
            await judge.evaluate_faithfulness(
                response="Test",
                context=["Context"],
            )

    @pytest.mark.asyncio
    async def test_non_text_response_type(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test handling of non-text response type."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="tool_use", text=None)]
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        with pytest.raises(JudgeResponseError, match="Expected text response"):
            await judge.evaluate_faithfulness(
                response="Test",
                context=["Context"],
            )


class TestClaudeJudgeJsonParsing:
    """Tests for JSON parsing edge cases."""

    @pytest.mark.asyncio
    async def test_json_in_markdown_code_block(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="text",
                text='```json\n{"score": 0.75, "reasoning": "Wrapped in markdown"}\n```',
            )
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.evaluate_faithfulness(
            response="Test",
            context=["Context"],
        )

        assert result.score == 0.75
        assert result.reasoning == "Wrapped in markdown"

    @pytest.mark.asyncio
    async def test_json_with_whitespace(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test parsing JSON with surrounding whitespace."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="text", text='  \n  {"score": 0.80, "reasoning": "With whitespace"}  \n  '
            )
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.evaluate_faithfulness(
            response="Test",
            context=["Context"],
        )

        assert result.score == 0.80

    @pytest.mark.asyncio
    async def test_missing_reasoning_field(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test that missing reasoning field defaults to empty string."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text='{"score": 0.6}')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.evaluate_faithfulness(
            response="Test",
            context=["Context"],
        )

        assert result.score == 0.6
        assert result.reasoning == ""


@pytest.mark.usefixtures("mock_anthropic_client")
class TestClaudeJudgePromptBuilding:
    """Tests for prompt construction."""

    def test_faithfulness_prompt_contains_context(self) -> None:
        """Test that faithfulness prompt includes context documents."""
        judge = ClaudeJudge(api_key="test-key")
        system_prompt, user_prompt = judge._build_faithfulness_prompt(
            response="Test response",
            context=["Doc 1", "Doc 2"],
        )

        assert "Document 1" in user_prompt
        assert "Doc 1" in user_prompt
        assert "Document 2" in user_prompt
        assert "Doc 2" in user_prompt
        assert "Test response" in user_prompt
        assert "faithfulness" in system_prompt.lower()

    def test_relevance_prompt_contains_query(self) -> None:
        """Test that relevance prompt includes query and response."""
        judge = ClaudeJudge(api_key="test-key")
        system_prompt, user_prompt = judge._build_relevance_prompt(
            query="What is X?",
            response="X is Y.",
        )

        assert "What is X?" in user_prompt
        assert "X is Y." in user_prompt
        assert "relevance" in system_prompt.lower()


class TestClaudeJudgeApiCallParameters:
    """Tests for API call parameter handling."""

    @pytest.mark.asyncio
    async def test_uses_config_parameters(
        self,
        mock_anthropic_client: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Test that API calls use configuration parameters."""
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        config = JudgeConfig(model="claude-opus-4-20250514", temperature=0.5, max_tokens=2048)
        judge = ClaudeJudge(config=config, api_key="test-key")

        await judge.evaluate_faithfulness(
            response="Test",
            context=["Context"],
        )

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-20250514"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 2048


class TestClaudeJudgeExtractClaims:
    """Tests for extract_claims method."""

    @pytest.mark.asyncio
    async def test_extract_claims_success(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test successful claim extraction."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="text",
                text='{"claims": ["Paris is the capital", "France is in Europe"]}',
            )
        ]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 30
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.extract_claims("Paris is the capital of France.")

        assert isinstance(result, ClaimsResult)
        assert len(result.claims) == 2
        assert "Paris is the capital" in result.claims
        assert "France is in Europe" in result.claims
        assert result.tokens_used == 80

    @pytest.mark.asyncio
    async def test_extract_claims_empty_response(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test claim extraction from empty response."""
        judge = ClaudeJudge(api_key="test-key")
        result = await judge.extract_claims("")

        assert result.claims == []
        assert result.tokens_used == 0
        # Should not call the API
        mock_anthropic_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_anthropic_client")
    async def test_extract_claims_whitespace_only(self) -> None:
        """Test claim extraction from whitespace-only response."""
        judge = ClaudeJudge(api_key="test-key")
        result = await judge.extract_claims("   \n\t  ")

        assert result.claims == []
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_extract_claims_invalid_claims_type(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test handling of invalid claims type in response."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text='{"claims": "not a list"}')]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 30
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        with pytest.raises(JudgeResponseError, match="Expected 'claims' to be a list"):
            await judge.extract_claims("Some response")

    @pytest.mark.asyncio
    async def test_extract_claims_filters_empty_items(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test that empty claims are filtered out."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="text",
                text='{"claims": ["Valid claim", "", null, "Another claim"]}',
            )
        ]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 30
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.extract_claims("Response with claims")

        assert len(result.claims) == 2
        assert "Valid claim" in result.claims
        assert "Another claim" in result.claims


class TestClaudeJudgeVerifyClaim:
    """Tests for verify_claim method."""

    @pytest.mark.asyncio
    async def test_verify_claim_supported(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test verifying a supported claim."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="text",
                text='{"verdict": "SUPPORTED", "evidence": "Context confirms this"}',
            )
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 40
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.verify_claim(
            claim="Paris is the capital of France",
            context=["France is a country. Its capital is Paris."],
        )

        assert isinstance(result, ClaimVerdict)
        assert result.verdict == "SUPPORTED"
        assert result.evidence == "Context confirms this"

    @pytest.mark.asyncio
    async def test_verify_claim_contradicted(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test verifying a contradicted claim."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="text",
                text='{"verdict": "CONTRADICTED", "evidence": "Context says otherwise"}',
            )
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 40
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.verify_claim(
            claim="Paris is in Germany",
            context=["Paris is the capital of France."],
        )

        assert result.verdict == "CONTRADICTED"

    @pytest.mark.asyncio
    async def test_verify_claim_not_enough_info(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test verifying a claim with insufficient context."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="text",
                text='{"verdict": "NOT_ENOUGH_INFO", "evidence": "Not mentioned in context"}',
            )
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 40
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.verify_claim(
            claim="Paris has 2 million people",
            context=["Paris is the capital of France."],
        )

        assert result.verdict == "NOT_ENOUGH_INFO"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_anthropic_client")
    async def test_verify_claim_empty_context(self) -> None:
        """Test verifying a claim with empty context."""
        judge = ClaudeJudge(api_key="test-key")
        result = await judge.verify_claim(
            claim="Some claim",
            context=[],
        )

        assert result.verdict == "NOT_ENOUGH_INFO"
        assert "No context provided" in result.evidence

    @pytest.mark.asyncio
    async def test_verify_claim_invalid_verdict(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test handling of invalid verdict in response."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="text",
                text='{"verdict": "MAYBE", "evidence": "Not sure"}',
            )
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 40
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        with pytest.raises(JudgeResponseError, match="Invalid verdict"):
            await judge.verify_claim(
                claim="Some claim",
                context=["Some context"],
            )

    @pytest.mark.asyncio
    async def test_verify_claim_lowercase_verdict_normalized(
        self,
        mock_anthropic_client: MagicMock,
    ) -> None:
        """Test that lowercase verdicts are normalized to uppercase."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="text",
                text='{"verdict": "supported", "evidence": "Works with lowercase"}',
            )
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 40
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judge = ClaudeJudge(api_key="test-key")
        result = await judge.verify_claim(
            claim="Some claim",
            context=["Some context"],
        )

        assert result.verdict == "SUPPORTED"
