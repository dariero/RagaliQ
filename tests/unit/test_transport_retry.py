"""Unit tests for ClaudeTransport retry/backoff behavior.

Tests the tenacity configuration in isolation: which errors trigger retries,
how many attempts are made, and that non-retryable errors fail immediately.

These tests mock at the AsyncAnthropic.messages.create level to exercise
the actual tenacity decorator without making network calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import APIConnectionError, APIStatusError

from ragaliq.judges.base import JudgeAPIError
from ragaliq.judges.models import DEFAULT_JUDGE_MODEL, GOLD_STANDARD_JUDGE_MODEL
from ragaliq.judges.transport import ClaudeTransport


def _make_api_status_error(status_code: int, message: str = "error") -> APIStatusError:
    """Build an APIStatusError with the given status code."""
    resp = MagicMock()
    resp.status_code = status_code
    return APIStatusError(message=message, response=resp, body={"error": message})


def _make_success_response():
    """Build a mock Anthropic Message with valid content."""
    response = MagicMock()
    response.content = [MagicMock(type="text", text="response text")]
    response.usage.input_tokens = 40
    response.usage.output_tokens = 20
    return response


@pytest.fixture
def mock_client():
    """Patch AsyncAnthropic and return the mock client instance."""
    with patch("ragaliq.judges.transport.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


class TestRetryOn429:
    """429 Rate Limit errors must trigger retries."""

    @pytest.mark.asyncio
    async def test_429_retried_3_times_then_raises(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=_make_api_status_error(429, "rate limited")
        )
        transport = ClaudeTransport(api_key="test")

        with pytest.raises(JudgeAPIError) as exc_info:
            await transport.send("system", "user")

        assert exc_info.value.status_code == 429
        assert mock_client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_429_succeeds_on_second_attempt(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=[
                _make_api_status_error(429),
                _make_success_response(),
            ]
        )
        transport = ClaudeTransport(api_key="test")

        result = await transport.send("system", "user")

        assert result.text == "response text"
        assert mock_client.messages.create.call_count == 2


class TestRetryOn5xx:
    """5xx Server errors must trigger retries."""

    @pytest.mark.asyncio
    async def test_500_retried_3_times(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=_make_api_status_error(500, "internal server error")
        )
        transport = ClaudeTransport(api_key="test")

        with pytest.raises(JudgeAPIError):
            await transport.send("system", "user")

        assert mock_client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_502_retried(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=_make_api_status_error(502, "bad gateway")
        )
        transport = ClaudeTransport(api_key="test")

        with pytest.raises(JudgeAPIError):
            await transport.send("system", "user")

        assert mock_client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_503_retried(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=_make_api_status_error(503, "service unavailable")
        )
        transport = ClaudeTransport(api_key="test")

        with pytest.raises(JudgeAPIError):
            await transport.send("system", "user")

        assert mock_client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_500_succeeds_on_third_attempt(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=[
                _make_api_status_error(500),
                _make_api_status_error(500),
                _make_success_response(),
            ]
        )
        transport = ClaudeTransport(api_key="test")

        result = await transport.send("system", "user")

        assert result.text == "response text"
        assert mock_client.messages.create.call_count == 3


class TestRetryOnConnectionError:
    """APIConnectionError must trigger retries."""

    @pytest.mark.asyncio
    async def test_connection_error_retried_3_times(self, mock_client):
        mock_client.messages.create = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))
        transport = ClaudeTransport(api_key="test")

        with pytest.raises(JudgeAPIError, match="Connection to Claude API failed"):
            await transport.send("system", "user")

        assert mock_client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_connection_error_succeeds_on_retry(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=[
                APIConnectionError(request=MagicMock()),
                _make_success_response(),
            ]
        )
        transport = ClaudeTransport(api_key="test")

        result = await transport.send("system", "user")

        assert result.text == "response text"
        assert mock_client.messages.create.call_count == 2


class TestNoRetryOnClientErrors:
    """4xx errors (except 429) must NOT be retried."""

    @pytest.mark.asyncio
    async def test_400_not_retried(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=_make_api_status_error(400, "bad request")
        )
        transport = ClaudeTransport(api_key="test")

        with pytest.raises(JudgeAPIError) as exc_info:
            await transport.send("system", "user")

        assert exc_info.value.status_code == 400
        assert mock_client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_401_not_retried(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=_make_api_status_error(401, "unauthorized")
        )
        transport = ClaudeTransport(api_key="test")

        with pytest.raises(JudgeAPIError) as exc_info:
            await transport.send("system", "user")

        assert exc_info.value.status_code == 401
        assert mock_client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_403_not_retried(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=_make_api_status_error(403, "forbidden")
        )
        transport = ClaudeTransport(api_key="test")

        with pytest.raises(JudgeAPIError) as exc_info:
            await transport.send("system", "user")

        assert exc_info.value.status_code == 403
        assert mock_client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_404_not_retried(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=_make_api_status_error(404, "not found")
        )
        transport = ClaudeTransport(api_key="test")

        with pytest.raises(JudgeAPIError) as exc_info:
            await transport.send("system", "user")

        assert exc_info.value.status_code == 404
        assert mock_client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_422_not_retried(self, mock_client):
        mock_client.messages.create = AsyncMock(
            side_effect=_make_api_status_error(422, "unprocessable entity")
        )
        transport = ClaudeTransport(api_key="test")

        with pytest.raises(JudgeAPIError):
            await transport.send("system", "user")

        assert mock_client.messages.create.call_count == 1


class TestTransportResponseMapping:
    """Verify transport correctly maps API responses to TransportResponse."""

    @pytest.mark.asyncio
    async def test_successful_response_mapping(self, mock_client):
        mock_client.messages.create = AsyncMock(return_value=_make_success_response())
        transport = ClaudeTransport(api_key="test")

        result = await transport.send("sys", "usr", model="test-model")

        assert result.text == "response text"
        assert result.input_tokens == 40
        assert result.output_tokens == 20
        assert result.model == "test-model"

    @pytest.mark.asyncio
    async def test_empty_response_raises(self, mock_client):
        response = MagicMock()
        response.content = []
        mock_client.messages.create = AsyncMock(return_value=response)
        transport = ClaudeTransport(api_key="test")

        from ragaliq.judges.base import JudgeResponseError

        with pytest.raises(JudgeResponseError, match="Empty response"):
            await transport.send("sys", "usr")

    @pytest.mark.asyncio
    async def test_non_text_content_raises(self, mock_client):
        response = MagicMock()
        response.content = [MagicMock(type="image")]
        mock_client.messages.create = AsyncMock(return_value=response)
        transport = ClaudeTransport(api_key="test")

        from ragaliq.judges.base import JudgeResponseError

        with pytest.raises(JudgeResponseError, match="Expected text response"):
            await transport.send("sys", "usr")

    @pytest.mark.asyncio
    async def test_mixed_content_uses_text_block(self, mock_client):
        response = MagicMock()
        response.content = [
            MagicMock(type="thinking"),
            MagicMock(type="text", text='{"score": 1.0, "reasoning": "ok"}'),
        ]
        response.usage.input_tokens = 40
        response.usage.output_tokens = 20
        mock_client.messages.create = AsyncMock(return_value=response)
        transport = ClaudeTransport(api_key="test")

        result = await transport.send("sys", "usr")
        assert result.text == '{"score": 1.0, "reasoning": "ok"}'

    @pytest.mark.asyncio
    async def test_max_tokens_stop_raises_naming_model_and_limit(self, mock_client):
        response = _make_success_response()
        response.stop_reason = "max_tokens"
        mock_client.messages.create = AsyncMock(return_value=response)
        transport = ClaudeTransport(api_key="test")

        from ragaliq.judges.base import JudgeResponseError

        with pytest.raises(JudgeResponseError, match="test-model stopped at max_tokens=2048"):
            await transport.send("sys", "usr", model="test-model", max_tokens=2048)
        assert mock_client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_max_tokens_stop_with_only_thinking_names_the_limit(self, mock_client):
        response = MagicMock()
        response.content = [MagicMock(type="thinking")]
        response.stop_reason = "max_tokens"
        mock_client.messages.create = AsyncMock(return_value=response)
        transport = ClaudeTransport(api_key="test")

        from ragaliq.judges.base import JudgeResponseError

        with pytest.raises(JudgeResponseError) as exc_info:
            await transport.send("sys", "usr", model=GOLD_STANDARD_JUDGE_MODEL)
        assert "claude-opus-5-5 stopped at max_tokens=16000" in str(exc_info.value)
        assert "Expected text response" not in str(exc_info.value)


class TestTransportRequestSettings:
    """Verify which sampling and thinking settings reach messages.create."""

    @pytest.mark.asyncio
    async def test_no_temperature_sends_no_extra_body(self, mock_client):
        mock_client.messages.create = AsyncMock(return_value=_make_success_response())
        transport = ClaudeTransport(api_key="test")

        await transport.send("sys", "usr")

        assert mock_client.messages.create.call_args.kwargs["extra_body"] is None

    @pytest.mark.asyncio
    async def test_explicit_temperature_goes_into_extra_body(self, mock_client):
        mock_client.messages.create = AsyncMock(return_value=_make_success_response())
        transport = ClaudeTransport(api_key="test")

        await transport.send("sys", "usr", temperature=0.0)

        assert mock_client.messages.create.call_args.kwargs["extra_body"] == {"temperature": 0.0}

    @pytest.mark.asyncio
    async def test_default_tier_turns_thinking_off_with_between_tools(self, mock_client):
        mock_client.messages.create = AsyncMock(return_value=_make_success_response())
        transport = ClaudeTransport(api_key="test")

        await transport.send("sys", "usr", model=DEFAULT_JUDGE_MODEL)

        assert mock_client.messages.create.call_args.kwargs["thinking"] == {"type": "between_tools"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", [GOLD_STANDARD_JUDGE_MODEL, "claude-sonnet-4-6"])
    async def test_other_models_get_no_thinking_keyword(self, mock_client, model):
        mock_client.messages.create = AsyncMock(return_value=_make_success_response())
        transport = ClaudeTransport(api_key="test")

        await transport.send("sys", "usr", model=model)

        assert "thinking" not in mock_client.messages.create.call_args.kwargs
