"""Transport layer for LLM judge API calls.

A transport owns only provider HTTP calls, retries, and token counting. Prompt
building, response parsing, and score clamping all live in `BaseJudge`.
"""

from typing import Protocol

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from ragaliq.judges.models import DEFAULT_JUDGE_MODEL


class TransportResponse(BaseModel):
    """Normalized response every transport returns: text plus token/model metadata."""

    text: str = Field(..., description="Raw text response from LLM")
    input_tokens: int = Field(..., ge=0, description="Tokens in prompt")
    output_tokens: int = Field(..., ge=0, description="Tokens in response")
    model: str = Field(..., description="Model identifier used")

    model_config = {"frozen": True, "extra": "forbid"}


class JudgeTransport(Protocol):
    """Structural type for a transport `BaseJudge` can call.

    Any object with a matching `send()` works, so providers can be swapped
    without touching judge logic.
    """

    async def send(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = DEFAULT_JUDGE_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> TransportResponse:
        """Send the prompts to the LLM and return a normalized TransportResponse."""
        ...


class ClaudeTransport:
    """
    Transport implementation for Anthropic's Claude API.

    Handles API calls with automatic retry on rate limits and server errors.
    Uses tenacity for exponential backoff (1s, 2s, 4s, max 10s).
    """

    def __init__(self, api_key: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

    async def send(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = DEFAULT_JUDGE_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> TransportResponse:
        """Call the Claude API, retrying on 429 / 5xx / connection errors.

        Raises:
            JudgeAPIError: If the call fails after retries are exhausted.
        """
        from anthropic import APIConnectionError, APIStatusError
        from anthropic.types import Message
        from tenacity import (
            retry,
            retry_if_exception,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
            wait_random,
        )

        from ragaliq.judges.base import JudgeAPIError, JudgeResponseError

        def _is_retryable_api_error(exc: BaseException) -> bool:
            """Check if an exception is a retryable API status error (429 or 5xx)."""
            return isinstance(exc, APIStatusError) and (
                exc.status_code == 429 or exc.status_code >= 500
            )

        @retry(
            retry=(
                retry_if_exception_type(APIConnectionError)
                | retry_if_exception(_is_retryable_api_error)
            ),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10) + wait_random(min=0, max=1),
            reraise=True,
        )
        async def _call_with_retry() -> Message:
            """Make the API call with retry logic."""
            return await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
            )

        try:
            response: Message = await _call_with_retry()

            # Extract text content. Claude may return non-text blocks (for example,
            # thinking/tool blocks) before the final text answer.
            if not response.content:
                raise JudgeResponseError("Empty response from Claude API")

            text_blocks = [block.text for block in response.content if block.type == "text"]
            if not text_blocks:
                block_types = ", ".join(block.type for block in response.content)
                raise JudgeResponseError(f"Expected text response, got {block_types}")

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            return TransportResponse(
                text="\n".join(text_blocks),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
            )

        except APIStatusError as e:
            # Map all status errors to our exception hierarchy
            # (tenacity has already retried 429/5xx if needed)
            raise JudgeAPIError(
                f"Claude API error: {e.message}",
                status_code=e.status_code,
            ) from e
        except APIConnectionError as e:
            # Exhausted retries on connection errors
            raise JudgeAPIError(f"Connection to Claude API failed: {e}") from e
