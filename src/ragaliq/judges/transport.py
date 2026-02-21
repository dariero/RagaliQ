"""
Transport layer for LLM judge API calls.

This module defines the transport protocol and implementations for
communicating with LLM APIs. The transport layer handles:
- Making HTTP requests to LLM providers
- Retry logic on failures
- Token counting

It does NOT handle:
- Prompt building (done by BaseJudge)
- Response parsing (done by BaseJudge)
- Score clamping (done by BaseJudge)
"""

from typing import Protocol

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field


class TransportResponse(BaseModel):
    """
    Response from an LLM transport call.

    This is a normalized response format that all transports must return.
    It contains the raw text output and metadata about token usage.

    Attributes:
        text: The raw text response from the LLM.
        input_tokens: Number of tokens in the prompt.
        output_tokens: Number of tokens in the response.
        model: The model identifier that was used.
    """

    text: str = Field(..., description="Raw text response from LLM")
    input_tokens: int = Field(..., ge=0, description="Tokens in prompt")
    output_tokens: int = Field(..., ge=0, description="Tokens in response")
    model: str = Field(..., description="Model identifier used")

    model_config = {"frozen": True, "extra": "forbid"}


class JudgeTransport(Protocol):
    """
    Protocol for LLM transport implementations.

    Any class that implements this protocol can be used as a transport
    for BaseJudge. This enables swapping out API providers without
    changing the judge logic.

    Example:
        class MyCustomTransport:
            async def send(
                self,
                system_prompt: str,
                user_prompt: str,
                model: str = "my-model",
                temperature: float = 0.0,
                max_tokens: int = 1024,
            ) -> TransportResponse:
                # Call your LLM API
                response = await my_api.complete(...)
                return TransportResponse(
                    text=response.text,
                    input_tokens=response.usage.input,
                    output_tokens=response.usage.output,
                    model=model,
                )
    """

    async def send(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> TransportResponse:
        """
        Send a prompt to the LLM and return the response.

        Args:
            system_prompt: System message defining the LLM's role.
            user_prompt: User message with the task.
            model: Model identifier to use.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.

        Returns:
            TransportResponse with text and token counts.

        Raises:
            Exception: If the API call fails (transport-specific exception).
        """
        ...


class ClaudeTransport:
    """
    Transport implementation for Anthropic's Claude API.

    Handles API calls with automatic retry on rate limits and server errors.
    Uses tenacity for exponential backoff (1s, 2s, 4s, max 10s).
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialize Claude transport.

        Args:
            api_key: Anthropic API key for authentication.
        """
        self._client = AsyncAnthropic(api_key=api_key)

    async def send(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> TransportResponse:
        """
        Send a prompt to Claude API with retry logic.

        Automatically retries on:
        - 429 Rate Limit errors
        - 500+ Server errors
        - Connection errors

        Args:
            system_prompt: System message defining Claude's role.
            user_prompt: User message with the task.
            model: Claude model identifier.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.

        Returns:
            TransportResponse with text and token counts.

        Raises:
            JudgeAPIError: If API call fails after retries.
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

            # Calculate total tokens
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
