"""Transport layer for LLM judge API calls.

A transport owns only provider HTTP calls, retries, and token counting. Prompt
building, response parsing, and score clamping all live in `BaseJudge`.
"""

import asyncio
import threading
import weakref
from collections.abc import AsyncGenerator
from types import TracebackType
from typing import Protocol, Self

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from ragaliq.judges.models import DEFAULT_JUDGE_MODEL, JUDGE_THINKING


class TransportResponse(BaseModel):
    """Normalized response every transport returns: text plus token/model metadata."""

    text: str = Field(..., description="Raw text response from LLM")
    input_tokens: int = Field(..., ge=0, description="Tokens in prompt")
    output_tokens: int = Field(..., ge=0, description="Output tokens: thinking plus text")
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
        temperature: float | None = None,
        max_tokens: int = 16000,
    ) -> TransportResponse:
        """Send the prompts to the LLM and return a normalized TransportResponse."""
        ...


async def _close_at_loop_shutdown(
    client: AsyncAnthropic,
    owner: weakref.ref[ClaudeTransport],
    loop: asyncio.AbstractEventLoop,
) -> AsyncGenerator[None]:
    """Hold `client` open until its event loop shuts down, then close it there.

    asyncio.run() and asyncio.Runner.close() call loop.shutdown_asyncgens() before
    they close the loop, and it closes every async generator still suspended in that
    loop. The finally block below then runs while the loop is still running.
    """
    try:
        yield
    finally:
        try:
            await client.close()
        finally:
            # The generator holds its loop, so the transport's entry would keep the
            # closed loop and its client alive. A weak reference adds no cycle.
            transport = owner()
            if transport is not None:
                transport._forget(loop)


class ClaudeTransport:
    """
    Transport implementation for Anthropic's Claude API.

    Handles API calls with automatic retry on rate limits and server errors.
    Uses tenacity for exponential backoff (1s, 2s, 4s, max 10s).

    The transport keeps one SDK client per event loop, created on the first send
    in that loop. A pooled HTTP connection belongs to the loop that opened it, and
    the sync API (`RagaliQ.evaluate()`, the CLI) runs every call in a new loop with
    asyncio.run(). A client shared across loops would send each new loop's first
    request on a dead connection and could not be closed from a later loop (#179).
    Each loop's client is closed when that loop shuts down, or earlier by
    `aclose()` in that loop. A loop closed without loop.shutdown_asyncgens() keeps
    its client, and the client's sockets, until the transport is garbage-collected.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        # Bound now, so a patch of AsyncAnthropic that is active while the transport
        # is built also covers the clients it creates later.
        self._client_class = AsyncAnthropic
        self._closed = False
        self._clients: dict[
            asyncio.AbstractEventLoop, tuple[AsyncAnthropic, AsyncGenerator[None]]
        ] = {}
        # Loops in different threads can bind at the same time.
        self._clients_lock = threading.Lock()

    async def _client_for_loop(self) -> AsyncAnthropic:
        """Return the running loop's client, creating it on the first call in that loop.

        Raises:
            RuntimeError: If the transport was closed.
        """
        if self._closed:
            raise RuntimeError("ClaudeTransport is closed")
        loop = asyncio.get_running_loop()
        with self._clients_lock:
            bound = self._clients.get(loop)
        if bound is not None:
            return bound[0]
        client = self._client_class(api_key=self._api_key)
        closer = _close_at_loop_shutdown(client, weakref.ref(self), loop)
        # The first step registers the generator with this loop and runs to its
        # yield without suspending, so no other task of this loop can bind between
        # the lookup above and the store below.
        await anext(closer)
        with self._clients_lock:
            self._clients[loop] = (client, closer)
        return client

    def _forget(self, loop: asyncio.AbstractEventLoop) -> None:
        """Drop the entry of a loop whose client was closed."""
        with self._clients_lock:
            self._clients.pop(loop, None)

    async def send(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = DEFAULT_JUDGE_MODEL,
        temperature: float | None = None,
        max_tokens: int = 16000,
    ) -> TransportResponse:
        """Call the Claude API, retrying on 429 / 5xx / connection errors.

        Raises:
            JudgeAPIError: If the call fails after retries are exhausted.
            JudgeResponseError: If the response stopped at ``max_tokens``, was a refusal,
                hit the model's context window, or holds no text.
            RuntimeError: If the transport was closed.
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

        # anthropic has no `temperature` keyword on messages.create; under API version
        # 2026-10-21 (anthropic 2.x) any value returns HTTP 400 (ADR-149, amendment).
        extra_body = {"temperature": temperature} if temperature is not None else None
        # A model with no JUDGE_THINKING entry is sent no `thinking` keyword at all.
        thinking = JUDGE_THINKING.get(model)
        client = await self._client_for_loop()

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
            if thinking is None:
                return await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": user_prompt}],
                    system=system_prompt,
                    extra_body=extra_body,
                )
            return await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
                thinking=thinking,
                extra_body=extra_body,
            )

        try:
            response: Message = await _call_with_retry()
            # A rejected response was still billed, so each rejection below carries
            # its usage for the judge's trace.
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # max_tokens covers thinking plus text, so a cut-off response may hold
            # only thinking blocks or truncated JSON. Name the limit, not the symptom.
            if response.stop_reason == "max_tokens":
                raise JudgeResponseError(
                    f"Claude response from {model} stopped at max_tokens={max_tokens}",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            # These two stops also leave no usable verdict. Without them the error
            # names a symptom: missing text, or text that does not parse as JSON.
            if response.stop_reason == "refusal":
                raise JudgeResponseError(
                    f"Claude response from {model} was a refusal (stop_reason=refusal)",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            if response.stop_reason == "model_context_window_exceeded":
                raise JudgeResponseError(
                    f"Claude response from {model} stopped at the model's context window "
                    "(stop_reason=model_context_window_exceeded)",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            # Extract text content. Claude may return non-text blocks (for example,
            # thinking/tool blocks) before the final text answer.
            if not response.content:
                raise JudgeResponseError(
                    "Empty response from Claude API",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            text_blocks = [block.text for block in response.content if block.type == "text"]
            if not text_blocks:
                block_types = ", ".join(block.type for block in response.content)
                raise JudgeResponseError(
                    f"Expected text response, got {block_types}",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

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

    async def aclose(self) -> None:
        """Close the running loop's client. The transport cannot send after this.

        A client another event loop opened is closed when that loop shuts down: its
        connections cannot be closed from this loop. Closing twice is safe.
        """
        self._closed = True
        with self._clients_lock:
            bound = self._clients.pop(asyncio.get_running_loop(), None)
        if bound is not None:
            # Closing the generator runs its finally block, which closes the client.
            await bound[1].aclose()

    async def __aenter__(self) -> Self:
        """Return the transport; the HTTP client is created on the first send."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the HTTP client when leaving an `async with` block."""
        await self.aclose()
