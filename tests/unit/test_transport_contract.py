"""Contract test between ClaudeTransport and the installed anthropic SDK.

The other transport tests mock `messages.create`, so they accept any keyword.
This test reads the real signature of the installed SDK's
`AsyncMessages.create` and fails if the transport passes a keyword the SDK
does not declare (anthropic 1.x removed `temperature`, for example). No
network: the signature test mocks the client; the close test builds a real
client and sends nothing.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.resources.messages import AsyncMessages

from ragaliq.judges.transport import ClaudeTransport


@pytest.mark.asyncio
async def test_send_passes_only_keywords_the_installed_sdk_declares() -> None:
    declared = set(inspect.signature(AsyncMessages.create).parameters) - {"self"}

    response = MagicMock()
    response.content = [MagicMock(type="text", text="ok")]
    response.usage.input_tokens = 1
    response.usage.output_tokens = 1

    with patch("ragaliq.judges.transport.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=response)
        mock_cls.return_value = client
        await ClaudeTransport(api_key="test").send("system", "user", temperature=0.3)

    passed = set(client.messages.create.call_args.kwargs)
    undeclared = sorted(passed - declared)
    assert not undeclared, f"messages.create does not declare: {undeclared}"


@pytest.mark.asyncio
async def test_aclose_closes_the_sdk_client() -> None:
    # A real client this time; no request is sent. is_closed is a method on the
    # SDK client, so the first assert also fails if the parentheses go missing.
    transport = ClaudeTransport(api_key="test")
    assert not transport._client.is_closed()

    await transport.aclose()
    assert transport._client.is_closed()

    await transport.aclose()  # closing twice is safe


@pytest.mark.asyncio
async def test_async_with_returns_the_transport_and_closes_its_client() -> None:
    transport = ClaudeTransport(api_key="test")
    async with transport as entered:
        assert entered is transport
        assert not transport._client.is_closed()
    assert transport._client.is_closed()

    # __aexit__ must close the client on the error path too.
    failing = ClaudeTransport(api_key="test")
    with pytest.raises(RuntimeError, match="boom"):
        async with failing:
            raise RuntimeError("boom")
    assert failing._client.is_closed()
