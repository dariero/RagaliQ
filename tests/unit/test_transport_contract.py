"""Contract test between ClaudeTransport and the installed anthropic SDK.

The other transport tests mock `messages.create`, so they accept any keyword.
This test reads the real signature of the installed SDK's
`AsyncMessages.create` and fails if the transport passes a keyword the SDK
does not declare (anthropic 1.x removed `temperature`, for example). No
network: the client is mocked, only the signature is real.
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
