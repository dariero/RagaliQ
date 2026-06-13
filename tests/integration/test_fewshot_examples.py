"""Integration test: few-shot examples actually reach the model.

Regression guard for the "unwired examples" defect (audit B1). Every prompt
template ships an ``examples:`` block, but they were never injected into any
prompt — ``get_examples_text()`` had a green unit test while the feature was
dead end-to-end. These tests drive a judge through a recording transport and
assert that example content appears in the system prompt sent over the wire.
"""

from __future__ import annotations

import pytest

from ragaliq.judges.base_judge import BaseJudge
from ragaliq.judges.transport import TransportResponse


class RecordingTransport:
    """JudgeTransport stub that records each send() and returns canned JSON."""

    def __init__(self, response_text: str) -> None:
        self.calls: list[tuple[str, str]] = []
        self._response_text = response_text

    async def send(
        self, system_prompt: str, user_prompt: str, **_kwargs: object
    ) -> TransportResponse:
        self.calls.append((system_prompt, user_prompt))
        return TransportResponse(
            text=self._response_text, input_tokens=10, output_tokens=5, model="test-model"
        )


@pytest.mark.asyncio
async def test_faithfulness_system_prompt_includes_examples() -> None:
    """Faithfulness must send its template's few-shot examples to the transport."""
    transport = RecordingTransport('{"score": 1.0, "reasoning": "ok"}')
    judge = BaseJudge(transport=transport)

    await judge.evaluate_faithfulness(
        response="Paris is the capital of France.",
        context=["Paris is the capital of France."],
    )

    assert transport.calls, "judge made no transport call"
    system_prompt = transport.calls[0][0]
    # "Examples:" is emitted only by the get_examples_text() wiring...
    assert "Examples:" in system_prompt
    # ..."Seine River" appears only in faithfulness.yaml's first example, so its
    # presence proves the authored example (not just any text) was injected.
    assert "Seine River" in system_prompt


@pytest.mark.asyncio
async def test_verify_claim_system_prompt_includes_examples() -> None:
    """Claim verification must also send its template's few-shot examples."""
    transport = RecordingTransport('{"verdict": "SUPPORTED", "evidence": "ok"}')
    judge = BaseJudge(transport=transport)

    await judge.verify_claim(
        claim="Paris is the capital of France.",
        context=["France's capital city is Paris."],
    )

    assert transport.calls
    assert "Examples:" in transport.calls[0][0]


@pytest.mark.asyncio
async def test_relevance_system_prompt_includes_examples() -> None:
    """Relevance must send its template's few-shot examples."""
    transport = RecordingTransport('{"score": 0.9, "reasoning": "ok"}')
    judge = BaseJudge(transport=transport)

    await judge.evaluate_relevance(
        query="What is the capital of France?",
        response="The capital of France is Paris.",
    )

    assert transport.calls
    assert "Examples:" in transport.calls[0][0]
