"""Fixtures and configuration for the meta-evaluation suite.

The live tests here make real API calls and cost money, so they are:
  - gated behind the `meta` marker
  - enabled explicitly with RAGALIQ_RUN_META=1
  - skipped automatically when ANTHROPIC_API_KEY is absent

The harness/metric tests (test_meta_metrics.py) are NOT gated — they run in CI
with no network using a StubJudge.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from tests.meta.meta_metrics import (
    GoldenCase,
    GoldenClaim,
    load_golden_cases,
    load_golden_claims,
    resolve_meta_model,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ragaliq.judges.base import LLMJudge


@pytest.fixture
def golden_claims() -> list[GoldenClaim]:
    """The human-labelled claim-verdict golden set."""
    return load_golden_claims()


@pytest.fixture
def golden_cases() -> list[GoldenCase]:
    """The case-level faithfulness golden set."""
    return load_golden_cases()


@pytest.fixture
async def live_judge() -> AsyncIterator[LLMJudge]:
    """A real ClaudeJudge, closed at teardown.

    Skips when the paid tier was not requested. Fails — deliberately, not
    skips — when it *was* requested but cannot run, so that "the gate could
    not return a verdict" is never reported as success.

    The fixture is async so the HTTP client is closed on the test's own event
    loop. A client left to the garbage collector is closed on a later loop and
    logs "Task exception was never retrieved ... Event loop is closed".

    RAGALIQ_META_MODEL picks the judge tier: unset keeps the default model, and
    a value that names neither tier fails rather than running the default.
    """
    if os.getenv("RAGALIQ_RUN_META") != "1":
        pytest.skip("set RAGALIQ_RUN_META=1 to enable paid meta-evaluation")
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.fail(
            "RAGALIQ_RUN_META=1 was set but ANTHROPIC_API_KEY is missing. "
            "The gate cannot return a verdict, so it must not report success.",
            pytrace=False,
        )
    from ragaliq.judges.base import JudgeConfig
    from ragaliq.judges.claude import ClaudeJudge

    try:
        model = resolve_meta_model(os.getenv("RAGALIQ_META_MODEL"))
    except ValueError as exc:
        pytest.fail(str(exc), pytrace=False)
    config = JudgeConfig(model=model) if model is not None else None

    async with ClaudeJudge(config=config) as judge:
        yield judge
