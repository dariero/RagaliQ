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
)

if TYPE_CHECKING:
    from ragaliq.judges.base import LLMJudge


def pytest_configure(config: pytest.Config) -> None:
    """Register the `meta` marker for live judge-quality benchmarks."""
    config.addinivalue_line(
        "markers",
        "meta: live judge meta-evaluation against the golden set (needs ANTHROPIC_API_KEY)",
    )


@pytest.fixture
def golden_claims() -> list[GoldenClaim]:
    """The human-labelled claim-verdict golden set."""
    return load_golden_claims()


@pytest.fixture
def golden_cases() -> list[GoldenCase]:
    """The case-level faithfulness golden set."""
    return load_golden_cases()


@pytest.fixture
def live_judge() -> LLMJudge:
    """A real ClaudeJudge. Skips the test cleanly if no API key is configured."""
    if os.getenv("RAGALIQ_RUN_META") != "1":
        pytest.skip("set RAGALIQ_RUN_META=1 to enable paid meta-evaluation")
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; skipping live meta-evaluation")
    from ragaliq.judges.claude import ClaudeJudge

    return ClaudeJudge()
