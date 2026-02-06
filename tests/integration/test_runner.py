"""Live integration tests for RagaliQ runner (require external API keys)."""

import os

import pytest

from ragaliq.core.runner import RagaliQ
from ragaliq.judges.claude import ClaudeJudge


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping live integration test",
)
class TestLiveIntegration:
    """Live integration tests that require API key."""

    def test_claude_judge_init_with_env_key(self):
        """ClaudeJudge initializes successfully with env API key."""
        runner = RagaliQ()
        runner._init_judge()

        assert runner._judge is not None
        assert isinstance(runner._judge, ClaudeJudge)
