"""Guard tests: judge model defaults stay centralized on the constants.

Closes the audit's E1 gap — nothing previously asserted that the transport
``send()`` signature defaults stayed in sync with ``JudgeConfig``'s default,
so they could silently drift apart.
"""

import inspect

from ragaliq.judges.base import JudgeConfig
from ragaliq.judges.models import DEFAULT_JUDGE_MODEL, GOLD_STANDARD_JUDGE_MODEL, JUDGE_THINKING
from ragaliq.judges.transport import ClaudeTransport, JudgeTransport


def test_judgeconfig_default_uses_constant() -> None:
    """The JudgeConfig default model resolves to the centralized constant."""
    assert JudgeConfig().model == DEFAULT_JUDGE_MODEL


def test_transport_signature_defaults_match_constant() -> None:
    """Both transport send() signatures default to the same centralized model."""
    for send in (JudgeTransport.send, ClaudeTransport.send):
        default = inspect.signature(send).parameters["model"].default
        assert default == DEFAULT_JUDGE_MODEL


def test_transport_signature_defaults_match_judgeconfig() -> None:
    """Both transport send() signatures default to JudgeConfig's temperature and max_tokens."""
    for send in (JudgeTransport.send, ClaudeTransport.send):
        parameters = inspect.signature(send).parameters
        for name in ("temperature", "max_tokens"):
            assert parameters[name].default == JudgeConfig.model_fields[name].default


def test_only_the_default_tier_has_a_thinking_setting() -> None:
    """Only the default tier is sent a thinking setting; every other model gets none."""
    assert JUDGE_THINKING == {DEFAULT_JUDGE_MODEL: {"type": "between_tools"}}


def test_model_constants_have_expected_values() -> None:
    """The two tiers are distinct and pinned to known-valid model ids."""
    assert DEFAULT_JUDGE_MODEL == "claude-sonnet-5-5"
    assert GOLD_STANDARD_JUDGE_MODEL == "claude-opus-5-5"
    assert DEFAULT_JUDGE_MODEL != GOLD_STANDARD_JUDGE_MODEL
