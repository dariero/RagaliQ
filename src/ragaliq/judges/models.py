"""Canonical judge model identifiers.

Single source of truth for the judge model strings so the identifier isn't
duplicated across :class:`~ragaliq.judges.base.JudgeConfig`, the transport
``send()`` signatures, and the trace pricing table. Two tiers are provided:

- ``DEFAULT_JUDGE_MODEL`` — the cost-efficient default used for routine
  evaluation (what :class:`JudgeConfig` resolves to when unset).
- ``GOLD_STANDARD_JUDGE_MODEL`` — a higher-capability model recommended for
  complex, multi-step, or gold-standard judging flows.

``JUDGE_THINKING`` holds the ``thinking`` setting the transport sends with a
model; a model with no entry is sent no ``thinking`` field.
"""

from anthropic.types import ThinkingConfigParam

# Default judge model for routine evaluation (cost-efficient).
DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"

# Higher-capability model for complex / gold-standard judging.
GOLD_STANDARD_JUDGE_MODEL = "claude-opus-5-5"

# The default tier runs without up-front thinking. The gold tier cannot turn
# thinking off, so it gets no entry and runs with the API's adaptive thinking.
JUDGE_THINKING: dict[str, ThinkingConfigParam] = {
    DEFAULT_JUDGE_MODEL: {"type": "between_tools"},
}
