"""Canonical judge model identifiers.

Single source of truth for the judge model strings so the identifier isn't
duplicated across :class:`~ragaliq.judges.base.JudgeConfig`, the transport
``send()`` signatures, and the trace pricing table. Two tiers are provided:

- ``DEFAULT_JUDGE_MODEL`` — the cost-efficient default used for routine
  evaluation (what :class:`JudgeConfig` resolves to when unset).
- ``GOLD_STANDARD_JUDGE_MODEL`` — a higher-capability model recommended for
  complex, multi-step, or gold-standard judging flows.
"""

# Default judge model for routine evaluation (cost-efficient).
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

# Higher-capability model for complex / gold-standard judging.
GOLD_STANDARD_JUDGE_MODEL = "claude-opus-4-8"
