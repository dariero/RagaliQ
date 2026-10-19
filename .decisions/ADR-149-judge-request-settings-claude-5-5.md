# ADR-149: Judge Request Settings for Claude Sonnet 5.5 and Opus 5.5

**Status:** Accepted
**Date:** 2026-10-19
**Issue:** #149 — [FEAT] Judge models to Claude Sonnet 5.5 and Opus 5.5

## Context

The judge has two tiers in `src/ragaliq/judges/models.py`: `DEFAULT_JUDGE_MODEL` was `claude-sonnet-4-6` and `GOLD_STANDARD_JUDGE_MODEL` was `claude-opus-4-8`. Every request sent `temperature=0.0` by default (`JudgeConfig.temperature`; the transport puts it in `extra_body` because anthropic 1.x dropped the keyword), `max_tokens=1024` (bound 4096), and no `thinking` field.

The tiers move to Claude Sonnet 5.5 and Claude Opus 5.5. These lines of the Claude docs (read 2026-10-19) decide which request settings those models accept:

- Model IDs, https://platform.claude.com/docs/en/models/overview: the "Claude API ID" row lists `claude-opus-5-5` and `claude-sonnet-5-5`.
- Sampling, https://platform.claude.com/docs/en/models/sonnet-5-5/overview: "Setting `temperature`, `top_p`, or `top_k` to a non-default value returns a 400 error."
- Sampling, https://platform.claude.com/docs/en/about-claude/model-deprecations: `temperature`, `top_p`, `top_k` "Returns a 400 error when set to a non-default value on Claude 4.7 and later models". The same page lists `claude-sonnet-4-6` and `claude-opus-4-8` as Active.
- Thinking on Sonnet 5.5, https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide:
  - "On Claude Sonnet 5.5, a request with no `thinking` field runs with adaptive thinking".
  - "To turn off up-front thinking on Claude Sonnet 5.5, send `thinking: {"type": "between_tools"}`." "Without tools, the response contains only text."
  - "It's accepted at `low`, `medium`, and `high` effort." "`between_tools` takes no other field".
  - "The default on the Claude API is `high`." (effort)
  - `max_tokens` "covers thinking plus text, and thinking tokens are billed as output tokens."
  - "Against Claude Sonnet 4.6, Claude Sonnet 4.5, and Claude Haiku 4.5, the same text produces about 30% more tokens".
- Thinking on Opus 5.5, https://platform.claude.com/docs/en/models/opus-5-5/migration-guide:
  - "Send no `thinking` field, or send `thinking: {"type": "adaptive"}`, which is equivalent: adaptive thinking is always on."
  - Effort: "the default is `medium`".
  - "`max_tokens` remains a hard limit on total output, thinking plus response text".
- Prices, https://platform.claude.com/docs/en/about-claude/pricing (input / output per MTok): Claude Sonnet 5.5 $2 / $10, Claude Opus 5.5 $4 / $20, Claude Sonnet 4.6 $3 / $15, Claude Opus 4.8 $5 / $25.
- Output budget:
  - https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost: "`max_tokens` is a hard cap on total output for the request, thinking and response text combined." "A `max_tokens` sized for a response with no thinking is often too small once Claude starts thinking on hard requests."
  - https://platform.claude.com/docs/en/api/rate-limits: "The `max_tokens` parameter does not factor into OTPM rate limit calculations, so there is no rate limit downside to setting a higher `max_tokens` value."
  - https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python: "The SDK will throw a `ValueError` if a non-streaming request is expected to take longer than approximately 10 minutes." In the installed anthropic 1.13.0 (`_base_client.py`, `_calculate_nonstreaming_timeout`) that is any `max_tokens` above 21,333 on the default timeout. `ClaudeTransport` sends non-streaming requests on the default timeout.

## Proposed Solution

1. **Model tiers.** `DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"` and `GOLD_STANDARD_JUDGE_MODEL = "claude-opus-5-5"`. The constants keep their names, so every existing import of them still works; the only import line that changes is `transport.py`'s, which also imports `JUDGE_THINKING` (item 3).
2. **Sampling: D3a (Darie, 2026-10-19).** `JudgeConfig.temperature` becomes `float | None`, default `None`, with the same 0.0 to 1.0 bounds when set. `JudgeTransport.send`, `ClaudeTransport.send` and the pytest plugin's `LatencyInjectionTransport.send` take `temperature: float | None = None`. The transport sends `extra_body={"temperature": t}` only when `t` is not `None`. There is no model table for temperature and no code path that drops a value the caller set: an explicit value reaches the API, and on a 5.5 model the API answers 400 to any non-default value. No `top_p` or `top_k`.
3. **Thinking: DTa (Darie, 2026-10-19).** `JUDGE_THINKING` in `judges/models.py`, typed `dict[str, anthropic.types.ThinkingConfigParam]`, maps `DEFAULT_JUDGE_MODEL` to `{"type": "between_tools"}`. `ClaudeTransport` passes `thinking=` to `messages.create` only for a model with an entry; a request for any other model has no `thinking` keyword. No request carries `effort` or `output_config`, so Sonnet 5.5 runs at the API default effort `high`, which `between_tools` accepts, and Opus 5.5 runs adaptive thinking at its default effort `medium`. No new `JudgeConfig` field and no new `send()` parameter.
4. **Output budget (agent's recommendation, accepted by the session lead's silence or GO).** `JudgeConfig.max_tokens` defaults to 16000 (was 1024) with an upper bound of 21333 (was 4096), and the three `send()` signatures default to 16000. 16000 is the `max_tokens` of the docs' thinking examples under "Turn off up-front thinking" (Sonnet 5.5 migration guide), "Thinking can't be disabled" (Opus 5.5 migration guide) and "Prompt caching" (thinking-steering-and-cost); the minimal requests in the two migration guides use 4096. 16000 leaves Opus 5.5 room to think before it answers. Above 21333 the SDK refuses a non-streaming request on its default timeout with a `ValueError`, before sending it. One default serves every model. When a response's `stop_reason` is `"max_tokens"`, `ClaudeTransport` raises `JudgeResponseError` naming the model and the `max_tokens` value. It checks this before it extracts text, so a response cut off inside its thinking gets this error, not "Expected text response".
5. **Pricing.** The trace pricing table prices `DEFAULT_JUDGE_MODEL` at (2.0, 10.0) and `GOLD_STANDARD_JUDGE_MODEL` at (4.0, 20.0), and keeps literal entries for `claude-sonnet-4-6` (3.0, 15.0) and `claude-opus-4-8` (5.0, 25.0), so a caller pinned to a 4.x model keeps its price.
6. **Gold-tier measurement: DGa (Darie, 2026-10-19).** The gold tier moves now and is measured. `tests/meta/meta_metrics.py` gains `resolve_meta_model()`, and the `live_judge` fixture builds its judge from `RAGALIQ_META_MODEL`: unset keeps the default tier, either tier's model id selects that tier, and any other value (an empty string included) fails the run, so a typo never runs the default tier and reports it as the gold tier. `src/` gains nothing for this. The weekly `meta-eval.yml` run sets no such variable and keeps measuring the default tier.
7. **Measurement protocol.** The 5.5 models reject a non-default temperature, so `temperature=0` no longer stands for a repeatable judge. The 3-run protocol, first used for the B0 baseline of 2026-10-11 (`claude-sonnet-4-6`), replaces it: 3 sequential paid runs per tier, run 1 a gate, compared per claim and per case. The re-baseline against B0 uses it before this change merges.

### Consequences

- A caller pinned to a model before 4.7, such as `claude-sonnet-4-6`, now gets the API's default temperature unless they set `temperature`. Before this change 0.0 was sent by default.
- Custom transports implementing `JudgeTransport` see the new temperature type: `BaseJudge` now passes `temperature=None` by default, and `max_tokens=16000`.
- Thinking tokens are billed as output tokens, and Opus 5.5 cannot turn thinking off. Gold-tier calls can therefore produce more output tokens per request than their text alone.
- Before this change the gold tier sent `temperature=0.0` to `claude-opus-4-8`, a "4.7 and later" model. This is INFERRED to fail with HTTP 400 (the docs give no numeric default for temperature) and stays unconfirmed until the probe that runs with the paid re-baseline: one `claude-opus-4-8` request with `temperature=0.0`. The meta suite never exercised the gold tier.
- A response that the transport rejects (stopped at `max_tokens`, empty, or holding no text) is recorded in the trace with 0 input and 0 output tokens, although its tokens are billed, so `TraceCollector.total_cost_estimate` and `--ragaliq-cost-limit` leave them out. Before this change a response cut off at `max_tokens` that still held text was traced with its real tokens and failed later as unparseable JSON.

## Principles Applied

- Fail loudly: the judge sends a caller's explicit setting as given and lets the API reject it, and a cut-off response names `max_tokens` instead of failing later as unparseable JSON.
- One source of truth: the model ids and the setting each model is sent live side by side in `judges/models.py`.

## Alternatives Considered

- **D3b, a model table that drops temperature for models that reject it.** Not chosen: it silently discards a value the caller set, so a caller cannot tell that their setting had no effect, and the table needs an entry for every new model.
- **D3c, remove the `temperature` field.** Not chosen: callers pinned to `claude-sonnet-4-6` or earlier would lose a setting those models accept, a breaking change to `JudgeConfig` and to every transport signature.
- **DTb, the API's default thinking on the default tier (adaptive thinking at effort `high`).** Not chosen: adaptive thinking decides per request whether to think, and at effort `high` "Claude thinks on most requests that benefit from it" (https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost), so verdicts could carry thinking tokens and latency, and the default tier would move further from B0, which ran `claude-sonnet-4-6` without thinking.
- **DGb, keep `claude-opus-4-8` as the gold tier.** Not chosen: the two tiers would sit on different model generations, and the gold tier would stay unmeasured, since the meta suite has only ever run the default tier.
