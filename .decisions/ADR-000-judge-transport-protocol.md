# ADR-000: Judge Transport Protocol

**Status:** Accepted
**Date:** 2026-02-15
**Context:** Strategic refactoring (PR 4 of 6)

## Context

`ClaudeJudge` mixed three distinct concerns in a single class:
1. **Domain logic**: Prompt building, JSON parsing, score clamping
2. **Transport**: HTTP calls to Claude API, retry logic
3. **Provider-specific logic**: Claude-specific error handling

Adding support for additional LLM providers (OpenAI, vLLM, local models) would require duplicating all the prompt building and parsing logic, violating DRY.

## Decision

Introduce a **transport protocol** to separate API communication from judge logic:

```
┌─────────────────────────────────────────┐
│ LLMJudge (ABC - evaluator-facing API)  │ ← Unchanged
└─────────────────┬───────────────────────┘
                  │
         ┌────────▼────────┐
         │   BaseJudge     │ ← New: prompt building, parsing, clamping
         └────────┬────────┘
                  │
         ┌────────▼────────────┐
         │ JudgeTransport      │ ← New: Protocol (structural typing)
         │   (Protocol)        │
         └─────────────────────┘
                  ▲
        ┌─────────┴──────────┐
        │                    │
┌───────▼────────┐  ┌────────▼────────┐
│ ClaudeTransport│  │ OpenAITransport │ ← Future
└────────────────┘  └─────────────────┘
```

### Key Components

**`JudgeTransport` (Protocol):**
- Structural subtyping (not ABC) — any class with `async def send(...)` qualifies
- Single method: `send(system_prompt, user_prompt, ...) -> TransportResponse`
- Returns normalized `TransportResponse(text, input_tokens, output_tokens, model)`

**`BaseJudge(LLMJudge)`:**
- Extracts shared logic from `ClaudeJudge`: `_build_*_prompt()`, `_parse_json_response()`, score clamping
- Accepts `transport: JudgeTransport` in constructor
- Implements all 4 abstract methods from `LLMJudge` using the transport

**`ClaudeJudge(BaseJudge)`:**
- Thin wrapper: creates `ClaudeTransport(api_key)` and passes to `super().__init__()`
- **100% API-compatible** with old implementation
- Constructor signature unchanged: `ClaudeJudge(config, api_key)`

**`ClaudeTransport`:**
- Encapsulates `AsyncAnthropic` client
- Retry logic (tenacity with exponential backoff)
- Claude-specific error handling (429, 5xx)

## Principles Applied

1. **Separation of Concerns**: Transport ≠ Domain Logic
2. **Open/Closed**: Open for extension (new transports), closed for modification (BaseJudge unchanged)
3. **Dependency Inversion**: High-level `BaseJudge` depends on `JudgeTransport` abstraction, not concrete `ClaudeTransport`
4. **Interface Segregation**: `JudgeTransport` has single responsibility (send prompt → get text)

## Alternatives Considered

### Alternative 1: ABC inheritance hierarchy
```
LLMJudge (ABC)
 ├─ BaseJudge (ABC with shared logic)
 │   ├─ ClaudeJudge
 │   └─ OpenAIJudge
 └─ CustomJudge (user's custom implementation)
```

**Rejected:** Forces all judges into inheritance hierarchy. Makes it hard for users to bring their own transport without subclassing.

### Alternative 2: Adapter pattern with explicit interfaces
```
JudgeAdapter(ABC)
 ├─ ClaudeAdapter
 └─ OpenAIAdapter

BaseJudge uses JudgeAdapter
```

**Rejected:** More boilerplate. Protocol achieves same goal with structural typing (duck typing with type safety).

### Alternative 3: Keep ClaudeJudge monolithic, duplicate for OpenAI
**Rejected:** Violates DRY. Prompt engineering improvements would need to be duplicated across all providers.

## Consequences

### Positive
- **Reusability**: All prompt logic shared across providers
- **Testability**: Can mock `JudgeTransport` for unit testing without hitting real APIs
- **Extensibility**: Adding OpenAI = 50 lines (just the transport), not 500 lines (full judge)
- **Backward compatibility**: Existing `ClaudeJudge` API unchanged

### Negative
- **Indirection**: 3 layers instead of 1 (`ClaudeJudge` → `BaseJudge` → `ClaudeTransport`)
- **Learning curve**: New contributors need to understand transport abstraction

### Neutral
- Test mocking changes: `patch("ragaliq.judges.claude.AsyncAnthropic")` → `patch("ragaliq.judges.transport.AsyncAnthropic")`

## Verification

All 327 existing tests pass without logic changes (only mock patch paths updated).

**Test coverage:**
- `BaseJudge`: Prompt building, JSON parsing, score clamping (reused tests from old `ClaudeJudge`)
- `ClaudeTransport`: Retry logic, error handling, token counting
- `ClaudeJudge`: Integration (thin layer, delegates to `BaseJudge`)

## Migration Path

**Phase 1 (This PR):** Introduce transport, refactor `ClaudeJudge` to use it
**Phase 2 (Future):** Add `OpenAITransport` and `OpenAIJudge`
**Phase 3 (Future):** Add support for local models (vLLM, Ollama)

Users can implement custom judges by:
1. Creating a transport class with `async def send(...) -> TransportResponse`
2. Passing it to `BaseJudge(transport, config)`

No changes required to evaluators or runner.
