# new-judge

Create a new LLM judge backend.

## Files to Create

```
src/ragaliq/judges/{provider}.py
src/ragaliq/judges/prompts/{provider}/  (if custom prompts needed)
tests/unit/judges/test_{provider}.py
tests/integration/judges/test_{provider}.py
```

## Required Pattern

```python
from ragaliq.judges.base import LLMJudge, JudgeConfig
from tenacity import retry, stop_after_attempt, wait_exponential

class {Provider}Judge(LLMJudge):
    """Judge implementation using {Provider} API."""

    def __init__(
        self,
        api_key: str | None = None,
        config: JudgeConfig | None = None
    ):
        self.client = AsyncClient(api_key=api_key or os.getenv("{PROVIDER}_API_KEY"))
        self.config = config or JudgeConfig()
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10)
    )
    async def _call_llm(self, system: str, user: str) -> str:
        response = await self.client.chat(...)
        # Track token usage
        self._usage["prompt_tokens"] += response.usage.prompt_tokens
        self._usage["completion_tokens"] += response.usage.completion_tokens
        return response.content

    # Implement all abstract methods from LLMJudge
    async def extract_claims(self, response: str) -> list[str]: ...
    async def verify_claim(self, claim: str, context: list[str]) -> ClaimVerification: ...
    # etc.
```

## Requirements

- All API calls must be async
- Retry logic with exponential backoff (tenacity)
- Track token usage for cost monitoring
- Handle malformed JSON responses gracefully
- API key from env var with configurable override

## Update Exports

Add to `src/ragaliq/judges/__init__.py`

## Verify

```bash
hatch run test tests/unit/judges/test_{provider}.py
hatch run typecheck
```
