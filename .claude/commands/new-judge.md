# new-judge

## Purpose
Create a new LLM judge implementation for RagaliQ. Judges are the LLM backends that perform claim extraction, verification, and scoring operations.

## Usage
Invoke when:
- Adding support for new LLM providers (OpenAI, Cohere, Mistral, local models)
- Creating specialized judges (fast/cheap for CI, accurate for production)
- Implementing mock judges for testing

## Automated Steps

1. **Analyze existing judge patterns**
   - Read `src/ragaliq/judges/base.py` for abstract interface
   - Review `src/ragaliq/judges/claude.py` for implementation patterns
   - Check prompt templates in `src/ragaliq/judges/prompts/`

2. **Generate judge implementation**
   ```
   src/ragaliq/judges/{provider}.py
   ```
   - Inherit from `LLMJudge` base class
   - Implement all abstract methods (async)
   - Add retry logic with tenacity
   - Track token usage for cost monitoring

3. **Create/adapt prompt templates**
   ```
   src/ragaliq/judges/prompts/{provider}/
   ```
   - Customize prompts for model's strengths
   - Optimize for response format consistency
   - Add model-specific few-shot examples

4. **Export in __init__.py**
   - Add to `src/ragaliq/judges/__init__.py`

5. **Create comprehensive tests**
   ```
   tests/unit/test_{provider}_judge.py
   tests/integration/test_{provider}_judge.py
   ```

6. **Document provider setup**
   - Add to README.md with API key setup
   - Document model selection and pricing

## Domain Expertise Applied

### LLM Integration Best Practices
- **Async-first**: All API calls must be async for parallelization
- **Structured outputs**: Use JSON mode where available
- **Retry logic**: Exponential backoff for rate limits
- **Timeout handling**: Configurable timeouts with sensible defaults
- **Token tracking**: Monitor usage for cost control

### Judge Implementation Pattern
```python
class {Provider}Judge(LLMJudge):
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
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type(APIError)
    )
    async def _call_llm(self, system: str, user: str) -> str:
        response = await self.client.chat(...)
        self._usage["prompt_tokens"] += response.usage.prompt_tokens
        return response.content

    async def extract_claims(self, response: str) -> list[str]:
        prompt = get_prompt("extract_claims")
        result = await self._call_llm(prompt.system, prompt.user.format(response=response))
        return json.loads(result)
```

### Provider-Specific Considerations
- **OpenAI**: Use `response_format={"type": "json_object"}` for reliability
- **Anthropic**: Leverage XML tags for structured responses
- **Local models**: Handle longer timeouts, batch for efficiency
- **Mock judges**: Deterministic responses for unit testing

### Pitfalls to Avoid
- Don't forget to handle malformed JSON responses
- Don't ignore rate limit headers - implement backoff
- Don't hardcode model names - use config
- Don't skip token tracking - essential for cost management

## Interactive Prompts

**Ask for:**
- Provider name: `{provider}`
- SDK/client library to use
- Default model (and alternatives)
- API key environment variable name
- Any provider-specific prompt optimizations needed?

**Suggest:**
- Appropriate retry configuration
- Token limits for this provider
- Cost-effective model alternatives

**Validate:**
- All abstract methods implemented
- Error handling covers common API failures
- Token usage is tracked accurately

## Success Criteria
- [ ] Judge class implements all LLMJudge abstract methods
- [ ] Async client properly initialized
- [ ] Retry logic with exponential backoff
- [ ] Token usage tracking works
- [ ] Unit tests with mocked client
- [ ] Integration test (skipped without API key)
- [ ] `make test && make typecheck` passes
- [ ] Provider documented in README.md
