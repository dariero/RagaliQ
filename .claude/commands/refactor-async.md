# refactor-async

## Purpose
Refactor synchronous code to async/await patterns. Essential for LLM integrations where parallel API calls dramatically improve performance.

## Usage
Invoke when:
- Converting sync functions to async for LLM operations
- Adding concurrency to batch evaluations
- Implementing proper timeout and cancellation handling
- Optimizing slow sequential API calls

## Automated Steps

1. **Analyze current implementation**
   - Identify sync code making I/O calls
   - Map call dependencies to find parallelization opportunities
   - Check for blocking operations

2. **Refactor to async**
   - Convert functions to `async def`
   - Replace blocking calls with async equivalents
   - Use `asyncio.gather()` for parallel operations
   - Add proper timeout handling

3. **Update callers**
   - Propagate async through call chain
   - Add sync wrappers where needed for backwards compatibility
   - Update tests to use `pytest-asyncio`

4. **Add error handling**
   - Implement timeout with `asyncio.wait_for()`
   - Handle `asyncio.CancelledError` properly
   - Add retry logic for transient failures

5. **Test async behavior**
   - Verify parallelization works
   - Test timeout handling
   - Test cancellation scenarios

## Domain Expertise Applied

### Python 3.14+ Async Patterns

**1. Basic Async Conversion**
```python
# Before (sync)
def evaluate_all(test_cases: list[RAGTestCase]) -> list[RAGTestResult]:
    results = []
    for tc in test_cases:
        result = evaluate(tc)  # Blocking!
        results.append(result)
    return results

# After (async with parallelization)
async def evaluate_all(test_cases: list[RAGTestCase]) -> list[RAGTestResult]:
    tasks = [evaluate_async(tc) for tc in test_cases]
    return await asyncio.gather(*tasks)
```

**2. Semaphore for Rate Limiting**
```python
class RateLimitedJudge:
    def __init__(self, max_concurrent: int = 5):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def call_api(self, prompt: str) -> str:
        async with self._semaphore:
            return await self._client.complete(prompt)
```

**3. Timeout Handling**
```python
async def evaluate_with_timeout(
    test_case: RAGTestCase,
    timeout: float = 30.0
) -> RAGTestResult:
    try:
        return await asyncio.wait_for(
            evaluate_async(test_case),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return RAGTestResult(
            test_case=test_case,
            status="error",
            error="Evaluation timed out"
        )
```

**4. Graceful Cancellation**
```python
async def evaluate_batch(
    test_cases: list[RAGTestCase]
) -> list[RAGTestResult]:
    tasks = [asyncio.create_task(evaluate_async(tc)) for tc in test_cases]

    try:
        return await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        # Cancel all pending tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        # Wait for cancellation to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
```

**5. Sync Wrapper for Backwards Compatibility**
```python
def evaluate(test_case: RAGTestCase) -> RAGTestResult:
    """Sync wrapper for async evaluate."""
    return asyncio.run(evaluate_async(test_case))

# Or for existing event loop:
def evaluate_sync(test_case: RAGTestCase) -> RAGTestResult:
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - can't use asyncio.run
        raise RuntimeError("Use evaluate_async() in async context")
    except RuntimeError:
        # No running loop - safe to use asyncio.run
        return asyncio.run(evaluate_async(test_case))
```

**6. Async Context Manager**
```python
class AsyncJudge:
    async def __aenter__(self):
        self._client = await self._create_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.close()

# Usage:
async with AsyncJudge() as judge:
    result = await judge.evaluate(test_case)
```

### Performance Optimization Tips
- **Batch requests**: Group API calls when provider supports it
- **Connection pooling**: Reuse HTTP connections via `httpx.AsyncClient`
- **Lazy initialization**: Don't create clients until needed
- **Progress tracking**: Use `asyncio.as_completed()` for progress bars

### Pitfalls to Avoid
- Don't mix `asyncio.run()` with existing event loops
- Don't forget to close async clients (use context managers)
- Don't ignore `CancelledError` - propagate it
- Don't create too many concurrent tasks - use semaphores
- Don't block the event loop with CPU-bound work

## Interactive Prompts

**Ask for:**
- Which code paths need async conversion?
- Maximum concurrency desired?
- Timeout requirements?
- Need sync wrapper for backwards compatibility?

**Suggest:**
- Parallelization opportunities
- Appropriate semaphore limits
- Error handling strategy

**Validate:**
- All I/O operations are non-blocking
- Timeouts are reasonable
- Cancellation is handled properly

## Success Criteria
- [ ] All I/O operations are async
- [ ] `asyncio.gather()` used for parallelization
- [ ] Timeouts implemented on external calls
- [ ] Cancellation handled gracefully
- [ ] Sync wrappers provided if needed
- [ ] Tests use `pytest-asyncio`
- [ ] `make test && make typecheck` passes
- [ ] Performance improvement measurable
