# ADR-011: Evaluator Registry with Decorator API

**Status:** Accepted
**Date:** 2026-02-12
**Issue:** #11

## Context

RagaliQ ships with built-in evaluators (faithfulness, relevance, hallucination) and needs a mechanism for:
1. Discovering available evaluators at runtime
2. Looking up evaluator classes by name (e.g., "faithfulness" → `FaithfulnessEvaluator`)
3. Enabling users to register custom evaluators without modifying library code
4. Maintaining backward compatibility with existing `EVALUATOR_REGISTRY` dict

The initial implementation (PR #35) used an inline dict in `evaluators/__init__.py`, but this approach lacks:
- **Discoverability**: No public API for listing available evaluators
- **Extensibility**: Users can't register custom evaluators without forking
- **Validation**: No enforcement that registered classes are actually `Evaluator` subclasses
- **Separation of concerns**: Registration logic mixed with package initialization

## Proposed Solution

Create a dedicated `evaluators/registry.py` module with:

### 1. Module-level registry dict
```python
_REGISTRY: dict[str, type[Evaluator]] = {}  # Private
EVALUATOR_REGISTRY = _REGISTRY  # Backward-compat public alias
```

### 2. Decorator API for registration
```python
@register_evaluator("faithfulness")
class FaithfulnessEvaluator(Evaluator):
    ...
```

Built-in evaluators use this decorator to auto-register at import time. The decorator:
- Validates that the class is a subclass of `Evaluator`
- Rejects empty or duplicate names
- Returns the class unchanged (type-preserving via `TypeVar`)

### 3. Programmatic registration API
```python
register_evaluator_class(name: str, cls: type[Evaluator]) -> None
```

Imperative alternative for dynamic registration or when decorators are inconvenient.

### 4. Lookup and discovery functions
```python
get_evaluator(name: str) -> type[Evaluator]  # Raises ValueError if unknown
list_evaluators() -> list[str]  # Sorted list of registered names
```

### 5. Runner migration
`runner.py` migrates from:
```python
evaluator_class = EVALUATOR_REGISTRY[name]  # Manual dict lookup
```
to:
```python
evaluator_class = get_evaluator(name)  # Function with validation
```

Error message format preserved: `"Unknown evaluator: {name!r}. Available evaluators: {available}"`

## Principles Applied

### 1. **Evaluator Pattern** (from CLAUDE.md)
Each metric is a separate `Evaluator` class. The registry provides a centralized lookup mechanism without violating the pattern.

### 2. **Pydantic Everywhere** (from CLAUDE.md)
While the registry itself doesn't use Pydantic (it's a simple dict), all evaluators return `EvaluationResult` (Pydantic model) as specified in the base class contract.

### 3. **Open/Closed Principle**
Built-in evaluators are auto-registered via decorator (no manual dict maintenance). Users extend functionality by registering custom evaluators, not by modifying library code.

### 4. **Fail Fast**
Validation happens at registration time, not evaluation time:
- Reject non-`Evaluator` classes immediately
- Reject duplicate names to prevent silent overwrites
- Raise clear errors for unknown evaluator names

### 5. **Backward Compatibility**
Existing code using `from ragaliq.evaluators import EVALUATOR_REGISTRY` continues to work. The dict is the same object as `_REGISTRY`, so mutations (if any) are reflected.

## Alternatives Considered

### 1. Singleton registry class
```python
class EvaluatorRegistry:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**Rejected:** Over-engineered for a simple lookup dict. Module-level dict is simpler, testable, and Pythonic. No need for object lifecycle management.

### 2. Class attribute on base `Evaluator`
```python
class Evaluator(ABC):
    _registry: dict[str, type[Evaluator]] = {}
```

**Rejected:** Violates single responsibility — `Evaluator` should define the interface, not manage registration. Harder to test in isolation (requires mocking class attributes).

### 3. Manual registration in `__init__.py`
```python
EVALUATOR_REGISTRY = {
    "faithfulness": FaithfulnessEvaluator,
    "relevance": RelevanceEvaluator,
    ...
}
```

**Rejected:** Current implementation. Requires manual updates when adding evaluators. No validation. No extensibility for users.

### 4. Entry points / plugins
```python
# setup.py
entry_points={
    "ragaliq.evaluators": [
        "faithfulness = ragaliq.evaluators:FaithfulnessEvaluator"
    ]
}
```

**Rejected:** Overkill for MVP. Adds setuptools/packaging complexity. Decorator API is simpler for both built-ins and user extensions. Can revisit if plugin ecosystem emerges.

### 5. Allowing duplicate names (last-wins)
```python
if name in _REGISTRY:
    warnings.warn(f"Overwriting evaluator {name!r}")
_REGISTRY[name] = cls
```

**Rejected:** Silent overwrites are error-prone. Explicit is better than implicit. If a user wants to replace a built-in, they should use a different name or explicitly unregister first.

### 6. Permissive validation (allow non-Evaluator classes)
**Rejected:** Duck typing would allow registering arbitrary classes, leading to runtime errors when `runner.py` calls `evaluator.evaluate()`. Strict validation catches errors early.

## Implementation Notes

### Import order (no circular dependencies)
```
registry.py → core/evaluator.py (only for type checking)
faithfulness.py → registry.py
__init__.py → registry.py + faithfulness.py (triggers decorator)
runner.py → evaluators/__init__.py → get_evaluator()
```

No circular imports: `registry.py` only imports `Evaluator` for runtime validation (inside function), not at module level.

### Decorator is type-preserving
```python
_E = TypeVar("_E", bound=type[Evaluator])

def register_evaluator(name: str) -> Callable[[_E], _E]:
    ...
```

This ensures type checkers understand that:
```python
@register_evaluator("faithfulness")
class FaithfulnessEvaluator(Evaluator):
    ...

# Type checker knows FaithfulnessEvaluator is still FaithfulnessEvaluator
assert FaithfulnessEvaluator.name == "faithfulness"  # No type error
```

### Test isolation
Custom evaluator tests use `clean_registry` fixture to snapshot/restore `_REGISTRY`:
```python
@pytest.fixture
def clean_registry():
    import ragaliq.evaluators.registry as reg
    original = reg._REGISTRY.copy()
    yield
    reg._REGISTRY.clear()
    reg._REGISTRY.update(original)
```

This prevents test pollution when registering custom evaluators.

### Error message consistency
Runner's existing error format is preserved:
```python
# Before (runner.py)
raise ValueError(f"Unknown evaluator: {name!r}. Available evaluators: {available}")

# After (registry.py get_evaluator())
raise ValueError(f"Unknown evaluator: {name!r}. Available evaluators: {available}")
```

Existing test `test_runner.py::TestEvaluatorInitialization::test_raises_on_unknown_evaluator` continues to pass.

## Trade-offs

### ✅ Pros
- **Discoverability**: `list_evaluators()` enables dynamic UIs, CLI help, documentation generation
- **Extensibility**: Users can register custom evaluators without forking
- **Validation**: Type safety enforced at registration time
- **Separation of concerns**: Registration logic isolated in `registry.py`
- **Backward compatible**: Existing `EVALUATOR_REGISTRY` dict access still works
- **Type-safe**: Decorator preserves types for static analysis

### ⚠️ Cons
- **Global state**: Registry is module-level (not thread-local, not immutable after import)
  - Mitigation: Documentation clarifies registry is initialized at import time. Tests use `clean_registry` fixture.
- **Implicit registration**: Evaluators register via decorator side-effect at import time
  - Mitigation: Explicit imports in `__init__.py` make registration predictable
- **No unregistration API**: Once registered, can't remove an evaluator
  - Mitigation: Not needed for MVP. Can add `unregister_evaluator(name)` if use case emerges.

## Consequences

### For Built-in Evaluators
- Add two lines to each evaluator:
  ```python
  from ragaliq.evaluators.registry import register_evaluator

  @register_evaluator("faithfulness")
  class FaithfulnessEvaluator(Evaluator):
      ...
  ```
- No other changes required

### For Runner
- Simplified lookup: `get_evaluator(name)` instead of manual dict access + validation
- Error messages remain identical (backward compat for tests)

### For Users
- Can now list available evaluators: `from ragaliq.evaluators import list_evaluators`
- Can register custom evaluators:
  ```python
  from ragaliq.evaluators import register_evaluator
  from ragaliq.core.evaluator import Evaluator

  @register_evaluator("my_metric")
  class MyEvaluator(Evaluator):
      ...

  tester = RagaliQ(evaluators=["faithfulness", "my_metric"])
  ```

### For Testing
- Test isolation via `clean_registry` fixture
- Comprehensive coverage of decorator, programmatic API, validation, lookup, backward compat

## Future Work

- **Auto-discovery**: If plugin ecosystem emerges, consider entry points
- **Unregistration API**: Add `unregister_evaluator(name)` if needed
- **Registry immutability**: Lock registry after initial import (prevent accidental mutations)
- **Thread-local registries**: If multi-tenant use case emerges (unlikely for CLI/pytest plugin)

## References

- Issue #11: Evaluator Registry
- PR #35: Initial inline registry dict
- `core/evaluator.py`: Base `Evaluator` class
- `core/runner.py`: Runner uses `_init_evaluators()` to look up evaluators
- Python `TypeVar` documentation: Type-preserving decorators
