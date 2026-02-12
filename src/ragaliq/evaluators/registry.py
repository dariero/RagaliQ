"""
Evaluator registry for RagaliQ.

This module provides a centralized registry for evaluator classes with a
decorator-based API for registration. Built-in evaluators use the decorator
to auto-register themselves at import time, and users can register custom
evaluators using either the decorator or the imperative API.

Example:
    # Using the decorator on a custom evaluator
    from ragaliq.evaluators.registry import register_evaluator
    from ragaliq.core.evaluator import Evaluator

    @register_evaluator("my_custom_metric")
    class MyCustomEvaluator(Evaluator):
        name = "my_custom_metric"
        description = "My custom metric"
        ...

    # Or using the imperative API
    from ragaliq.evaluators.registry import register_evaluator_class

    register_evaluator_class("my_custom_metric", MyCustomEvaluator)

    # Retrieving evaluators
    from ragaliq.evaluators.registry import get_evaluator, list_evaluators

    evaluator_class = get_evaluator("faithfulness")
    all_evaluators = list_evaluators()  # ['faithfulness', 'hallucination', 'relevance']
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ragaliq.core.evaluator import Evaluator

# Private registry dict
_REGISTRY: dict[str, type[Evaluator]] = {}

# Type variable for the decorator to preserve class types
_E = TypeVar("_E", bound=type["Evaluator"])


def register_evaluator(name: str) -> Callable[[_E], _E]:
    """
    Class decorator to register an evaluator in the global registry.

    This decorator validates that the class is a subclass of Evaluator and
    that the name is valid (non-empty, no duplicates). Built-in evaluators
    use this decorator for auto-registration at import time.

    Args:
        name: Unique identifier for the evaluator (e.g., "faithfulness").
              Must be non-empty and not already registered.

    Returns:
        The decorator function that registers and returns the class unchanged.

    Raises:
        ValueError: If name is empty, already registered, or class is not an Evaluator.

    Example:
        @register_evaluator("my_metric")
        class MyMetricEvaluator(Evaluator):
            name = "my_metric"
            description = "My custom metric"
            ...
    """

    def decorator(cls: _E) -> _E:
        register_evaluator_class(name, cls)
        return cls

    return decorator


def register_evaluator_class(name: str, cls: type[Evaluator]) -> None:
    """
    Programmatically register an evaluator class in the global registry.

    This is the imperative alternative to the @register_evaluator decorator.
    Useful for dynamic registration or when decorators are not convenient.

    Args:
        name: Unique identifier for the evaluator (e.g., "faithfulness").
              Must be non-empty and not already registered.
        cls: The evaluator class to register. Must be a subclass of Evaluator.

    Raises:
        ValueError: If name is empty, already registered, or cls is not an Evaluator.

    Example:
        register_evaluator_class("my_metric", MyMetricEvaluator)
    """
    # Import here to avoid circular dependency
    from ragaliq.core.evaluator import Evaluator

    # Validation
    if not name or not name.strip():
        raise ValueError("Evaluator name cannot be empty")

    if name in _REGISTRY:
        raise ValueError(f"Evaluator {name!r} is already registered")

    if not issubclass(cls, Evaluator):
        raise ValueError(
            f"Class {cls.__name__!r} must be a subclass of Evaluator, got {cls.__mro__}"
        )

    # Register
    _REGISTRY[name] = cls


def get_evaluator(name: str) -> type[Evaluator]:
    """
    Retrieve an evaluator class by name from the registry.

    Args:
        name: The evaluator name (e.g., "faithfulness", "relevance").

    Returns:
        The registered evaluator class.

    Raises:
        ValueError: If the evaluator name is not found in the registry.

    Example:
        evaluator_class = get_evaluator("faithfulness")
        evaluator = evaluator_class(threshold=0.8)
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown evaluator: {name!r}. Available evaluators: {available}")

    return _REGISTRY[name]


def list_evaluators() -> list[str]:
    """
    List all registered evaluator names.

    Returns:
        Sorted list of registered evaluator names.

    Example:
        all_evaluators = list_evaluators()
        # ['faithfulness', 'hallucination', 'relevance']
    """
    return sorted(_REGISTRY.keys())


# Backward compatibility: expose the registry dict directly
# (same object, so mutations are reflected)
EVALUATOR_REGISTRY = _REGISTRY
