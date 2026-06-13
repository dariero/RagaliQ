"""Global registry mapping evaluator names to `Evaluator` classes.

Built-in evaluators auto-register at import time via the `@register_evaluator`
decorator; custom ones can use the decorator or `register_evaluator_class`.

Example:
    @register_evaluator("my_metric")
    class MyEvaluator(Evaluator):
        name = "my_metric"
        ...
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragaliq.core.evaluator import Evaluator

_REGISTRY: dict[str, type[Evaluator]] = {}


def register_evaluator[E: type[Evaluator]](name: str) -> Callable[[E], E]:
    """Class decorator registering an evaluator under `name`.

    Args:
        name: Unique, non-empty identifier (e.g. "faithfulness").

    Raises:
        ValueError: If `name` is empty/duplicate or the class is not an Evaluator.
    """

    def decorator(cls: E) -> E:
        register_evaluator_class(name, cls)
        return cls

    return decorator


def register_evaluator_class(name: str, cls: type[Evaluator]) -> None:
    """Register an evaluator class imperatively (the non-decorator path).

    Args:
        name: Unique, non-empty identifier (e.g. "faithfulness").
        cls: The `Evaluator` subclass to register.

    Raises:
        ValueError: If `name` is empty/duplicate or `cls` is not an Evaluator.
    """
    # Imported here (not at module level) to break a circular dependency.
    from ragaliq.core.evaluator import Evaluator

    if not name or not name.strip():
        raise ValueError("Evaluator name cannot be empty")
    if name in _REGISTRY:
        raise ValueError(f"Evaluator {name!r} is already registered")
    if not issubclass(cls, Evaluator):
        raise ValueError(
            f"Class {cls.__name__!r} must be a subclass of Evaluator, got {cls.__mro__}"
        )

    _REGISTRY[name] = cls


def get_evaluator(name: str) -> type[Evaluator]:
    """Return the registered evaluator class for `name`.

    Raises:
        ValueError: If `name` is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown evaluator: {name!r}. Available evaluators: {available}")

    return _REGISTRY[name]


def list_evaluators() -> list[str]:
    """Return all registered evaluator names, sorted."""
    return sorted(_REGISTRY.keys())
