from typing import Any

from tetherai._version import __version__
from tetherai.circuit_breaker import enforce_budget
from tetherai.config import TetherConfig, load_config
from tetherai.exceptions import (
    BudgetExceededError,
    TetherError,
    TokenCountError,
    TurnLimitError,
    UnknownModelError,
)


class Tether:
    """TetherAI namespace class."""

    enforce_budget = staticmethod(enforce_budget)


tether = Tether


def protect_crew(*args: Any, **kwargs: Any) -> Any:
    from tetherai.crewai.integration import protect_crew as _protect_crew

    return _protect_crew(*args, **kwargs)


__all__ = [
    "BudgetExceededError",
    "TetherConfig",
    "TetherError",
    "TokenCountError",
    "TurnLimitError",
    "UnknownModelError",
    "__version__",
    "enforce_budget",
    "load_config",
    "protect_crew",
    "tether",
]
