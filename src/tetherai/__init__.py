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


class tether:
    enforce_budget = staticmethod(enforce_budget)


from typing import Any


def protect_crew(*args: Any, **kwargs: Any) -> Any:
    from tetherai.crewai.integration import protect_crew as _protect_crew
    return _protect_crew(*args, **kwargs)


__all__ = [
    "__version__",
    "enforce_budget",
    "tether",
    "protect_crew",
    "BudgetExceededError",
    "TurnLimitError",
    "TokenCountError",
    "TetherError",
    "UnknownModelError",
    "TetherConfig",
    "load_config",
]
