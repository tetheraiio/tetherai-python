class TetherError(Exception):
    """Base exception for all TetherAI errors."""


class BudgetExceededError(TetherError):
    """Raised when a run's accumulated cost exceeds its budget."""

    def __init__(
        self,
        message: str,
        run_id: str,
        budget_usd: float,
        spent_usd: float,
        last_model: str,
        trace_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.budget_usd = budget_usd
        self.spent_usd = spent_usd
        self.last_model = last_model
        self.trace_url = trace_url

    def __str__(self) -> str:
        return (
            f"Budget exceeded: ${self.spent_usd:.2f} / ${self.budget_usd:.2f} on run {self.run_id}"
        )

    def __reduce__(self) -> tuple[type, tuple]:  # type: ignore[type-arg]
        return (
            self.__class__,
            (
                self.args[0],
                self.run_id,
                self.budget_usd,
                self.spent_usd,
                self.last_model,
                self.trace_url,
            ),
        )


class TurnLimitError(TetherError):
    """Raised when an agent exceeds max allowed LLM calls."""

    def __init__(
        self,
        message: str,
        run_id: str,
        max_turns: int,
        current_turn: int,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.max_turns = max_turns
        self.current_turn = current_turn

    def __str__(self) -> str:
        return f"Turn limit exceeded: {self.current_turn} / {self.max_turns} on run {self.run_id}"

    def __reduce__(self) -> tuple[type, tuple]:  # type: ignore[type-arg]
        return (
            self.__class__,
            (
                self.args[0],
                self.run_id,
                self.max_turns,
                self.current_turn,
            ),
        )


class TokenCountError(TetherError):
    """Raised when token counting fails (e.g., unknown encoding)."""

    def __init__(self, message: str, model: str | None = None) -> None:
        super().__init__(message)
        self.model = model

    def __reduce__(self) -> tuple[type, tuple]:  # type: ignore[type-arg]
        return (self.__class__, (self.args[0], self.model))


class UnknownModelError(TetherError):
    """Raised when a model is not found in the pricing registry."""

    def __init__(self, message: str, model: str) -> None:
        super().__init__(message)
        self.model = model

    def __reduce__(self) -> tuple[type, tuple]:  # type: ignore[type-arg]
        return (self.__class__, (self.args[0], self.model))
