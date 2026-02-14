import threading
from dataclasses import dataclass
from typing import Any

from tetherai.exceptions import BudgetExceededError, TurnLimitError


@dataclass
class CallRecord:
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float
    duration_ms: float


class BudgetTracker:
    def __init__(
        self,
        run_id: str,
        max_usd: float,
        max_turns: int | None = None,
    ):
        self.run_id = run_id
        self.max_usd = max_usd
        self.max_turns = max_turns
        self._spent_usd = 0.0
        self._turn_count = 0
        self._calls: list[CallRecord] = []
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        with self._lock:
            return max(0.0, self.max_usd - self._spent_usd)

    @property
    def turn_count(self) -> int:
        with self._lock:
            return self._turn_count

    @property
    def is_exceeded(self) -> bool:
        with self._lock:
            return self._spent_usd >= self.max_usd

    def pre_check(self, estimated_cost: float, model: str = "unknown") -> None:
        with self._lock:
            projected = self._spent_usd + estimated_cost
            if projected >= self.max_usd:
                raise BudgetExceededError(
                    message=f"Budget exceeded: ${projected:.6f} >= ${self.max_usd:.6f}",
                    run_id=self.run_id,
                    budget_usd=self.max_usd,
                    spent_usd=projected,
                    last_model=model,
                )

    def record_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cost_usd: float,
        duration_ms: float,
    ) -> None:
        if cost_usd < 0:
            raise ValueError("cost_usd must be non-negative")

        with self._lock:
            if self.max_turns is not None and self._turn_count >= self.max_turns:
                raise TurnLimitError(
                    message=f"Turn limit exceeded: {self._turn_count} >= {self.max_turns}",
                    run_id=self.run_id,
                    max_turns=self.max_turns,
                    current_turn=self._turn_count + 1,
                )

            self._spent_usd += cost_usd
            self._turn_count += 1

            if self._spent_usd > self.max_usd:
                self._spent_usd = self.max_usd

            self._calls.append(
                CallRecord(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                    cost_usd=cost_usd,
                    duration_ms=duration_ms,
                )
            )

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "budget_usd": self.max_usd,
                "spent_usd": self._spent_usd,
                "remaining_usd": max(0.0, self.max_usd - self._spent_usd),
                "turn_count": self._turn_count,
                "calls": [
                    {
                        "input_tokens": call.input_tokens,
                        "output_tokens": call.output_tokens,
                        "model": call.model,
                        "cost_usd": call.cost_usd,
                        "duration_ms": call.duration_ms,
                    }
                    for call in self._calls
                ],
            }
