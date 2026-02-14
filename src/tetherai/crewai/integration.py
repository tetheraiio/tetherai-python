from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crewai import Crew


def _check_crewai_installed() -> None:
    try:
        import crewai  # noqa: F401
    except ImportError:
        raise ImportError(
            "crewai is not installed. Install it with: pip install tetherai[crewai]"
        ) from None


class ProtectedCrew:
    """Wrapper around CrewAI crew with budget enforcement."""

    def __init__(
        self,
        crew: "Crew",
        max_usd: float,
        max_turns: int | None = None,
        trace_export: str | None = None,
    ):
        from tetherai.circuit_breaker import enforce_budget

        self._crew = crew
        self._max_usd = max_usd
        self._max_turns = max_turns

        self.kickoff = enforce_budget(
            max_usd=max_usd, max_turns=max_turns, trace_export=trace_export
        )(crew.kickoff)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._crew, name)


def protect_crew(
    crew: "Crew",
    max_usd: float,
    max_turns: int | None = None,
    trace_export: str | None = None,
) -> ProtectedCrew:
    _check_crewai_installed()
    return ProtectedCrew(crew, max_usd, max_turns, trace_export)


def tether_step_callback(step_output: Any) -> None:
    pass


def tether_task_callback(task_output: Any) -> None:
    pass
