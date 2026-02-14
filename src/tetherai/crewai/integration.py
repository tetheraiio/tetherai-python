from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crewai import Crew


def _check_crewai_installed() -> None:
    try:
        import crewai
    except ImportError:
        raise ImportError(
            "crewai is not installed. Install it with: pip install tetherai[crewai]"
        )


def protect_crew(
    crew: "Crew",
    max_usd: float,
    max_turns: int | None = None,
) -> "Crew":
    _check_crewai_installed()

    from tetherai.circuit_breaker import enforce_budget

    original_kickoff = crew.kickoff

    @enforce_budget(max_usd=max_usd, max_turns=max_turns)
    def wrapped_kickoff(*args: Any, **kwargs: Any):
        return original_kickoff(*args, **kwargs)

    crew.kickoff = wrapped_kickoff

    for agent in crew.agents:
        original_step_callback = agent.step_callback

        def make_callback(original):
            def callback(step_output: Any) -> None:
                if original:
                    original(step_output)
            return callback

        agent.step_callback = make_callback(original_step_callback)

    for task in crew.tasks:
        original_task_callback = task.callback

        def make_task_callback(original):
            def callback(task_output: Any) -> None:
                if original:
                    original(task_output)
            return callback

        task.callback = make_task_callback(original_task_callback)

    return crew


def tether_step_callback(step_output: Any) -> None:
    pass


def tether_task_callback(task_output: Any) -> None:
    pass
