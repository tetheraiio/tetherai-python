# TetherAI

[![CI](https://github.com/tetherai/tetherai-python/actions/workflows/ci.yml/badge.svg)](https://github.com/tetherai/tetherai-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tetherai-python)](https://pypi.org/project/tetherai-python/)
[![Python](https://img.shields.io/pypi/pyversions/tetherai-python)](https://pypi.org/project/tetherai-python/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**Financial circuit breaker for AI agents. Stop runaway LLM costs before they happen.**

AI agents don't crash when they fail — they loop, hallucinate, and retry. A stuck CrewAI crew can silently burn hundreds of dollars in LLM API calls over a weekend. Traditional monitoring tools like Datadog will show you an API cost spike on Monday morning. TetherAI kills the agent before it spends your money.

<!-- TODO: Replace with asciinema embed or GIF -->
<!-- Record with: asciinema rec demo.cast then upload to asciinema.org -->

## Install

```bash
pip install tetherai-python
```

With CrewAI support:

```bash
pip install tetherai-python[crewai]
```

## Quick Start

The `@enforce_budget` decorator wraps any function with budget guardrails:

```python
from tetherai import tether, BudgetExceededError
import litellm


@tether.enforce_budget(max_usd=0.05)
def my_workflow():
    for i in range(100):
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Count to {i}"}]
        )
        print(f"Call {i+1}: {response.choices[0].message.content[:50]}...")


try:
    my_workflow()
except BudgetExceededError as e:
    print(f"\n🛑 Stopped! Spent ${e.spent_usd:.4f} of ${e.budget_usd:.2f} budget")
```

Expected output:
```
Call 1: 0...
Call 2: 0, 1...
Call 3: 0, 1, 2...

🛑 Stopped! Spent $0.0501 of $0.05 budget
```

## CrewAI Integration

For CrewAI crews, use `protect_crew()` to wrap budget enforcement around the entire crew:

```python
from tetherai import protect_crew, BudgetExceededError
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Research Analyst",
    goal="Find comprehensive information on AI observability tools",
    backstory="You are a thorough research analyst known for detailed analysis.",
    verbose=True,
)

task = Task(
    description="Research the competitive landscape of AI observability tools",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential)

protected_crew = protect_crew(crew, max_usd=0.10)

try:
    result = protected_crew.kickoff()
except BudgetExceededError as e:
    print(f"Budget exceeded: ${e.spent_usd:.2f} of ${e.budget_usd:.2f}")
```

## What Happens When Budget Is Exceeded

When the budget is exceeded, TetherAI raises `BudgetExceededError` with details about the run:

```json
{
  "run_id": "run_abc123",
  "budget_usd": 0.10,
  "spent_usd": 0.13,
  "turns": 7,
  "spans": [
    {
      "span_type": "llm_call",
      "model": "gpt-4o-mini",
      "input_tokens": 1250,
      "output_tokens": 340,
      "cost_usd": 0.0019,
      "status": "ok"
    }
  ]
}
```

## How It Works

TetherAI patches `litellm.completion` at runtime to intercept every LLM call your agent makes. Before each call, it counts input tokens locally using tiktoken and checks the projected cost against your budget. If the budget would be exceeded, the call is blocked and a `BudgetExceededError` is raised. After each successful call, actual token usage from the LLM response is recorded for accurate cost tracking.

## Supported Frameworks

| Framework | Status | Integration |
|-----------|--------|-------------|
| CrewAI    | ✅ Supported | `protect_crew()` |
| LiteLLM (direct) | ✅ Supported | `@enforce_budget` decorator |
| LangChain | 🔜 Coming soon | — |
| smolagents | 🔜 Coming soon | — |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0
