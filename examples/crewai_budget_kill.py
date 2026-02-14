from tetherai import protect_crew, BudgetExceededError
from crewai import Agent, Task, Crew, Process

# Create a research agent that will loop when given a broad task
researcher = Agent(
    role="Research Analyst",
    goal="Find comprehensive information on AI observability tools",
    backstory="You are a thorough research analyst known for detailed analysis.",
    verbose=True,
)

# Task that will trigger multiple agent reasoning steps
task = Task(
    description="Research the competitive landscape of AI observability tools",
    expected_output="A comprehensive report on AI observability tools",
    agent=researcher,
)

# Create the crew
crew = Crew(
    agents=[researcher],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
)

# Protect the crew with a low budget ($0.10)
protected_crew = protect_crew(crew, max_usd=0.10)

try:
    result = protected_crew.kickoff()
    print(f"\nCrew completed successfully: {result}")
except BudgetExceededError as e:
    print(f"\n⚠️  TetherAI Budget Exceeded!")
    print(f"   Budget:  ${e.budget_usd:.2f}")
    print(f"   Spent:   ${e.spent_usd:.2f}")
    print(f"   Model:   {e.last_model}")
    print(f"\n💾 Full execution trace saved to: ./tetherai_traces/{e.run_id}.json")
    print(f"   Open it to see exactly which step blew the budget.")
