from tetherai import protect_crew, BudgetExceededError
from crewai import Agent, Task, Crew, Process
import os

print("Starting CrewAI budget demo...")
print(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")

# Create a research agent
researcher = Agent(
    role="Research Analyst",
    goal="Find comprehensive information on AI observability tools",
    backstory="You are a thorough research analyst known for detailed analysis.",
    verbose=True,
    llm="gpt-4o-mini",
)

task = Task(
    description="Research the competitive landscape of AI observability tools",
    expected_output="A comprehensive report on AI observability tools",
    agent=researcher,
)

crew = Crew(
    agents=[researcher],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
)

# Before protection - let's check what crew uses
print(f"Crew type: {type(crew)}")
print(f"Agents: {crew.agents}")
if crew.agents:
    print(f"Agent LLM: {crew.agents[0].llm}")
    print(f"Agent LLM type: {type(crew.agents[0].llm)}")

# Check what happens when we call kickoff
print("\n--- Testing without protection first ---")
try:
    result = crew.kickoff()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

print("\n--- Now with protection ---")
protected_crew = protect_crew(crew, max_usd=0.10, trace_export="json")

try:
    result = protected_crew.kickoff()
    print(f"Result: {result}")
except BudgetExceededError as e:
    print(f"\n⚠️  TetherAI Budget Exceeded!")
    print(f"   Budget:  ${e.budget_usd:.2f}")
    print(f"   Spent:   ${e.spent_usd:.2f}")
    print(f"   Model:   {e.last_model}")
    print(f"\n💾 Full execution trace saved to: ./tetherai_traces/{e.run_id}.json")
