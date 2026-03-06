from crewai import Agent

docs = Agent(
    role="Technical Writer",
    goal="Maintain documentation",
    backstory="Expert documentation engineer",
    verbose=True
)
