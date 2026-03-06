from crewai import Agent

architect = Agent(
    role="Software Architect",
    goal="Design system architecture for new features",
    backstory="Senior software architect with experience in AI systems",
    verbose=True
)
