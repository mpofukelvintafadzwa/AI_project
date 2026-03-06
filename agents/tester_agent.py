from crewai import Agent
from tools.test_tools import run_tests

tester = Agent(
    role="QA Engineer",
    goal="Run automated tests",
    backstory="Expert in software validation",
    tools=[run_tests],
    verbose=True
)
