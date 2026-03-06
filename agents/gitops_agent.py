from crewai import Agent
from tools.git_tools import commit_and_push

gitops = Agent(
    role="DevOps Engineer",
    goal="Commit and push code changes to GitHub",
    backstory="Expert DevOps engineer specializing in CI/CD pipelines",
    tools=[commit_and_push],
    verbose=True
)
