from crewai import Crew, Task

from agents.architect_agent import architect
from agents.developer_agent import developer
from agents.tester_agent import tester
from agents.docs_agent import docs
from agents.gitops_agent import gitops


architect_task = Task(
    description="Design architecture for a nanoparticle size prediction module",
    expected_output="Architecture description",
    agent=architect
)

developer_task = Task(
    description="""
Create a Python module called nanoparticle_model.py
that predicts nanoparticle size using a RandomForest model.

Save the file using the write_file tool.
""",
    expected_output="Python code saved as nanoparticle_model.py",
    agent=developer
)

test_task = Task(
    description="Write and run tests for the nanoparticle predictor save locally",
    expected_output="Test results",
    agent=tester
)

docs_task = Task(
    description="Update README documentation for the nanoparticle predictor,save this locally",
    expected_output="Updated documentation",
    agent=docs
)

git_task = Task(
    description="Commit and push the new code to the GitHub repository,https://github.com/mpofukelvintafadzwa, create a repository",
    expected_output="Git commit",
    agent=gitops
)


crew = Crew(
    agents=[architect, developer, tester, docs, gitops],
    tasks=[
        architect_task,
        developer_task,
        test_task,
        docs_task,
        git_task
    ],
    verbose=True,
    tracing=True
)


result = crew.kickoff()

print(result)
