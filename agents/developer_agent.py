from crewai import Agent
from tools.file_tools import write_file

developer = Agent(
    role="Python Developer",
    goal="Write high quality Python code and save it to the project",
    backstory="Expert machine learning engineer",
    tools=[write_file],
    verbose=True
)
