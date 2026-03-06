from crewai.tools import tool
import subprocess

@tool
def run_tests() -> str:
    """Run pytest on the project."""

    result = subprocess.run(
        ["pytest"],
        capture_output=True,
        text=True
    )

    return result.stdout
