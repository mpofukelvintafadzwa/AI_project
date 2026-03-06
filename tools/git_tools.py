from crewai.tools import tool
import git

@tool
def commit_and_push() -> str:
    """
    Commit all changes and push them to GitHub.
    """

    repo = git.Repo(".")

    repo.git.add(all=True)

    repo.index.commit("AI agent update")

    origin = repo.remote(name="origin")
    origin.push()

    return "Changes committed and pushed to GitHub."
