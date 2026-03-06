import git
@tool
def push_changes(repo_path):

    repo = git.Repo(repo_path)

    repo.git.add(all=True)
    repo.index.commit("AI agent update")

    origin = repo.remote(name="origin")
    origin.push()
