import git
import os
import shutil

def git_pull(repo_url, local_path):
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
        print(f"Deleted existing directory: {local_path}")
    
    os.makedirs(local_path)

    repo = git.Repo.clone_from(repo_url, local_path)
    print(f"Cloned repository from {repo_url} to {local_path}")
    
    readme_path = os.path.join(local_path, 'README.md')
    readme_cn_path = os.path.join(local_path, 'README_CN.md')

    return readme_path, readme_cn_path
    
def git_origin(local_path):
    repo = git.Repo(local_path)
 
    repo.git.add(all=True)
    
    commit_message = 'Add all changes'
    repo.index.commit(commit_message)

    origin = repo.remote(name='origin')
    origin.push()
    
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
        print(f"Local path '{local_path}' has been deleted.")

    print("Changes have been pushed to the remote repository. PLZ check content on your remote repository.")
