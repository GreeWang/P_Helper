import git
import os
import shutil

def git_pull(repo_url, local_path):
    # 如果本地文件夹存在，删除它
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
        print(f"Deleted existing directory: {local_path}")
    
    # 创建新的本地文件夹
    os.makedirs(local_path)

    # 克隆远端仓库
    repo = git.Repo.clone_from(repo_url, local_path)
    print(f"Cloned repository from {repo_url} to {local_path}")
    
    readme_path = os.path.join(local_path, 'README.md')
    readme_cn_path = os.path.join(local_path, 'README_CN.md')

    return readme_path, readme_cn_path
    
def git_origin(local_path):
    repo = git.Repo(local_path)
 
    # 暂存本地仓库的文件
    repo.git.add(all=True)
    
    # 提交变更
    commit_message = 'Add all changes'
    repo.index.commit(commit_message)

    # 推送到远端
    origin = repo.remote(name='origin')
    origin.push()

    print("Changes have been pushed to the remote repository.")
