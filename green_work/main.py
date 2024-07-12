from green_work.main_struct.file_processor.filter import process_markdown
from green_work.main_struct.api_prompt.api_prompt import summarize_paper
from green_work.main_struct.picture_processor.find_picture import get_picture_paths
from green_work.main_struct.picture_processor.picture_filter import find_largest_resolusion_ratio
from green_work.main_struct.picture_processor.update_img_repo import transfer_folder_contents
from green_work.main_struct.file_processor.output import parse_response_to_md
from green_work.main_struct.file_processor.find_all_files import find_md_files
from green_work.main_struct.git_processor.git_to_romote import git_origin, git_pull
from green_work.translation.main import translating
from green_work.translation.translation_order import need_translation

import time

def green(folder):
    api_key = 'your_api_key'
    api_url = 'your_api_url'
    repo_url = 'your_remote_repo_url'
    local_repo = 'local_repo'
    order = need_translation() 
    temporary_path, temporary_cn_path = git_pull(repo_url, local_repo)
    md_files = find_md_files(folder)
    
    for md_file in md_files:
        
        paper_content = process_markdown(md_file)
        result = summarize_paper(api_key, api_url, paper_content)
        
        if result is not None:
            picture_paths = get_picture_paths(paper_content, md_file, folder)
            intro_picture_path = find_largest_resolusion_ratio(picture_paths)
            parse_response_to_md(intro_picture_path, result, temporary_path)
            if order:
                translating(api_key, api_url, result, intro_picture_path, temporary_cn_path)
            time.sleep(7)
        else:
            continue
        
    transfer_folder_contents(folder, f'{local_repo}/img')
    git_origin('local_repo')
    
#this is test   
#green('/home/dongpeijie/workspace/marker/output_example')
    