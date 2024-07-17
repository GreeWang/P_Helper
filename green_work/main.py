from main_struct.file_processor.filter import process_markdown
from main_struct.api_prompt.api_prompt import summarize_paper
from main_struct.picture_processor.find_picture import get_picture_paths
from main_struct.picture_processor.picture_filter import find_largest_resolusion_ratio
from main_struct.picture_processor.update_img_repo import transfer_folder_contents
from main_struct.file_processor.output import parse_response_to_md
from main_struct.file_processor.find_all_files import find_md_files
from main_struct.git_processor.git_to_romote import git_origin, git_pull
from translation.main import translating
from translation.translation_order import need_translation

import time

def green(folder):
    api_key = 'sk-qan80EN2mmbBmngj475851C4619f45E99d771e71Df13A1Ba'
    api_url = 'https://vip.yi-zhan.top/v1/chat/completions'
    repo_url = 'https://github.com/GreeWang/summer_reshearch_out_test.git'
    local_repo = 'local_repo'
    order = need_translation() 
    temporary_path, temporary_cn_path = git_pull(repo_url, local_repo)
    md_files = find_md_files(folder)
    
    for md_file in md_files:
        
        paper_content = process_markdown(md_file)
        result = summarize_paper(api_key, api_url, paper_content)
        
        if result is not None:
            picture_paths = get_picture_paths(paper_content, md_file, folder)
            intro_picture_path = find_largest_resolusion_ratio(picture_paths, local_repo)
            parse_response_to_md(intro_picture_path, result, temporary_path)
            if order:
                translating(api_key, api_url, result, intro_picture_path, temporary_cn_path)
            time.sleep(7)
        else:
            continue
        
    transfer_folder_contents(folder, f'{local_repo}/img')
    git_origin(local_repo)
    
#this is test   
green('/home/dongpeijie/workspace/marker1/sr/output_example')
    