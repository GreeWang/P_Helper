from green_work.main_struct.file_processor.filter import process_markdown
from green_work.main_struct.api_prompt.api_prompt import summarize_paper
from green_work.main_struct.picture_processor.find_picture import get_picture_paths
from green_work.main_struct.picture_processor.picture_filter import find_largest_resolusion_ratio
from green_work.main_struct.file_processor.output import parse_response_to_md
from green_work.main_struct.file_processor.find_all_files import find_md_files
from green_work.main_struct.git_processor.git_to_romote import git_origin, git_pull
from green_work.translation.main import translating
from green_work.translation.translation_order import need_translation

def green(folder, current_directory):
    api_key = 'sk-KYe0arAHXoNtKv77A98d28Cd1d314c6899Fe898a8b46Fe94'
    api_url = 'https://vip.yi-zhan.top/v1/chat/completions'
    order = need_translation() 
    temporary_path, temporary_cn_path = git_pull('https://github.com/GreeWang/summer_reshearch_out_test.git', 'local_repo')
    md_files = find_md_files(folder)
    
    for md_file in md_files:
        
        paper_content = process_markdown(md_file)
        result = summarize_paper(api_key, api_url, paper_content)
        
        if result is not None:
            picture_paths = get_picture_paths(paper_content, md_file, current_directory)
            intro_picture_path = find_largest_resolusion_ratio(picture_paths)
            parse_response_to_md(intro_picture_path, result, temporary_path)
            if order:
                translating(api_key, api_url, result, intro_picture_path, temporary_cn_path)
        else:
            continue
        
    git_origin('local_repo')
    
#this is test   
#green('/home/dongpeijie/workspace/marker/output_example', '/home/dongpeijie/workspace/marker')
    