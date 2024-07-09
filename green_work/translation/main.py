from green_work.translation.api import translate_paper
from green_work.translation.output import parse_response_to_md

def translating(api_key, api_url, api_response, intro_picture_path, temporary_cn_path):
    result = translate_paper(api_key, api_url, api_response)
    parse_response_to_md(intro_picture_path, result, temporary_cn_path)
    
    