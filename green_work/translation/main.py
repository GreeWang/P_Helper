from .api import translate_paper
from .output import parse_response_to_md
from .translate_title import translate_title

def translating(api_key, api_url, api_response, intro_picture_path, temporary_cn_path, title, be_in_file):
    title = translate_title(api_key, api_url, title)
    title = title['choices'][0].get('message', {}).get('content', '').strip()
    result = translate_paper(api_key, api_url, api_response)
    parse_response_to_md(be_in_file, title, intro_picture_path, result, temporary_cn_path)
    
    