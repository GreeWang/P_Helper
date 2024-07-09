import os
def remove_asterisks(text):
    text = text.replace('*', '')
    text = text.replace('\n', '')
    return text

def parse_response_to_md(intro_picture_path, api_response, output_filename="output.md"):
    try:
        if not api_response.get('choices') or len(api_response['choices']) == 0:
            raise ValueError("API响应中 'choices' 为空或不存在。")
        
        content = api_response['choices'][0].get('message', {}).get('content', '').strip()
        content = remove_asterisks(content)
        
        title_start = "title:"
        summary_start = "summary:"
        
        title_index = content.lower().find(title_start)
        summary_index = content.lower().find(summary_start)
        
        if title_index == -1 or summary_index == -1:
            raise ValueError("无法在 'content' 中找到标题或摘要。")
        
        title = content[title_index + len(title_start):summary_index].strip()
        summary = content[summary_index + len(summary_start):].strip()

        special_marker = "<!-- end_marker -->"
        file_exists = os.path.exists(output_filename)
        file_has_marker = False
        
        if file_exists:
            file_size = os.path.getsize(output_filename)
            if file_size >= len(special_marker):
                with open(output_filename, "rb") as f:
                    f.seek(-len(special_marker), os.SEEK_END)
                    if f.read().decode('utf-8').strip() == special_marker.strip():
                        file_has_marker = True
                        
        if file_has_marker:
            with open(output_filename, "rb+") as f:
                f.seek(-len(special_marker), os.SEEK_END)
                f.truncate()
        
        # 使用追加模式打开文件
        with open(output_filename, "a") as md_file:
            # 检查文件是否为空或者刚刚被清空
            md_file.seek(0, os.SEEK_END) # 移动到文件末尾
            if md_file.tell() == 0 or not file_has_marker:
                # 文件为空，写入表头
                md_file.write("| Title | Introduction | Summary |\n")
                md_file.write("| :---- | :----------- | :------ |\n")
            
            # 根据图片路径写入内容
            if intro_picture_path is None:
                md_file.write(f"| {title} |  | {summary} |\n")
            else:
                md_file.write(f"| {title} | ![introduction]({intro_picture_path}) | {summary} |\n")
            
            md_file.write(special_marker)

        print(f"Markdown文件 '{output_filename}' 已成功保存。")
    
    except ValueError as e:
        print(f"解析错误：{e}")
    except KeyError as e:
        print(f"解析错误：找不到键 {e}。请检查API返回的内容结构。")
    except Exception as e:
        print(f"发生错误：{e}")

# 使用示例
# parse_response_to_md(None, {
#     'id': 'chatcmpl-02fdd57a1017426aa13041fdc474c265',
#     'model': 'gemini-pro',
#     'object': 'chat.completion',
#     'created': 1720428536,
#     'choices': [{
#         'index': 0,
#         'message': {
#             'role': 'assistant',
#             'content': 'Title: Sample Title\n\nSummary: Research Object: The study focuses on the analysis of text data. Main Content: The article explores the use of natural language processing (NLP) techniques to extract insights from text data, such as sentiment analysis, topic modeling, and machine translation.'
#         },
#         'finish_reason': 'stop'
#     }],
#     'usage': {'prompt_tokens': 250, 'completion_tokens': 56, 'total_tokens': 306}
# })