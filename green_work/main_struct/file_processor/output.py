import os

def remove_asterisks(text):
    text = text.replace('*', '')
    text = text.replace('\n', '')
    return text

def title_exists_in_file(title, filename):
    """检查标题是否存在于文件中"""
    if not os.path.exists(filename):
        return False  # 文件不存在，标题肯定不存在
    with open(filename, "r", encoding="utf-8") as f:
        if remove_asterisks(title) in f.read():
            return True
    return False

def parse_response_to_md(title, intro_picture_path, api_response, output_filename="output.md"):
    try:
        # 解析 title（假设 title 仍然是字典，需要从中提取内容）
        title_content = title['choices'][0].get('message', {}).get('content', '').strip()
        title_content = remove_asterisks(title_content)

        # 直接使用 api_response 字符串
        summary = remove_asterisks(api_response.strip())

        # 检查标题是否已存在
        if title_exists_in_file(title_content, output_filename):
            print(f"标题 '{title_content}' 已存在于文件 '{output_filename}' 中，跳过写入。")
            return 1

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
        with open(output_filename, "a", encoding="utf-8") as md_file:
            # 检查文件是否为空或者刚刚被清空
            md_file.seek(0, os.SEEK_END) # 移动到文件末尾
            if md_file.tell() == 0 or not file_has_marker:
                # 文件为空，写入表头
                md_file.write("| Title | Introduction | Summary |\n")
                md_file.write("| :---- | :----------- | :------ |\n")
            
            # 根据图片路径写入内容
            if intro_picture_path is None:
                md_file.write(f"| {title_content} |  | {summary} |\n")
            else:
                md_file.write(f"| {title_content} | ![introduction]({intro_picture_path}) | {summary} |\n")
            
            md_file.write(special_marker)

        print(f"Markdown文件 '{output_filename}' 已成功保存。")
    
    except KeyError as e:
        print(f"解析错误：找不到键 {e}。请检查API返回的内容结构。")
    except Exception as e:
        print(f"发生错误：{e}")
        
    return 0
