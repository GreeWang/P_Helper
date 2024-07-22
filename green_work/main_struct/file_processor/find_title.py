def find_heading(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read(300)  # 读取前300个字符
        
        # 按行拆分内容
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith("# ") or line.startswith("## "):  # 检查是否为一级或二级标题
                return line.replace("#", "")
        
        return "No first-level or second-level heading found in the first 300 characters."
    
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"


# 使用示例
# file_path = 'example.md'  # 替换为你的Markdown文件路径
# heading = find_heading('/home/dongpeijie/workspace/marker1/sr/result_of_vit_prune/2308.04657v1/2308.04657v1.md')
# print(heading)
