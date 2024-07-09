import re

def extract_png_names(content):
    result_dict = content
    intro_section = None
    png_names_list = []
    for key, value in result_dict.items():
        if key == "introduction":
            intro_section = value
            break
        
    if intro_section is None:
        return []
            
    # 在提取的内容中查找所有PNG图片文件的名称
    png_pattern = r'!\[.*?\]\((.*?.png)\)'
    png_names_list = re.findall(png_pattern, intro_section)

    # 设置布尔值，如果没有找到图片，则为0；否则为1
    has_images = 1 if png_names_list else 0

    return png_names_list

def get_relative_path_from_absolute(absolute_path, current_directory):
    """
    Removes the current directory part from an absolute path to get a relative path.
    
    :param absolute_path: The absolute path that needs to be converted.
    :param current_directory: The current directory to be removed from the absolute path.
    :return: The relative path obtained by removing the current directory part from the absolute path.
    """
    # 确保当前目录以斜杠结尾，以便正确移除 win可能出现兼容问题
    if not current_directory.endswith('/'):
        current_directory += '/'
    
    if absolute_path.startswith(current_directory):
        return absolute_path.replace(current_directory, '', 1)
    else:
        # 如果绝对路径不是以当前目录开始的，则返回原始绝对路径
        return absolute_path

def get_picture_paths(content, path, current_directory):
    png_names_list = extract_png_names(content)
    png_paths_list = []
    path1 = get_relative_path_from_absolute(path, current_directory)
    for name in png_names_list:
        png_paths_list.append(f"{path1}/../{name}")
    return png_paths_list


# if has_images:
#     print(f"Found PNG images: {png_paths_list}")
# else:
#     print("No PNG images found.")