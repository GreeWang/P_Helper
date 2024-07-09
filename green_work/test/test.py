def get_relative_path_from_absolute(absolute_path, current_directory):
    """
    Removes the current directory part from an absolute path to get a relative path.
    
    :param absolute_path: The absolute path that needs to be converted.
    :param current_directory: The current directory to be removed from the absolute path.
    :return: The relative path obtained by removing the current directory part from the absolute path.
    """
    # 确保当前目录以斜杠结尾，以便正确移除
    if not current_directory.endswith('/'):
        current_directory += '/'
    
    # 移除当前目录部分
    if absolute_path.startswith(current_directory):
        return absolute_path.replace(current_directory, '', 1)
    else:
        # 如果绝对路径不是以当前目录开始的，则返回原始绝对路径
        return absolute_path

# 示例使用
current_directory = "/path/to"
absolute_path = "/path/to/your/file.txt"
relative_path = get_relative_path_from_absolute(absolute_path, current_directory)
print("Relative Path:", relative_path)