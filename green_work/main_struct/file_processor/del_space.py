import os

def replace_spaces_with_underscore(path):
    """Replace spaces in the last component of the path with underscores."""
    base_name = os.path.basename(path)
    new_base_name = base_name.replace(' ', '_')
    return os.path.join(os.path.dirname(path), new_base_name)

def generate_unique_path(path):
    """Generate a unique path by appending an index if the path already exists."""
    base_name, extension = os.path.splitext(path)
    counter = 1
    new_path = f"{base_name}_{counter}{extension}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_name}_{counter}{extension}"
    return new_path

def replace_spaces_in_folder_names(root_folder):
    for root, dirs, files in os.walk(root_folder, topdown=True):  # Set topdown to True
        # 处理文件夹名称
        for dir_index, dir_name in enumerate(dirs):
            if ' ' in dir_name:
                old_dir_path = os.path.join(root, dir_name)
                new_dir_path = replace_spaces_with_underscore(old_dir_path)
                
                # 如果新路径与旧路径不相同且已存在，则生成一个唯一的新路径
                if os.path.exists(new_dir_path) and old_dir_path != new_dir_path:
                    new_dir_path = generate_unique_path(new_dir_path)
                
                if old_dir_path != new_dir_path:
                    os.rename(old_dir_path, new_dir_path)
                    print(f"Renamed directory: {old_dir_path} -> {new_dir_path}")
                    # 更新dirs列表以反映新的路径
                    dirs[dir_index] = os.path.basename(new_dir_path)

        # 处理文件名称，文件不需要在这里处理路径更新，因为它们不会影响os.walk的遍历
        for file_name in files:
            if ' ' in file_name:
                old_file_path = os.path.join(root, file_name)
                new_file_path = replace_spaces_with_underscore(old_file_path)
                
                # 如果新路径与旧路径不相同且已存在，则跳过重命名
                if old_file_path != new_file_path and not os.path.exists(new_file_path):
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed file: {old_file_path} -> {new_file_path}")