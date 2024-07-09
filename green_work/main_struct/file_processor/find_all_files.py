import os
import glob


def find_md_files(directory):
    # 使用glob.glob找到所有.md文件
    md_files = glob.glob(os.path.join(directory, '**', '*.md'), recursive=True)
    return md_files

