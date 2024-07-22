import os
import shutil

def update_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted existing directory: {folder}")
    
    os.makedirs(folder)
