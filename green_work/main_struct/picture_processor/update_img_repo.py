import shutil
import os

def transfer_folder_contents(source_folder, destination_folder):
    
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)
        
        if os.path.exists(destination_path):
            if os.path.isfile(destination_path):
                os.remove(destination_path)
            elif os.path.isdir(destination_path):
                shutil.rmtree(destination_path)
        
        # 复制文件或文件夹，而不是移动
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)
        elif os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path)
            
#transfer_folder_contents('/home/dongpeijie/workspace/marker1/sr/result_of_vit_prune', 'local_repo/img')