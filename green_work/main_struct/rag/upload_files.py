import os
import requests

def upload_files(user_id, kb_id, folder_path, mode="soft", localhost='0.0.0.0'):
    data = {
        "user_id": user_id,
        "kb_id": kb_id,
        "mode": mode
    }
    url = f"http://{localhost}:8777/api/local_doc_qa/upload_files"
    files = []
    for root, dirs, file_names in os.walk(folder_path):
        for file_name in file_names:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                files.append(("files", open(file_path, "rb")))

    response = requests.post(url, files=files, data=data)
    print(response.text)

#upload_files("zzp", "KBb64798d0b81f41889dcea8b5f2b1bb23", "/home/dongpeijie/workspace/marker1/QAnything/test")