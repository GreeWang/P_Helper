import requests
import json

def get_files_status(user_id, kb_id, localhost='0.0.0.0'):
    url = f"http://{localhost}:8777/api/local_doc_qa/list_files"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "user_id": user_id,
        "kb_id": kb_id
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    res = response.json()
    
    if "data" in res and "total" in res["data"] and "gray" in res["data"]["total"]:
        return get_files_status(user_id, kb_id, localhost)
    else:
        print(response.status_code)
        print(response.text)
        return None
    
#get_files_status("zzp", "KBf6b2f74db02c49bfb496a771598f63ee")
    

