import requests
import json

def delete_kb(user_id, kb_ids, localhost="0.0.0.0"):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "user_id": user_id,
        "kb_ids": kb_ids
    }
    url = f"http://{localhost}:8777/api/local_doc_qa/delete_knowledge_base"
    response = requests.post(url, headers=headers, data=json.dumps(data))

    print(response.status_code)
    print(response.text)

#delete_kb("zzp", ["KBb64798d0b81f41889dcea8b5f2b1bb23"])