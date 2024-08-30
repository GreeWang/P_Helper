import requests
import json

def create_kb(user_id, kb_name, localhost= '0.0.0.0'):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "user_id": user_id,
        "kb_name": kb_name
    }
    url = f"http://{localhost}:8777/api/local_doc_qa/new_knowledge_base"
    response = requests.post(url, headers=headers, data=json.dumps(data))

    print(response.status_code)
    print(response.text)

    # 将response.text解析为字典
    response_data = json.loads(response.text)

    # 使用get方法获取kb_id
    kb_id = response_data.get('data', {}).get('kb_id')
    #test
    #print(kb_id)
    return kb_id

#create_kb("zzp", "1")
