import sys
import requests
import time

def send_request_without_history(user_id, kb_ids, question, custom_prompt, localhost='0.0.0.0'):
    headers = {
        'content-type': 'application/json'
    }
    data = {
        "user_id": user_id,
        "kb_ids": kb_ids,
        "question": question,
        "custom_prompt": custom_prompt
    }
    url = f"http://{localhost}:8777/api/local_doc_qa/local_doc_chat"
    try:
        start_time = time.time()
        response = requests.post(url=url, headers=headers, json=data, timeout=60)
        end_time = time.time()
        res = response.json()
        last_history = res['history'][-1]
        last_answer = last_history[-1]
        #print(last_answer)
        print(f"响应状态码: {response.status_code}, 响应时间: {end_time - start_time}秒")
    except Exception as e:
        print(f"请求发送失败: {e}")
    return question, last_answer
 
        
def send_request_with_history(user_id, kb_ids, question, history, custom_prompt, localhost="0.0.0.0"):
    headers = {
        'content-type': 'application/json'
    }
    data = {
        "user_id": user_id,
        "kb_ids": kb_ids,
        "question": question,
        "custom_prompt": custom_prompt,
        "history": history
    }
    url = f"http://{localhost}:8777/api/local_doc_qa/local_doc_chat"
    try:
        start_time = time.time()
        response = requests.post(url=url, headers=headers, json=data, timeout=60)
        end_time = time.time()
        res = response.json()
        last_history = res['history'][-1]
        last_answer = last_history[-1]
        #print(last_answer)
        print(f"响应状态码: {response.status_code}, 响应时间: {end_time - start_time}秒")
    except Exception as e:
        print(f"请求发送失败: {e}")
    return question, last_answer

#send_request_without_history("zzp", ["KBf6b2f74db02c49bfb496a771598f63ee"], "总结", "questioner")
        
