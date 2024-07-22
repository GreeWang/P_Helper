import requests

def inspect_title(api_key, api_url, title):

    TASK = f"""
    example: Width & Depth Pruning For Vision Transformers Fang Yu1,2, Kun Huang3, Meng Wang3, Yuan Cheng3*, Wei Chu3**, Li Cui**1 -> Width & Depth Pruning For Vision Transformers
    task:
    
    check whether the following title has unnecessary text. If so, remove it. (for example, some people's names, etc.)
    if the title is already concise, return the title itself.
    {title} 
    ONLY return the title itself, without any unnecessary text.
    """

    PROMPT = f"{TASK}"
    IDENTITY = "You are a checker."

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    # Set request data
    data = {
        'model': 'gemini-pro',  # Use the chat model
        'messages': [
            {'role': 'system', 'content': IDENTITY},
            {'role': 'user', 'content': PROMPT}
        ],
        'max_tokens': 50,
    }

    # Send request
    response = requests.post(api_url, headers=headers, json=data)

    # Parse response
    if response.status_code == 200:
        result = response.json()
        #print(result) #this is test
        return result
    else:
        print(f'Request failed, status code: {response.status_code}')
        print(response.text)
        return None

# Example usage:
# api_key = 'your_api_key_here'
# api_url = 'https://vip.yi-zhan.top/v1/chat/completions'
# paper_content = {'title': 'Sample Title', 'abstract': 'Sample Abstract'}  # replace with your actual paper content
# result = summarize_paper(api_key, api_url, paper_content)
# print(result)
