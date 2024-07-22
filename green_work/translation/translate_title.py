import requests

def remove_asterisks(text):
    text = text.replace('*', '')
    text = text.replace('\n', '')
    return text

def translate_title(api_key, api_url, title):
    title = title['choices'][0].get('message', {}).get('content', '').strip()
    title = remove_asterisks(title)
    
    TASK = f"""task:
    translate the following text into Chinese.
    use the least words to translate the text.
    {title} 
    """

    PROMPT = f"{TASK}"
    IDENTITY = "You are a professional translater with a background in computer science."

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
        'max_tokens': 150,
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
