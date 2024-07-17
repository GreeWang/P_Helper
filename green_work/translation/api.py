import requests

def translate_paper(api_key, api_url, api_response):
    # Convert the filtered content from the dictionary into string form
    content = api_response['choices'][0].get('message', {}).get('content', '').strip()
    print(content)
    REQUIREMENT = """requirements:
    
    MUST return content like this format example: 标题：大语言模型的模型压缩技术 摘要：这篇调查报告全面概述了针对 LLM 的模型压缩技术，包括量化、剪枝、知识蒸馏和低秩分解。它探索了每种技术中的最新进展和创新方法，对现有工作进行分类，并讨论用于评估压缩 LLM 有效性的基准策略和评估指标。该论文强调了模型压缩对于提高 LLM 的效率和实际适用性的重要意义。
    USING the least words to translate the text.
    DO not change the original meaning of the text.
    DO not add any new information.
    DO not change the original formatting.
    """

    TASK = f"""task:
    translate the following text into Chinese.
    {content} 
    """

    PROMPT = f"{TASK}" + f"{REQUIREMENT}"
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
        'max_tokens': 350,
    }

    # Send request
    response = requests.post(api_url, headers=headers, json=data)

    # Parse response
    if response.status_code == 200:
        result = response.json()
        print(result) #this is test
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
