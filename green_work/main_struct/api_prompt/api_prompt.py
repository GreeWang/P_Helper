import requests

def summarize_paper(api_key, api_url, paper_content):
    # Convert the filtered content from the dictionary into string form
    paper_content_str = "\n\n".join([f"{key}: {value}" for key, value in paper_content.items()])

    REQUIREMENT = """must answer in this structure:
    title:[title (the text of the primary title, REMOVE ALL LINE BREAK AND *)],
    summary: Research Object: The study focuses on [research object]. Main Content: The article explores [main content].
    requirements:
    MUST REMOVE all line breaks and *.
    use more short sentences.
    Try to avoid using the author and research team as the subject, and instead use the research object as the subject to intuitively and succinctly summarize their nature or characteristics.
    take a deep breath!
    Combine all the texts into one paragraph.
    """

    TASK = f"""task:
    MUST Summarize in no more than 75 words, so please choose more important parts to summarize.
    Summarize the following article with emphasis on its research object and main content and method.
    Ignore the unrelated parts (for example, why the author research on this question)
    find out the paper's title
    {paper_content_str} 
    """

    PROMPT = f"{TASK}" + f"{REQUIREMENT}"
    IDENTITY = "You are a professional researcher with a background in computer science. You have a certain understanding of deep learning, natural language processing, and related fields. You possess excellent summarization and reading skills, enabling you to accurately identify the key points of a research paper."

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
        'max_tokens': 250,
    }

    # Send request
    response = requests.post(api_url, headers=headers, json=data)

    # Parse response
    if response.status_code == 200:
        result = response.json()
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
