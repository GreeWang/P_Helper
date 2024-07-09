import re

def process_markdown(file_path):
    # Read Markdown file content
    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    # Split content by lines
    lines = md_content.splitlines()
    filtered_sections = []
    skip_next = False

    # Filter out acknowledgment and reference sections
    for line in lines:
        if line.lower().startswith('## acknowledgment') or line.lower().startswith('## reference'):
            skip_next = True
            continue
        if not skip_next:
            filtered_sections.append(line)

    # Recombine filtered content
    filtered_md_content = '\n'.join(filtered_sections)

    # Extract content before the first numbered secondary heading
    abstract_pattern = r'(?s)(.*?)(?=\n## \d+)'
    abstract_match = re.search(abstract_pattern, filtered_md_content)

    filtered_dict = {}
    if abstract_match:
        abstract_content = abstract_match.group(1).strip()
        if '## abstract' in abstract_content.lower():
            abstract_start = abstract_content.lower().index('## abstract')
            abstract_content = abstract_content[abstract_start:]
        filtered_dict["abstract"] = abstract_content

    # Extract all secondary headings with content
    pattern = r'##\s+(\d+)(?:\.|\s+)(.*?)\n(.*?)(?=\n## \d+|\Z)'
    matches = re.findall(pattern, filtered_md_content, re.DOTALL)

    method_content = ""
    introduction_added = False

    wanted_words = []
    unwanted_words = ["conclu", "related work", "background", "framework", "discussion", "metric", "benchmark", "future", "challenge", "acknowledgment", "reference", "other"]

    # Process matches for introduction and methods
    for number, title, content in matches:
        lower_title = title.strip().lower()
        if not introduction_added:
            filtered_dict["introduction"] = content.strip()
            introduction_added = True
            continue

        matched_wanted_word = next((word for word in wanted_words if word in lower_title), None)
        matched_unwanted_word = next((word for word in unwanted_words if word in lower_title), None)

        if matched_wanted_word:
            filtered_dict[matched_wanted_word] = content.strip()
        elif not matched_wanted_word and not matched_unwanted_word:
            method_content += content.strip() + "\n\n"

    if method_content:
        filtered_dict["method"] = method_content.strip()

    return filtered_dict

# Example usage:
# file_path = 'path/to/your/file.md'
# result = process_markdown(file_path)
# print(result)
