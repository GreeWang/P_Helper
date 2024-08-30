import requests
import time
import os
import shutil
import tempfile
from green_work.main_struct.rag.agent import summarize

def remove_asterisks(text):
    return text.replace('*', '')

def create_temp_folder_and_write(content: str, filename: str = "output.txt") -> str:
    # 创建临时文件夹
    temp_dir = tempfile.mkdtemp()
    
    # 创建txt文件并写入内容
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"文件已写入: {file_path}")
    return temp_dir  # 返回临时文件夹路径

def delete_temp_folder(temp_dir: str) -> None:
    # 检测文件夹是否存在
    if os.path.exists(temp_dir):
        # 删除文件夹及其中的所有文件
        shutil.rmtree(temp_dir)
        print(f"已删除临时文件夹: {temp_dir}")
    else:
        print(f"文件夹不存在: {temp_dir}")

def summarize_paper(paper_content, localhost = '0.0.0.0'):
    # Convert the filtered content from the dictionary into string form
    paper_content_str = "\n".join([f"{key}: {value}" for key, value in paper_content.items()])
    paper_content_str = remove_asterisks(paper_content_str)
    temp_dir = create_temp_folder_and_write(paper_content_str)
    result = summarize(localhost, temp_dir)
    delete_temp_folder(temp_dir)
    #print(result)
    return result

# paper_dict = {
#     'title': 'The Impact of Machine Learning Algorithms on Predictive Modeling for Climate Change Analysis',
#     'abstract': 'This study investigates the impact of machine learning algorithms on predictive modeling for climate change analysis. Through a comparative evaluation of different models, including linear regression, decision trees, and neural networks, the research aims to identify the most effective techniques for forecasting temperature and precipitation patterns. The data used in this study encompasses global climate datasets from 1950 to 2020, allowing for a comprehensive examination of long-term trends. The findings suggest that neural networks outperform traditional models in capturing the complexity of climate systems, leading to more accurate predictions. This paper contributes to the growing body of research on the application of machine learning in environmental sciences, providing insights into the future of climate modeling.',
#     'introduction': 'Climate change is one of the most pressing global issues of the 21st century, affecting ecosystems, economies, and societies on a worldwide scale. Accurate prediction of climate trends is critical for developing effective mitigation and adaptation strategies. Traditional climate models, while useful, often fail to capture the complex, non-linear relationships within climate systems. Recent advancements in machine learning (ML) have opened new possibilities for improving predictive accuracy. Machine learning algorithms, particularly those employing neural networks, have demonstrated an ability to model complex patterns in large datasets, offering potential improvements over conventional statistical methods. Previous studies have applied machine learning to various environmental datasets, showing promise in enhancing the resolution and reliability of climate projections. However, there remains a gap in comparative evaluations of different machine learning techniques in the context of climate modeling. This research seeks to fill that gap by conducting a rigorous comparison of multiple algorithms on the same dataset. The objective is to determine which method most effectively predicts key climate variables, such as temperature and precipitation, under various future scenarios.',
#     'method': 'The study utilizes global climate data sourced from the National Centers for Environmental Information (NCEI) and the Coupled Model Intercomparison Project (CMIP6). The dataset spans from 1950 to 2020, covering a range of climate variables, including temperature, humidity, and precipitation. Preprocessing involved cleaning and normalizing the data, removing outliers, and handling missing values using interpolation techniques. Data were then divided into training (80%) and test (20%) sets to evaluate model performance. Three machine learning models were chosen for comparison: linear regression, decision trees, and neural networks. Each model was implemented using Python’s scikit-learn and TensorFlow libraries. The linear regression model served as a baseline due to its simplicity and widespread use in climate studies. Decision trees were included for their ability to model non-linear relationships, while neural networks, particularly a deep feedforward architecture, were tested for their capacity to learn complex patterns. Model performance was evaluated using metrics such as mean squared error (MSE), R-squared, and root mean squared error (RMSE). Cross-validation was applied to ensure the robustness of the results, with hyperparameter tuning performed through grid search. Additionally, feature importance was analyzed to identify the most significant variables contributing to the predictions.'
# }

# summarize_paper(paper_dict)

# Example usage:
# api_key = 'your_api_key_here'
# api_url = 'https://vip.yi-zhan.top/v1/chat/completions'
# paper_content = {'title': 'Sample Title', 'abstract': 'Sample Abstract'}  # replace with your actual paper content
# result = summarize_paper(api_key, api_url, paper_content)
# print(result)
