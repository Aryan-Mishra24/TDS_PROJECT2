# /// script
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "scipy",
#   "matplotlib",
#   "numpy",
#   "chardet",
#   "tabulate",
#   "requests",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import chardet
from scipy.stats import skew, kurtosis

# Load AIPROXY_TOKEN from environment variable
AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

# API endpoint for OpenAI proxy
API_ENDPOINT = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def detect_file_encoding(file_path):
    """Detect file encoding to handle diverse datasets gracefully."""
    with open(file_path, 'rb') as file:
        encoding_result = chardet.detect(file.read())
    return encoding_result.get('encoding', 'utf-8')

def calculate_statistics(dataframe):
    """Calculate advanced statistics like skewness and kurtosis."""
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64'])
    if numeric_columns.empty:
        return "No numeric columns found."
    return {
        "Skewness": numeric_columns.apply(skew).to_dict(),
        "Kurtosis": numeric_columns.apply(kurtosis).to_dict(),
    }

def generate_visualizations(dataframe, output_folder):
    """Generate visualizations and save image files."""
    visualization_paths = []
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64'])
    os.makedirs(output_folder, exist_ok=True)

    # Correlation Heatmap
    if not numeric_columns.empty:
        try:
            plt.figure(figsize=(8, 8))
            correlation_matrix = numeric_columns.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
            plt.title("Correlation Heatmap", fontsize=14)
            plt.xlabel("Features", fontsize=12)
            plt.ylabel("Features", fontsize=12)
            heatmap_path = os.path.join(output_folder, "correlation_heatmap.png")
            plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
            visualization_paths.append(heatmap_path)
            plt.close()
        except Exception as error:
            print(f"Error generating correlation heatmap: {error}")

    # Missing Values Heatmap
    try:
        plt.figure(figsize=(8, 5))
        sns.heatmap(dataframe.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap", fontsize=14)
        missing_values_path = os.path.join(output_folder, "missing_values_heatmap.png")
        plt.savefig(missing_values_path, dpi=150, bbox_inches="tight")
        visualization_paths.append(missing_values_path)
        plt.close()
    except Exception as error:
        print(f"Error generating missing values heatmap: {error}")

    return visualization_paths

def construct_prompt(summary_stats, advanced_stats, correlation_matrix, dataset_name):
    """Generate a detailed prompt for the LLM based on dataset analysis."""
    return f"""
    Below is the analysis summary for the dataset {dataset_name}:

    **Summary Statistics:**
    ```json
    {summary_stats}
    ```

    **Advanced Statistics:**
    ```json
    {advanced_stats}
    ```

    **Correlation Matrix:**
    ```json
    {correlation_matrix}
    ```

    Key Insights:
    - Describe relationships, gaps, and trends in the dataset.
    - Recommend actions based on findings.
    - Highlight strategic implications of these insights.

    Please provide a business-focused report in Markdown format.
    """

def call_llm(prompt):
    """Make an API call to the LLM with the generated prompt."""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000
    }
    response = requests.post(
        API_ENDPOINT,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        },
        json=payload
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        sys.exit(1)

def analyze_dataset(file_path):
    """Main function to analyze a dataset and generate insights and visualizations."""
    # Detect encoding and load dataset
    encoding = detect_file_encoding(file_path)
    dataframe = pd.read_csv(file_path, encoding=encoding)
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = dataset_name
    os.makedirs(output_folder, exist_ok=True)

    # Generate statistics
    summary_stats = dataframe.describe(include='all').to_dict()
    advanced_stats = calculate_statistics(dataframe)
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_columns.corr().to_dict() if not numeric_columns.empty else "N/A"

    # Generate visualizations
    visualization_paths = generate_visualizations(dataframe, output_folder)

    # Construct prompt and call LLM
    prompt = construct_prompt(summary_stats, advanced_stats, correlation_matrix, dataset_name)
    insights = call_llm(prompt)

    # Save results to README.md
    readme_path = os.path.join(output_folder, "README.md")
    with open(readme_path, "w", encoding="utf-8") as readme_file:
        readme_file.write(f"# Analysis of {dataset_name}\n\n")
        readme_file.write("## Insights and Recommendations\n\n")
        readme_file.write("### Business Report\n")
        readme_file.write(insights)
        readme_file.write("\n\n### Visualizations\n")
        for vis_path in visualization_paths:
            readme_file.write(f"![{os.path.basename(vis_path)}]({os.path.basename(vis_path)})\n")

    print(f"Analysis complete for {dataset_name}. Results saved in {readme_path}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)
    analyze_dataset(sys.argv[1])
