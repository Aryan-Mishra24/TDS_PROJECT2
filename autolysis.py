import base64
import logging
import os
import shutil
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import json
import requests
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import stat

# Load environment variables
load_dotenv()

def load_data(filepath, encoding='ISO-8859-1'):
    """
    Load CSV file and perform initial data exploration.
    Supports both local files and URLs.
    """
    try:
        if filepath.startswith("http://") or filepath.startswith("https://"):
            response = requests.get(filepath)
            response.raise_for_status()
            temp_file = "temp_dataset.csv"
            with open(temp_file, "wb") as f:
                f.write(response.content)
            filepath = temp_file

        df = pd.read_csv(filepath, encoding=encoding)
        print(f"Data Loaded Successfully\nDataset Shape: {df.shape}")
        print(df.info())
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

def analyze_data(df):
    """
    Perform statistical analysis and visualize data.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    if numeric_cols.empty:
        logging.warning("No numeric columns found for analysis.")
        return {}

    analyses = {
        'summary_stats': df.describe(),
        'missing_values': df.isnull().sum(),
        'data_types': df.dtypes,
        'correlation_matrix': df[numeric_cols].corr()
    }

    for col in numeric_cols:
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.savefig(f'{col}_histogram.png')
        plt.close()

        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.savefig(f'{col}_boxplot.png')
        plt.close()

    sns.heatmap(analyses['correlation_matrix'], annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()

    return analyses

def create_visualizations(df, analyses):
    """
    Generate visualizations based on the data and analyses.
    """
    if not analyses.get('missing_values').empty:
        analyses['missing_values'].plot(kind='bar')
        plt.title('Missing Values')
        plt.savefig('missing_values.png')
        plt.close()

    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
        if not df['Date'].empty:
            df.set_index('Date').resample('M').mean().plot()
            plt.title('Time Series Analysis')
            plt.savefig('time_series_analysis.png')
            plt.close()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def send_request_with_retry(api_token, context):
    """
    Send a POST request to the LLM API with retries on failure.
    """
    response = httpx.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_token}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a data scientist tasked with creating README.md."},
                {"role": "user", "content": json.dumps(context)}
            ]
        },
        timeout=600
    )
    response.raise_for_status()
    return response

def generate_readme(df, analyses):
    """
    Use an LLM to generate a README.md based on analysis.
    """
    context = {
        'column_info': df.info(),
        'summary_stats': analyses.get('summary_stats', {}).to_string(),
        'missing_values': analyses.get('missing_values', {}).to_string(),
        'data_types': analyses.get('data_types', {}).to_string(),
        'correlation_matrix': analyses.get('correlation_matrix', {}).to_string() if analyses.get('correlation_matrix') is not None else "None"
    }

    api_token = os.getenv("AIPROXY_TOKEN")
    if not api_token:
        logging.error("API token not found.")
        sys.exit(1)

    try:
        response = send_request_with_retry(api_token, context)
        narrative = response.json()['choices'][0]['message']['content']
        with open('README.md', 'w') as f:
            f.write(narrative)
        print("README.md created successfully.")
    except httpx.RequestError as e:
        logging.error(f"Request error: {e}")
        sys.exit(1)

def onerror(func, path, exc_info):
    """
    Error handler for shutil.rmtree to handle readonly files.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

def move_files(output_folder):
    """
    Move generated files to the specified output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    for file in [f for f in os.listdir() if f.endswith(('.png', '.md'))]:
        shutil.move(file, os.path.join(output_folder, file))

def main():
    filepath = input("Enter the path to the dataset: ").strip()
    encoding = input("Enter the file encoding (default 'ISO-8859-1'): ").strip() or 'ISO-8859-1'

    if filepath.startswith("http://") or filepath.startswith("https://"):
        logging.info("Downloading dataset from URL...")
    elif not os.path.exists(filepath):
        logging.error("Input file not found.")
        sys.exit(1)

    df = load_data(filepath, encoding=encoding)
    analyses = analyze_data(df)
    create_visualizations(df, analyses)
    generate_readme(df, analyses)

    try:
        move_files(os.path.dirname(filepath))
    except PermissionError as e:
        logging.error(f"Permission error during file move: {e}")
        shutil.rmtree(os.path.dirname(filepath), onerror=onerror)

if __name__ == "__main__":
    main()

