# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "openai",
#   "scikit-learn",
#   "python-dotenv",
#   "pytest-shutil",
#   "logging"
# ]
# ///

import os
import sys
import logging
import shutil
import base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import json
import argparse
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv
from multiprocessing import Pool

load_dotenv()

MAX_COLUMNS = 5  # Limit visualizations to top N columns


def read_csv_data(file_path):
    """
    Reads a CSV file and provides an overview of the data.
    """
    try:
        dataset = pd.read_csv(file_path, encoding='ISO-8859-1')
        print("Dataset loaded successfully.")
        print(f"Dimensions of the dataset: {dataset.shape}")
        print("\nColumns Overview:")
        print(dataset.info())
        return dataset
    except Exception as error:
        print(f"Failed to load data: {error}")
        sys.exit(1)


def create_visuals(data, analysis):
    """
    Generates visual outputs including histograms, boxplots, missing values, and correlation heatmaps.
    """
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = numeric_columns[:MAX_COLUMNS]  # Limit to top N columns

    # Histograms
    if not numeric_columns.empty:
        plt.figure(figsize=(15, 6))
        for idx, column in enumerate(numeric_columns):
            plt.subplot(1, len(numeric_columns), idx + 1)
            sns.histplot(data[column], kde=True)
            plt.title(f'Distribution of {column}')
        plt.tight_layout()
        plt.savefig('numeric_histograms.png')
        plt.close()

    # Boxplots
    if not numeric_columns.empty:
        plt.figure(figsize=(15, 6))
        for idx, column in enumerate(numeric_columns):
            plt.subplot(1, len(numeric_columns), idx + 1)
            sns.boxplot(x=data[column])
            plt.title(f'Boxplot for {column}')
        plt.tight_layout()
        plt.savefig('numeric_boxplots.png')
        plt.close()

    # Missing Values
    plt.figure(figsize=(10, 6))
    analysis['missing_counts'][:MAX_COLUMNS].plot(kind='bar', color='skyblue')
    plt.title('Missing Values by Column')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('missing_values.png')
    plt.close()

    # Correlation Heatmap
    if analysis['correlation'] is not None:
        plt.figure(figsize=(12, 8))
        sns.heatmap(analysis['correlation'], annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()


def perform_analysis(data):
    """
    Executes various data analyses, including summary statistics and correlation studies.
    """
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    analysis_results = {
        'summary': data.describe(include='all'),
        'missing_counts': data.isnull().sum().sort_values(ascending=False),
        'data_types': data.dtypes,
        'correlation': numeric_data.corr() if not numeric_data.empty else None
    }

    # Generate visuals
    create_visuals(data, analysis_results)

    return analysis_results


def generate_readme(data, analysis):
    """
    Utilizes an LLM to create a structured README file based on data insights and visual outputs.
    """
    context = {
        'column_metadata': str(data.info()),
        'stat_summary': str(analysis['summary']),
        'null_counts': str(analysis['missing_counts']),
        'types_of_data': str(analysis['data_types']),
        'corr_matrix': str(analysis['correlation']) if analysis['correlation'] is not None else "No correlations available.",
        'visuals': "The following visuals were generated: histograms, boxplots, missing values bar chart, and correlation heatmap."
    }

    api_key = os.getenv("AIPROXY_TOKEN")
    if not api_key:
        print("Error: Missing API key. Ensure the 'AIPROXY_TOKEN' environment variable is set.")
        sys.exit(1)

    try:
        response = httpx.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a skilled data science communicator. Use the provided dataset analysis to create a clear, "
                            "informative README file, emphasizing insights and practical relevance."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Dataset Analysis Context: {json.dumps(context)}\n"
                            "Create a README.md with:\n"
                            "- Dataset overview\n"
                            "- Key findings and visuals\n"
                            "- Practical applications\n"
                            "- Limitations."
                        )
                    }
                ]
            },
            timeout=300
        )

        response.raise_for_status()

        output = response.json().get('choices', [{}])[0].get('message', {}).get('content', "")

        if not output.strip():
            print("Error: Empty response received.")
            sys.exit(1)

        with open("README.md", "w") as readme_file:
            readme_file.write(output)

        print("README.md created successfully.")
    except httpx.RequestError as req_error:
        print(f"Request Error: {req_error}")
        sys.exit(1)


def get_directory(file_path):
    """
    Determines the folder path for the input file.
    """
    return os.path.dirname(file_path)


def relocate_files(target_folder, filenames):
    """
    Moves specified files to the given folder.
    """
    os.makedirs(target_folder, exist_ok=True)
    for filename in filenames:
        if os.path.exists(filename):
            shutil.move(filename, os.path.join(target_folder, filename))


def process_file(file_path):
    """
    Processes a single CSV file.
    """
    dataset = read_csv_data(file_path)
    analysis_outcomes = perform_analysis(dataset)
    generate_readme(dataset, analysis_outcomes)

    generated_files = ['README.md'] + [file for file in os.listdir() if file.endswith('.png')]

    if not generated_files:
        print("Warning: No files generated.")

    relocate_files(get_directory(file_path), generated_files)
    print(f"All generated files moved to '{get_directory(file_path)}'.")


def run():
    """
    Entry point for the data analysis script.
    """
    parser = argparse.ArgumentParser(description="Execute data analysis from CSV file.")
    parser.add_argument('file_path', type=str, nargs='+', help="Path(s) to the CSV file(s).")
    args = parser.parse_args()

    with Pool() as pool:
        pool.map(process_file, args.file_path)


if __name__ == "__main__":
    run()
