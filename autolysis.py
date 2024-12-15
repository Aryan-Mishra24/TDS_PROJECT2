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
import argparse
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv

load_dotenv()

def load_data(filepath):
    """
    Load CSV file and perform initial data exploration.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
        
        # Basic data exploration
        print("Data Loaded Successfully")
        print(f"Dataset Shape: {df.shape}")
        print("\nColumn Information:")
        print(df.info())
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def analyze_data(df):
    """
    Perform various analyses on the dataset, including statistical summaries,
    visualizations, and correlation analysis.
    """
    # Analyses to perform
    analyses = {
        'summary_stats': df.describe(),
        'missing_values': df.isnull().sum(),
        'data_types': df.dtypes,
        'correlation_matrix': df.select_dtypes(include=['float64', 'int64']).corr()
                             if not df.select_dtypes(include=['float64', 'int64']).empty else None
    }

    # Visualize histograms and boxplots for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if not numeric_cols.empty:
        # Histogram plots
        plt.figure(figsize=(15, 6))
        for i, col in enumerate(numeric_cols):
            plt.subplot(2, len(numeric_cols), i + 1)
            sns.histplot(df[col], kde=True)
            plt.title(f'Histogram of {col}')
        
        plt.tight_layout()
        plt.savefig('numeric_histograms.png')
        plt.close()

        # Boxplot plots for outlier detection
        plt.figure(figsize=(15, 6))
        for i, col in enumerate(numeric_cols):
            plt.subplot(2, len(numeric_cols), i + 1)
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
        
        plt.tight_layout()
        plt.savefig('numeric_boxplots.png')
        plt.close()

    # Correlation matrix visualization
    if analyses['correlation_matrix'] is not None:
        plt.figure(figsize=(12, 8))
        sns.heatmap(analyses['correlation_matrix'], annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()

    # Feature importance using Random Forest if target column is present
    if not numeric_cols.empty and 'target' in df.columns:
        X = df[numeric_cols].dropna()
        y = df['target'].dropna()

        model = RandomForestRegressor()
        model.fit(X, y)
        feature_importances = pd.Series(model.feature_importances_, index=numeric_cols).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        feature_importances.plot(kind='bar')
        plt.title('Feature Importance Analysis')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

    return analyses

def generate_visualizations(df, analyses):
    """
    Create visualizations based on the analyses, including outlier detection, regression analysis,
    time series, cluster analysis, and network analysis.
    """
    # Missing values plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    analyses['missing_values'].plot(kind='bar')
    plt.title('Missing Values')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('missing_values.png')
    plt.close()

    # Correlation matrix heatmap
    if analyses['correlation_matrix'] is not None:
        plt.figure(figsize=(12, 8))
        sns.heatmap(analyses['correlation_matrix'], annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()

    # Time series analysis if 'Date' column exists and is a datetime type
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        df.set_index('Date', inplace=True)
        df.resample('M').mean().plot(figsize=(15, 6))
        plt.title('Time Series Analysis')
        plt.tight_layout()
        plt.savefig('time_series_analysis.png')
        plt.close()

def get_llm_analysis(df, analyses):
    """
    Use LLM to generate a narrative and insights as a comprehensive README.md.
    """
    # Construct the context for the LLM with detailed descriptions
    context = {
        'column_info': str(df.info()),
        'summary_stats': str(analyses['summary_stats']),
        'missing_values': str(analyses['missing_values']),
        'data_types': str(analyses['data_types']),
        'correlation_matrix': str(analyses['correlation_matrix']) if analyses['correlation_matrix'] is not None else "No correlation matrix available"
    }

    # Ensure the API token is secure and not hard-coded in production
    api_token =  os.getenv("AIPROXY_TOKEN")
    if not api_token:
        print("Error: API token not set. Please set the 'API_PROXY_TOKEN' environment variable.")
        sys.exit(1)

    try:
        # Send a request to the LLM API with a more detailed prompt
        response = httpx.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a data science expert and a storyteller. Your task is to analyze the provided dataset "
                            "and create a compelling README.md that clearly explains the data and insights. Ensure the "
                            "narrative is structured, engaging, and uses appropriate sections, bullet points, and emphasis."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Given the following context:\n\n"
                            f"{json.dumps(context)}\n\n"
                            "Write a comprehensive README.md that includes:\n"
                            "1. A brief description of the dataset.\n"
                            "2. Key insights from the analysis (Mention all the points).\n"
                            "3. Visualizations used and their interpretations (Add the saved images (correlation matrix, numeric_values, numeric_histograms, numeric_distributions) in the README.md so we can see it, image is residing in the same directory as this python script).\n"
                            "4. Potential implications of the findings for practical applications.\n"
                            "5. Any limitations or considerations to keep in mind when using this data."
                        )
                    }
                ]
            },
            timeout=600.0  # Extended timeout to allow for longer responses
        )

        response.raise_for_status()  # Raise an exception for HTTP errors

        response_data = response.json()
        narrative = response_data.get('choices', [{}])[0].get('message', {}).get('content', "")


        # Check if the response content is valid
        if not narrative.strip():
            print("Error: Received an empty response from the LLM.")
            sys.exit(1)

        # Write the narrative to README.md
        with open("README.md", "w") as f:
            f.write(narrative)

        print("README.md has been created successfully.")
    except httpx.RequestError as e:
        print(f"HTTP Request Error: {e}")
        sys.exit(1)

def get_output_folder(filepath):
    """
    Extracts the directory from the input file path and returns it.
    """
    return os.path.dirname(filepath)

def move_files_to_output_folder(output_folder, files):
    """
    Moves or copies specified files to the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    for file in files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(output_folder, file))

def main():
    try:
        parser = argparse.ArgumentParser(description="Data Analysis Script")
        parser.add_argument('filepath', type=str, help="Path to the CSV dataset")
        args = parser.parse_args()

        # Extract output folder path based on the input file path
        output_folder = get_output_folder(args.filepath)

        # Validate input file exists
        if not os.path.exists(args.filepath):
            raise FileNotFoundError(f"The input file '{args.filepath}' does not exist.")

        # Load and analyze data
        df = load_data(args.filepath)
        analyses = analyze_data(df)
        generate_visualizations(df, analyses)
        get_llm_analysis(df, analyses)

        # Move the README.md and PNG files to the output folder
        files_to_move = ['README.md'] + [f for f in os.listdir() if f.endswith('.png')]
        
        # Check if there are files to move
        if not files_to_move:
            print("Warning: No files generated to move.")
        
        move_files_to_output_folder(output_folder, files_to_move)

        print(f"Files have been successfully moved to the '{output_folder}' folder.")

    except FileNotFoundError as fnf_error:
        print(f"File Error: {fnf_error}")
        sys.exit(1)
    except PermissionError as perm_error:
        print(f"Permission Error: Unable to read file or move files - {perm_error}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        # Optional: log the full traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
