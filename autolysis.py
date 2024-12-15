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
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv
from multiprocessing import Pool

load_dotenv()

MAX_COLUMNS = 5  # Limit visualizations to top N columns

# Helper function for time-series plotting
def plot_time_series(data):
    """
    Generates time-series plots if a datetime column exists.
    """
    datetime_cols = data.select_dtypes(include=['datetime64']).columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns[:MAX_COLUMNS]

    if not datetime_cols.any():
        print("No datetime columns found for time-series plotting.")
        return

    for datetime_col in datetime_cols:
        plt.figure(figsize=(15, 6))
        for idx, num_col in enumerate(numeric_cols):
            plt.subplot(1, len(numeric_cols), idx + 1)
            sns.lineplot(x=data[datetime_col], y=data[num_col])
            plt.title(f"{num_col} Over Time")
        plt.tight_layout()
        plt.savefig(f"time_series_{datetime_col}.png")
        plt.close()

# Helper function for regression analysis
def regression_analysis(data):
    """
    Performs regression analysis and generates an actual vs predicted plot.
    """
    numeric_cols = data.select_dtypes(include=['float64', 'int64'])

    if numeric_cols.shape[1] < 2:
        print("Not enough numeric columns for regression analysis.")
        return

    target_col = numeric_cols.columns[0]  # Assume the first column as target
    feature_cols = numeric_cols.columns[1:]  # Remaining as features

    X = data[feature_cols].dropna()
    y = data[target_col].dropna()

    if X.shape[0] != y.shape[0]:
        print("Mismatch in feature and target sizes for regression.")
        return

    model = RandomForestRegressor()
    model.fit(X, y)

    y_pred = model.predict(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig('actual_vs_predicted.png')
    plt.close()

    print(f"Regression Analysis - R^2: {r2_score(y, y_pred):.4f}, MAE: {mean_absolute_error(y, y_pred):.4f}")

# Helper function for clustering analysis
def clustering_analysis(data):
    """
    Performs K-means clustering and generates cluster visualizations.
    """
    numeric_cols = data.select_dtypes(include=['float64', 'int64'])

    if numeric_cols.empty:
        print("No numeric columns available for clustering.")
        return

    X = numeric_cols.dropna()
    inertia = []
    range_n_clusters = range(1, 11)

    for k in range_n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range_n_clusters, inertia, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.savefig('kmeans_elbow.png')
    plt.close()

    # Perform clustering with optimal k (e.g., k=3 for demonstration)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X)

    if X.shape[1] >= 2:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=clusters, palette='viridis')
        plt.title(f'K-Means Clustering (k={optimal_k})')
        plt.savefig('kmeans_clusters.png')
        plt.close()

# Updated function to include new analyses
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

    # Add time-series plot
    plot_time_series(data)

    # Perform regression analysis
    regression_analysis(data)

    # Perform clustering analysis
    clustering_analysis(data)

    return analysis_results

# Other functions remain unchanged

if __name__ == "__main__":
    run()
