import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_distribution(data, column, title, xlabel, ylabel, output_path: Path, log_scale=False):
    """Plots a histogram distribution and saves it."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, log_scale=log_scale)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def plot_bar(data, column, title, xlabel, ylabel, output_path: Path):
    """Plots a bar chart and saves it."""
    plt.figure(figsize=(10, 6))
    counts = data[column].value_counts().sort_index()
    counts.plot(kind='bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def get_stats(df, columns):
    """Returns descriptive statistics for specified columns."""
    return df[columns].describe(percentiles=[.25, .5, .75])

def check_quality(df, columns_to_check):
    """Checks for missing and duplicate values."""
    quality_report = []
    for col in columns_to_check:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        # For duplicates, we check if the column is a primary key candidate
        dup_count = df[col].duplicated().sum() if col in df.columns else 0
        quality_report.append({
            "Field": col,
            "Missing Count": missing_count,
            "Missing Percentage": f"{missing_pct:.2f}%",
            "Duplicate Count": dup_count
        })
    return pd.DataFrame(quality_report)
