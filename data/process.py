"""
Processes all raw datasets required for the project.

Run this script from the project root directory:
    `python data/process.py`
"""

import os
import subprocess
import sys
import datasets
import pandas as pd
import re
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

# ----- file paths ---------------
BASE = os.path.join(os.path.dirname(__file__))

def ensure_dirs():
    sentiment_raw_dir = os.path.join(BASE, "raw", "sentiment")
    # get all the raw files under sentiment directory
    sentiment_raw_csv_files = []
    for dirs in os.listdir(sentiment_raw_dir):
        directory = os.path.join(sentiment_raw_dir, dirs)
        for file in os.listdir(directory):
            if file == "raw.csv": sentiment_raw_csv_files.append(os.path.join(directory, file))

    processed_dir_sentiment = os.path.join(BASE, "processed", "sentiment")
    os.makedirs(processed_dir_sentiment, exist_ok=True)

    print("Raw sentiment files:", sentiment_raw_csv_files)
    return {
        "sentiment_raw_files": sentiment_raw_csv_files,
        "processed_sentiment": processed_dir_sentiment,
    }

# ----- normalization helpers ------------------
def normalize_sentiment(value):
    """
    Normalize sentiment labels into:
        positive / negative / neutral
    Supports strings and numeric values.
    """
    if pd.isna(value):
        return None

    # numeric case
    if isinstance(value, (int, float)):
        if value > 0:
            return "positive"
        if value < 0:
            return "negative"
        return "neutral"

    s = str(value).strip().lower()

    mapping = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "1": "positive",
        "0": "neutral",
        "-1": "negative",
    }

    return mapping.get(s, None)


def standardize_fpb(fpb_raw_path, output_path):
    """
    Standardize Financial PhraseBank / Kaggle dataset into:
        data, datatype, sentiment
    """
    # The Kaggle CSV is typically headerless with:
    # sentiment, headline
    df = pd.read_csv(
        fpb_raw_path,
        header=None,
        names=["sentiment", "data"],
        encoding="latin1"
    )

    df["datatype"] = "sentence"
    df["sentiment"] = df["sentiment"].apply(normalize_sentiment)

    df = df[["data", "datatype", "sentiment"]].copy()
    df = df.dropna(subset=["data", "sentiment"])
    df["data"] = df["data"].astype(str).str.strip()
    df = df[df["data"] != ""]

    df.to_csv(output_path, index=False)
    return df


def standardize_fmb(fmb_raw_path, output_path):
    """
    Standardize baptle/financial_headlines_market_based into:
        data, datatype, sentiment
    """
    df = pd.read_csv(fmb_raw_path, encoding="latin1")

    # Expected useful columns:
    # Title -> headline
    # Global Sentiment -> label (-1, 0, 1)
    if "Title" not in df.columns:
        raise ValueError("Expected column 'Title' not found in FMB dataset.")
    if "Global Sentiment" not in df.columns:
        raise ValueError("Expected column 'Global Sentiment' not found in FMB dataset.")

    df = df.rename(columns={
        "Title": "data",
        "Global Sentiment": "sentiment"
    })

    df["datatype"] = "headline"
    df["sentiment"] = df["sentiment"].apply(normalize_sentiment)

    df = df[["data", "datatype", "sentiment"]].copy()
    df = df.dropna(subset=["data", "sentiment"])
    df["data"] = df["data"].astype(str).str.strip()
    df = df[df["data"] != ""]

    df.to_csv(output_path, index=False)
    return df

def stitch_datasets(dfs, output_path):
    combined = pd.concat(dfs, ignore_index=True)
    # optional deduplication
    combined = combined.drop_duplicates(subset=["data", "datatype", "sentiment"])
    combined.to_csv(output_path, index=False)
    return combined

def standardize_sentiment():
    paths = ensure_dirs()
    # creating file names for the standardized file names
    fpb_std_path = os.path.join(paths["processed_sentiment"], "fpb_standardized.csv")
    fmb_std_path = os.path.join(paths["processed_sentiment"], "fmb_standardized.csv")
    stitched_path = os.path.join(paths["processed_sentiment"], "stitched_sentiment.csv")

    # standardizing all the raw files
    standardized_dfs = []
    for sentiment_raw_file_path in paths["sentiment_raw_files"]:
        if re.search("FPB", sentiment_raw_file_path):
            fpb_df = standardize_fpb(sentiment_raw_file_path, fpb_std_path)
            standardized_dfs.append(fpb_df)
        elif re.search("FMB", sentiment_raw_file_path):
            fmb_df = standardize_fmb(sentiment_raw_file_path, fmb_std_path)
            standardized_dfs.append(fmb_df)
    combined_df = stitch_datasets(standardized_dfs, stitched_path)

    print("\nSaved files:")
    print(f" - {fpb_std_path}")
    print(f" - {fmb_std_path}")
    print(f" - {stitched_path}")
    print(f"\nCombined shape: {combined_df.shape}")

if __name__ == "__main__":
    print("Standardizing datasets...")
    standardize_sentiment()