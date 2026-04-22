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
import unicodedata
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

# ----- file paths ---------------
BASE = os.path.join(os.path.dirname(__file__))

def ensure_dirs():
    # sentiment stuff
    sentiment_raw_dir = os.path.join(BASE, "raw", "sentiment")
    # get all the raw files under sentiment directory
    sentiment_raw_csv_files = []
    if os.path.exists(sentiment_raw_dir):
        for dirs in os.listdir(sentiment_raw_dir):
            directory = os.path.join(sentiment_raw_dir, dirs)
            if os.path.isdir(directory):
                for file in os.listdir(directory):
                    if file == "raw.csv": sentiment_raw_csv_files.append(os.path.join(directory, file))

    processed_dir_sentiment = os.path.join(BASE, "processed", "sentiment")
    os.makedirs(processed_dir_sentiment, exist_ok=True)

    # summary stuff
    summary_raw_dir = os.path.join(BASE, "raw", "summary")
    processed_dir_summary = os.path.join(BASE, "processed", "summary")
    os.makedirs(processed_dir_summary, exist_ok=True)

    return {
        "sentiment_raw_files": sentiment_raw_csv_files,
        "processed_sentiment": processed_dir_sentiment,
        "summary_raw_dir": summary_raw_dir,
        "processed_summary": processed_dir_summary,
    }

# helper functions
def clean_text(text):
    # nicode normalisation -> ASCII where possible
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # strip basic HTML tags / entities
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    # expand common English contractions
    contractions = {
        r"won\'t": "will not", r"can\'t": "cannot", r"n\'t": " not",
        r"\'re": " are",      r"\'s":  " is",       r"\'d":  " would",
        r"\'ll": " will",     r"\'ve": " have",      r"\'m":  " am",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return (str(text)
                .replace("\n", " ")
                .replace("\r", " ")
                .replace("?", "")
                .strip()
            )

# sentiment helpers
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


def standardize_sentiment(paths):
    # creating file names for the standardized file names
    fpb_std_path = os.path.join(paths["processed_sentiment"], "fpb_standardized.csv")
    fmb_std_path = os.path.join(paths["processed_sentiment"], "fmb_standardized.csv")
    stitched_path = os.path.join(paths["processed_sentiment"], "stitched_sentiment.csv")
    # standardizing all the raw files
    standardized_dfs = []
    print("Raw sentiment files:", paths["sentiment_raw_files"])
    for sentiment_raw_file_path in paths["sentiment_raw_files"]:
        if re.search("FPB", sentiment_raw_file_path):
            fpb_df = standardize_fpb(sentiment_raw_file_path, fpb_std_path)
            fpb_df["data"] = fpb_df["data"].apply(clean_text)
            fpb_df = fpb_df[fpb_df["data"] != ""]
            standardized_dfs.append(fpb_df)
        elif re.search("FMB", sentiment_raw_file_path):
            fmb_df = standardize_fmb(sentiment_raw_file_path, fmb_std_path)
            fmb_df["data"] = fmb_df["data"].apply(clean_text)
            fmb_df = fmb_df[fmb_df["data"] != ""]
            standardized_dfs.append(fmb_df)
    if standardized_dfs:
        combined_df = stitch_datasets(standardized_dfs, stitched_path)
        print("\nSaved files:")
        print(f" - {fpb_std_path}")
        print(f" - {fmb_std_path}")
        print(f" - {stitched_path}")
        print(f"\nCombined shape: {combined_df.shape} with columns [data, datatype (sentence | headline), sentiment (positive | negative | neutral)]")
    else:
        print("No sentiment data found to process.")



def process_summary_dataset(paths):
    # dataset 1: news summary
    def process_ns():
        processed_ns = os.path.join(paths["processed_summary"], "NS")
        os.makedirs(processed_ns, exist_ok=True)
        raw_ns = os.path.join(paths["summary_raw_dir"], "NS")
        src = os.path.join(raw_ns, "full.csv")
        dst = os.path.join(processed_ns, "full.csv")
        if not os.path.exists(src):
            print(f"File not found: {src}")
            return
        df = pd.read_csv(src, encoding="latin-1")
        df = df.rename(
            columns={
                "ctext": "article",
                "text": "summary"
            }
        )
        df = df[["article", "summary"]]
        # cleaning
        def clean(text):
            return (
                str(text)
                .replace("\n", " ")
                .replace("\r", " ")
                .replace("?", "")
                .strip()
            )
        df["article"] = df["article"].apply(clean)
        df["summary"] = df["summary"].apply(clean)
        # save processed data
        df.to_csv(dst, index=False)
        print(f"Saved processed NS to {dst}")

    # dataset 2: CNN-DailyMail News Text Summarisation
    def process_cnn_dm():
        processed_cnn_dm = os.path.join(paths["processed_summary"], "CNN_DM")
        os.makedirs(processed_cnn_dm, exist_ok=True)
        raw_cnn_dm = os.path.join(paths["summary_raw_dir"], "CNN_DM")
        for split in ["test", "train", "validation"]:
            src = os.path.join(raw_cnn_dm, f"{split}.csv")
            dst = os.path.join(processed_cnn_dm, f"{split}.csv")
            if not os.path.exists(src):
                print(f"File not found: {src}")
                continue
            df = pd.read_csv(src)
            df = df.rename(
                columns={
                    "highlights": "summary"
                }
            )
            df = df[["article", "summary"]]
            # cleaning
            df["article"] = df["article"].str.strip()
            df["summary"] = df["summary"].str.strip()
            # save processed data
            df.to_csv(dst, index=False)
            print(f"Saved processed CNN_DM {split} to {dst}")

    process_ns()
    process_cnn_dm()

# main
if __name__ == "__main__":
    print("Processing datasets...")
    paths = ensure_dirs()
    print("\n\nStandardizing sentiment datasets...")
    standardize_sentiment(paths)
    print("\n\nProcessing news summary datasets...")
    process_summary_dataset(paths)
    print("\n\nDatasets processed successfully.")
