"""
Downloads, standardizes, and stitches raw sentiment datasets.

Run this script from the project root directory:
    python data/retrieve.py
"""

import os
import subprocess
import sys
import datasets
import pandas as pd

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

# ----- file paths ---------------
BASE = os.path.join(os.path.dirname(__file__))


# ----- helpers ------------------
def authenticate_kaggle():
    """Checks for Kaggle credentials and provides setup instructions if missing."""
    has_config = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    has_env = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ

    if not (has_config or has_env):
        print("\n" + "=" * 50)
        print("[ERROR] Kaggle authentication missing!")
        print("To fix this, you can:")
        print("1. Place your 'kaggle.json' file in ~/.kaggle/")
        print("2. Set environment variables in your terminal:")
        print("   export KAGGLE_USERNAME=your_username")
        print("   export KAGGLE_KEY=your_api_key")
        print("3. (Not recommended) Hardcode them at the top of this script:")
        print("   os.environ['KAGGLE_USERNAME'] = '...'")
        print("   os.environ['KAGGLE_KEY'] = '...'")
        print("=" * 50 + "\n")
        sys.exit(1)


def requirements():
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Kaggle CLI not found. Please install it with 'pip install kaggle'.")
        sys.exit(1)
    authenticate_kaggle()


def kaggle(dataset, dest):
    print(f"\n[Kaggle] Downloading {dataset} -> {dest}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", dest, "--unzip"],
        check=True
    )
    print(f"[Kaggle] Done: {dataset}")


def ensure_dirs():
    raw_dir = os.path.join(BASE, "raw")
    raw_dir_sentiment = os.path.join(raw_dir, "sentiment")
    processed_dir = os.path.join(BASE, "processed")
    processed_dir_sentiment = os.path.join(processed_dir, "sentiment")

    fpb_dir = os.path.join(raw_dir_sentiment, "FPB")
    fmb_dir = os.path.join(raw_dir_sentiment, "FMB")


    os.makedirs(fpb_dir, exist_ok=True)
    os.makedirs(fmb_dir, exist_ok=True)
    os.makedirs(processed_dir_sentiment, exist_ok=True)

    return {
        "fpb_raw": fpb_dir,
        "fmb_raw": fmb_dir,
        "processed_sentiment": processed_dir_sentiment,
        "raw_sentiment": raw_dir_sentiment,
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


def standardize_fmb(output_path):
    """
    Standardize baptle/financial_headlines_market_based into:
        data, datatype, sentiment
    """
    ds = datasets.load_dataset("baptle/financial_headlines_market_based")
    df = ds["train"].to_pandas()

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


# ----- main workflow ---------------------
def get_sentiment_dataset():
    paths = ensure_dirs()

    # ---------- FPB / Kaggle ----------
    kaggle("ankurzing/sentiment-analysis-for-financial-news", paths["fpb_raw"])

    fpb_src = os.path.join(paths["fpb_raw"], "all-data.csv")
    fpb_raw = os.path.join(paths["fpb_raw"], "raw.csv")
    if os.path.exists(fpb_src):
        os.replace(fpb_src, fpb_raw)
    elif not os.path.exists(fpb_raw):
        raise FileNotFoundError("Could not find FPB raw file after Kaggle download.")

    # ---------- FMB / HuggingFace ----------
    # Save raw copy too, if you still want it
    fmb_raw = os.path.join(paths["fmb_raw"], "raw.csv")
    fmb_ds = datasets.load_dataset("baptle/financial_headlines_market_based")
    fmb_ds["train"].to_csv(fmb_raw, index=False)

    # ---------- Standardize ----------
    fpb_std_path = os.path.join(paths["processed_sentiment"], "fpb_standardized.csv")
    fmb_std_path = os.path.join(paths["processed_sentiment"], "fmb_standardized.csv")
    stitched_path = os.path.join(paths["processed_sentiment"], "stitched_sentiment.csv")

    fpb_df = standardize_fpb(fpb_raw, fpb_std_path)
    fmb_df = standardize_fmb(fmb_std_path)
    combined_df = stitch_datasets([fpb_df, fmb_df], stitched_path)

    print("\nSaved files:")
    print(f" - {fpb_std_path}")
    print(f" - {fmb_std_path}")
    print(f" - {stitched_path}")
    print(f"\nCombined shape: {combined_df.shape}")


if __name__ == "__main__":
    print("Downloading and standardizing datasets...\n")
    requirements()
    get_sentiment_dataset()
    print("\nDatasets successfully downloaded, standardized, and stitched.")