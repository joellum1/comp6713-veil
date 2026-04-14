"""
Downloads raw sentiment and summary datasets.

Run this script from the project root directory:
    python data/retrieve.py
"""

import os
import subprocess
import sys
import shutil
import datasets
import pandas as pd

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(call_pdb=False)


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


def flatten_folder(base):
    # search for subdirectories in the base directory
    subdirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]

    for sd in subdirs:
        nested = os.path.join(base, sd)

        # move contents in subdirectory to base directory
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(base, item)

            shutil.move(src, dst)

        # remove subdirectory
        os.rmdir(nested)


def ensure_dirs():
    base = os.path.join(os.path.dirname(__file__))
    raw_dir = os.path.join(base, "raw")
    sentiment_dir = os.path.join(raw_dir, "sentiment")
    summary_dir = os.path.join(raw_dir, "summary")

    fpb_dir = os.path.join(sentiment_dir, "FPB")
    fmb_dir = os.path.join(sentiment_dir, "FMB")
    ns_dir = os.path.join(summary_dir, "NS")
    cnn_dm_dir = os.path.join(summary_dir, "CNN_DM")

    for d in [fpb_dir, fmb_dir, ns_dir, cnn_dm_dir]:
        os.makedirs(d, exist_ok=True)

    return {
        "fpb_raw": fpb_dir,
        "fmb_raw": fmb_dir,
        "ns_raw": ns_dir,
        "cnn_dm_raw": cnn_dm_dir,
    }


# ----- main workflow ---------------------
def get_sentiment_dataset(paths):
    # ---------- FPB / Kaggle ----------
    kaggle("ankurzing/sentiment-analysis-for-financial-news", paths["fpb_raw"])

    flatten_folder(paths["fpb_raw"])

    fpb_src = os.path.join(paths["fpb_raw"], "all-data.csv")
    fpb_raw = os.path.join(paths["fpb_raw"], "raw.csv")
    if os.path.exists(fpb_src):
        os.replace(fpb_src, fpb_raw)

    # ---------- FMB / HuggingFace ----------
    fmb_raw = os.path.join(paths["fmb_raw"], "raw.csv")
    fmb_ds = datasets.load_dataset("baptle/financial_headlines_market_based")
    fmb_ds["train"].to_csv(fmb_raw, index=False)


def get_summary_dataset(paths):
    # ---------- NS / Kaggle ----------
    kaggle("sunnysai12345/news-summary", paths["ns_raw"])

    # standardised name
    src = os.path.join(paths["ns_raw"], "news_summary.csv")
    dst = os.path.join(paths["ns_raw"], "full.csv")
    if os.path.exists(src):
        os.rename(src, dst)

    # ---------- CNN_DM / Kaggle ----------
    kaggle("gowrishankarp/newspaper-text-summarization-cnn-dailymail", paths["cnn_dm_raw"])

    flatten_folder(paths["cnn_dm_raw"])


if __name__ == "__main__":
    print("Downloading datasets...\n")
    
    # pre-req stuff
    requirements()
    paths = ensure_dirs()

    # sentiment dataset
    print("\nRetrieving news sentiment datasets...")
    get_sentiment_dataset(paths)

    # summary dataset
    print("\nRetrieving news summary datasets...")
    get_summary_dataset(paths)

    print("\nDatasets successfully downloaded.")
