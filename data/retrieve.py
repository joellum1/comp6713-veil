"""
Downloads all raw datasets required for the project.

Run this script from the project root directory:
    `python data/retrieve.py`
"""

import os
import subprocess
import sys

# ----- file paths ---------------

BASE = os.path.join(os.path.dirname(__file__))

RAW = os.path.join(BASE, "raw")
os.makedirs(RAW, exist_ok=True)

SENTIMENT = os.path.join(RAW, "sentiment")
os.makedirs(SENTIMENT, exist_ok=True)
SUMMARY = os.path.join(RAW, "summary")
os.makedirs(SUMMARY, exist_ok=True)


# ----- helpers ------------------

def requirements():
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Kaggle CLI not found.")
        sys.exit(1)

def kaggle(dataset, dest):
    print(f"\n[Kaggle] Downloading {dataset} → {dest}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", dest, "--unzip"],
        check=True
    )
    print(f"[Kaggle] Done: {dataset}")


# --- news sentiment
# dataset 1: financial phrasebank
def download_fpb():
    FPB = os.path.join(SENTIMENT, "FPB")
    os.makedirs(FPB, exist_ok=True)

    kaggle("ankurzing/sentiment-analysis-for-financial-news", FPB)

    # standardised name
    src = os.path.join(FPB, "all-data.csv")
    dst = os.path.join(FPB, "raw.csv")
    if os.path.exists(src):
        os.rename(src, dst)


# --- news summary
# dataset 1: news summary
def download_ns():
    NS = os.path.join(SUMMARY, "NS")
    os.makedirs(NS, exist_ok=True)

    kaggle("sunnysai12345/news-summary", NS)

    # standardised name
    src = os.path.join(NS, "news_summary.csv")
    dst = os.path.join(NS, "raw.csv")
    if os.path.exists(src):
        os.rename(src, dst)

# dataset 2: CNN-DailyMail News Text Summarisation
def download_CNN_DM():
    CNN_DM = os.path.join(SUMMARY, "CNN_DM")
    os.makedirs(CNN_DM, exist_ok=True)

    kaggle("gowrishankarp/newspaper-text-summarization-cnn-dailymail", CNN_DM)


# ----- main ---------------------

if __name__ == "__main__":
    print("Downloading datasets...")

    requirements()

    print("\n\nRetrieving news sentiment datasets...")
    download_fpb()

    print("\n\nRetrieving news summary datasets...")
    download_ns()
    download_CNN_DM()

    print("\n\nDatasets successfully downloaded.")
