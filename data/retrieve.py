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
SUMMARY = os.path.join(RAW, "sentiment")
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


# ----- main ---------------------

if __name__ == "__main__":
    print("Downloading datasets...\n")

    requirements()
    download_fpb()

    print("\n\nDatasets successfully downloaded.")
