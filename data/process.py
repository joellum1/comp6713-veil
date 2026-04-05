"""
Processes all raw datasets required for the project.

Run this script from the project root directory:
    `python data/process.py`
"""

import os
import pandas as pd

# ----- file paths ---------------

BASE = os.path.join(os.path.dirname(__file__))

RAW = os.path.join(BASE, "raw")
RAW_SENTIMENT = os.path.join(RAW, "sentiment")
RAW_SUMMARY = os.path.join(RAW, "summary")

PROCESSED = os.path.join(BASE, "processed")
os.makedirs(PROCESSED, exist_ok=True)
PROCESSED_SENTIMENT = os.path.join(PROCESSED, "sentiment")
os.makedirs(PROCESSED_SENTIMENT, exist_ok=True)
PROCESSED_SUMMARY = os.path.join(PROCESSED, "summary")
os.makedirs(PROCESSED_SUMMARY, exist_ok=True)


# ----- helpers ------------------

def requirements():
    pass


# --- news summary
# dataset 1: news summary
def process_NS():
    PROCESSED_NS = os.path.join(PROCESSED_SUMMARY, "NS")
    os.makedirs(PROCESSED_NS, exist_ok=True)

    RAW_NS = os.path.join(RAW_SUMMARY, "NS")
    src = os.path.join(RAW_NS, "full.csv")
    dst = os.path.join(PROCESSED_NS, "full.csv")

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

# dataset 2: CNN-DailyMail News Text Summarisation
def process_CNN_DM():
    PROCESSED_CNN_DM = os.path.join(PROCESSED_SUMMARY, "CNN_DM")
    os.makedirs(PROCESSED_CNN_DM, exist_ok=True)

    RAW_CNN_DM = os.path.join(RAW_SUMMARY, "CNN_DM")
    for split in ["test", "train", "validation"]:
        src = os.path.join(RAW_CNN_DM, f"{split}.csv")
        dst = os.path.join(PROCESSED_CNN_DM, f"{split}.csv")

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


# ----- main ---------------------

if __name__ == "__main__":
    print("Processing datasets...")

    requirements()

    print("\n\nProcessing news sentiment datasets...")

    print("\n\nProcessing news summary datasets...")
    process_NS()
    process_CNN_DM()

    print("\n\nDatasets processed successfully.")