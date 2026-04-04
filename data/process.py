"""
Processes all raw datasets required for the project.

Run this script from the project root directory:
    `python data/process.py`
"""

import os

# ----- file paths ---------------

BASE = os.path.join(os.path.dirname(__file__))

PROCESSED = os.path.join(BASE, "processed")
os.makedirs(PROCESSED, exist_ok=True)

SENTIMENT = os.path.join(PROCESSED, "sentiment")
os.makedirs(SENTIMENT, exist_ok=True)
SUMMARY = os.path.join(PROCESSED, "sentiment")
os.makedirs(SUMMARY, exist_ok=True)


# ----- helpers ------------------

def requirements():
    pass


# ----- main ---------------------

if __name__ == "__main__":
    print("Processing datasets...\n")

    requirements()

    print("\n\nDatasets processed successfully.")