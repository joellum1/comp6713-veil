"""
Sentiment Analysis + Bias Detection for Financial News

Architecture:
  - FinBERT  : sentiment classification (positive / negative / neutral)
  - L&M      : bias/reliability scoring (uncertainty, litigious, constraining)

These two are fully independent.
The final output gives investors:
  1. Market sentiment   — what direction is this news pointing?
  2. Bias score         — how much should they trust it?
"""

import os
import ast
import json
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

REPO_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MODEL_NAME  = "ProsusAI/finbert"
SAVE_DIR    = os.path.join(REPO_ROOT, "results", "finbert_model")
REPORT_DIR  = os.path.join(REPO_ROOT, "results", "report")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "checkpoints")
MAX_LEN     = 128

LABEL_MAP    = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_DECODE = {0: "negative", 1: "neutral", 2: "positive"}

# L&M categories used for bias detection
# - uncertainty  : "may", "possible", "pending" — hedging language
# - litigious    : "litigation", "regulatory", "claim" — legal risk language  
# - constraining : "restrict", "limit", "requirement" — barrier language
BIAS_CATEGORIES = ["uncertainty", "litigious", "constraining"]

# threshold: if bias_score exceeds this, flag the news as potentially biased
BIAS_THRESHOLD = 0.15

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load omar's CSV.
    Required columns: data, sentiment, sentiment_list
    """
    df = pd.read_csv(csv_path)

    required = {"data", "sentiment", "sentiment_list"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["data", "sentiment", "sentiment_list"])

    if isinstance(df["sentiment_list"].iloc[0], str):
        df["sentiment_list"] = df["sentiment_list"].apply(ast.literal_eval)

    return df

def compute_bias_score(sentiment_list: list) -> dict:
    """
    Compute a bias/reliability score from L&M per-token tags.

    Uses only uncertainty, litigious, and constraining categories because:
    - uncertainty  : hedging language : investor can't rely on this news
    - litigious    : legal risk signals : situation is contested
    - constraining : barrier language : outcome is restricted/limited

    Returns:
        {
            "uncertainty":  float,   # proportion of uncertainty tokens
            "litigious":    float,   # proportion of litigious tokens
            "constraining": float,   # proportion of constraining tokens
            "bias_score":   float,   # combined score (mean of above)
            "bias_flag":    bool     # True if bias_score > BIAS_THRESHOLD
        }
    """
    total_tokens = len([t for t in sentiment_list if t != "cns"])

    scores = {}
    for cat in BIAS_CATEGORIES:
        count = sentiment_list.count(cat)
        scores[cat] = count / total_tokens if total_tokens > 0 else 0.0

    bias_score = sum(scores.values()) / len(BIAS_CATEGORIES)
    scores["bias_score"] = bias_score
    scores["bias_flag"]  = bias_score > BIAS_THRESHOLD

    return scores

class FinancialDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        self.data      = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row   = self.data.iloc[idx]
        text  = str(row["data"])
        label = int(row["label"])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels":         torch.tensor(label, dtype=torch.long),
        }
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1":       f1_score(labels, predictions, average="weighted"),
    }

def analyze(text: str, sentiment_list: list, model, tokenizer, device="cpu") -> dict:
    """
    Full analysis for a single news article.

    Args:
        text           : raw news text
        sentiment_list : L&M per-token tags from pipeline
        model          : fine-tuned FinBERT
        tokenizer      : FinBERT tokenizer

    Returns:
        {
            "sentiment":    "positive" | "negative" | "neutral",
            "confidence":   float,        # FinBERT softmax max prob
            "uncertainty":  float,        # L&M uncertainty ratio
            "litigious":    float,        # L&M litigious ratio
            "constraining": float,        # L&M constraining ratio
            "bias_score":   float,        # combined bias score
            "bias_flag":    bool          # True = potentially biased
        }
    """
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits
        probs  = torch.softmax(logits, dim=-1).squeeze()

    sentiment_idx = torch.argmax(probs).item()
    confidence    = probs[sentiment_idx].item()
    sentiment     = LABEL_DECODE[sentiment_idx]

    # L&M bias scoring
    bias = compute_bias_score(sentiment_list)

    return {
        "sentiment":    sentiment,
        "confidence":   round(confidence, 4),
        "uncertainty":  round(bias["uncertainty"], 4),
        "litigious":    round(bias["litigious"], 4),
        "constraining": round(bias["constraining"], 4),
        "bias_score":   round(bias["bias_score"], 4),
        "bias_flag":    bias["bias_flag"],
    }

def save_training_report(trainer, save_path=REPORT_DIR):
    os.makedirs(save_path, exist_ok=True)
    history    = trainer.state.log_history
    train_loss = [x["loss"]          for x in history if "loss"          in x]
    eval_loss  = [x["eval_loss"]     for x in history if "eval_loss"     in x]
    eval_acc   = [x["eval_accuracy"] for x in history if "eval_accuracy" in x]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(
        np.linspace(0, len(train_loss) - 1, len(eval_loss)),
        eval_loss, label="Val Loss", color="#ff7f0e", lw=2,
    )
    plt.title("Model Loss Trajectory", fontsize=14)
    plt.xlabel("Steps"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(eval_acc, label="Val Accuracy", color="#2ca02c", marker="o")
    plt.title("Validation Accuracy per Epoch", fontsize=14)
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/learning_curve.png", dpi=300)
    plt.close()


def save_confusion_matrix(trainer, test_dataset, save_path=REPORT_DIR):
    os.makedirs(save_path, exist_ok=True)
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    cm     = confusion_matrix(y_true, y_pred)
    labels = ["Negative", "Neutral", "Positive"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        cmap=plt.cm.Blues, ax=ax,
    )
    plt.title("Sentiment Analysis Confusion Matrix")
    plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300)
    plt.close()

def main(sample_ratio: float = 1.0):
    """
    sample_ratio: fraction of data to use (0.0 < sample_ratio <= 1.0)
        e.g. 0.01 = 1%  for a quick test run
             1.0  = full dataset for real training
    """
    CSV_PATH = os.path.join(
        REPO_ROOT, "data", "processed", "sentiment", "sentiment_list.csv"
    )
    df = load_data(CSV_PATH)

    # sample if requested — stratified by sentiment to keep class balance
    if sample_ratio < 1.0:
        df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        print(f"Using {sample_ratio*100:.1f}% of data → {len(df)} rows")

    # split
    train_val, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["sentiment"],
    )
    train, val = train_test_split(
        train_val, test_size=0.2, random_state=42, stratify=train_val["sentiment"],
    )
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    for split in [train, val, test]:
        split["label"] = split["sentiment"].map(LABEL_MAP)

    # tokenizer & datasets
    tokenizer     = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = FinancialDataset(train, tokenizer)
    val_dataset   = FinancialDataset(val,   tokenizer)
    test_dataset  = FinancialDataset(test,  tokenizer)

    # FinBERT — sentiment only, no L&M concat
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    )

    # training
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # reload best checkpoint log
    state_files = glob.glob(f"{RESULTS_DIR}/checkpoint-*/trainer_state.json")
    if state_files:
        latest = max(state_files, key=os.path.getmtime)
        with open(latest) as f:
            trainer.state.log_history = json.load(f)["log_history"]

    # reports
    save_training_report(trainer)
    save_confusion_matrix(trainer, test_dataset)

    test_results = trainer.evaluate(test_dataset)
    pd.DataFrame([test_results]).to_csv(f"{REPORT_DIR}/final_metrics.csv", index=False)

    # save model
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\nModel saved to {SAVE_DIR}")

    # ── bias scoring on full test set ────────────────────────────────────────
    print("\nComputing sentiment + bias scores on test set...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    records = []
    for col, row in test.iterrows():
        result = analyze(
            row["data"],
            row["sentiment_list"],
            model,
            tokenizer,
            device=device,
        )
        result["true_sentiment"] = row["sentiment"]
        records.append(result)

    results_df = pd.DataFrame(records)
    os.makedirs(REPORT_DIR, exist_ok=True)
    results_df.to_csv(f"{REPORT_DIR}/test_results_with_bias.csv", index=False)

    print("\n--- Sample Output (first 5 rows) ---")
    print(results_df[[
        "true_sentiment", "sentiment", "confidence",
        "bias_score", "bias_flag"
    ]].head(10).to_string(index=False))

    # bias flag summary
    flagged = results_df["bias_flag"].sum()
    total   = len(results_df)
    print(f"\nBias flagged: {flagged}/{total} ({flagged/total*100:.1f}%) of test articles")
    print(f"Results saved to {REPORT_DIR}/test_results_with_bias.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample", type=float, default=1.0,
        help="Fraction of data to use (e.g. 0.01 = 1%%). Default: 1.0 (full dataset)"
    )
    args = parser.parse_args()
    main(sample_ratio=args.sample)