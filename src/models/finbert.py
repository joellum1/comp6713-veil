"""
Sentiment Analysis & Bias Detection Pipeline for Financial News
- Model: FinBERT (Sequence Classification)
- Bias Scoring: Loughran-McDonald (L&M) Dictionary based heuristics
- Strategy: Full Fine-tuning with Class Weights for Imbalanced Data
"""

import os
import ast
import json
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# Constants & Configuration
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MODEL_NAME = "ProsusAI/finbert"
SAVE_DIR = os.path.join(REPO_ROOT, "results", "finbert_model")
REPORT_DIR = os.path.join(REPO_ROOT, "results", "report")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "checkpoints")
MAX_LEN = 128

LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2}
LABEL_DECODE = {0: "positive", 1: "negative", 2: "neutral"}

BIAS_CATEGORIES = ["uncertainty", "litigious", "constraining"]
BIAS_THRESHOLD = 0.15

# Utilities
def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate the dataset."""
    df = pd.read_csv(csv_path)

    required_cols = {"data", "sentiment", "sentiment_list"}
    if missing := required_cols - set(df.columns):
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["data", "sentiment", "sentiment_list"])

    if isinstance(df["sentiment_list"].iloc[0], str):
        df["sentiment_list"] = df["sentiment_list"].apply(ast.literal_eval)

    return df

def compute_bias_score(sentiment_list: list) -> dict:
    """Calculate bias score based on L&M dictionary categories."""
    total_tokens = len([t for t in sentiment_list if t != "cns"])
    scores = {}
    
    for cat in BIAS_CATEGORIES:
        count = sentiment_list.count(cat)
        scores[cat] = count / total_tokens if total_tokens > 0 else 0.0

    bias_score = sum(scores.values()) / len(BIAS_CATEGORIES)
    scores["bias_score"] = bias_score
    scores["bias_flag"] = bias_score > BIAS_THRESHOLD

    return scores

# Dataset & Trainer
class FinancialDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row["data"])
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
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

class WeightedTrainer(Trainer):
    """Custom Trainer to support class weights for imbalanced datasets."""
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

# Analysis & Reporting
def analyze(text: str, sentiment_list: list, model, tokenizer, device="cpu") -> dict:
    """Analyze a single text for sentiment and bias."""
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
        probs = torch.softmax(logits, dim=-1).squeeze()

    sentiment_idx = torch.argmax(probs).item()
    confidence = probs[sentiment_idx].item()
    
    bias = compute_bias_score(sentiment_list)

    return {
        "sentiment": LABEL_DECODE[sentiment_idx],
        "confidence": round(confidence, 4),
        "uncertainty": round(bias["uncertainty"], 4),
        "litigious": round(bias["litigious"], 4),
        "constraining": round(bias["constraining"], 4),
        "bias_score": round(bias["bias_score"], 4),
        "bias_flag": bias["bias_flag"],
    }

def save_training_report(trainer, save_path=REPORT_DIR):
    os.makedirs(save_path, exist_ok=True)
    history = trainer.state.log_history
    
    train_loss = [x["loss"] for x in history if "loss" in x]
    eval_loss = [x["eval_loss"] for x in history if "eval_loss" in x]
    eval_acc = [x["eval_accuracy"] for x in history if "eval_accuracy" in x]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(np.linspace(0, len(train_loss) - 1, len(eval_loss)), eval_loss, label="Val Loss", color="#ff7f0e", lw=2)
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
    
    cm = confusion_matrix(predictions.label_ids, y_pred)
    labels = ["Positive", "Negative", "Neutral"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Sentiment Analysis Confusion Matrix")
    plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300)
    plt.close()

# Main
def main(sample_ratio: float = 1.0):
    set_seed(42)

    csv_path = os.path.join(REPO_ROOT, "data", "processed", "sentiment", "sentiment_list.csv")
    df = load_data(csv_path)

    if sample_ratio < 1.0:
        df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        print(f"Using {sample_ratio*100:.1f}% of data → {len(df)} rows")

    # Data splitting
    train_val, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])
    train, val = train_test_split(train_val, test_size=0.2, random_state=42, stratify=train_val["sentiment"])
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    for split in [train, val, test]:
        split["label"] = split["sentiment"].map(LABEL_MAP)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = FinancialDataset(train, tokenizer)
    val_dataset = FinancialDataset(val, tokenizer)
    test_dataset = FinancialDataset(test, tokenizer)

    # Model initialization
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    for param in model.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total:,} | Trainable: {trainable:,} ({trainable/total*100:.1f}%)\n")

    # Class weights calculation
    labels_array = train["label"].values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_array),
        y=labels_array,
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print(f"Class weights: {dict(zip(LABEL_DECODE.values(), class_weights.round(3)))}\n")

    # Training configuration
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-6,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.05,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        logging_steps=10,
    )

    trainer = WeightedTrainer(
        class_weights=weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Load best checkpoint logs and save reports
    state_files = glob.glob(f"{RESULTS_DIR}/checkpoint-*/trainer_state.json")
    if state_files:
        latest = max(state_files, key=os.path.getmtime)
        with open(latest) as f:
            trainer.state.log_history = json.load(f)["log_history"]

    save_training_report(trainer)
    save_confusion_matrix(trainer, test_dataset)

    test_results = trainer.evaluate(test_dataset)
    pd.DataFrame([test_results]).to_csv(f"{REPORT_DIR}/final_metrics.csv", index=False)

    # Save finalized model
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\nModel saved to {SAVE_DIR}")

    # Final test evaluation
    print("\nComputing sentiment + bias scores on test set...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    records = []
    for _, row in test.iterrows():
        result = analyze(row["data"], row["sentiment_list"], model, tokenizer, device=device)
        result["true_sentiment"] = row["sentiment"]
        records.append(result)

    results_df = pd.DataFrame(records)
    os.makedirs(REPORT_DIR, exist_ok=True)
    results_df.to_csv(f"{REPORT_DIR}/test_results_with_bias.csv", index=False)

    flagged = results_df["bias_flag"].sum()
    total = len(results_df)
    print(f"\nBias flagged: {flagged}/{total} ({flagged/total*100:.1f}%) of test articles")
    print(f"Results saved to {REPORT_DIR}/test_results_with_bias.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of data to use (0.0 to 1.0)")
    args = parser.parse_args()
    main(sample_ratio=args.sample)