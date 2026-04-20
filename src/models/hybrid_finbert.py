"""
Hybrid FinBERT + L&M feature classifier.

Changes from original:
  1. Data loading  — reads teammate's CSV (needs: data, sentiment, sentiment_list columns)
  2. FinancialDataset — appends 7-dim L&M feature vector to each sample
  3. HybridFinBERT   — concat L&M features onto CLS embedding before classification

Everything else (TrainingArguments, Trainer, metrics, reports) is unchanged.
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
    AutoModel,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PretrainedConfig,
)

MODEL_NAME = "ProsusAI/finbert"
SAVE_DIR = "./my_finbert_lm_model"
REPORT_DIR = "./report"
RESULTS_DIR = "./results"
MAX_LEN = 128

LM_CATEGORIES = [
    "positive", "negative", "uncertainty",
    "litigious", "strong_modal", "weak_modal", "constraining",
]

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


# Expects CSV with columns: data, sentiment, sentiment_list
# sentiment_list is a Python list stored as a string, e.g.:
#   "['cns', 'positive', 'negative', 'uncertainty', 'cns']"
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"data", "sentiment", "sentiment_list"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Teammate's CSV is missing columns: {missing}")

    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["data", "sentiment", "sentiment_list"])

    if isinstance(df["sentiment_list"].iloc[0], str):
        df["sentiment_list"] = df["sentiment_list"].apply(ast.literal_eval)

    return df

def extract_lm_features(sentiment_list: list) -> list:
    """
    Convert a per-token sentiment_list into a normalised 7-dim count vector.

    E.g. ['cns','positive','negative','cns','uncertainty']
      -> [0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.25]  (after dropping 'cns', normalizing)

    Returns a list of 7 floats in LM_CATEGORIES order.
    """
    counts = {cat: 0 for cat in LM_CATEGORIES}
    for tag in sentiment_list:
        if tag in counts:
            counts[tag] += 1

    total = sum(counts.values())
    return [counts[cat] / total if total > 0 else 0.0 for cat in LM_CATEGORIES]

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

        lm_features = torch.tensor(
            extract_lm_features(row["sentiment_list"]),
            dtype=torch.float,
        )

        return {
            "input_ids":      encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "lm_features":    lm_features,          # shape: (7,)
            "labels":         torch.tensor(label, dtype=torch.long),
        }

# Model
class HybridConfig(PretrainedConfig):
    model_type = "hybrid_finbert"

    def __init__(self, base_model: str = MODEL_NAME, num_labels: int = 3,
                 lm_dim: int = 7, dropout: float = 0.1, **kwargs):
        super().__init__(num_labels=num_labels, **kwargs)
        self.base_model = base_model
        self.lm_dim     = lm_dim
        self.dropout    = dropout


class HybridFinBERT(PreTrainedModel):
    """
    FinBERT CLS embedding (768-dim)
    + L&M feature vector (7-dim)
    → Linear(775, 3)
    """
    config_class = HybridConfig

    def __init__(self, config: HybridConfig):
        super().__init__(config)
        self.bert      = AutoModel.from_pretrained(config.base_model)
        self.dropout   = nn.Dropout(config.dropout)
        combined_dim   = self.bert.config.hidden_size + config.lm_dim
        self.classifier = nn.Linear(combined_dim, config.num_labels)

    def forward(self, input_ids, attention_mask, lm_features, labels=None):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        combined   = torch.cat([cls_output, lm_features], dim=-1)
        combined   = self.dropout(combined)
        logits     = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1":       f1_score(labels, predictions, average="weighted"),
    }

def save_training_report(trainer, save_path=REPORT_DIR):
    os.makedirs(save_path, exist_ok=True)
    history  = trainer.state.log_history
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
    plt.show()

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
    plt.show()

def main():
    # TODO: replace with actual path
    CSV_PATH = "../data/processed/sentiment/lm_labeled.csv"
    df = load_data(CSV_PATH)

    train_val, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["sentiment"],
    )
    train, val = train_test_split(
        train_val, test_size=0.2, random_state=42, stratify=train_val["sentiment"],
    )
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    for split in [train, val, test]:
        split["label"] = split["sentiment"].map(LABEL_MAP)

    tokenizer     = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = FinancialDataset(train, tokenizer)
    val_dataset   = FinancialDataset(val,   tokenizer)
    test_dataset  = FinancialDataset(test,  tokenizer)

    config = HybridConfig(base_model=MODEL_NAME, num_labels=3, lm_dim=7)
    model  = HybridFinBERT(config)

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
    
    state_files = glob.glob(f"{RESULTS_DIR}/checkpoint-*/trainer_state.json")
    if state_files:
        latest = max(state_files, key=os.path.getmtime)
        with open(latest) as f:
            trainer.state.log_history = json.load(f)["log_history"]

    save_training_report(trainer)
    save_confusion_matrix(trainer, test_dataset)

    test_results = trainer.evaluate(test_dataset)
    pd.DataFrame([test_results]).to_csv(f"{REPORT_DIR}/final_metrics.csv", index=False)

    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\nModel saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()