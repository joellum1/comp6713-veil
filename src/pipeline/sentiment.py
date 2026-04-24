
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from langchain_core.runnables import RunnableLambda
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.finbert import LABEL_DECODE, MAX_LEN
from src.pipeline._checkpoint import resolve_local_or_hub
from src.pipeline.schema import SentimentSignal

REPO_ROOT = Path(__file__).resolve().parents[2]

# Local fine-tuned FinBERT checkpoints. First existing entry wins; only one
# slot today, list-shaped so adding a sentiment_model_v2 later is a one-liner.
CANDIDATE_LOCAL_DIRS = (
    REPO_ROOT / "results" / "finbert_model",
)
FALLBACK_HF_ID = "ProsusAI/finbert"


def _resolve_source(model_dir: Optional[str | Path]) -> str:
    return resolve_local_or_hub(model_dir, CANDIDATE_LOCAL_DIRS, FALLBACK_HF_ID)


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_finbert_runnable(
    *,
    model_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> RunnableLambda:
    """Return a Runnable mapping ``{'article': str}`` -> :class:`SentimentSignal`."""
    source = _resolve_source(model_dir)
    resolved_device = _resolve_device(device)

    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.to(resolved_device).eval()

    def _classify(payload: dict) -> SentimentSignal:
        encoding = tokenizer(
            payload["article"],
            return_tensors="pt",
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
        ).to(resolved_device)
        with torch.no_grad():
            logits = model(**encoding).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        return SentimentSignal(
            label=LABEL_DECODE[idx],
            confidence=round(float(probs[idx].item()), 4),
        )

    return RunnableLambda(_classify, name="finbert_classify")
