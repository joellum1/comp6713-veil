
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from transformers import BartForConditionalGeneration, BartTokenizer

from src.pipeline._checkpoint import pick_best_bart

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO_ROOT / "results" / "report"

CANDIDATE_LOCAL_DIRS = (
    REPO_ROOT / "results" / "bart-final-cnn_dm",
    REPO_ROOT / "results" / "bart-final-ns",
    REPO_ROOT / "results" / "bart_summariser",
)
FALLBACK_HF_ID = "facebook/bart-large-cnn"

NEUTRAL_PROMPT = PromptTemplate.from_template(
    "Provide a concise, factual summary. Avoid speculative language. "
    "Separate facts from opinions. Do not editorialise.\n\n"
    "Article:\n{article}"
)


def _resolve_source(model_dir: Optional[str | Path]) -> str:
    return pick_best_bart(
        model_dir,
        CANDIDATE_LOCAL_DIRS,
        FALLBACK_HF_ID,
        METRICS_DIR,
    )


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_summarizer_runnable(
    *,
    model_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
    max_input_length: int = 512,
    max_target_length: int = 128,
    min_length: int = 32,
    num_beams: int = 3,
    apply_neutral_prompt: bool = True,
) -> RunnableLambda:
    """Return a Runnable mapping ``{'article': str}`` -> ``{'summary': str}``."""
    source = _resolve_source(model_dir)
    resolved_device = _resolve_device(device)

    tokenizer = BartTokenizer.from_pretrained(source)
    model = BartForConditionalGeneration.from_pretrained(source)
    model.to(resolved_device).eval()

    model.generation_config.max_length = None
    model.generation_config.early_stopping = True
    model.generation_config.num_beams = num_beams

    def _summarize(payload: dict) -> dict:
        article = payload["article"]
        text_in = (
            NEUTRAL_PROMPT.format(article=article)
            if apply_neutral_prompt
            else article
        )
        inputs = tokenizer(
            text_in,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
        ).to(resolved_device)
        with torch.no_grad():
            ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_target_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        summary = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        return {"summary": summary}

    return RunnableLambda(_summarize, name="bart_summarize")
