
from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
)

from src.pipeline.lexicon import build_lexicon_runnable
from src.pipeline.reliability import build_reliability_runnable
from src.pipeline.sentiment import build_finbert_runnable
from src.pipeline.summarizer import build_summarizer_runnable


def _flatten(payload: dict) -> dict:
    """Collapse the RunnableParallel output into the flat state expected by
    :func:`src.pipeline.reliability._assess`."""
    return {
        "summary": payload["summary"]["summary"],
        "finbert": payload["finbert"],
        "lexicon": payload["lex"]["lexicon"],
        "bias": payload["lex"]["bias"],
    }


def build_pipeline(
    *,
    finbert_dir: Optional[str | Path] = None,
    bart_dir: Optional[str | Path] = None,
    lexicon_path: Optional[str | Path] = None,
    device: Optional[str] = None,
    apply_neutral_prompt: bool = True,
) -> Runnable:
    """Construct the full reliability pipeline.

    Input  : ``{"article": str}``
    Output : :class:`src.pipeline.schema.PipelineOutput`

    Heavy resources (BART, FinBERT, L&M dict) are loaded once at build time and
    reused across every ``invoke`` call.
    """
    summarizer = build_summarizer_runnable(
        model_dir=bart_dir,
        device=device,
        apply_neutral_prompt=apply_neutral_prompt,
    )
    finbert = build_finbert_runnable(model_dir=finbert_dir, device=device)
    lexicon = build_lexicon_runnable(lexicon_path=lexicon_path)

    fanout = RunnableParallel(
        summary=summarizer,
        finbert=finbert,
        lex=lexicon,
    )

    return fanout | RunnableLambda(_flatten, name="flatten") | build_reliability_runnable()
