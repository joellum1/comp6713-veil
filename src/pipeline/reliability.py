
from __future__ import annotations

from typing import List

from langchain_core.runnables import RunnableLambda

from src.models.finbert import BIAS_THRESHOLD
from src.pipeline.schema import (
    BiasSignal,
    LexiconSignal,
    PipelineOutput,
    SentimentSignal,
)

LOW_CONF_THRESHOLD = 0.55


def _assess(state: dict) -> PipelineOutput:
    summary: str = state["summary"]
    finbert: SentimentSignal = state["finbert"]
    lexicon: LexiconSignal = state["lexicon"]
    bias: BiasSignal = state["bias"]

    reasons: List[str] = []

    agreement = finbert.label == lexicon.label
    if not agreement:
        reasons.append("model_lexicon_disagreement")
    if finbert.confidence < LOW_CONF_THRESHOLD:
        reasons.append("low_finbert_confidence")
    if bias.flag or bias.score > BIAS_THRESHOLD:
        reasons.append("high_lexicon_bias")

    if len(reasons) >= 2:
        tier = "low"
    elif len(reasons) == 1:
        tier = "medium"
    else:
        tier = "high"

    return PipelineOutput(
        summary=summary,
        finbert=finbert,
        lexicon=lexicon,
        bias=bias,
        agreement=agreement,
        reliability=tier,
        reasons=reasons,
    )


def build_reliability_runnable() -> RunnableLambda:
    """Return a Runnable mapping the merged state -> :class:`PipelineOutput`."""
    return RunnableLambda(_assess, name="reliability_assess")
