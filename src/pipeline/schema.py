
from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field

SentimentLabel = Literal["positive", "negative", "neutral"]
ReliabilityTier = Literal["high", "medium", "low"]


class SentimentSignal(BaseModel):
    """FinBERT sentiment prediction."""

    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)


class LexiconSignal(BaseModel):
    """Loughran-McDonald rule-based sentiment vote."""

    label: SentimentLabel
    counts: Dict[str, int] = Field(default_factory=dict)


class BiasSignal(BaseModel):
    """L&M-derived bias indicators (matches src/models/finbert.compute_bias_score)."""

    uncertainty: float = Field(ge=0.0, le=1.0)
    litigious: float = Field(ge=0.0, le=1.0)
    constraining: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    flag: bool


class PipelineOutput(BaseModel):
    """Final structured reliability report returned to the caller."""

    summary: str
    finbert: SentimentSignal
    lexicon: LexiconSignal
    bias: BiasSignal
    agreement: bool
    reliability: ReliabilityTier
    reasons: List[str] = Field(default_factory=list)
