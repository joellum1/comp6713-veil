
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.runnables import RunnableLambda

from src.models.finbert import compute_bias_score
from src.pipeline.schema import BiasSignal, LexiconSignal, SentimentLabel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEXICON_PATH = (
    REPO_ROOT / "data" / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
)

# Same modified stopword list as notebooks/sentiment_model_rule_base.py.
# Dropped from traditional: A, I, S, T, DON, WILL, AGAINST. Added: AMONG.
STOPWORDS = frozenset(
    [
        "ME", "MY", "MYSELF", "WE", "OUR", "OURS", "OURSELVES", "YOU", "YOUR",
        "YOURS", "YOURSELF", "YOURSELVES", "HE", "HIM", "HIS", "HIMSELF", "SHE",
        "HER", "HERS", "HERSELF", "IT", "ITS", "ITSELF", "THEY", "THEM", "THEIR",
        "THEIRS", "THEMSELVES", "WHAT", "WHICH", "WHO", "WHOM", "THIS", "THAT",
        "THESE", "THOSE", "AM", "IS", "ARE", "WAS", "WERE", "BE", "BEEN", "BEING",
        "HAVE", "HAS", "HAD", "HAVING", "DO", "DOES", "DID", "DOING", "AN", "THE",
        "AND", "BUT", "IF", "OR", "BECAUSE", "AS", "UNTIL", "WHILE", "OF", "AT",
        "BY", "FOR", "WITH", "ABOUT", "BETWEEN", "INTO", "THROUGH", "DURING",
        "BEFORE", "AFTER", "ABOVE", "BELOW", "TO", "FROM", "UP", "DOWN", "IN",
        "OUT", "ON", "OFF", "OVER", "UNDER", "AGAIN", "FURTHER", "THEN", "ONCE",
        "HERE", "THERE", "WHEN", "WHERE", "WHY", "HOW", "ALL", "ANY", "BOTH",
        "EACH", "FEW", "MORE", "MOST", "OTHER", "SOME", "SUCH", "NO", "NOR",
        "NOT", "ONLY", "OWN", "SAME", "SO", "THAN", "TOO", "VERY", "CAN", "JUST",
        "SHOULD", "NOW", "AMONG",
    ]
)

_TOKEN_SPLIT = re.compile(r"[,:'\s.]")

# Priority order matches map_function in the rule-based baseline.
_PRIORITY = (
    ("positive", "Positive"),
    ("negative", "Negative"),
    ("uncertainty", "Uncertainty"),
    ("litigious", "Litigious"),
    ("strong_modal", "Strong_Modal"),
    ("weak_modal", "Weak_Modal"),
    ("constraining", "Constraining"),
)

# Coarse-grained mapping from L&M tag -> tri-class label, mirroring the
# SENTIMENT_NORMALISATION dict in the rule-based baseline (with 'litigious'
# treated as negative and the modal/uncertainty tags as neutral).
_TAG_TO_CLASS: Dict[str, SentimentLabel] = {
    "positive": "positive",
    "negative": "negative",
    "litigious": "negative",
    "constraining": "negative",
    "uncertainty": "neutral",
    "strong_modal": "neutral",
    "weak_modal": "neutral",
    "neutral": "neutral",
    "cns": "neutral",
}


def load_lexicon(path: Optional[str | Path] = None) -> Dict[str, Dict[str, int]]:
    """Read the L&M master dictionary into ``{WORD: {category: int}}``.

    Only the seven sentiment columns we need are kept; everything else in the
    9MB CSV is dropped to keep memory low and lookups fast.
    """
    src = Path(path) if path is not None else DEFAULT_LEXICON_PATH
    if not src.exists():
        raise FileNotFoundError(
            f"L&M dictionary not found at {src}. Pass lexicon_path=... or place "
            "the CSV at data/Loughran-McDonald_MasterDictionary_1993-2025.csv."
        )

    cols = ["Word"] + [c for _, c in _PRIORITY]
    df = pd.read_csv(src, usecols=cols)
    df["Word"] = df["Word"].astype(str).str.upper()
    for _, col in _PRIORITY:
        df[col] = df[col].fillna(0).astype(int)
    return df.set_index("Word")[[c for _, c in _PRIORITY]].to_dict("index")


def tag_tokens(text: str, lexicon: Dict[str, Dict[str, int]]) -> List[str]:
    """Tag each token with its L&M sentiment category (or ``cns``)."""
    tokens = [t for t in _TOKEN_SPLIT.split(text) if len(t) > 1]
    tags: List[str] = []
    for token in tokens:
        upper = token.upper()
        if upper in STOPWORDS:
            continue
        entry = lexicon.get(upper)
        if entry is None:
            tags.append("cns")
            continue
        for tag, col in _PRIORITY:
            if entry[col] > 0:
                tags.append(tag)
                break
        else:
            tags.append("cns")
    return tags


def _vote(tags: List[str]) -> Tuple[SentimentLabel, Dict[str, int]]:
    counts: Dict[str, int] = {}
    for tag in tags:
        if tag == "cns":
            continue
        counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return "neutral", counts
    top_tag = max(counts, key=counts.__getitem__)
    return _TAG_TO_CLASS.get(top_tag, "neutral"), counts


def build_lexicon_runnable(
    *, lexicon_path: Optional[str | Path] = None
) -> RunnableLambda:
    """Return a Runnable mapping ``{'article': str}`` -> ``{'lexicon', 'bias'}``."""
    lexicon = load_lexicon(lexicon_path)

    def _signal(payload: dict) -> dict:
        tags = tag_tokens(payload["article"], lexicon)
        label, counts = _vote(tags)
        bias = compute_bias_score(tags)
        return {
            "lexicon": LexiconSignal(label=label, counts=counts),
            "bias": BiasSignal(
                uncertainty=round(bias["uncertainty"], 4),
                litigious=round(bias["litigious"], 4),
                constraining=round(bias["constraining"], 4),
                score=round(bias["bias_score"], 4),
                flag=bool(bias["bias_flag"]),
            ),
        }

    return RunnableLambda(_signal, name="lexicon_signal")
