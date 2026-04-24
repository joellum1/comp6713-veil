"""Shared local-checkpoint resolution for the LangChain pipeline wrappers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

PathLike = Union[str, Path]


def resolve_local_or_hub(
    model_dir: Optional[PathLike],
    candidates: Iterable[Path],
    hub_fallback: str,
) -> str:
    """Return the first existing local candidate, else the Hub fallback model id."""
    if model_dir is not None:
        return str(model_dir)
    for c in candidates:
        if Path(c).exists():
            return str(c)
    return hub_fallback


def pick_best_bart(
    model_dir: Optional[PathLike],
    candidates: Iterable[Path],
    hub_fallback: str,
    metrics_dir: Path,
    metric_key: str = "final_test_rougeL",
) -> str:
    """Pick the local BART checkpoint with highest test rougeL, else fall back to the Hub."""
    if model_dir is not None:
        return str(model_dir)

    available = [Path(c) for c in candidates if Path(c).exists()]
    if not available:
        return hub_fallback

    scored: list[tuple[float, Path]] = []
    unscored: list[Path] = []

    for d in available:
        score = _read_score(d, metrics_dir, metric_key)
        if score is None:
            unscored.append(d)
        else:
            scored.append((score, d))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return str(scored[0][1])
    return str(unscored[0])


def _read_score(
    model_dir: Path, metrics_dir: Path, metric_key: str
) -> Optional[float]:
    """Return ``test.<metric_key>`` from final_metrics_<dataset>.json, or None if missing."""
    name = model_dir.name
    if not name.startswith("bart-final-"):
        return None
    suffix = name[len("bart-final-") :]
    metrics_path = metrics_dir / f"final_metrics_{suffix}.json"
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open() as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    test = data.get("test")
    if not isinstance(test, dict):
        return None
    score = test.get(metric_key)
    return float(score) if isinstance(score, (int, float)) else None
