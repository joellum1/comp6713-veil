"""Shared local-checkpoint resolution for the LangChain pipeline wrappers.

Both :mod:`src.pipeline.summarizer` and :mod:`src.pipeline.sentiment` need the
same logic: try a list of local checkpoint directories first, fall back to a
Hugging Face Hub model ID when none of them exist. For BART we additionally
pick the *best* of multiple local fine-tunes by reading the side-car metric
JSON the team's training notebooks emit.
"""

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
    """Return the first existing candidate when ``model_dir`` is None.

    * Explicit ``model_dir`` always wins.
    * Otherwise the first directory in ``candidates`` that exists is used.
    * If none exist, ``hub_fallback`` is returned (Hugging Face Hub model id).
    """
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
    """Like :func:`resolve_local_or_hub`, but for BART picks the highest-scoring
    locally-trained checkpoint based on the metrics JSON next to it.

    Mapping convention (matches notebooks/summarisation*.ipynb):
        ``results/bart-final-<dataset>``    ↔
        ``results/report/final_metrics_<dataset>.json``

    A directory without a readable metrics file is still eligible but ranked
    after any scored candidate.
    """
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
    """Look up ``results/report/final_metrics_<dataset>.json`` for the given
    ``bart-final-<dataset>`` directory and return its ``test.<metric_key>``.

    Returns ``None`` if the file or key is missing or unreadable.
    """
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
