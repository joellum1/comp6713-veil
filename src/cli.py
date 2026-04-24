"""Command-line interface for the VEIL LangChain reliability pipeline.

Usage (from the repo root with the venv activated)::

    python -m src.cli --input path/to/article.txt --pretty
    python -m src.cli --text "Apple beats earnings expectations..." --pretty
    cat article.txt | python -m src.cli --pretty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.pipeline.chain import build_pipeline


def _read_article(args: argparse.Namespace) -> str:
    if args.input:
        return Path(args.input).read_text(encoding="utf-8")
    if args.text:
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veil",
        description=(
            "Run the VEIL LangChain pipeline over a financial news article and "
            "print a structured reliability report as JSON."
        ),
    )
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument(
        "--input", "-i",
        help="Path to a UTF-8 text file containing the article.",
    )
    src_group.add_argument(
        "--text", "-t",
        help="Inline article text (use quotes).",
    )

    parser.add_argument(
        "--finbert-dir", default=None,
        help="Override path to a fine-tuned FinBERT checkpoint.",
    )
    parser.add_argument(
        "--bart-dir", default=None,
        help="Override path to a fine-tuned BART checkpoint.",
    )
    parser.add_argument(
        "--lexicon", default=None,
        help="Override path to the L&M Master Dictionary CSV.",
    )
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda", "mps"],
        help="Force a specific torch device. Auto-detects when omitted.",
    )
    parser.add_argument(
        "--no-neutral-prompt", action="store_true",
        help="Disable the neutral-prompt prefix on the BART summarizer.",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the JSON output (indent=2).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    article = _read_article(args).strip()
    if not article:
        parser.error(
            "No article provided. Use --input/--text or pipe text via stdin."
        )

    pipeline = build_pipeline(
        finbert_dir=args.finbert_dir,
        bart_dir=args.bart_dir,
        lexicon_path=args.lexicon,
        device=args.device,
        apply_neutral_prompt=not args.no_neutral_prompt,
    )

    result = pipeline.invoke({"article": article})
    indent = 2 if args.pretty else None
    print(json.dumps(result.model_dump(), indent=indent, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
