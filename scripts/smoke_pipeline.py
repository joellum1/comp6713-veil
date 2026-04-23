"""End-to-end smoke test using the public Hub fallbacks.

Run from the repo root:
    python scripts/smoke_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Allow `python scripts/smoke_pipeline.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import build_pipeline  # noqa: E402

SAMPLES = {
    "positive": (
        "Apple reported record fourth-quarter revenue of $94.9 billion on Thursday, "
        "exceeding analyst expectations and posting strong growth in services and wearables. "
        "Earnings per share rose 13% year over year, and management raised its full-year guidance, "
        "citing robust iPhone demand and continued momentum in emerging markets. "
        "Shares climbed more than 4% in after-hours trading."
    ),
    "negative": (
        "Boeing slashed its annual delivery forecast on Wednesday after a fresh round of "
        "production defects and supplier shortages forced the planemaker to halt 737 MAX shipments. "
        "The company warned of a $1.6 billion loss for the quarter, suspended its 2025 guidance, "
        "and disclosed an SEC inquiry into its quality controls. Shares plunged 9% on the news."
    ),
    "speculative": (
        "Tesla may face additional regulatory scrutiny over its full self-driving claims, "
        "which could result in litigation and possible restrictions on future deployments. "
        "Analysts caution that the outcome remains uncertain and is subject to ongoing investigations "
        "by federal authorities. Restrictions imposed on similar programs in the past have constrained "
        "vehicle deliveries and limited near-term revenue growth."
    ),
}


def main() -> None:
    t0 = time.time()
    pipeline = build_pipeline(device="cpu")
    print(f"[setup] pipeline built in {time.time() - t0:.1f}s\n", flush=True)

    for name, article in SAMPLES.items():
        t = time.time()
        out = pipeline.invoke({"article": article})
        elapsed = time.time() - t
        print(f"=== {name}  ({elapsed:.1f}s) ===")
        print(json.dumps(out.model_dump(), indent=2, ensure_ascii=False))
        print(flush=True)


if __name__ == "__main__":
    main()
