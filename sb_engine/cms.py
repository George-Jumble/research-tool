from __future__ import annotations

import math
import random
from typing import Any


def run_cms(n: int = 4000, amp: float = 0.18, seed: int = 7) -> dict[str, Any]:
    """SYNTHETIC ILLUSTRATION ONLY.
    Generates counts from a 7-fold modulation and recovers it.
    Not a real CMS Open Data analysis. Do not treat chi2 as a detection.
    """
    n = max(200, min(int(n), 20000))
    amp = min(max(float(amp), 0.0), 0.8)
    rng = random.Random(seed)
    weights = [1.0 + amp * math.cos(2.0 * math.pi * k / 7.0) for k in range(7)]
    total = sum(weights)
    cdf = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cdf.append(acc)
    counts = [0] * 7
    for _ in range(n):
        u = rng.random()
        slot = next((i for i, c in enumerate(cdf) if u <= c), 6)
        counts[slot] += 1
    expected = n / 7.0
    chi2_null = sum((c - expected) ** 2 / expected for c in counts)
    pred = [n * w / total for w in weights]
    chi2 = sum((c - p) ** 2 / p for c, p in zip(counts, pred))
    return {
        "status": "SYNTHETIC",
        "warning": "This module plants the 7-fold signal it recovers. Not real CMS data.",
        "n": n,
        "amp": amp,
        "counts": counts,
        "chi2_clockwork": chi2,
        "chi2_uniform": chi2_null,
        "note": "7 active clockwork slots; Φ8 is reset and is not a CMS bin",
    }
