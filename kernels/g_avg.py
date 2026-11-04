"""Equal-tick domain average of 1/gamma^2. R-independent. Not a halo."""

from __future__ import annotations

import numpy as np

from sb_engine.constants import V_ORB

GAMMA = tuple(float(row["gamma"]) for row in V_ORB)
G_AVG_OVER_GN = sum(1.0 / g / g for g in GAMMA) / 6.0


def g_avg_over_gn() -> float:
    return float(G_AVG_OVER_GN)


def v_sb(v_baryon):
    arr = np.asarray(v_baryon, dtype=float)
    return (g_avg_over_gn() ** 0.5) * np.abs(arr)
