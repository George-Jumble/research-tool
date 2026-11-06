"""RETIRED 2026-10-31 (T1) / 2026-11-03 (N15): equal-tick domain average of 1/gamma^2.

This was the Weyl/conformal averaging kernel with Scheme A weights. v5 retires it, and the quark-ladder
gamma drops out of gravity. Kept importable for legacy reports only; Template B now uses
``kernels.l_res_rl`` (hierarchical L_res, v5 Section 6).
"""

from __future__ import annotations

import numpy as np

from sb_engine.constants import V_ORB

KERNEL_STATUS = "RETIRED"

GAMMA = tuple(float(row["gamma"]) for row in V_ORB)
G_AVG_OVER_GN = sum(1.0 / g / g for g in GAMMA) / 6.0


def g_avg_over_gn() -> float:
    return float(G_AVG_OVER_GN)


def v_sb(v_baryon):
    arr = np.asarray(v_baryon, dtype=float)
    return (g_avg_over_gn() ** 0.5) * np.abs(arr)
