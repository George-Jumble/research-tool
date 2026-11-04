from __future__ import annotations

import math
from typing import Any

from .constants import R_M_LOCKED_M

PREF = 1.299e-27  # N m^2, perfect-conductor |F/A| ~ 1/d^4


def casimir_curve(amp: float = 0.04, phase: float = 0.0, n: int = 48) -> list[dict[str, Any]]:
    """SYNTHETIC ILLUSTRATION ONLY. Adds a hand-injected log7 ripple."""
    out = []
    for i in range(n):
        d_nm = 20.0 * (400.0 / 20.0) ** (i / (n - 1))
        d = d_nm * 1e-9
        lif = -PREF / d**4
        log7 = math.log(d / R_M_LOCKED_M) / math.log(7.0)
        ripple = 1.0 + amp * math.sin(2.0 * math.pi * log7 + phase)
        out.append({"d_nm": d_nm, "lifshitz": lif, "mirron": lif * ripple})
    return out


def shape_battery(amp: float = 0.04, phase: float = 0.0) -> dict[str, Any]:
    """SYNTHETIC ILLUSTRATION ONLY.
    Injects a log7 modulation then measures steps and slope.
    Not a real dielectric-response analysis.
    """
    pts = casimir_curve(amp, phase, 64)
    y = [p["mirron"] for p in pts]
    x = [math.log10(p["d_nm"]) for p in pts]
    dy = [y[i + 1] - y[i] for i in range(len(y) - 1)]
    mu = sum(dy) / len(dy)
    var = sum((v - mu) ** 2 for v in dy) / len(dy)
    sd = math.sqrt(var)
    steps = sum(1 for v in dy if abs(v - mu) > 2 * sd)
    logy = [math.log10(abs(v)) for v in y]
    xbar = sum(x) / len(x)
    ybar = sum(logy) / len(logy)
    num = sum((x[i] - xbar) * (logy[i] - ybar) for i in range(len(x)))
    den = sum((x[i] - xbar) ** 2 for i in range(len(x)))
    slope = num / den
    return {
        "status": "SYNTHETIC",
        "warning": "This module plants a log7 ripple then recovers shape features. Not real Casimir data.",
        "discrete_steps": steps,
        "m_plus_4": abs(slope + 4.0),
        "note": "shape battery only; not ΔE = E_obs - E_Lifshitz (dropped 2026-09-15)",
    }
