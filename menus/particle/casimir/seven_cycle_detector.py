"""
Independent 7-cycle matched-filter detector.

Quadrature template correlator for a sinusoid of period 7 in the sweep
coordinate. This module never writes primary signature flags. Its threshold
is calibrated on decay-null Monte Carlo so max FPR stays < 0.05.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

SEVEN_CYCLE_PERIOD: float = 7.0
# Conservative starter; overwritten by the decay-null calibration.
DEFAULT_SEVEN_CYCLE_MF_THRESHOLD: float = 0.105
SEVEN_CYCLE_FLAGS: tuple[str, ...] = (
    "seven_phase_clockwork",
    "tav_resonance_detected",
)


def seven_cycle_mf_threshold() -> float:
    raw = os.getenv("TAV_CASIMIR_SEVEN_CYCLE_MF_THRESHOLD")
    try:
        value = float(raw) if raw is not None else DEFAULT_SEVEN_CYCLE_MF_THRESHOLD
    except ValueError:
        return DEFAULT_SEVEN_CYCLE_MF_THRESHOLD
    return value if 0.0 <= value <= 1.0 else DEFAULT_SEVEN_CYCLE_MF_THRESHOLD


def _unit(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-30:
        return None
    return vec / norm


def matched_filter_coherence(x: np.ndarray, residuals: np.ndarray, *, period: float) -> dict[str, float]:
    """
    Phase-invariant coherence of residuals with sin/cos of the given period.

    Templates are demeaned and orthonormalized on the actual sampling grid.
    Score is ||P r|| / ||r|| in [0, 1].
    """
    x = np.asarray(x, dtype=float).ravel()
    r = np.asarray(residuals, dtype=float).ravel()
    n = min(x.size, r.size)
    empty = {"score": 0.0, "n": int(n), "period": float(period), "amplitude": 0.0, "phase": 0.0}
    if n < 16 or period <= 0:
        return empty
    x = x[:n]
    r = r[:n] - float(np.mean(r[:n]))
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 1e-30:
        return empty

    omega = 2.0 * np.pi / float(period)
    sine = np.sin(omega * x)
    cosine = np.cos(omega * x)
    sine = sine - float(np.mean(sine))
    cosine = cosine - float(np.mean(cosine))
    s_hat = _unit(sine)
    if s_hat is None:
        return empty
    cosine = cosine - float(np.dot(cosine, s_hat)) * s_hat
    c_hat = _unit(cosine)
    if c_hat is None:
        return empty

    a = float(np.dot(r, s_hat))
    b = float(np.dot(r, c_hat))
    proj = float(np.hypot(a, b))
    score = float(min(proj / r_norm, 1.0))
    return {
        "score": score,
        "n": int(n),
        "period": float(period),
        "amplitude": proj,
        "phase": float(np.arctan2(b, a)),
    }


def detect_seven_cycle_matched_filter(
    x: np.ndarray,
    residuals: np.ndarray,
    *,
    threshold: float | None = None,
    period: float = SEVEN_CYCLE_PERIOD,
) -> dict[str, Any]:
    """
    Independent 7-cycle call. Returns its own flags; does not mutate callers.

    Both family flags share this single statistic. They are not primary claims.
    """
    gate = float(threshold) if threshold is not None else seven_cycle_mf_threshold()
    stats = matched_filter_coherence(x, residuals, period=period)
    detected = bool(stats["score"] >= gate)
    return {
        "analysis": "seven_cycle_matched_filter",
        "detector": "quadrature_template_correlator",
        "independent_of": "primary_signature_summary",
        "period": float(period),
        "score": stats["score"],
        "amplitude": stats["amplitude"],
        "phase": stats["phase"],
        "n": stats["n"],
        "threshold": gate,
        "detected": detected,
        "seven_phase_clockwork": detected,
        "tav_resonance_detected": detected,
        "note": (
            "Independent matched-filter correlator for period-7 structure. "
            "Never used for primary claims. Own decay-null FPR is documented "
            "in the null-hypothesis report."
        ),
    }
