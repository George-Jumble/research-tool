from __future__ import annotations

from typing import Any

from .constants import (
    DELTA_N_PLASMA,
    RD_OBSERVED_ERR_MPC,
    RD_OBSERVED_MPC,
)


def sound_horizon(gamma: float) -> dict[str, Any]:
    """r_d = ΔN_plasma * r_d,obs / γ. Residual is identically zero when γ = ΔN_plasma (by construction). No Tau compact period. No n_hier."""
    g = max(float(gamma), 1e-6)
    rd = (DELTA_N_PLASMA * RD_OBSERVED_MPC) / g
    residual = rd - RD_OBSERVED_MPC
    return {
        "gamma": g,
        "rd_predicted_mpc": rd,
        "rd_observed_mpc": RD_OBSERVED_MPC,
        "residual_mpc": residual,
        "residual_sigma": residual / RD_OBSERVED_ERR_MPC,
        "delta_n_plasma": DELTA_N_PLASMA,
        "note": "residual identically zero when gamma equals DELTA_N_PLASMA (by construction)",
    }


def calibrate_gamma() -> float:
    """Default calibration. No n_hier factor."""
    return DELTA_N_PLASMA
