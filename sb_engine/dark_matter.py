"""Dark-to-ordinary ratio under v5 (Ansatz A, B_lin co-orbit weights as fixed T_priv weightings).

Rulings: R5/T4 (lock 5.4015, gamma^1 on B_lin), N15 (quark-ladder gamma drops out of gravity),
N17 (each envelope contributes gamma*M*c^2 at the moment of action on T_priv), N18 (every domain is
Prime to itself), N19 (co-orbit gamma is a fixed T_priv weighting, not a composable velocity).
Test write-ups: local_gravity_coorbit_test.md, domain_relativity_test.md, scheme_A_light_speed_test.md.
"""

from __future__ import annotations

import math
from typing import Any

from .constants import B_LIN_BETA, B_LIN_GAMMA, DARK_TO_ORDINARY_LOCK, SCHEME_A_GAMMA


def _gamma(beta: float) -> float:
    return 1.0 / math.sqrt(1.0 - float(beta) ** 2)


def dark_to_ordinary(reading: str = "tpriv_energy") -> float:
    """Hidden/Prime gravitating ratio for one reading.

    tpriv_energy      gamma^1 on B_lin (each envelope gamma M c^2 on T_priv)  -> 5.4015 (live, N17)
    local_rest_mass   equal weights (strict local rest mass)                 -> 5.00   (fails)
    kernel_gamma3     literal boosted-dust kernel gamma^3 on B_lin           -> 6.44   (not used)
    ladder_gamma1     quark-ladder gamma as gravitating weight               -> retired (N15)
    """
    hidden = B_LIN_GAMMA[1:]
    if reading == "tpriv_energy":
        return float(sum(hidden))
    if reading == "local_rest_mass":
        return float(len(hidden))
    if reading == "kernel_gamma3":
        return float(sum(g ** 3 for g in hidden))
    if reading == "ladder_gamma1":
        return float(sum(v for k, v in SCHEME_A_GAMMA.items() if k != "u"))
    raise KeyError(reading)


def viewpoint_ratios_tpriv() -> dict[int, float]:
    """N18/N19: every domain sees itself at rest and the other five with the same B_lin weights."""
    same = dark_to_ordinary("tpriv_energy")
    return {i: same for i in range(len(B_LIN_GAMMA))}


def viewpoint_ratios_composed() -> dict[int, float]:
    """RETIRED reading (domain_relativity_test): B_lin as real collinear velocities, composed relativistically."""
    out: dict[int, float] = {}
    for j, bj in enumerate(B_LIN_BETA):
        total = 0.0
        for i, bi in enumerate(B_LIN_BETA):
            if i == j:
                continue
            total += _gamma((bi - bj) / (1.0 - bi * bj))
        out[j] = total
    return out


def pull_sigma(model: float, obs: float, obs_unc: float, model_unc: float = 0.0) -> float:
    return (float(model) - float(obs)) / math.hypot(float(obs_unc), float(model_unc))


def ratio_report(obs: float, obs_unc: float) -> dict[str, Any]:
    live = dark_to_ordinary("tpriv_energy")
    return {
        "lock": DARK_TO_ORDINARY_LOCK,
        "computed": live,
        "pull_sigma": pull_sigma(live, obs, obs_unc),
        "local_rest_mass": dark_to_ordinary("local_rest_mass"),
        "local_rest_mass_pull_sigma": pull_sigma(dark_to_ordinary("local_rest_mass"), obs, obs_unc),
        "gamma1_derived": False,
        "weights_input": "B_lin beta_i = (m_i - 11)/30 (convention)",
    }
