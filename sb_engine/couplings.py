"""Coupling constants are local (ruling N14). Pure numbers are frame-map invariant.

A frame map multiplies every quantity of dimension M^a L^b T^c by gamma^(a*mu + b*lam + c*tau).
Any dimensionless combination has total exponent zero for every (mu, lam, tau), so alpha, alpha_s,
G_F m_p^2 and alpha_G are identical in every domain. Port of Part 1 of
scheme_A_relative_couplings_test.py. The superseded literal gamma_q * alpha reading is kept as a
negative control (alpha ~ 583 in the t domain; alpha_s > 4 pi from the s domain on).
"""

from __future__ import annotations

import math

from .constants import SCHEME_A_GAMMA

DIM = {
    "c": (0, 1, -1),
    "hbar": (1, 2, -1),
    "e2_4pi_eps0": (1, 3, -2),
    "G": (-1, 3, -2),
    "G_F_Jm3": (1, 5, -2),
    "mass": (1, 0, 0),
}
DIMLESS = {
    "alpha": {"e2_4pi_eps0": 1, "hbar": -1, "c": -1},
    "alpha_s": {"e2_4pi_eps0": 1, "hbar": -1, "c": -1},
    "G_F_mp2": {"G_F_Jm3": 1, "mass": 2, "c": 1, "hbar": -3},
    "alpha_G": {"G": 1, "mass": 2, "hbar": -1, "c": -1},
}
ALPHA_EM = 1.0 / 137.035999
ALPHA_S_MZ = 0.1180


def dimensionless_exponents(mu: float, lam: float, tau: float) -> dict[str, float]:
    e = {k: a * mu + b * lam + c * tau for k, (a, b, c) in DIM.items()}
    return {name: sum(e[c] * p for c, p in combo.items()) for name, combo in DIMLESS.items()}


def literal_scaled_couplings() -> dict[str, dict[str, float]]:
    """Superseded reading: alpha_q = gamma_q * alpha (and the same for alpha_s)."""
    return {
        q: {"alpha": ALPHA_EM * g, "alpha_s": ALPHA_S_MZ * g}
        for q, g in SCHEME_A_GAMMA.items()
    }
