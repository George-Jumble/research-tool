from __future__ import annotations

import math
from typing import Any

from .constants import GAMMA_HEX, V_ORB

G_KPC = 4.30091e-6
D_BETA2 = V_ORB[1]["beta"] ** 2


def _miyamoto(r: float, mass: float, a: float, b: float) -> float:
    den = (r * r + (a + b) ** 2) ** 1.5
    return math.sqrt(max(G_KPC * mass * r * r / den, 0.0))


def _nfw(r: float, rho_s: float, rs: float) -> float:
    x = r / rs
    f = math.log(1.0 + x) - x / (1.0 + x)
    v2 = 4.0 * math.pi * G_KPC * rho_s * rs**3 * f / max(r, 1e-6)
    return math.sqrt(max(v2, 0.0))


def emergent_dm(r: float, beta2: float, phase: float) -> float:
    r_c = 2.2 + 5.4 * beta2 + 0.7 * math.sin(phase)
    v_scale = 38.0 + 95.0 * math.sqrt(max(beta2, 0.0)) * (GAMMA_HEX / 1.4)
    ripple = 1.0 + 0.1 * math.sin(phase + r / 6.0)
    return v_scale * math.sqrt(r / (r + r_c)) * ripple


def run_sparc(beta2: float = D_BETA2, phase: float = 0.4) -> dict[str, Any]:
    """SYNTHETIC ILLUSTRATION ONLY.
    Generates synthetic NGC3198 velocities then fits them with emergent_dm.
    Not a confrontation with published SPARC rotation curves.
    """
    points = []
    chi2 = 0.0
    for i in range(28):
        r = 0.6 + i * 1.35
        v_disk = _miyamoto(r, 3.05e10, 3.14, 0.35)
        v_gas = _miyamoto(r, 1.15e10, 8.4, 0.25)
        v_bar = math.hypot(v_disk, v_gas)
        v_obs = math.hypot(v_bar, _nfw(r, 4.2e6, 16.5))
        e_obs = 6.5 + 0.04 * r
        v_dm = emergent_dm(r, beta2, phase)
        v_th = math.hypot(v_bar, v_dm)
        chi2 += ((v_obs - v_th) / e_obs) ** 2
        points.append(
            {
                "r_kpc": r,
                "v_obs": v_obs,
                "v_bar": v_bar,
                "v_dm": v_dm,
                "v_theory": v_th,
            }
        )
    ndof = max(len(points) - 2, 1)
    return {
        "status": "SYNTHETIC",
        "warning": "This module synthesizes NGC3198 data and fits it. Not real SPARC curves.",
        "galaxy": "NGC3198 (synthetic)",
        "beta2": beta2,
        "phase": phase,
        "chi2": chi2,
        "reduced_chi2": chi2 / ndof,
        "n": len(points),
        "points": points,
    }
