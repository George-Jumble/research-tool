"""Hierarchical L_res rotation-curve kernel (v5 Section 6, 'Rotation curves').

V_RL(r) = V_flat * sqrt(r / (r + R_core)), added in quadrature to V_bar (M/L = 1), with V_flat the
mean V_obs of the outer 30 % of points, one free R_core per galaxy, and eta = 0.

Replaces the retired Template B kernel G_avg = <1/gamma^2> over the quark ladder (kernels/g_avg.py):
the Weyl/conformal averaging kernel with Scheme A weights was retired 2026-10-31 (T1), and the
quark-ladder gamma drops out of gravity (N15).
"""

from __future__ import annotations

import numpy as np

KERNEL_NAME = "L_res_V_RL_hierarchical"
R_CORE_GRID_KPC = np.logspace(-2, 2.5, 400)
OUTER_FRACTION = 0.30


def v_baryon(v_gas, v_disk, v_bul) -> np.ndarray:
    """Newtonian baryonic speed, M/L = 1, signed squares as in SPARC (negative V_gas allowed)."""
    terms = [np.asarray(v, dtype=float) for v in (v_gas, v_disk, v_bul)]
    v2 = sum(np.sign(t) * t * t for t in terms)
    return np.sqrt(np.clip(v2, 0.0, None))


def v_flat_anchor(v_obs, outer_fraction: float = OUTER_FRACTION) -> float:
    v = np.asarray(v_obs, dtype=float)
    k = max(1, int(np.ceil(outer_fraction * v.size)))
    return float(np.mean(v[-k:]))


def v_rl(r, v_flat: float, r_core: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return float(v_flat) * np.sqrt(r / (r + float(r_core)))


def fit_galaxy(r, v_obs, v_bar, sigma=None, grid=R_CORE_GRID_KPC) -> dict:
    """Grid fit of R_core for one galaxy (radii sorted ascending). Returns chi2 for L_res and baryons."""
    r = np.asarray(r, dtype=float)
    v_obs = np.asarray(v_obs, dtype=float)
    v_bar = np.asarray(v_bar, dtype=float)
    sig = np.ones_like(v_obs) if sigma is None else np.asarray(sigma, dtype=float)
    order = np.argsort(r)
    r, v_obs, v_bar, sig = r[order], v_obs[order], v_bar[order], sig[order]
    v_flat = v_flat_anchor(v_obs)
    chi2_bar = float(np.sum(((v_obs - v_bar) / sig) ** 2))
    best = (float("inf"), float("nan"))
    for rc in grid:
        model = np.sqrt(v_bar ** 2 + v_rl(r, v_flat, rc) ** 2)
        c2 = float(np.sum(((v_obs - model) / sig) ** 2))
        if c2 < best[0]:
            best = (c2, float(rc))
    n = int(v_obs.size)
    return {
        "kernel": KERNEL_NAME,
        "n": n,
        "v_flat": v_flat,
        "r_core_kpc": best[1],
        "chi2_SB": best[0],
        "chi2_baryon": chi2_bar,
        "reduced_chi2_SB": best[0] / max(n - 1, 1),
        "reduced_chi2_baryon": chi2_bar / max(n - 1, 1),
        "eta": 0.0,
        "mass_to_light": 1.0,
    }
