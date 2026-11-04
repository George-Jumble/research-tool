"""
Approximate planar Lifshitz free-energy density for a vacuum gap.

This is NOT a scattering-code integrator. It is a first-working Matsubara
sum with TE/TM Fresnel coefficients for two identical planar mirrors
separated by vacuum. Every approximation is listed in
``LIFSHITZ_APPROXIMATIONS`` and copied into JSON reports.

Units
-----
* ``xi_n`` from Fig2-style tables is inferred: median ≲ 1e6 → eV, else rad/s.
* Separation ``d`` is metres.
* Output ``F/A`` is J/m² (negative = attractive).
* Temperature default 300 K; n_max is the length of the supplied ξ table
  (600 for Fig2), not a freshly generated Matsubara ladder.
"""

from __future__ import annotations

from typing import Any

import numpy as np

K_B = 1.380649e-23
HBAR = 1.054571817e-34
C_LIGHT = 299792458.0
E_CHARGE = 1.602176634e-19

LIFSHITZ_APPROXIMATIONS: list[str] = [
    "planar identical mirrors, vacuum gap, μ = 1 (non-magnetic)",
    "TE/TM Fresnel reflection at a single dielectric–vacuum interface; r^2 for two identical plates",
    "no roughness, no finite conductivity correction beyond tabulated eps(i ξ), no retardation truncation beyond k_max",
    "Matsubara frequencies taken from the supplied table (n_max = table length), not recomputed from T",
    "n = 0 term weighted 1/2 (standard Matsubara prime); TE → 0 at ξ = 0",
    "k_perp integral: trapezoid on a finite grid k ∈ [0, max(50/d, 20 ξ/c)]",
    "NOT a full scattering / sphere-plane / proximity-force scattering code",
]


def infer_xi_rad_s(xi_n: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert tabulated ξ to rad/s. Document the inferred unit."""
    xi = np.asarray(xi_n, dtype=np.float64).ravel()
    positive = xi[np.isfinite(xi) & (xi > 0)]
    if positive.size == 0:
        return xi, "unknown"
    med = float(np.median(positive))
    if med < 1.0e6:
        return xi * (E_CHARGE / HBAR), "eV"
    return xi, "rad/s"


def _fresnel_vacuum_dielectric(
    eps: float,
    xi_over_c: float,
    k_perp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k2 = np.square(k_perp)
    kappa0 = np.sqrt(k2 + xi_over_c**2)
    eps_clamped = max(float(eps), 1e-30)
    kappa1 = np.sqrt(np.maximum(k2 + eps_clamped * xi_over_c**2, 0.0))
    den_te = kappa0 + kappa1
    den_tm = eps_clamped * kappa0 + kappa1
    r_te = np.divide(kappa0 - kappa1, den_te, out=np.zeros_like(kappa0), where=den_te != 0)
    r_tm = np.divide(
        eps_clamped * kappa0 - kappa1,
        den_tm,
        out=np.zeros_like(kappa0),
        where=den_tm != 0,
    )
    if xi_over_c <= 0.0:
        r_te = np.zeros_like(r_te)
        r_tm = np.full_like(r_tm, (eps_clamped - 1.0) / (eps_clamped + 1.0))
        kappa0 = np.abs(k_perp)
    return r_te, r_tm, kappa0


def _k_integrand(r: np.ndarray, kappa0: np.ndarray, d: float) -> np.ndarray:
    damp = np.exp(-2.0 * kappa0 * d)
    arg = r * r * damp
    arg = np.clip(arg, 0.0, 1.0 - 1e-15)
    return np.log(np.maximum(1.0 - arg, 1e-300))


def _one_term_integral(
    eps: float,
    xi_rad: float,
    d: float,
    *,
    n_k: int = 96,
) -> float:
    """Integrate in dimensionless p = k_perp * d so the grid resolves exp(-2 κ d)."""
    gap = max(float(d), 1e-18)
    xi_over_c = float(xi_rad) / C_LIGHT
    p_max = 50.0
    p = np.linspace(0.0, p_max, int(n_k))
    k_perp = p / gap
    r_te, r_tm, kappa0 = _fresnel_vacuum_dielectric(eps, xi_over_c, k_perp)
    integrand = p * (
        _k_integrand(r_te, kappa0, gap) + _k_integrand(r_tm, kappa0, gap)
    )
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    # ∫ k dk = d^{-2} ∫ p dp
    return float(trapz(integrand, p) / (gap * gap))


def free_energy_density(
    xi_n: np.ndarray,
    eps_pri: np.ndarray,
    d: np.ndarray,
    *,
    T: float = 300.0,
    n_k: int = 96,
) -> dict[str, Any]:
    """
    Approximate Lifshitz free-energy per area F/A(d) from tabulated eps(i ξ).

    Parameters
    ----------
    xi_n
        Matsubara frequencies (eV or rad/s; inferred).
    eps_pri
        Primary dielectric ε(i ξ_n), dimensionless.
    d
        Separation grid in metres.
    T
        Temperature in kelvin (prefactor k_B T / 2π only; ξ taken from table).
    n_k
        Trapezoid nodes in the k_perp integral.
    """
    xi_rad, xi_unit = infer_xi_rad_s(xi_n)
    eps = np.asarray(eps_pri, dtype=np.float64).ravel()
    d_m = np.asarray(d, dtype=np.float64).ravel()
    n = min(xi_rad.size, eps.size)
    xi_rad = xi_rad[:n]
    eps = eps[:n]
    order = np.argsort(xi_rad)
    xi_rad = xi_rad[order]
    eps = eps[order]
    finite = np.isfinite(xi_rad) & np.isfinite(eps) & (eps > 0)
    xi_rad = xi_rad[finite]
    eps = eps[finite]
    n_max = int(xi_rad.size)

    pref = (K_B * float(T)) / (2.0 * np.pi)
    F = np.zeros(d_m.size, dtype=np.float64)
    for i, gap in enumerate(d_m):
        acc = 0.0
        for n_idx in range(n_max):
            weight = 0.5 if n_idx == 0 else 1.0
            acc += weight * _one_term_integral(
                float(eps[n_idx]),
                float(xi_rad[n_idx]),
                float(gap),
                n_k=n_k,
            )
        F[i] = pref * acc

    return {
        "d": d_m,
        "F_over_A": F,
        "n_max": n_max,
        "T": float(T),
        "xi_unit": xi_unit,
        "n_k": int(n_k),
        "lifshitz_approximations": list(LIFSHITZ_APPROXIMATIONS),
    }


def free_energy_density_quadrature_check(
    xi_n: np.ndarray,
    eps_pri: np.ndarray,
    d: np.ndarray,
    *,
    T: float = 300.0,
    n_k: int = 64,
) -> dict[str, Any]:
    """Re-integrate on a denser k-grid; report relative quadrature mismatch."""
    coarse = free_energy_density(xi_n, eps_pri, d, T=T, n_k=n_k)
    fine = free_energy_density(xi_n, eps_pri, d, T=T, n_k=n_k + 24)
    f1 = np.asarray(coarse["F_over_A"], dtype=np.float64)
    f2 = np.asarray(fine["F_over_A"], dtype=np.float64)
    scale = np.maximum(np.max(np.abs(f1)), 1e-300)
    rel = float(np.max(np.abs(f1 - f2)) / scale)
    return {
        "F_coarse": f1,
        "F_fine": f2,
        "quadrature_rel": rel,
        "n_k_coarse": n_k,
        "n_k_fine": n_k + 24,
        "xi_unit": coarse["xi_unit"],
        "n_max": coarse["n_max"],
        "T": coarse["T"],
        "lifshitz_approximations": coarse["lifshitz_approximations"],
    }
