"""Template C (Mirron / Planck-Kerr) numeric kernel — author ruling 2026-11-04.

    S_C(q²) = A · M² / (M² + q²),   M = M_mir = Δ = 313.1 MeV (locked floor, sb_engine.constants),
    A = α_s(μ = ħc / r_M = 313.1 MeV).

Zero free parameters: A is computed, never fitted. q² is taken in GeV² (LHCb convention) and M² is
converted to GeV². α_s(μ) is MS-bar running from α_s(M_Z) = 0.1180 with n_f flavour thresholds at the
MS-bar masses m_b(m_b) = 4.18 GeV and m_c(m_c) = 1.27 GeV (PDG); n_f = 3 below m_c.
loops=4 uses 3-loop decoupling at the thresholds, loops<=2 uses continuous matching.
μ = 313.1 MeV is near/below Λ_MS^(3), so the perturbative running may hit the Landau pole; this is
reported (``stable`` flag / ``status``) instead of hidden.
"""

from __future__ import annotations

import math

import numpy as np

from sb_engine.constants import DELTA_FLOOR_MEV

KERNEL_NAME = "Mirron_C_propagator_alpha_s"
ALPHA_S_MZ = 0.1180
M_Z_GEV = 91.1876
M_B_GEV = 4.18
M_C_GEV = 1.27
M_MIR_GEV = DELTA_FLOOR_MEV / 1000.0
_ZETA3 = 1.2020569031595942


def _beta(nf: int) -> tuple[float, float, float, float]:
    """β_i for da/dln μ² = -Σ β_i a^(i+2), a = α_s/π."""
    z3 = _ZETA3
    b0 = (11.0 - 2.0 * nf / 3.0) / 4.0
    b1 = (102.0 - 38.0 * nf / 3.0) / 16.0
    b2 = (2857.0 / 2.0 - 5033.0 * nf / 18.0 + 325.0 * nf**2 / 54.0) / 64.0
    b3 = (
        (149753.0 / 6.0 + 3564.0 * z3)
        - (1078361.0 / 162.0 + 6508.0 * z3 / 27.0) * nf
        + (50065.0 / 162.0 + 6472.0 * z3 / 81.0) * nf**2
        + 1093.0 / 729.0 * nf**3
    ) / 256.0
    return b0, b1, b2, b3


def _run(a: float, mu0: float, mu1: float, nf: int, loops: int, steps: int = 4000) -> float:
    """RK4 in t = ln μ² from mu0 to mu1 (GeV). Returns a = α_s/π, or inf past the Landau pole."""
    b = _beta(nf)[:loops]

    def f(x: float) -> float:
        return -sum(bi * x ** (i + 2) for i, bi in enumerate(b))

    t0, t1 = 2.0 * math.log(mu0), 2.0 * math.log(mu1)
    h = (t1 - t0) / steps
    for _ in range(steps):
        k1 = f(a)
        k2 = f(a + 0.5 * h * k1)
        k3 = f(a + 0.5 * h * k2)
        k4 = f(a + h * k3)
        a = a + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if not math.isfinite(a) or a <= 0 or a > 10.0:
            return math.inf
    return a


def _decouple_down(a: float, nl: int, loops: int) -> float:
    """α_s^(nl) from α_s^(nl+1) at μ = m_q(m_q), MS-bar mass, a = α_s/π."""
    if loops < 3:
        return a
    c2 = 11.0 / 72.0
    c3 = 564731.0 / 124416.0 - 82043.0 / 27648.0 * _ZETA3 - 2633.0 / 31104.0 * nl
    return a * (1.0 + c2 * a**2 + (c3 * a**3 if loops >= 4 else 0.0))


def alpha_s(mu_gev: float = M_MIR_GEV, loops: int = 4, alpha_mz: float = ALPHA_S_MZ) -> dict:
    """α_s(μ) in MS-bar; deterministic. Returns value, status and per-step values."""
    a = alpha_mz / math.pi
    steps = []
    a = _run(a, M_Z_GEV, M_B_GEV, 5, loops)
    steps.append(("m_b", 5, math.pi * a))
    a = _decouple_down(a, 4, loops) if math.isfinite(a) else a
    if mu_gev < M_C_GEV:
        a = _run(a, M_B_GEV, M_C_GEV, 4, loops) if math.isfinite(a) else a
        steps.append(("m_c", 4, math.pi * a))
        a = _decouple_down(a, 3, loops) if math.isfinite(a) else a
        a = _run(a, M_C_GEV, mu_gev, 3, loops) if math.isfinite(a) else a
    else:
        a = _run(a, M_B_GEV, mu_gev, 4, loops) if math.isfinite(a) else a
    val = math.pi * a
    stable = math.isfinite(val) and val < 1.0
    status = "ok" if stable else ("landau_pole" if not math.isfinite(val) else "non_perturbative_(>=1)")
    return {"mu_GeV": mu_gev, "loops": loops, "alpha_s": val, "stable": stable,
            "status": status, "thresholds": steps}


def amplitude(loops: int = 4) -> dict:
    """A = α_s(313.1 MeV) at the requested loop order (no fallback here)."""
    return alpha_s(M_MIR_GEV, loops=loops)


def s_c(q2_gev2, a: float, m_gev: float = M_MIR_GEV) -> np.ndarray:
    q2 = np.asarray(q2_gev2, dtype=float)
    m2 = m_gev * m_gev
    return a * m2 / (m2 + q2)


def score(r, q2_gev2, sigma=None, *, loops: int = 4) -> dict:
    """χ² of r = D − B against S_C (sign +, A not fitted) and the B-baseline χ² (S = 0).

    Verdict follows Template B: compare chi2_SB with chi2_Lambda on the same bins.
    A diagnostic least-squares A ± σ for the fixed shape is reported but not used in the score.
    """
    r = np.asarray(r, dtype=float)
    q2 = np.asarray(q2_gev2, dtype=float)
    sig = np.ones_like(r) if sigma is None else np.asarray(sigma, dtype=float)
    used = {}
    for lo in (4, 3, 2, 1):
        res = alpha_s(M_MIR_GEV, loops=lo)
        used[lo] = res
    chosen = next((lo for lo in (4, 3, 2, 1) if used[lo]["stable"]), None)
    if chosen is None:
        return {"kernel": KERNEL_NAME, "chi2_SB": None, "verdict": "inconclusive",
                "alpha_s_by_loops": {k: v["alpha_s"] for k, v in used.items()},
                "kernel_note": "α_s(313.1 MeV) not perturbatively stable at any loop order"}
    A = used[chosen]["alpha_s"]
    shape = s_c(q2, 1.0)
    model = A * shape
    chi2_sb = float(np.sum(((r - model) / sig) ** 2))
    chi2_l = float(np.sum((r / sig) ** 2))
    w = 1.0 / sig**2
    a_fit = float(np.sum(w * r * shape) / np.sum(w * shape * shape))
    a_err = float(1.0 / math.sqrt(np.sum(w * shape * shape)))
    return {
        "kernel": KERNEL_NAME,
        "N": int(r.size),
        "M_mir_GeV": M_MIR_GEV,
        "A": A,
        "A_loops_used": chosen,
        "alpha_s_by_loops": {k: v["alpha_s"] for k, v in used.items()},
        "alpha_s_status_by_loops": {k: v["status"] for k, v in used.items()},
        "S_k": model.tolist(),
        "chi2_SB": chi2_sb,
        "chi2_Lambda": chi2_l,
        "diagnostic_best_fit_A": a_fit,
        "diagnostic_best_fit_A_sigma": a_err,
        "verdict": "SB_better" if chi2_sb < chi2_l else "baseline_better",
    }
