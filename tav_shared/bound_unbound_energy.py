"""
Bound vs unbound energy partition for TAU-SB.

Convention (content, spectral): E_b := observed power/energy in discrete
bound bins that pass T1–T5. Continuum / free / radiative remainder is E_u.
Deficit convention is available via ``ledger="deficit"`` (E_b := ΔE_bind).

Raw inputs are never overwritten. Clean fields: E_bound_clean, E_unbound_clean,
bound_flag_clean, delta_e_bind, f_b, f_u, theta_flag, S.
"""

from __future__ import annotations

from typing import Any

import numpy as np

S_MIN_DEFAULT: float = 3.0
N_7: int = 7
W_K_WEIGHT6: int = 6

SYMBOL_KEY: dict[str, str] = {
    "E_b": "bound energy locked in a hierarchical binding well",
    "E_u": "unbound continuum / free / radiative / kinetic energy",
    "E_tot": "raw total; E_tot = E_b + E_u + E_noise",
    "E_free": "energy of the same d.o.f. if fully unbound",
    "ΔE_bind": "binding residual E_free − E_observed; bound requires ΔE_bind > 0",
    "E_th": "continuum / unbinding / ionization threshold",
    "E_noise": "noise/background not booked as bound",
    "w_k": "spectral weight of harmonic k (Weight-6: k=6)",
    "n_7": "7-fold Tav resonance index",
    "τ_bind": "binding / coherence lifetime",
    "f_b": "bound fraction E_b / E_tot after clean split",
    "f_u": "unbound fraction E_u / E_tot after clean split",
    "θ_flag": "boolean bound-state flag after T1–T5",
    "S": "significance of ΔE_bind in σ",
    "S_min": "minimum significance (default 3)",
    "B_contam": "sites that mixed E_u into E_b",
}


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def partition_energy(
    e_observed: np.ndarray,
    *,
    e_free: np.ndarray | None = None,
    e_th: np.ndarray | float | None = None,
    is_discrete_harmonic: np.ndarray | bool | None = None,
    tau_bind_ok: np.ndarray | bool | None = None,
    from_total_aggregator: bool = False,
    significance: np.ndarray | float | None = None,
    s_min: float = S_MIN_DEFAULT,
    e_noise: np.ndarray | float = 0.0,
    ledger: str = "content",
) -> dict[str, Any]:
    """
    Split observed energy into bound / unbound / ambiguous.

    ``ledger='content'``: E_b := E_observed on bound rows.
    ``ledger='deficit'``: E_b := ΔE_bind on bound rows.
    AMBIGUOUS (unevaluable tests) never stays in the bound ledger.
    """
    if ledger not in {"content", "deficit"}:
        raise ValueError("ledger must be 'content' or 'deficit'")

    e_obs = _as_float_array(e_observed)
    n = int(e_obs.size)
    noise = np.broadcast_to(_as_float_array(e_noise), (n,)) if n else np.asarray([], dtype=float)

    if e_free is None:
        e_free_arr = np.full(n, np.nan)
        t3_known = False
        delta = np.full(n, np.nan)
    else:
        e_free_arr = np.broadcast_to(_as_float_array(e_free), (n,))
        t3_known = True
        delta = e_free_arr - e_obs

    if e_th is None:
        t2_known = False
        t2 = np.zeros(n, dtype=bool)
    else:
        t2_known = True
        t2 = e_obs < np.broadcast_to(_as_float_array(e_th), (n,))

    if is_discrete_harmonic is None:
        t1_known = False
        t1 = np.zeros(n, dtype=bool)
    else:
        t1_known = True
        t1 = np.broadcast_to(np.asarray(is_discrete_harmonic, dtype=bool).ravel(), (n,))

    if tau_bind_ok is None:
        t4_known = False
        t4 = np.zeros(n, dtype=bool)
    else:
        t4_known = True
        t4 = np.broadcast_to(np.asarray(tau_bind_ok, dtype=bool).ravel(), (n,))

    t5 = not bool(from_total_aggregator)
    t5_known = True

    if significance is None:
        if t3_known:
            sigma = np.maximum(np.abs(noise), 1e-30)
            s_arr = np.where(np.isfinite(delta), delta / sigma, np.nan)
            t3s_known = True
        else:
            s_arr = np.full(n, np.nan)
            t3s_known = False
    else:
        s_arr = np.broadcast_to(_as_float_array(significance), (n,))
        t3s_known = True

    t3 = t3_known & np.isfinite(delta) & (delta > 0)
    if t3s_known:
        t3 = t3 & np.isfinite(s_arr) & (s_arr >= float(s_min))

    tests_known = bool(t1_known and t2_known and t3_known and t4_known and t5_known)
    theta = t1 & t2 & t3 & t4 & np.full(n, t5)
    ambiguous = np.full(n, not tests_known, dtype=bool)
    # Unevaluable tests never stay in the bound ledger.
    theta = theta & ~ambiguous

    if ledger == "deficit":
        e_b = np.where(theta, np.clip(np.nan_to_num(delta, nan=0.0), 0.0, None), 0.0)
    else:
        e_b = np.where(theta, e_obs, 0.0)
    e_tot = e_obs
    e_u = np.clip(e_tot - e_b - noise, 0.0, None)
    tot = np.maximum(e_tot, 1e-30)
    return {
        "ledger": ledger,
        "E_bound_clean": e_b,
        "E_unbound_clean": e_u,
        "bound_flag_clean": theta.astype(bool),
        "theta_flag": theta.astype(bool),
        "delta_e_bind": delta,
        "S": s_arr,
        "f_b": e_b / tot,
        "f_u": e_u / tot,
        "ambiguous": ambiguous,
        "n_bound": int(np.sum(theta)),
        "n_unbound": int(np.sum(~theta & ~ambiguous)),
        "n_ambiguous": int(np.sum(ambiguous)),
        "sum_E_b": float(np.sum(e_b)),
        "sum_E_u": float(np.sum(e_u)),
        "sum_E_tot": float(np.sum(e_tot)),
        "tests_fully_known": bool(tests_known),
        "t1_discrete": t1,
        "t2_threshold": t2,
        "t3_residual": t3,
        "t4_stability": t4,
        "t5_provenance": np.full(n, t5),
    }


def partition_periodogram(
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    f_bound: float,
    df_tol: float | None = None,
    s_min: float = S_MIN_DEFAULT,
    ledger: str = "content",
) -> dict[str, Any]:
    """
    Split a periodogram into discrete-harmonic bound power vs continuum.

    T1: bin within one df of f_bound (or 1/7 family).
    T2: power below Nyquist-dominated max is not required; threshold is
        local continuum (E_th := 2 × median continuum).
    T3: ΔE_bind := power − continuum; S := ΔE_bind / continuum.
    T4: treated True for a static spectrum (no decay tag).
    T5: False if this vector is an unsplit total (caller sets from_aggregator).
    """
    f = _as_float_array(freqs)
    p = _as_float_array(power)
    if f.size != p.size or f.size == 0:
        empty = partition_energy(np.zeros(0), ledger=ledger)
        empty["note"] = "empty or mismatched periodogram"
        return empty
    if df_tol is None:
        diffs = np.diff(np.sort(f))
        diffs = diffs[diffs > 0]
        df_tol = float(np.median(diffs)) if diffs.size else abs(float(f_bound)) * 0.05
    discrete = np.abs(f - float(f_bound)) <= (df_tol + 1e-15)
    continuum = float(np.median(p[~discrete])) if np.any(~discrete) else float(np.median(p))
    if not np.isfinite(continuum) or continuum <= 0:
        continuum = float(np.mean(p)) or 1e-30
    # Spectral T3 uses continuum-subtracted excess, not a missing E_free formula.
    excess = p - continuum
    s_arr = excess / max(continuum, 1e-30)
    part = partition_energy(
        p,
        e_free=p + np.maximum(excess, 0.0),
        e_th=np.inf,
        is_discrete_harmonic=discrete,
        tau_bind_ok=True,
        from_total_aggregator=False,
        significance=s_arr,
        s_min=s_min,
        e_noise=0.0,
        ledger=ledger,
    )
    part["local_continuum"] = continuum
    part["f_bound"] = float(f_bound)
    part["df_tol"] = float(df_tol)
    part["delta_e_bind"] = excess
    return part


def partition_periodogram_totals(
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    f_bound: float,
    s_min: float = S_MIN_DEFAULT,
) -> dict[str, float | int | str]:
    """Scalar ledger for spectral aggregators that previously did np.sum(power)."""
    part = partition_periodogram(freqs, power, f_bound=f_bound, s_min=s_min, ledger="content")
    return {
        "ledger": "content",
        "E_bound_clean": float(part["sum_E_b"]),
        "E_unbound_clean": float(part["sum_E_u"]),
        "bound_flag_clean": bool(part["n_bound"] > 0),
        "f_b": float(part["sum_E_b"] / max(part["sum_E_tot"], 1e-30)),
        "f_u": float(part["sum_E_u"] / max(part["sum_E_tot"], 1e-30)),
        "n_bound": int(part["n_bound"]),
        "n_unbound": int(part["n_unbound"]),
        "n_ambiguous": int(part["n_ambiguous"]),
        "sum_E_tot": float(part["sum_E_tot"]),
        "local_continuum": float(part.get("local_continuum") or 0.0),
    }
