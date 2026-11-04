"""
Blind-path feature detectors.

This module is the only analysis executed when --blind is set.
It contains no preferred period and no theory-specific vocabulary.
Generic periodicity: strongest residual-periodogram peak + Fisher g /
local-continuum / bootstrap p-value. No target frequency is injected.

Forbidden-token grep for this file is listed in BLIND_ANALYSIS_PROTOCOL.md
(keep those tokens out of this module, including comments).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import periodogram

from menus.particle.casimir.series_integrity import clean_xy
from menus.particle.casimir.significance import (
    fit_smooth_trend,
    holm_adjust,
    hysteresis_pvalue,
    p_threshold,
    pelt_changepoints_with_bootstrap,
    residual_step_pvalue,
)


def generic_strongest_period(
    residuals: np.ndarray,
    x: np.ndarray,
    *,
    n_bootstrap: int = 200,
    seed: int = 17,
) -> dict[str, Any]:
    """Return the strongest residual period and its significance. No preferred f."""
    y = np.asarray(residuals, dtype=float)
    x = np.asarray(x, dtype=float)
    empty = {
        "p_value": 1.0,
        "p_local": 1.0,
        "p_fisher_g": 1.0,
        "p_bootstrap": 1.0,
        "strongest_frequency": None,
        "strongest_period": None,
        "peak_power": 0.0,
        "local_continuum": 0.0,
        "freqs": [],
        "power": [],
    }
    if y.size < 16:
        return empty

    dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
    fs = 1.0 / dx if dx > 0 else 1.0
    freqs, power = periodogram(y, fs=fs)
    if freqs.size < 4:
        return empty

    work_f = freqs[1:]
    work_p = power[1:]
    idx = int(np.argmax(work_p))
    f_peak = float(work_f[idx])
    i_peak = float(work_p[idx])
    period = (1.0 / f_peak) if f_peak > 0 else None

    half = max(3, work_p.size // 20)
    lo = max(0, idx - 3 * half)
    hi = min(work_p.size, idx + 3 * half)
    mask = np.ones(work_p.size, dtype=bool)
    mask[max(0, idx - 2) : min(work_p.size, idx + 3)] = False
    neighborhood = work_p[lo:hi][mask[lo:hi]]
    if neighborhood.size < 4:
        neighborhood = work_p[mask] if mask.any() else work_p
    continuum = float(np.median(neighborhood))
    if continuum <= 0:
        continuum = float(np.mean(work_p)) or 1e-30
    p_local = float(np.exp(-i_peak / continuum))

    total = float(np.sum(work_p)) or 1e-30
    g_obs = float(np.max(work_p) / total)
    p_fisher = _fisher_g_pvalue(g_obs, int(work_p.size))

    rng = np.random.default_rng(int(seed))
    exceed = 0
    scale = float(np.std(y)) or 1.0
    for _ in range(int(n_bootstrap)):
        boot = rng.normal(0.0, scale, size=y.size)
        _, boot_p = periodogram(boot, fs=fs)
        if boot_p.size > 1 and float(np.max(boot_p[1:])) >= i_peak:
            exceed += 1
    p_boot = (exceed + 1.0) / (n_bootstrap + 1.0)

    from tav_shared.bound_unbound_energy import partition_periodogram_totals

    energy_split = partition_periodogram_totals(work_f, work_p, f_bound=float(f_peak))
    return {
        "p_value": float(min(p_local, p_fisher, p_boot)),
        "p_local": p_local,
        "p_fisher_g": p_fisher,
        "p_bootstrap": p_boot,
        "fisher_g": g_obs,
        "strongest_frequency": f_peak,
        "strongest_period": period,
        "peak_power": i_peak,
        "local_continuum": continuum,
        "snr_local": float(i_peak / continuum),
        "n_freq_bins": int(work_p.size),
        "freqs": work_f.tolist(),
        "power": work_p.tolist(),
        "E_bound_clean": energy_split["E_bound_clean"],
        "E_unbound_clean": energy_split["E_unbound_clean"],
        "bound_flag_clean": energy_split["bound_flag_clean"],
        "f_b": energy_split["f_b"],
        "f_u": energy_split["f_u"],
        "energy_ledger": "content",
        "note_energy": (
            "Blind peak is not automatically BOUND. E_bound_clean is only discrete "
            "power that passes T1–T5 vs local continuum; the rest is E_unbound_clean."
        ),
    }


def _fisher_g_pvalue(g_obs: float, n: int) -> float:
    if n < 2 or g_obs <= 0:
        return 1.0
    if g_obs >= 1.0:
        return 0.0
    kmax = min(int(np.floor(1.0 / g_obs)), n)
    pval = 0.0
    sign = 1.0
    for k in range(1, kmax + 1):
        term = sign * _binom_coef(n, k) * (1.0 - k * g_obs) ** (n - 1)
        pval += term
        sign *= -1.0
    return float(min(max(pval, 0.0), 1.0))


def _binom_coef(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    k = min(k, n - k)
    out = 1.0
    for i in range(k):
        out *= (n - i) / (i + 1)
    return out


def detect_generic_signatures(
    df,
    param_col: str | None = None,
    value_col: str | None = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Blind detector: trend removal + generic peak search. No preferred period."""
    from menus.particle.casimir.scanner import _auto_select_columns

    results: dict[str, Any] = {
        "detected_features": [],
        "param_col": None,
        "value_col": None,
        "diagnostics": {},
        "analysis_mode": "blind",
    }
    if param_col is None or value_col is None:
        param_col, value_col = _auto_select_columns(df)
        if verbose:
            print(f"Auto-selected param={param_col}, value={value_col}")
    results["param_col"] = param_col
    results["value_col"] = value_col

    x = df[param_col].values.astype(float)
    y = df[value_col].values.astype(float)
    x_clean, y_clean = clean_xy(x, y)

    trend_fit = fit_smooth_trend(x_clean, y_clean)
    residuals = np.asarray(trend_fit["residuals"], dtype=float)
    results["diagnostics"]["trend"] = {
        "model": trend_fit["model"],
        "rmse": trend_fit.get("rmse"),
        "params": trend_fit.get("params") or {},
        "trend": np.asarray(trend_fit["trend"], dtype=float).tolist(),
        "residuals": residuals.tolist(),
    }

    spec = generic_strongest_period(residuals, x_clean)
    results["diagnostics"]["freqs"] = spec.get("freqs") or []
    results["diagnostics"]["power"] = spec.get("power") or []
    results["diagnostics"]["residual_periodogram"] = {
        k: spec[k] for k in spec if k not in {"freqs", "power"}
    }

    p_period = float(spec.get("p_value") or 1.0)
    results["detected_features"].append(
        {
            "type": "periodicity_search",
            "p_value": p_period,
            "p_local": spec.get("p_local"),
            "p_fisher_g": spec.get("p_fisher_g"),
            "p_bootstrap": spec.get("p_bootstrap"),
            "strongest_frequency": spec.get("strongest_frequency"),
            "strongest_period": spec.get("strongest_period"),
            "peak_power": spec.get("peak_power"),
            "local_continuum": spec.get("local_continuum"),
            "snr_local": spec.get("snr_local"),
            "trend_model": trend_fit["model"],
            "note": "Strongest residual-periodogram peak; no preferred period",
        }
    )

    step = residual_step_pvalue(residuals)
    step_locations = (
        x_clean[step["indices"] + 1] if step["n_steps"] and len(x_clean) > 1 else np.array([])
    )
    p_steps = float(step["p_value"])
    results["detected_features"].append(
        {
            "type": "residual_steps",
            "p_value": p_steps,
            "locations": step_locations[:10].tolist() if hasattr(step_locations, "tolist") else [],
            "n_steps": step["n_steps"],
            "note": "Residual increment exceedances (no alignment to a preferred grid)",
        }
    )

    hyst = hysteresis_pvalue(residuals)
    p_hyst = float(hyst["p_value"])
    results["detected_features"].append(
        {
            "type": "residual_asymmetry",
            "p_value": p_hyst,
            "magnitude": hyst.get("magnitude"),
            "n_perm": hyst.get("n_perm"),
            "note": "Permutation test of residual half-sweep asymmetry",
        }
    )

    cp = pelt_changepoints_with_bootstrap(residuals, x_clean)
    p_cp = float(cp.get("p_value") or 1.0)
    results["diagnostics"]["changepoints"] = cp
    results["detected_features"].append(
        {
            "type": "changepoints",
            "p_value": p_cp,
            "locations": cp.get("locations") or [],
            "n_breaks": cp.get("n_breaks") or 0,
            "method": cp.get("method"),
            "penalty": cp.get("penalty"),
            "penalty_formula": cp.get("penalty_formula"),
            "bootstrap_intervals": cp.get("bootstrap_intervals") or [],
            "note": cp.get("note") or "PELT on residuals with bootstrap location CI",
        }
    )

    alpha = p_threshold()
    p_raw = [p_period, p_steps, p_hyst, p_cp]
    p_holm = holm_adjust(p_raw)
    holm_by_type = {
        "periodicity_search": p_holm[0],
        "residual_steps": p_holm[1],
        "residual_asymmetry": p_holm[2],
        "changepoints": p_holm[3],
    }
    for feat in results["detected_features"]:
        if feat["type"] in holm_by_type:
            feat["p_value_holm"] = holm_by_type[feat["type"]]

    results["diagnostics"]["p_values"] = {"raw": {
        "periodicity_search": p_period,
        "residual_steps": p_steps,
        "residual_asymmetry": p_hyst,
        "changepoints": p_cp,
    }, "holm": holm_by_type, "alpha": alpha}

    raw_summary = {
        feat["type"]: bool(float(feat.get("p_value_holm", feat.get("p_value", 1.0))) < alpha)
        for feat in results["detected_features"]
    }
    results["raw_signature_summary"] = raw_summary
    results["signature_summary"] = dict(raw_summary)
    results["diagnostics"]["raw_signature_summary"] = raw_summary
    results["tav_harmonics"] = None
    return results
