#!/usr/bin/env python3
"""
TSB Casimir Test Framework — statistical falsification tests for TSB predictions.

Shape tests on one (x, y) pair, with mandatory audits/controls:

1. Discrete thresholds vs smooth Lifshitz
2. Hysteresis / sorted-sweep asymmetry (bootstrap; loop area if two-pass)
3. Hierarchical log-modulations (permutation p; never self-normalized χ²)
4. powerlaw_index (force/energy vs separation only) and anisotropy_channels
5. Periodicity with sampling-grid lock

Bound-energy residuals ΔE = E_obs − E_Lifshitz live in bound_energy.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import chi2, kruskal, norm, t as student_t

from menus.particle.casimir.diagnostics import (
    TSB_EVIDENCE_ALPHA,
    TSB_TEST_RESULTS_DIR,
    classify_casimir_table,
    find_two_pass_columns,
    json_safe,
    load_family_lock,
    print_casimir_load_context,
    print_falsification_report,
    preregister_split,
    record_hash_ledger,
    score_s_tsb,
)
from menus.particle.casimir.scanner import (
    DATASETS_DIR,
    _auto_select_columns,
    discover_local_casimir_files,
    generate_mock_casimir_data,
    load_data,
    resolve_local_casimir_file,
)
from menus.particle.casimir.series_integrity import (
    build_raw_series_integrity,
    clean_xy,
    register_series_hash,
)
from menus.particle.casimir.v_ppr import (
    DAMPING_V_PPR,
    evolve_v_ppr_trajectory,
    latency_to_pN_scale,
)

FALSIFICATION_FIXTURE_DIR = DATASETS_DIR / "fixtures"
FALSIFICATION_FIXTURE_CSV = FALSIFICATION_FIXTURE_DIR / "tsb_falsification_mock.csv"
FALSIFICATION_FIXTURE_PARAM_COL = "magnetic_field_mT"
FALSIFICATION_FIXTURE_VALUE_COL = "force_pN"

# Re-export: tests and callers import TSB_EVIDENCE_ALPHA from this module.
__all__ = [
    "TSB_EVIDENCE_ALPHA",
    "TSB_TEST_RESULTS_DIR",
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _prepare_xy(
    df: pd.DataFrame,
    param_col: str | None,
    value_col: str | None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if param_col is None or value_col is None:
        param_col, value_col = _auto_select_columns(df)
    x = df[param_col].values.astype(float)
    y = df[value_col].values.astype(float)
    x, y = clean_xy(x, y)
    return x, y, param_col, value_col


def _finite_sigma(sigma: np.ndarray | None, n: int) -> np.ndarray | None:
    if sigma is None:
        return None
    arr = np.asarray(sigma, dtype=np.float64).ravel()
    if arr.size < n:
        return None
    arr = arr[:n]
    if not np.all(np.isfinite(arr) & (arr > 0)):
        return None
    return arr


def _not_applicable(reason: str) -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "p_value": None,
        "evidence_for_TSB": False,
        "counted_in_S_TSB": False,
        "not_applicable_reason": reason,
    }


def detect_steps(
    x: np.ndarray,
    y: np.ndarray,
    *,
    threshold_sigma: float = 2.0,
) -> tuple[np.ndarray, float, int]:
    """Prediction 1: discrete steps vs smooth Lifshitz."""
    _ = x
    dy = np.diff(y)
    std_dy = float(np.std(dy)) if len(dy) else 0.0
    if std_dy <= 0:
        return np.array([], dtype=int), 1.0, 0
    step_mask = np.abs(dy) > threshold_sigma * std_dy
    step_indices = np.where(step_mask)[0]
    n_steps = int(len(step_indices))
    expected_steps = max(len(dy) * 0.05, 1.0)
    z_score = (n_steps - expected_steps) / np.sqrt(expected_steps)
    p_value = float(1.0 - norm.cdf(z_score))
    return step_indices, p_value, n_steps


def _grid_locked_ratio(ratio: float, *, tol: float = 0.02) -> bool:
    if not np.isfinite(ratio) or ratio <= 0:
        return False
    nearest = int(round(ratio))
    if nearest < 1:
        return abs(ratio - 1.0) <= tol
    return abs(ratio - nearest) / float(nearest) <= tol


def test_periodicity(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Predictions 1 & 5: periodogram + sampling-grid lock."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 20:
        return {
            "status": "too_few_points",
            "p_value": 1.0,
            "evidence_for_TSB": False,
            "dominant_period": None,
            "Delta_xi": None,
            "dx": None,
            "ratio": None,
            "grid_locked": False,
            "fs": None,
        }
    dx = float(np.mean(np.diff(x)))
    delta_xi = dx
    fs = (1.0 / dx) if dx > 0 else 1.0
    freqs, power = signal.periodogram(y, fs=fs)
    peaks, _ = signal.find_peaks(power, height=float(np.max(power)) * 0.2 if np.max(power) > 0 else 0.0)
    if len(peaks) == 0 or not np.isfinite(np.max(power)):
        return {
            "status": "no_peak",
            "p_value": 1.0,
            "evidence_for_TSB": False,
            "dominant_period": None,
            "Delta_xi": delta_xi,
            "dx": dx,
            "ratio": None,
            "grid_locked": False,
            "fs": fs,
        }
    dominant_freq = float(freqs[peaks[int(np.argmax(power[peaks]))]])
    period = (1.0 / dominant_freq) if dominant_freq > 0 else float("inf")
    ratio = (period / delta_xi) if delta_xi > 0 and np.isfinite(period) else None
    grid_locked = bool(ratio is not None and _grid_locked_ratio(float(ratio)))
    p_value = float(1.0 - chi2.cdf(float(np.max(power)), df=2))
    evidence = bool(p_value < TSB_EVIDENCE_ALPHA) and not grid_locked
    reason = None
    if grid_locked:
        reason = "period matches sampling grid, not a physical modulation"
        evidence = False
    return {
        "status": "ok",
        "p_value": p_value,
        "evidence_for_TSB": evidence,
        "dominant_period": None if not np.isfinite(period) else float(period),
        "Delta_xi": delta_xi,
        "dx": dx,
        "ratio": None if ratio is None else float(ratio),
        "grid_locked": grid_locked,
        "grid_locked_reason": reason,
        "fs": fs,
        "dominant_frequency": dominant_freq,
    }


def _hysteresis_point(y: np.ndarray) -> tuple[float, float]:
    if len(y) < 10:
        return 0.0, 1.0
    mid = len(y) // 2
    left = y[:mid]
    right = y[mid:][::-1]
    n = min(len(left), len(right))
    if n <= 0:
        return 0.0, 1.0
    diff = float(np.mean(np.abs(left[:n] - right[:n])))
    std_y = float(np.std(y))
    effect_size = diff / std_y if std_y > 0 else 0.0
    p_value = float(1.0 - norm.cdf(effect_size))
    return effect_size, p_value


def _block_bootstrap_y(y: np.ndarray, rng: np.random.Generator, *, block: int) -> np.ndarray:
    n = len(y)
    block = max(1, min(int(block), n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
    chunks = [y[s : s + block] for s in starts]
    out = np.concatenate(chunks)[:n]
    return out


def _loop_area(x: np.ndarray, y_fwd: np.ndarray, y_rev: np.ndarray) -> float:
    n = min(len(x), len(y_fwd), len(y_rev))
    if n < 3:
        return 0.0
    xx, a, b = x[:n], y_fwd[:n], y_rev[:n]
    order = np.argsort(xx)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(a[order] - b[order], xx[order]))


def test_hysteresis(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 7,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Prediction 2: hysteresis.

    Default statistic is a shape-asymmetry of a *sorted* sweep, NOT a measured
    up/down Casimir loop. If the table has a real two-pass pair (column names
    containing up/down, forward/reverse, retract/approach), the loop area is
    used as e_hyst.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fwd_col, rev_col = find_two_pass_columns(df)
    two_pass = bool(fwd_col and rev_col and df is not None)
    mode = "sorted_sweep_shape_asymmetry"
    note = (
        "shape-asymmetry of a sorted sweep, NOT a measured up/down Casimir loop"
    )
    e_hyst, p_point = _hysteresis_point(y)
    if two_pass:
        y_fwd = pd.to_numeric(df[fwd_col], errors="coerce").to_numpy(dtype=float)
        y_rev = pd.to_numeric(df[rev_col], errors="coerce").to_numpy(dtype=float)
        n = min(len(x), len(y_fwd), len(y_rev))
        area = _loop_area(x[:n], y_fwd[:n], y_rev[:n])
        scale = float(np.std(y_fwd[:n])) or 1.0
        e_hyst = float(abs(area) / (scale * max(np.ptp(x[:n]), 1e-30)))
        p_point = float(1.0 - norm.cdf(e_hyst))
        mode = "two_pass_loop_area"
        note = f"measured two-pass loop area from columns {fwd_col!r} / {rev_col!r}"

    rng = np.random.default_rng(seed)
    block = max(4, len(y) // 20) if len(y) else 1
    effects: list[float] = []
    p_boot: list[float] = []
    n_boot = max(0, int(n_bootstrap))
    for _ in range(n_boot):
        if two_pass and df is not None:
            idx = rng.integers(0, len(y), size=len(y))
            eb, pb = _hysteresis_point(y[idx])
        else:
            yb = _block_bootstrap_y(y, rng, block=block)
            eb, pb = _hysteresis_point(yb)
        effects.append(eb)
        p_boot.append(pb)
    if effects:
        lo, hi = np.percentile(effects, [2.5, 97.5])
        p_lo, p_hi = np.percentile(p_boot, [2.5, 97.5])
    else:
        lo = hi = e_hyst
        p_lo = p_hi = p_point
    evidence = bool(p_point < TSB_EVIDENCE_ALPHA)
    return {
        "status": "ok",
        "effect_size": float(e_hyst),
        "e_hyst": float(e_hyst),
        "p_value": float(p_point),
        "p_point": float(p_point),
        "effect_size_ci_lo": float(lo),
        "effect_size_ci_hi": float(hi),
        "p_ci_lo": float(p_lo),
        "p_ci_hi": float(p_hi),
        "n_bootstrap": n_boot,
        "bootstrap_seed": seed,
        "mode": mode,
        "note": note,
        "two_pass_columns": [fwd_col, rev_col] if two_pass else None,
        "evidence_for_TSB": evidence,
    }


def _sin_model(t: np.ndarray, a: float, f: float, phi: float, c: float) -> np.ndarray:
    return a * np.sin(2.0 * np.pi * f * t + phi) + c


def _fit_log_sine(log_x: np.ndarray, y: np.ndarray) -> dict[str, Any] | None:
    """Four-parameter sine in log10(x). Residuals are never divided by std(r)."""
    try:
        rms = float(np.std(y)) or 1.0
        y_mean = float(np.mean(y))
        p0 = [rms, 1.0 / 7.0, 0.0, y_mean]
        bounds = (
            [-10.0 * rms, 1e-6, -np.pi, y_mean - 10.0 * rms],
            [10.0 * rms, 20.0, np.pi, y_mean + 10.0 * rms],
        )
        popt, _ = curve_fit(_sin_model, log_x, y, p0=p0, bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError, TypeError):
        # Linear fallback at f = 1/7 (A sin + B cos + c); still not a χ² p-value.
        t = log_x
        f0 = 1.0 / 7.0
        design = np.column_stack(
            [np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t), np.ones_like(t)]
        )
        try:
            coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:
            return None
        a = float(np.hypot(coef[0], coef[1]))
        phi = float(np.arctan2(coef[1], coef[0]))
        c = float(coef[2])
        f = f0
        fitted = _sin_model(t, a, f, phi, c)
        resid = y - fitted
        return {
            "a": a,
            "f": f,
            "phi": phi,
            "c": c,
            "chi2_raw": float(np.sum(np.square(resid))),
            "N": int(len(y)),
            "residuals": resid,
            "status": "linear_fallback",
        }
    a, f, phi, c = (float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]))
    fitted = _sin_model(log_x, a, f, phi, c)
    resid = y - fitted
    return {
        "a": a,
        "f": f,
        "phi": phi,
        "c": c,
        "chi2_raw": float(np.sum(np.square(resid))),
        "N": int(len(y)),
        "residuals": resid,
        "status": "ok",
    }


def _linear_amp_at_f(log_x: np.ndarray, y: np.ndarray, f: float) -> float:
    design = np.column_stack(
        [np.sin(2 * np.pi * f * log_x), np.cos(2 * np.pi * f * log_x), np.ones_like(log_x)]
    )
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(np.hypot(coef[0], coef[1]))


def _drude_like_surrogate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    xpos = np.maximum(xx, np.min(np.abs(xx[np.isfinite(xx)])) if np.any(np.isfinite(xx)) else 1.0)
    x0 = float(np.median(xpos)) or 1.0
    tail = 1.0 / (1.0 + np.square(xpos / x0))
    # Decreasing Drude-like tail scaled to y; no sine planted.
    t_norm = (tail - float(np.min(tail))) / (float(np.ptp(tail)) or 1.0)
    y_lo, y_hi = float(np.min(yy)), float(np.max(yy))
    decreasing = True
    if len(xx) > 2:
        decreasing = bool(np.polyfit(xx, yy, 1)[0] <= 0)
    if decreasing:
        return y_lo + (1.0 - t_norm) * (y_hi - y_lo)
    return y_lo + t_norm * (y_hi - y_lo)


def test_log_modulations(
    x: np.ndarray,
    y: np.ndarray,
    *,
    y_sigma: np.ndarray | None = None,
    n_perm: int = 199,
    seed: int = 20260915,
) -> dict[str, Any]:
    """
    Prediction 3: hierarchical Δn modulations in log-separation.

    Never divides residuals by their own sample std for a p-value.
    Returns the raw fit (a, f, phi, c, chi2_raw, N) plus a mandatory audit:
    (i) data, (ii) y shuffled vs fixed x, (iii) monotone Drude-like surrogate.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pos_mask = x > 0
    n_pos = int(np.sum(pos_mask))
    if n_pos < 10:
        return {
            "status": "too_few_points",
            "p_value": 1.0,
            "evidence_for_TSB": False,
            "identically_null": True,
            "a": None,
            "f": None,
            "phi": None,
            "c": None,
            "chi2_raw": None,
            "N": n_pos,
        }
    log_x = np.log10(x[pos_mask])
    y_pos = y[pos_mask]
    n = int(len(y_pos))
    sigma = _finite_sigma(None if y_sigma is None else np.asarray(y_sigma)[pos_mask], n)

    fit_data = _fit_log_sine(log_x, y_pos)
    if fit_data is None:
        return {
            "status": "fit_failed",
            "p_value": 1.0,
            "evidence_for_TSB": False,
            "identically_null": True,
            "a": None,
            "f": None,
            "phi": None,
            "c": None,
            "chi2_raw": None,
            "N": n,
        }

    rng = np.random.default_rng(seed)
    y_shuffle = rng.permutation(y_pos)
    fit_shuffle = _fit_log_sine(log_x, y_shuffle) or fit_data
    y_surr = _drude_like_surrogate(x[pos_mask], y_pos)
    fit_surr = _fit_log_sine(log_x, y_surr) or fit_data

    chi2_raw = float(fit_data["chi2_raw"])
    chi2_shuffle = float(fit_shuffle["chi2_raw"])
    mean_chi2 = chi2_raw / max(n, 1)
    mean_chi2_shuffle = chi2_shuffle / max(n, 1)
    s_resid = float(np.std(fit_data["residuals"], ddof=0))
    rms_y = float(np.sqrt(np.mean(np.square(y_pos)))) or 1.0
    effect_size = abs(float(fit_data["a"])) / rms_y

    # Permutation p-value from |a| at the data frequency (linear, not self-χ²).
    f_hat = float(fit_data["f"]) if np.isfinite(fit_data["f"]) else (1.0 / 7.0)
    stat_data = abs(float(fit_data["a"]))
    n_perm = max(int(n_perm), 1)
    exceed = 0
    for _ in range(n_perm):
        amp = _linear_amp_at_f(log_x, rng.permutation(y_pos), f_hat)
        if amp >= stat_data:
            exceed += 1
    p_perm = float((1 + exceed) / (n_perm + 1))
    stat_shuffle = abs(float(fit_shuffle["a"]))
    exceed_sh = 0
    rng2 = np.random.default_rng(seed + 1)
    for _ in range(n_perm):
        amp = _linear_amp_at_f(log_x, rng2.permutation(y_pos), f_hat)
        if amp >= stat_shuffle:
            exceed_sh += 1
    p_shuffle = float((1 + exceed_sh) / (n_perm + 1))

    p_analytic: float | None = None
    if sigma is not None:
        chi2_weighted = float(np.sum(np.square(fit_data["residuals"] / sigma)))
        p_analytic = float(1.0 - chi2.cdf(chi2_weighted, df=max(n - 4, 1)))
        p_value = p_analytic
        p_method = "external_sigma_chi2"
    else:
        p_value = p_perm
        p_method = "permutation"
        chi2_weighted = None

    identically_null = bool(
        round(float(p_value), 3) == round(float(p_shuffle), 3)
        or round(mean_chi2, 3) == round(mean_chi2_shuffle, 3)
    )
    evidence = bool(p_value < TSB_EVIDENCE_ALPHA) and not identically_null
    return {
        "status": str(fit_data["status"]),
        "a": float(fit_data["a"]),
        "f": float(fit_data["f"]),
        "phi": float(fit_data["phi"]),
        "c": float(fit_data["c"]),
        "chi2_raw": chi2_raw,
        "chi2_weighted": chi2_weighted,
        "N": n,
        "s_resid": s_resid,
        "s_resid_note": "np.std(residuals, ddof=0); MUST NOT be used as chi2 sigma",
        "effect_size": effect_size,
        "p_value": float(p_value),
        "p_method": p_method,
        "p_permutation": p_perm,
        "p_analytic_external_sigma": p_analytic,
        "p_shuffle": float(p_shuffle),
        "shuffle_seed": seed,
        "identically_null": identically_null,
        "audit": {
            "data": {
                "a": float(fit_data["a"]),
                "f": float(fit_data["f"]),
                "chi2_raw": chi2_raw,
                "chi2_raw_over_N": mean_chi2,
            },
            "shuffle": {
                "a": float(fit_shuffle["a"]),
                "f": float(fit_shuffle["f"]),
                "chi2_raw": chi2_shuffle,
                "chi2_raw_over_N": mean_chi2_shuffle,
                "seed": seed,
            },
            "monotone_surrogate": {
                "kind": "decreasing_drude_like_tail",
                "a": float(fit_surr["a"]),
                "f": float(fit_surr["f"]),
                "chi2_raw": float(fit_surr["chi2_raw"]),
                "chi2_raw_over_N": float(fit_surr["chi2_raw"]) / max(n, 1),
            },
        },
        "evidence_for_TSB": evidence,
    }


def _ols_slope(log_x: np.ndarray, log_y: np.ndarray) -> tuple[float, float, int]:
    n = int(len(log_x))
    if n < 3:
        return float("nan"), float("nan"), n
    x_mean = float(np.mean(log_x))
    y_mean = float(np.mean(log_y))
    dx = log_x - x_mean
    sxx = float(np.sum(dx * dx))
    if sxx <= 0:
        return float("nan"), float("nan"), n
    m = float(np.sum(dx * (log_y - y_mean)) / sxx)
    intercept = y_mean - m * x_mean
    resid = log_y - (intercept + m * log_x)
    s2 = float(np.sum(resid * resid) / max(n - 2, 1))
    se = float(np.sqrt(s2 / sxx)) if sxx > 0 else float("nan")
    return m, se, n


def test_powerlaw_index(
    x: np.ndarray,
    y: np.ndarray,
    *,
    param_col: str = "",
    value_col: str = "",
) -> dict[str, Any]:
    """
    Legal only when x is a separation/distance column AND y is force, energy,
    or pressure. Frequency / dielectric tables return not_applicable and are
    not counted in S_TSB.
    """
    from menus.particle.casimir.diagnostics import (
        expected_lifshitz_index,
        powerlaw_index_is_legal,
    )

    if not powerlaw_index_is_legal(param_col, value_col):
        return _not_applicable(
            "powerlaw_index requires x=separation/distance and y=force/energy/pressure; "
            f"got param_col={param_col!r} value_col={value_col!r}"
        )
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (x > 0) & np.isfinite(x) & np.isfinite(y) & (np.abs(y) > 0)
    if int(np.sum(mask)) < 10:
        return {
            "status": "too_few_points",
            "p_value": 1.0,
            "evidence_for_TSB": False,
            "m": None,
            "delta_4": None,
        }
    log_x = np.log(x[mask])
    log_y = np.log(np.abs(y[mask]))
    m, se, n = _ols_slope(log_x, log_y)
    m0 = expected_lifshitz_index(value_col)
    if m0 is None:
        m0 = -4.0
    delta = abs(float(m) - float(m0))
    delta_4 = abs(float(m) + 4.0)
    if np.isfinite(se) and se > 0 and np.isfinite(m):
        t_stat = (float(m) - float(m0)) / se
        p_value = float(2.0 * (1.0 - student_t.cdf(abs(t_stat), df=max(n - 2, 1))))
    else:
        t_stat = None
        p_value = 1.0
    # Consistent with the Lifshitz index → not TSB evidence.
    evidence = bool(p_value < TSB_EVIDENCE_ALPHA)
    return {
        "status": "ok",
        "m": float(m),
        "m_se": None if not np.isfinite(se) else float(se),
        "m0": float(m0),
        "delta_index": float(delta),
        "delta_4": float(delta_4),
        "t_stat": None if t_stat is None else float(t_stat),
        "p_value": float(p_value),
        "evidence_for_TSB": evidence,
        "note": (
            f"OLS log|y| vs log x compared to Lifshitz index {m0:g} "
            f"({'force ~ d^{-4}' if m0 == -4 else 'energy ~ d^{-3}'})"
        ),
    }


def test_anisotropy_channels(
    df: pd.DataFrame | None,
    *,
    primary: str | None = None,
    x: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Spread of auxiliary dielectric channels about eps_pri.

    Negative control (every run): broadcast eps_pri into every aux column and
    recompute. If the control still flags evidence_for_TSB, mark control_failed
    and force evidence_for_TSB = false.
    """
    _ = x
    from menus.particle.casimir.diagnostics import aux_dielectric_columns, dielectric_columns

    if df is None or df.empty:
        return _not_applicable("no table supplied for anisotropy_channels")
    diel = dielectric_columns(df)
    if len(diel) < 2:
        return _not_applicable(
            "anisotropy_channels requires two or more dielectric_response_* columns"
        )
    pri_name = primary if primary in df.columns else (
        "dielectric_response_primary" if "dielectric_response_primary" in df.columns else diel[0]
    )
    aux_names = aux_dielectric_columns(df, primary=pri_name)
    if not aux_names:
        aux_names = [c for c in diel if c != pri_name]
    if not aux_names:
        return _not_applicable("no auxiliary dielectric channels")

    eps_pri = pd.to_numeric(df[pri_name], errors="coerce").to_numpy(dtype=float)
    aux = [
        pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in aux_names
    ]
    n = min(len(eps_pri), *(len(a) for a in aux))
    eps_pri = eps_pri[:n]
    aux = [a[:n] for a in aux]
    mask = np.isfinite(eps_pri)
    for a in aux:
        mask &= np.isfinite(a)
    eps_pri = eps_pri[mask]
    aux = [a[mask] for a in aux]
    if len(eps_pri) < 8:
        return {
            "status": "too_few_points",
            "p_value": 1.0,
            "evidence_for_TSB": False,
            "control_failed": False,
        }

    def _spread(primary_arr: np.ndarray, channels: list[np.ndarray]) -> float:
        rms_pri = float(np.sqrt(np.mean(np.square(primary_arr)))) or 1.0
        acc = []
        for ch in channels:
            acc.append(np.square(ch - primary_arr))
        return float(np.sqrt(np.mean(np.concatenate(acc)))) / rms_pri

    def _kruskal_p(primary_arr: np.ndarray, channels: list[np.ndarray]) -> float:
        samples = [primary_arr, *channels]
        try:
            _, p_kw = kruskal(*samples)
            if not np.isfinite(p_kw):
                return 1.0
            return float(p_kw)
        except ValueError:
            return 1.0

    spread = _spread(eps_pri, aux)
    p_data = _kruskal_p(eps_pri, aux)
    evidence_data = bool(p_data < TSB_EVIDENCE_ALPHA) and spread > 0.0

    control_channels = [np.array(eps_pri, copy=True) for _ in aux]
    spread_ctrl = _spread(eps_pri, control_channels)
    p_ctrl = _kruskal_p(eps_pri, control_channels)
    evidence_ctrl = bool(p_ctrl < TSB_EVIDENCE_ALPHA) and spread_ctrl > 0.0
    control_failed = bool(evidence_ctrl)
    evidence = bool(evidence_data) and not control_failed
    if control_failed:
        evidence = False
    return {
        "status": "ok",
        "p_value": float(p_data),
        "spread": float(spread),
        "n_aux": len(aux_names),
        "aux_columns": list(aux_names),
        "primary": pri_name,
        "control_spread": float(spread_ctrl),
        "control_p_value": float(p_ctrl),
        "control_failed": control_failed,
        "control_note": "broadcast eps_pri into every aux column",
        "evidence_for_TSB": evidence,
    }


def test_anisotropy_or_backreaction(
    x: np.ndarray,
    y: np.ndarray,
    *,
    df: pd.DataFrame | None = None,
    param_col: str = "",
    value_col: str = "",
) -> dict[str, Any]:
    """Deprecated combined name. Split into powerlaw_index and anisotropy_channels."""
    return {
        "deprecated_name": "anisotropy_or_backreaction",
        "powerlaw_index": test_powerlaw_index(
            x, y, param_col=param_col, value_col=value_col
        ),
        "anisotropy_channels": test_anisotropy_channels(df, primary=value_col, x=x),
        "note": "Do not use this combined name; score the two tests separately.",
    }


def run_full_test_suite(
    df: pd.DataFrame,
    *,
    param_col: str | None = None,
    value_col: str | None = None,
    output_prefix: str = "tsb_falsification",
    plot: bool = True,
    show_popup: bool = False,
    file_path: str | Path | None = None,
    data_label: str = "",
    include_bound_energy: bool = False,
    n_bootstrap: int = 1000,
    n_perm: int = 199,
    relock: bool = False,
) -> dict[str, Any]:
    """Run TSB falsification tests; write JSON, PNG, and TXT report."""
    TSB_TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    lock = preregister_split(relock=relock)

    x, y, pcol, vcol = _prepare_xy(df, param_col, value_col)
    meta = classify_casimir_table(df, pcol, vcol)
    integrity = build_raw_series_integrity(
        x,
        y,
        file_path=file_path,
        data_label=data_label or output_prefix,
    )
    register_series_hash(integrity)
    H = str(integrity.get("sha256") or "")
    print_casimir_load_context(
        observable_kind_value=str(meta["observable_kind"]),
        n_aux=int(meta["n_aux_dielectric_channels"]),
        applicable=list(meta["applicable_tests"]),
    )
    ledger = record_hash_ledger(
        data_label=data_label or output_prefix,
        path=file_path,
        H=H,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        stamp=stamp,
    )

    step_idx, p_steps, n_steps = detect_steps(x, y)
    hyst = test_hysteresis(x, y, n_bootstrap=n_bootstrap, df=df)
    v_ppr = evolve_v_ppr_trajectory(n_max=8, damping=DAMPING_V_PPR)
    v_ppr_pred = latency_to_pN_scale(v_ppr["topological_vacuum_latency"])
    v_ppr_cal = (
        float(hyst["effect_size"]) - v_ppr_pred
        if np.isfinite(v_ppr_pred)
        else float("nan")
    )
    hyst["v_ppr_damping"] = DAMPING_V_PPR
    hyst["v_ppr_topological_vacuum_latency"] = v_ppr["topological_vacuum_latency"]
    hyst["v_ppr_predicted_hysteresis_pN"] = v_ppr_pred
    hyst["v_ppr_calibration_residual_pN"] = v_ppr_cal
    hyst["v_ppr_calibration_residual_pN_note"] = (
        "V_PPR latency mapped to pN; calibration residual, NOT Delta_E"
    )
    hyst["v_ppr_a_terminal"] = {
        "a_7": v_ppr["a_terminal_7"],
        "a_8": v_ppr["a_terminal_8"],
    }
    period = test_periodicity(x, y)
    logm = test_log_modulations(x, y, n_perm=n_perm)
    plaw = test_powerlaw_index(x, y, param_col=pcol, value_col=vcol)
    aniso = test_anisotropy_channels(df, primary=vcol, x=x)

    results: dict[str, dict[str, Any]] = {
        "discrete_steps": {
            "n_steps": n_steps,
            "step_indices": np.asarray(step_idx).tolist(),
            "p_value": p_steps,
            "evidence_for_TSB": bool(p_steps < TSB_EVIDENCE_ALPHA),
        },
        "hysteresis": hyst,
        "periodicity": period,
        "log_modulations": logm,
        "powerlaw_index": plaw,
        "anisotropy_channels": aniso,
    }
    s_block = score_s_tsb(results)

    bound_block: dict[str, Any] | None = None
    if include_bound_energy:
        from menus.particle.casimir.bound_energy import estimate_bound_energy

        bound_block = estimate_bound_energy(
            df,
            param_col=pcol,
            value_col=vcol,
            data_label=data_label or output_prefix,
        )

    plot_path: str | None = None
    if plot:
        plot_path = str(
            _save_falsification_plot(
                x,
                y,
                pcol,
                vcol,
                step_idx,
                float(hyst.get("effect_size") or 0.0),
                float(hyst.get("p_point") or hyst.get("p_value") or 1.0),
                results,
                output_prefix=output_prefix,
                stamp=stamp,
            )
        )
        if show_popup and plot_path:
            img = plt.imread(plot_path)
            plt.figure(figsize=(15, 10))
            plt.imshow(img)
            plt.axis("off")
            plt.title(Path(plot_path).name)
            plt.show()

    json_path = TSB_TEST_RESULTS_DIR / f"{output_prefix}_report_{stamp}.json"
    txt_path = TSB_TEST_RESULTS_DIR / f"{output_prefix}_summary_{stamp}.txt"

    report = {
        "module": "tsb_casimir_test_framework",
        "timestamp": stamp,
        "param_col": pcol,
        "value_col": vcol,
        "n_points": int(len(x)),
        "alpha": TSB_EVIDENCE_ALPHA,
        "raw_series_sha256": H,
        "raw_series_integrity": integrity,
        "hash_ledger": ledger,
        "pool_allowed": bool(ledger.get("pool_allowed", True)),
        "observable_kind": meta["observable_kind"],
        "n_aux_dielectric_channels": meta["n_aux_dielectric_channels"],
        "applicable_tests": meta["applicable_tests"],
        "preregister": lock,
        "predictions": {
            "1_discrete_steps": "Discrete thresholds vs smooth Lifshitz",
            "2_hysteresis": "Arrow-of-time / dissipative asymmetry (bootstrap)",
            "3_log_modulations": "Hierarchical Δn modulations in log-x (permutation + audit)",
            "4a_powerlaw_index": "Force/energy vs separation index vs Lifshitz −4/−3",
            "4b_anisotropy_channels": "Aux dielectric spread vs eps_pri with identity control",
            "5_periodicity": "Periodogram with sampling-grid lock",
        },
        "results": results,
        "S_TSB": s_block["S_TSB"],
        "S_TSB_n_applicable": s_block["n_applicable"],
        "S_TSB_n_evidence": s_block["n_evidence"],
        "bound_energy": json_safe(bound_block) if bound_block else None,
        "plot_path": plot_path,
        "data_label": data_label or output_prefix,
        "file_path": str(file_path) if file_path else None,
    }
    json_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")

    lines = [
        "=== TSB Casimir Falsification Test Report ===",
        f"timestamp: {stamp}",
        f"columns: {pcol} vs {vcol}",
        f"observable_kind: {meta['observable_kind']}",
        f"n_points: {len(x)}",
        f"raw_series_sha256: {H}",
        "",
    ]
    for test, res in results.items():
        pval = res.get("p_value")
        p_txt = "null" if pval is None else f"{pval:.4g}"
        flag = "TSB evidence" if res.get("evidence_for_TSB") else "null-consistent"
        lines.append(f"{test}: p={p_txt} ({flag})")
    if s_block["S_TSB"] is not None:
        lines.append(f"S_TSB: {s_block['S_TSB']:.4g}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print_falsification_report(results, s_tsb=s_block["S_TSB"], bound_energy=bound_block)
    if plot_path:
        print(f"\nPlot saved to: {plot_path}")
    print(f"JSON: {json_path}")
    print(f"TXT:  {txt_path}")

    report["report_path"] = str(json_path)
    report["summary_path"] = str(txt_path)
    return report


def _save_falsification_plot(
    x: np.ndarray,
    y: np.ndarray,
    param_col: str,
    value_col: str,
    step_idx: np.ndarray,
    effect_hyst: float,
    p_hyst: float,
    results: dict[str, dict[str, Any]],
    *,
    output_prefix: str,
    stamp: str,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    axes_flat[0].plot(x, y, "b.-", alpha=0.7)
    axes_flat[0].set_title(f"Raw Data ({param_col} vs {value_col})")
    axes_flat[0].set_xlabel(param_col)
    axes_flat[0].set_ylabel(value_col)

    axes_flat[1].plot(x[1:], np.diff(y), "r.-")
    for idx in step_idx:
        axes_flat[1].axvline(x[idx], color="green", linestyle="--", alpha=0.5)
    axes_flat[1].set_title("Derivative + Detected Steps")

    if len(x) > 20:
        freqs, power = signal.periodogram(y)
        axes_flat[2].semilogy(freqs, power)
    per = results.get("periodicity") or {}
    axes_flat[2].set_title(
        "Periodogram"
        + (
            f"  P={per.get('dominant_period')} grid_locked={per.get('grid_locked')}"
            if per
            else ""
        )
    )

    hyst = results.get("hysteresis") or {}
    axes_flat[3].text(
        0.5,
        0.5,
        (
            f"Hysteresis effect: {effect_hyst:.3f}\n"
            f"p_point={p_hyst:.3g}\n"
            f"CI=[{hyst.get('effect_size_ci_lo')}, {hyst.get('effect_size_ci_hi')}]\n"
            f"{hyst.get('mode', '')}"
        ),
        transform=axes_flat[3].transAxes,
        ha="center",
        va="center",
        fontsize=9,
    )
    axes_flat[3].set_title("Hysteresis Test")
    axes_flat[3].axis("off")

    if len(x) > 10 and np.all(x > 0):
        axes_flat[4].plot(np.log10(x), y, "g.-")
    axes_flat[4].set_title("Log-scale Data (hierarchical modulations)")
    axes_flat[4].set_xlabel(f"log10({param_col})")

    def _p_txt(res: dict[str, Any]) -> str:
        pval = res.get("p_value")
        if pval is None:
            return "null"
        try:
            return f"{float(pval):.3g}"
        except (TypeError, ValueError):
            return str(pval)

    summary_text = "\n".join(
        f"{k}: p={_p_txt(v)}"
        + (" (TSB)" if v.get("evidence_for_TSB") else "")
        + (" n/a" if v.get("status") == "not_applicable" else "")
        for k, v in results.items()
    )
    axes_flat[5].text(
        0.1,
        0.5,
        summary_text,
        transform=axes_flat[5].transAxes,
        fontsize=9,
        va="center",
    )
    axes_flat[5].set_title("TSB Falsification Summary")
    axes_flat[5].axis("off")

    fig.tight_layout()
    out = TSB_TEST_RESULTS_DIR / f"{output_prefix}_report_{stamp}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def ensure_falsification_fixture_csv(*, force: bool = False) -> Path:
    """
    Write (or reuse) a synthetic Casimir sweep CSV for the falsification suite.

    The file lives under datasets/casimir/fixtures/ and is auto-injected by the
    menu pre-form hook so no manual path entry is required.
    """
    FALSIFICATION_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    if FALSIFICATION_FIXTURE_CSV.is_file() and not force:
        return FALSIFICATION_FIXTURE_CSV
    mock_df = generate_mock_casimir_data()
    mock_df.to_csv(FALSIFICATION_FIXTURE_CSV, index=False)
    print(f"[Superblock Residual Engine] Falsification fixture CSV: {FALSIFICATION_FIXTURE_CSV}")
    return FALSIFICATION_FIXTURE_CSV


def falsification_default_options() -> dict[str, str]:
    """Menu/runner defaults: prefer downloaded tables, else a generated fixture."""
    found = discover_local_casimir_files()
    if found:
        return {
            "local_csv": str(found[0]),
            "param_col": "",
            "value_col": "",
            "output_prefix": "tsb_falsification",
            "show_graphics": "artifacts" if len(found) > 1 else "popup",
            "auto_discover": "yes",
        }
    path = ensure_falsification_fixture_csv()
    return {
        "local_csv": str(path),
        "param_col": FALSIFICATION_FIXTURE_PARAM_COL,
        "value_col": FALSIFICATION_FIXTURE_VALUE_COL,
        "output_prefix": "tsb_falsification",
        "show_graphics": "popup",
        "auto_discover": "no",
    }


def run_falsification_from_source(
    *,
    data_source: str = "mock",
    local_csv: str | Path | None = None,
    param_col: str | None = None,
    value_col: str | None = None,
    output_prefix: str = "tsb_falsification",
    plot: bool = True,
    show_popup: bool = False,
    include_bound_energy: bool = False,
    relock: bool = False,
) -> dict[str, Any]:
    """Load Casimir data (mock, local CSV, or GitHub cache) and run falsification suite."""
    data_source = (data_source or "mock").strip().lower()
    source_path: str | Path | None = None
    if data_source == "mock":
        loaded = generate_mock_casimir_data()
        label = "mock magnetic-field Casimir sweep"
    elif data_source in {"local", "csv", "file", "auto"}:
        source_path = resolve_local_casimir_file(local_csv)
        loaded = load_data(source_path)
        label = source_path.name
    elif data_source in {"github", "github_magnetic_fluid"}:
        if local_csv:
            source_path = Path(local_csv)
        else:
            cache = DATASETS_DIR / "github_magnetic_fluid"
            candidates = [
                p
                for p in discover_local_casimir_files()
                if cache in p.parents or p.parent == cache
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"No GitHub magnetic-fluid files under {cache}. "
                    "Run Download GitHub Magnetic-Fluid Data first."
                )
            source_path = candidates[0]
        loaded = load_data(source_path)
        label = Path(source_path).name
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    report = run_full_test_suite(
        loaded,
        param_col=param_col,
        value_col=value_col,
        output_prefix=output_prefix,
        plot=plot,
        show_popup=show_popup,
        file_path=source_path,
        data_label=label,
        include_bound_energy=include_bound_energy,
        relock=relock,
    )
    report["data_label"] = label
    return report


def run_falsification_on_local_downloads(
    *,
    param_col: str | None = None,
    value_col: str | None = None,
    output_prefix: str = "tsb_falsification",
    plot: bool = True,
    show_popup: bool = False,
    include_bound_energy: bool = False,
    relock: bool = False,
) -> dict[str, Any]:
    """
    Run the falsification suite on every downloaded Casimir table.

    Falls back to the generated mock fixture only when the cache is empty.
    Refuses to pool labels whose hash ledger records a mismatch.
    """
    preregister_split(relock=relock)
    files = discover_local_casimir_files()
    if not files:
        print(
            f"[Superblock Residual Engine] No downloaded Casimir tables under {DATASETS_DIR}; "
            "using generated mock fixture."
        )
        report = run_falsification_from_source(
            data_source="mock",
            param_col=param_col,
            value_col=value_col,
            output_prefix=output_prefix,
            plot=plot,
            show_popup=show_popup,
            include_bound_energy=include_bound_energy,
            relock=relock,
        )
        return {"files": [], "reports": [report], "used_fixture": True}

    print(f"[Superblock Residual Engine] Auto-discovered {len(files)} Casimir table(s) under {DATASETS_DIR}")
    reports: list[dict[str, Any]] = []
    for index, path in enumerate(files):
        prefix = f"{output_prefix}_{path.stem}" if len(files) > 1 else output_prefix
        print(f"[Superblock Residual Engine] Falsification tests: {path.name}")
        reports.append(
            run_falsification_from_source(
                data_source="local",
                local_csv=path,
                param_col=param_col,
                value_col=value_col,
                output_prefix=prefix,
                plot=plot,
                show_popup=show_popup and index == len(files) - 1,
                include_bound_energy=include_bound_energy,
                relock=relock,
            )
        )
    poolable = [r for r in reports if r.get("pool_allowed", True)]
    refused = [r.get("data_label") for r in reports if not r.get("pool_allowed", True)]
    if refused:
        print(f"[CASIMIR LOAD] refusing to pool mismatched labels: {refused}")
    return {
        "files": [str(p) for p in files],
        "reports": reports,
        "used_fixture": False,
        "pooled_reports": poolable,
        "refused_pool_labels": refused,
        "preregister": load_family_lock(),
        "report_path": reports[-1].get("report_path") if reports else None,
        "plot_path": reports[-1].get("plot_path") if reports else None,
    }


# Analysis helpers keep the historical test_* names; they are not pytest cases.
test_periodicity.__test__ = False
test_hysteresis.__test__ = False
test_log_modulations.__test__ = False
test_powerlaw_index.__test__ = False
test_anisotropy_channels.__test__ = False
test_anisotropy_or_backreaction.__test__ = False


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TSB Casimir falsification test framework")
    p.add_argument("--data", type=str, default=None, help="Path to Casimir CSV")
    p.add_argument("--param_col", type=str, default=None)
    p.add_argument("--value_col", type=str, default=None)
    p.add_argument("--mock", action="store_true", help="Run on synthetic mock data")
    p.add_argument("--prefix", default="tsb_falsification")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--relock", action="store_true", help="Rewrite family preregistration lockfile")
    p.add_argument(
        "--bound-energy",
        action="store_true",
        help="Also compute ΔE = E_obs − E_Lifshitz (approximate integrator)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    _ = args
    from superblock_mode import print_retired

    print_retired("TSB_CASIMIR", "falsification CLI")
    print(
        "[Superblock Residual Engine] Casimir falsification CLI is retired. "
        "Tau/5D templates were not applied."
    )
    return 0
    if False and args.mock:
        run_falsification_from_source(
            data_source="mock",
            param_col=args.param_col,
            value_col=args.value_col,
            output_prefix=args.prefix,
            plot=not args.no_plot,
            include_bound_energy=args.bound_energy,
            relock=args.relock,
        )
    elif args.data:
        loaded = load_data(args.data)
        run_full_test_suite(
            loaded,
            param_col=args.param_col,
            value_col=args.value_col,
            output_prefix=args.prefix,
            plot=not args.no_plot,
            file_path=args.data,
            data_label=Path(args.data).name,
            include_bound_energy=args.bound_energy,
            relock=args.relock,
        )
    else:
        run_falsification_on_local_downloads(
            param_col=args.param_col,
            value_col=args.value_col,
            output_prefix=args.prefix,
            plot=not args.no_plot,
            include_bound_energy=args.bound_energy,
            relock=args.relock,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
