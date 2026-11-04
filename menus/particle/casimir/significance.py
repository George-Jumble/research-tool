"""
Residual significance tests for TSB Casimir signature detectors.

Trend (power-law or Drude+interband) is removed first. Periodogram power at
the pre-specified 1/7 frequency is scored against the local continuum.
Alignment and step counts use a random-placement null. Changepoints report
the PELT penalty and a residual-bootstrap interval on break locations.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import periodogram
from scipy.stats import binom

TAU_RESONANCE_PERIOD: float = 7.0
DEFAULT_P_THRESHOLD: float = 0.01
DEFAULT_ONE_SEVENTH_POWER_RATIO: float = 0.55
DEFAULT_SEVEN_CYCLE_POWER_RATIO: float = 0.53
SEVEN_CYCLE_FLAGS: tuple[str, ...] = (
    "seven_phase_clockwork",
    "tav_resonance_detected",
)


def p_threshold() -> float:
    raw = os.getenv("TAV_CASIMIR_P_THRESHOLD", str(DEFAULT_P_THRESHOLD))
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_P_THRESHOLD
    return value if 0.0 < value < 1.0 else DEFAULT_P_THRESHOLD


def _clip_power_ratio(raw: str | None, default: float) -> float:
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        return default
    return value if 0.0 <= value <= 1.0 else default


def one_seventh_power_ratio_threshold() -> float:
    """Primary I(f_1/7)/max(I) gate for the validated claim pipeline."""
    return _clip_power_ratio(
        os.getenv("TAV_CASIMIR_ONE_SEVENTH_POWER_RATIO"),
        DEFAULT_ONE_SEVENTH_POWER_RATIO,
    )


def seven_cycle_power_ratio_threshold() -> float:
    """Secondary I(f_1/7)/max(I) gate for the independent 7-cycle analysis."""
    return _clip_power_ratio(
        os.getenv("TAV_CASIMIR_SEVEN_CYCLE_POWER_RATIO"),
        DEFAULT_SEVEN_CYCLE_POWER_RATIO,
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    adjusted = [1.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, float(p_values[idx]) * (n - rank))
        running = max(running, adj)
        adjusted[idx] = running
    return adjusted


def _power_law(x: np.ndarray, amp: float, exponent: float, offset: float) -> np.ndarray:
    return offset + amp * np.power(np.clip(x, 1e-30, None), -exponent)


def _drude_interband(
    x: np.ndarray,
    offset: float,
    plasma: float,
    gamma: float,
    inter_amp: float,
    inter_w2: float,
) -> np.ndarray:
    xi = np.clip(x, 1e-30, None)
    drude = (plasma * plasma) / (xi * (xi + np.abs(gamma) + 1e-30))
    inter = inter_amp / (xi * xi + np.abs(inter_w2) + 1e-30)
    return offset + drude + inter


def fit_smooth_trend(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """
    Fit power-law and Drude+interband trends; keep the lower-RMSE model.

    Returns trend, residuals, model name, and parameter dict.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    if n < 6:
        trend = np.full(n, float(np.mean(y)) if n else 0.0)
        return {
            "model": "mean",
            "trend": trend,
            "residuals": y - trend,
            "rmse": float(np.sqrt(np.mean((y - trend) ** 2))) if n else None,
            "params": {},
        }

    candidates: list[dict[str, Any]] = []
    y_med = float(np.median(y))
    x_med = float(np.median(x))
    amp0 = float(np.ptp(y)) or 1.0

    try:
        popt, _ = curve_fit(
            _power_law,
            x,
            y,
            p0=(max(amp0 * x_med, 1e-3), 1.0, min(y_med, np.min(y))),
            maxfev=8000,
            bounds=([-np.inf, 0.05, -np.inf], [np.inf, 6.0, np.inf]),
        )
        trend = _power_law(x, *popt)
        rmse = float(np.sqrt(np.mean((y - trend) ** 2)))
        candidates.append(
            {
                "model": "power_law",
                "trend": trend,
                "residuals": y - trend,
                "rmse": rmse,
                "params": {
                    "amplitude": float(popt[0]),
                    "exponent": float(popt[1]),
                    "offset": float(popt[2]),
                },
            }
        )
    except (RuntimeError, ValueError, TypeError):
        pass

    try:
        popt, _ = curve_fit(
            _drude_interband,
            x,
            y,
            p0=(1.0, max(amp0, 1.0), max(x_med, 0.1), max(amp0, 0.1), max(x_med**2, 0.1)),
            maxfev=12000,
            bounds=(
                [-np.inf, 0.0, 1e-6, 0.0, 1e-6],
                [np.inf, np.inf, np.inf, np.inf, np.inf],
            ),
        )
        trend = _drude_interband(x, *popt)
        rmse = float(np.sqrt(np.mean((y - trend) ** 2)))
        candidates.append(
            {
                "model": "drude_interband",
                "trend": trend,
                "residuals": y - trend,
                "rmse": rmse,
                "params": {
                    "offset": float(popt[0]),
                    "plasma": float(popt[1]),
                    "gamma": float(popt[2]),
                    "interband_amp": float(popt[3]),
                    "interband_w2": float(popt[4]),
                },
            }
        )
    except (RuntimeError, ValueError, TypeError):
        pass

    if not candidates:
        # log-log linear fallback
        pos = x > 0
        if int(np.sum(pos)) >= 4:
            slope, intercept = np.polyfit(np.log(x[pos]), np.log(np.clip(np.abs(y[pos]), 1e-30, None)), 1)
            trend = np.exp(intercept) * np.power(np.clip(x, 1e-30, None), slope)
            if float(np.median(y)) < 0:
                trend = -trend
        else:
            trend = np.full_like(y, y_med)
        return {
            "model": "loglog_fallback",
            "trend": trend,
            "residuals": y - trend,
            "rmse": float(np.sqrt(np.mean((y - trend) ** 2))),
            "params": {},
        }

    best = min(candidates, key=lambda item: item["rmse"])
    return best


def expected_one_seventh_frequency(x: np.ndarray) -> float:
    span = float(np.max(x) - np.min(x)) if len(x) else 0.0
    if span <= 0:
        return 1.0 / TAU_RESONANCE_PERIOD
    return TAU_RESONANCE_PERIOD / span


def residual_periodogram_significance(
    residuals: np.ndarray,
    x: np.ndarray,
    *,
    n_bootstrap: int = 200,
    seed: int = 7,
) -> dict[str, Any]:
    """
    Periodogram of residuals; score power at the expected 1/7 frequency.

    Local-continuum test: I(f_1/7) / median_neighborhood ~ Exp, p = exp(-I/μ).
    Fisher's g uses the global maximum as a search-calibrated companion.
    """
    y = np.asarray(residuals, dtype=float)
    x = np.asarray(x, dtype=float)
    empty = {
        "p_value": 1.0,
        "p_local": 1.0,
        "p_fisher_g": 1.0,
        "p_bootstrap": 1.0,
        "power_at_one_seventh": 0.0,
        "power_at_one_seventh_proxy": 0.0,
        "local_continuum": 0.0,
        "expected_frequency": expected_one_seventh_frequency(x),
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

    target = expected_one_seventh_frequency(x)
    # skip the DC bin
    work_f = freqs[1:]
    work_p = power[1:]
    idx = int(np.argmin(np.abs(work_f - target)))
    i_target = float(work_p[idx])

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
    # exponential tail (periodogram ordinate / continuum)
    p_local = float(np.exp(-i_target / continuum))

    # Fisher's g on the residual spectrum
    total = float(np.sum(work_p)) or 1e-30
    g_obs = float(np.max(work_p) / total)
    n = int(work_p.size)
    p_fisher = _fisher_g_pvalue(g_obs, n)

    p_boot = 1.0
    if int(n_bootstrap) > 0:
        rng = np.random.default_rng(int(seed))
        exceed = 0
        scale = float(np.std(y)) or 1.0
        for _ in range(int(n_bootstrap)):
            boot = rng.normal(0.0, scale, size=y.size)
            _, boot_p = periodogram(boot, fs=fs)
            if boot_p.size > 1 and float(boot_p[1:][min(idx, boot_p.size - 2)]) >= i_target:
                exceed += 1
        p_boot = (exceed + 1.0) / (n_bootstrap + 1.0)

    max_power = float(np.max(work_p)) if work_p.size else 0.0
    power_proxy = float(i_target / max_power) if max_power > 0 else 0.0
    from tav_shared.bound_unbound_energy import partition_periodogram_totals

    energy_split = partition_periodogram_totals(work_f, work_p, f_bound=float(target))
    return {
        "p_value": float(min(p_local, p_boot) if n_bootstrap else p_local),
        "p_local": p_local,
        "p_fisher_g": p_fisher,
        "p_bootstrap": p_boot,
        "fisher_g": g_obs,
        "power_at_one_seventh": i_target,
        "power_at_one_seventh_proxy": power_proxy,
        "local_continuum": continuum,
        "snr_local": float(i_target / continuum),
        "expected_frequency": float(target),
        "observed_frequency": float(work_f[idx]),
        "n_freq_bins": n,
        "freqs": work_f.tolist(),
        "power": work_p.tolist(),
        "E_bound_clean": energy_split["E_bound_clean"],
        "E_unbound_clean": energy_split["E_unbound_clean"],
        "bound_flag_clean": energy_split["bound_flag_clean"],
        "f_b": energy_split["f_b"],
        "f_u": energy_split["f_u"],
        "energy_ledger": "content",
        "fisher_g_uses_E_tot": True,
        "note_energy": (
            "Fisher g still uses sum(power)=E_tot (raw). Bound ledger is E_bound_clean "
            "(discrete 1/7 bins that pass T1–T5); continuum is E_unbound_clean."
        ),
    }


def _fisher_g_pvalue(g_obs: float, n: int) -> float:
    """Exact Fisher g p-value for a hidden periodicity (white-noise periodogram)."""
    if n < 2 or g_obs <= 0:
        return 1.0
    if g_obs >= 1.0:
        return 0.0
    kmax = int(np.floor(1.0 / g_obs))
    kmax = min(kmax, n)
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


def random_placement_pvalue(
    locations: np.ndarray,
    markers: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    tolerance: float,
) -> dict[str, Any]:
    """P(K ≥ k_obs) for k hits among n uniform points vs fixed marker windows."""
    locs = np.asarray(locations, dtype=float).ravel()
    marks = np.asarray(markers, dtype=float).ravel()
    span = float(x_max - x_min)
    n = int(locs.size)
    if n == 0 or marks.size == 0 or span <= 0 or tolerance <= 0:
        return {"k_obs": 0, "n": n, "p_hit": 0.0, "p_value": 1.0}

    k_obs = int(
        sum(np.any(np.abs(loc - marks) < tolerance) for loc in locs)
    )
    # Union of clipped marker windows / span
    intervals = []
    for mark in marks:
        lo = max(x_min, mark - tolerance)
        hi = min(x_max, mark + tolerance)
        if hi > lo:
            intervals.append((lo, hi))
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in intervals:
        if not merged or lo > merged[-1][1]:
            merged.append((lo, hi))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
    covered = sum(hi - lo for lo, hi in merged)
    p_hit = float(min(1.0, covered / span))
    p_value = float(binom.sf(k_obs - 1, n, p_hit)) if k_obs > 0 else 1.0
    return {
        "k_obs": k_obs,
        "n": n,
        "p_hit": p_hit,
        "p_value": p_value,
        "tolerance": float(tolerance),
    }


def residual_step_pvalue(residuals: np.ndarray, *, z_thresh: float = 2.0) -> dict[str, Any]:
    """
    Count |Δr| exceedances after trend removal.

    Under a Gaussian residual random-walk increment, P(|Δr| > z σ) = 2 Φ(-z).
    """
    r = np.asarray(residuals, dtype=float)
    if r.size < 6:
        return {"n_steps": 0, "p_value": 1.0, "indices": np.array([], dtype=int)}
    dy = np.diff(r)
    sigma = float(np.std(dy)) or 1e-30
    indices = np.where(np.abs(dy) > z_thresh * sigma)[0]
    n_steps = int(indices.size)
    n_trials = int(dy.size)
    p_hit = float(2.0 * (1.0 - _norm_cdf(z_thresh)))
    p_value = float(binom.sf(n_steps - 1, n_trials, p_hit)) if n_steps > 0 else 1.0
    return {
        "n_steps": n_steps,
        "n_trials": n_trials,
        "p_hit": p_hit,
        "p_value": p_value,
        "indices": indices,
        "sigma_dy": sigma,
        "z_thresh": float(z_thresh),
    }


def _norm_cdf(z: float) -> float:
    import math

    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def hysteresis_pvalue(residuals: np.ndarray, n_perm: int = 400, seed: int = 11) -> dict[str, Any]:
    """Permutation p-value for first-half vs reversed-second-half asymmetry."""
    r = np.asarray(residuals, dtype=float)
    if r.size < 20:
        return {"magnitude": 0.0, "p_value": 1.0}
    mid = r.size // 2
    obs = float(np.mean(np.abs(r[:mid] - r[mid:][::-1][:mid])))
    rng = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(int(n_perm)):
        shuf = rng.permutation(r)
        stat = float(np.mean(np.abs(shuf[:mid] - shuf[mid:][::-1][:mid])))
        if stat >= obs:
            exceed += 1
    return {
        "magnitude": obs,
        "p_value": (exceed + 1.0) / (n_perm + 1.0),
        "n_perm": int(n_perm),
    }


def pelt_changepoints_with_bootstrap(
    residuals: np.ndarray,
    x: np.ndarray,
    *,
    n_bootstrap: int = 40,
    seed: int = 13,
) -> dict[str, Any]:
    """PELT on residuals; report penalty and residual-bootstrap location CI."""
    y = np.nan_to_num(np.asarray(residuals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if y.size < 8 or x.size != y.size:
        return {
            "method": "none",
            "locations": [],
            "n_breaks": 0,
            "penalty": None,
            "p_value": 1.0,
        }

    variance = float(np.var(y)) or 1.0
    penalty = float(max(2.0 * np.log(max(y.size, 2)) * variance, 1e-12))
    locations, method, note = _pelt_or_fallback(y, x, penalty)

    boot_sets: list[list[float]] = []
    if int(n_bootstrap) > 0:
        rng = np.random.default_rng(int(seed))
        for _ in range(int(n_bootstrap)):
            boot = y[rng.integers(0, y.size, size=y.size)]
            locs, _, _ = _pelt_or_fallback(boot, x, penalty)
            boot_sets.append(locs)

    intervals: list[dict[str, float]] = []
    for loc in locations:
        nearest = []
        for locs in boot_sets:
            if not locs:
                continue
            nearest.append(min(locs, key=lambda v: abs(v - loc)))
        if nearest:
            intervals.append(
                {
                    "location": float(loc),
                    "ci_low": float(np.percentile(nearest, 2.5)),
                    "ci_high": float(np.percentile(nearest, 97.5)),
                    "n_bootstrap_hits": int(len(nearest)),
                }
            )

    n_obs = len(locations)
    if int(n_bootstrap) > 0 and n_obs:
        n_ge = sum(1 for locs in boot_sets if len(locs) >= n_obs)
        p_value = (n_ge + 1.0) / (n_bootstrap + 1.0)
    else:
        p_value = 1.0 if n_obs == 0 else 1.0

    return {
        "method": method,
        "locations": locations[:10],
        "n_breaks": n_obs,
        "penalty": penalty,
        "penalty_formula": "2 log(n) Var(residual)",
        "bootstrap_intervals": intervals[:10],
        "p_value": float(p_value),
        "n_bootstrap": int(n_bootstrap),
        "note": note,
    }


def _pelt_or_fallback(
    y: np.ndarray,
    x: np.ndarray,
    penalty: float,
) -> tuple[list[float], str, str]:
    y = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.asarray(x, dtype=float)
    if y.size < 8 or not np.isfinite(penalty) or penalty <= 0:
        return [], "skipped", "invalid series or penalty"
    try:
        import ruptures as rpt

        with np.errstate(all="ignore"):
            algo = rpt.Pelt(model="l2", min_size=max(5, y.size // 50)).fit(y.reshape(-1, 1))
            bkps = [b for b in algo.predict(pen=float(penalty)) if 0 < b < len(y)]
        locations = [float(x[max(0, idx - 1)]) for idx in bkps if idx > 0]
        return (
            locations,
            "ruptures_pelt",
            f"PELT l2 penalty={penalty:.4g}",
        )
    except ImportError:
        dy = np.diff(y)
        thresh = (np.std(dy) or 1.0) * 2.5
        idx = np.where(np.abs(dy) > thresh)[0]
        locations = [float(x[i + 1]) for i in idx[:10] if i + 1 < len(x)]
        return (
            locations,
            "derivative_fallback",
            "ruptures not installed; derivative threshold fallback",
        )
    except Exception:
        return [], "pelt_failed", "PELT raised; treated as no breaks"
