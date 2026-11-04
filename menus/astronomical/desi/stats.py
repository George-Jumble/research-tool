"""DESI Tau-SB statistical diagnostics: s-grid, bootstrap, null model, γ jackknife."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

# 1/7 Lomb-Scargle detection gates (AND; all must hold).
F_EXPECTED_1_7: float = 1.0 / 7.0
POWER_RATIO_MIN: float = 0.5
P_BOOT_MAX: float = 0.05
MIN_N_DEDUPED_DETECTION: int = 12
MIN_CYCLES_DETECTION: float = 2.0
MIN_N_LOMB: int = 8
ALLOWED_RESIDUAL_SOURCES: frozenset[str] = frozenset(
    {"poly_baseline", "polyfit", "tau_sb", "tau_sb_fit"}
)
FORBIDDEN_RESIDUAL_SOURCES: frozenset[str] = frozenset(
    {"raw_dh", "raw", "geometric_mu", "geometry_mu", "geometric"}
)

def oversample_s_phase_coherent(
    s: np.ndarray,
    residuals: np.ndarray,
    *,
    n_target: int = 96,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Interpolate (s, residuals) onto a uniform dense s-grid.

    Used when n BAO points is too small for 1/7 Lomb resolution (Nyquist washout).
    Returns (s_dense, y_dense, was_oversampled).
    """
    s = np.asarray(s, dtype=float)
    y = np.asarray(residuals, dtype=float)
    if len(s) >= n_target or len(s) < 3:
        return s, y, False
    order = np.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]
    s_dense = np.linspace(float(s_sorted[0]), float(s_sorted[-1]), int(n_target))
    y_dense = np.interp(s_dense, s_sorted, y_sorted)
    return s_dense, y_dense, True


def diagnose_frequency_grid(
    s_array: np.ndarray,
    *,
    period: float = 7.0,
    n_cycles_min: float = 0.8,
) -> dict[str, Any]:
    """Check whether s-space span supports a 1/period oscillation search."""
    s = np.asarray(s_array, dtype=float)
    s_min, s_max = float(s.min()), float(s.max())
    s_range = s_max - s_min
    cycles_possible = s_range / period if period > 0 else 0.0
    recommended_n_bins = max(50, int(cycles_possible * 25))

    return {
        "s_min": s_min,
        "s_max": s_max,
        "s_range": s_range,
        "period_s_units": period,
        "cycles_possible": cycles_possible,
        "recommended_n_bins": recommended_n_bins,
        "warning": cycles_possible < n_cycles_min,
        "warning_message": (
            "s-range spans < {:.1f} full 1/{} cycles; periodicity search is underpowered".format(
                n_cycles_min, int(period) if period == int(period) else period
            )
            if cycles_possible < n_cycles_min
            else None
        ),
    }


def lomb_scargle_cyclic(
    t: np.ndarray,
    y: np.ndarray,
    freqs_cyclic: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Lomb-Scargle power vs cyclic frequency (cycles per unit time).

    SciPy's ``lombscargle`` takes angular frequencies ``ω = 2π f``. Passing
    cyclic ``f`` (e.g. 1/7) as ``freqs`` makes a true 1/7 sinusoid pile up at
    the Nyquist edge of a [0.05, 0.5] grid — the 20:41 false-positive pattern.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    freqs = np.asarray(freqs_cyclic, dtype=float)
    if freqs.size == 0:
        return np.asarray([], dtype=float)
    omega = 2.0 * np.pi * freqs
    return np.asarray(signal.lombscargle(t, y, omega, normalize=normalize), dtype=float)


def frequency_bin_width(freqs: np.ndarray) -> float:
    """Median spacing of a frequency grid (one bin)."""
    freqs = np.asarray(freqs, dtype=float)
    if freqs.size < 2:
        return float("inf")
    diffs = np.diff(np.sort(freqs))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("inf")
    return float(np.median(diffs))


def evaluate_1_7_detection(
    *,
    freqs: np.ndarray,
    power: np.ndarray,
    p_boot: float | None,
    n_raw: int,
    n_deduped: int,
    cycles: float,
    residual_source: str = "poly_baseline",
    gamma_source: str = "fixed",
    f_expected: float = F_EXPECTED_1_7,
) -> dict[str, Any]:
    """
    Strict AND rule for a reportable 1/7 Lomb-Scargle detection.

    ``lomb_peak_1_7`` is True only if the periodogram *maximum* is at 1/7
    (not f=0.5, not harmonic 2 or 7 alone), the target is a dominant mode,
    bootstrap p at that same bin is < 0.05, n_deduped≥12, cycles≥2, n≥8,
    and residuals are Tau-SB / poly-baseline (never raw D_H/r_d or geometry μ).

    If any clause fails: ``lomb_peak_1_7=False`` and ``detection=False``.
    """
    freqs_arr = np.asarray(freqs, dtype=float)
    power_arr = np.asarray(power, dtype=float)
    f_exp = float(f_expected)
    fail_reasons: list[str] = []

    empty = {
        "f_expected": f_exp,
        "power_at_expected": float("nan"),
        "f_max": float("nan"),
        "power_max": float("nan"),
        "power_ratio": float("nan"),
        "p_boot": float(p_boot) if p_boot is not None else float("nan"),
        "n_raw": int(n_raw),
        "n_deduped": int(n_deduped),
        "cycles": float(cycles),
        "gamma_source": str(gamma_source),
        "residual_source": str(residual_source),
        "lomb_peak_1_7": False,
        "detection": False,
        "df_bin": float("nan"),
        "fail_reasons": fail_reasons,
        "near_expected_bin": False,
        "dominant_mode": False,
        "p_boot_ok": False,
        "n_ok": False,
        "cycles_ok": False,
        "residual_ok": False,
    }

    if freqs_arr.size < 2 or power_arr.size != freqs_arr.size:
        fail_reasons.append("empty_or_mismatched_periodogram")
        return empty

    i_exp = int(np.argmin(np.abs(freqs_arr - f_exp)))
    i_max = int(np.argmax(power_arr))
    f_max = float(freqs_arr[i_max])
    power_max = float(power_arr[i_max])
    power_at_expected = float(power_arr[i_exp])
    power_ratio = (
        float(power_at_expected / power_max) if power_max > 0 and np.isfinite(power_max) else 0.0
    )
    df_grid = frequency_bin_width(freqs_arr)
    # Independent LS resolution is 1/Δs (Rayleigh), not the oversampled linspace
    # spacing. f=0.5 and harmonic 2 (2/7) remain many resolution bins away.
    period = 1.0 / f_exp if f_exp > 0 else 7.0
    s_range = abs(float(cycles) * period)
    rayleigh = 1.0 / max(s_range, 1e-6)
    df_bin = float(max(df_grid, rayleigh))
    near_expected = bool(abs(f_max - f_exp) <= df_bin + 1e-15)
    dominant = bool(power_ratio >= POWER_RATIO_MIN)
    p_val = float(p_boot) if p_boot is not None and np.isfinite(p_boot) else 1.0
    p_ok = bool(p_val < P_BOOT_MAX)
    n_lomb_ok = bool(int(n_raw) >= MIN_N_LOMB)
    n_ok = bool(int(n_deduped) >= MIN_N_DEDUPED_DETECTION and n_lomb_ok)
    cycles_ok = bool(float(cycles) >= MIN_CYCLES_DETECTION)
    src = str(residual_source or "").strip().lower()
    residual_ok = src in ALLOWED_RESIDUAL_SOURCES and src not in FORBIDDEN_RESIDUAL_SOURCES

    if not near_expected:
        fail_reasons.append(
            f"f_max={f_max:.4f} not within one bin (Δf={df_bin:.4g}) of f_expected={f_exp:.4f}"
        )
    if not dominant:
        fail_reasons.append(
            f"power_ratio={power_ratio:.4g} < {POWER_RATIO_MIN} (target is not a dominant mode)"
        )
    if not p_ok:
        fail_reasons.append(f"p_boot={p_val:.4g} ≥ {P_BOOT_MAX}")
    if not n_lomb_ok:
        fail_reasons.append(f"n_raw={int(n_raw)} < {MIN_N_LOMB} (Lomb/Tav skipped)")
    if int(n_deduped) < MIN_N_DEDUPED_DETECTION:
        fail_reasons.append(f"n_deduped={int(n_deduped)} < {MIN_N_DEDUPED_DETECTION}")
    if not cycles_ok:
        fail_reasons.append(f"cycles={float(cycles):.3f} < {MIN_CYCLES_DETECTION}")
    if not residual_ok:
        fail_reasons.append(
            f"residual_source={residual_source!r} is not Tau-SB/poly-baseline "
            "(raw D_H/r_d and geometry μ never set the detection bit)"
        )

    lomb_peak = bool(
        near_expected and dominant and p_ok and n_ok and cycles_ok and residual_ok
    )
    detection = lomb_peak

    return {
        "f_expected": f_exp,
        "power_at_expected": power_at_expected,
        "f_max": f_max,
        "power_max": power_max,
        "power_ratio": power_ratio,
        "p_boot": p_val,
        "n_raw": int(n_raw),
        "n_deduped": int(n_deduped),
        "cycles": float(cycles),
        "gamma_source": str(gamma_source),
        "residual_source": str(residual_source),
        "lomb_peak_1_7": lomb_peak,
        "detection": detection,
        "df_bin": df_bin,
        "fail_reasons": fail_reasons,
        "near_expected_bin": near_expected,
        "dominant_mode": dominant,
        "p_boot_ok": p_ok,
        "n_ok": n_ok,
        "cycles_ok": cycles_ok,
        "residual_ok": residual_ok,
    }


def periodicity_json_fields(
    per: dict[str, Any] | None,
    *,
    tav_harmonics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Always-written 1/7 detection fields for JSON / plot annotations."""
    per = per or {}
    th = tav_harmonics if tav_harmonics is not None else (per.get("tav_harmonics") or {})
    f_expected = per.get("f_expected", per.get("expected_f", F_EXPECTED_1_7))
    f_max = per.get("f_max", per.get("max_freq", float("nan")))
    power_max = per.get("power_max", per.get("max_power", float("nan")))
    p_boot = per.get("p_boot", per.get("pval_bootstrap", per.get("pval_approx", 1.0)))
    lomb_peak = bool(per.get("lomb_peak_1_7", False))
    detection = bool(per.get("detection", False))
    return {
        "f_expected": f_expected,
        "power_at_expected": per.get("power_at_expected"),
        "f_max": f_max,
        "power_max": power_max,
        "power_ratio": per.get("power_ratio"),
        "p_boot": p_boot,
        "n_raw": per.get("n_raw"),
        "n_deduped": per.get("n_deduped"),
        "cycles": per.get("cycles", (per.get("s_grid_diagnostics") or {}).get("cycles_possible")),
        "gamma_source": per.get("gamma_source"),
        "lomb_peak_1_7": lomb_peak,
        "tav_resonance_detected": bool(th.get("tav_resonance_detected", False)),
        "harmonic_peaks": list(th.get("harmonic_peaks") or per.get("harmonic_peaks") or []),
        "detection": detection,
        # backward-compatible aliases
        "expected_f": f_expected,
        "max_freq": f_max,
        "max_power": power_max,
        "pval_bootstrap": p_boot,
        "pval_approx": per.get("pval_approx", p_boot),
        "lomb_detected": lomb_peak,
        "s_grid_diagnostics": per.get("s_grid_diagnostics"),
        "fail_reasons": list(per.get("fail_reasons") or []),
    }


def bootstrap_lomb_scargle_pvalue(
    s: np.ndarray,
    residual: np.ndarray,
    *,
    freq: float,
    n_bootstrap: int = 2000,
    n_freq: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Permutation bootstrap p-value for Lomb-Scargle power at a target frequency."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(residual, dtype=float)
    if len(s) < MIN_N_LOMB:
        return {"observed_power": 0.0, "bootstrap_p_value": 1.0, "n_bootstrap": 0}

    f_lo = max(0.02, freq * 0.4)
    f_hi = min(0.5, freq * 2.5)
    freqs = np.linspace(f_lo, f_hi, max(n_freq, 50))
    power = lomb_scargle_cyclic(s, y, freqs, normalize=True)
    idx = int(np.argmin(np.abs(freqs - freq)))
    observed = float(power[idx])

    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        perm = y[rng.permutation(len(y))]
        boot[i] = float(lomb_scargle_cyclic(s, perm, freqs, normalize=True)[idx])

    return {
        "observed_power": observed,
        "bootstrap_p_value": float(np.mean(boot >= observed)),
        "n_bootstrap": float(n_bootstrap),
    }


def _lomb_scargle_power_at_frequency(
    s: np.ndarray,
    y: np.ndarray,
    freq: float,
) -> tuple[float, str]:
    """Lomb-Scargle power at a single frequency (astropy preferred, scipy fallback)."""
    try:
        from astropy.timeseries import LombScargle

        power = float(LombScargle(s, y).power(float(freq)))
        return power, "astropy"
    except ImportError:
        f_lo = max(0.02, float(freq) * 0.4)
        f_hi = min(0.5, float(freq) * 2.5)
        freqs = np.linspace(f_lo, f_hi, 200)
        power_grid = lomb_scargle_cyclic(s, y, freqs, normalize=True)
        idx = int(np.argmin(np.abs(freqs - float(freq))))
        return float(power_grid[idx]), "scipy"


def search_higher_harmonics(
    residuals: np.ndarray,
    s: np.ndarray,
    *,
    fundamental_freq: float = 1.0 / 7.0,
    max_harmonic: int = 5,
    n_bootstrap: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Search for power at higher harmonics of the fundamental 1/7 frequency.

    Returns permutation-bootstrap p-values for each harmonic h=1..max_harmonic
    at frequency ``h * fundamental_freq`` on the (s, residuals) grid.

    Per-harmonic entries live in ``result["harmonics"][h]`` with keys
    ``frequency``, ``observed_power``, and ``bootstrap_p_value``.
    """
    s = np.asarray(s, dtype=float)
    resid = np.asarray(residuals, dtype=float)
    if len(s) != len(resid):
        raise ValueError(f"s length {len(s)} != residuals length {len(resid)}")
    if len(resid) < 3:
        return {
            "fundamental_freq": float(fundamental_freq),
            "max_harmonic": int(max_harmonic),
            "n_bootstrap": int(n_bootstrap),
            "harmonics": {},
            "any_significant_at_5pct": False,
            "significant_harmonics": [],
            "harmonic_1_significant": False,
            "E_bound_clean": 0.0,
            "E_unbound_clean": 0.0,
            "bound_flag_clean": False,
            "note": f"n={len(resid)} too few for Lomb-Scargle (need n≥3)",
        }

    fundamental = float(fundamental_freq)
    rng = np.random.default_rng(seed)
    backend = "unknown"
    harmonics: dict[int, dict[str, float]] = {}

    for h in range(1, max_harmonic + 1):
        freq = h * fundamental
        observed_power, backend = _lomb_scargle_power_at_frequency(s, resid, freq)

        bootstrap_powers = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            y_perm = resid[rng.permutation(len(resid))]
            bootstrap_powers[i], _ = _lomb_scargle_power_at_frequency(s, y_perm, freq)

        p_value = float(np.mean(bootstrap_powers >= observed_power))
        harmonics[h] = {
            "frequency": float(freq),
            "observed_power": float(observed_power),
            "bootstrap_p_value": p_value,
            "significant_at_5pct": p_value < 0.05,
            "is_fundamental_1_7": h == 1,
        }
        if verbose:
            tag = " [1/7 fundamental]" if h == 1 else " [higher harmonic — not a 1/7 detection]"
            print(
                f"Harmonic {h} (f={freq:.4f}): "
                f"Power = {observed_power:.2e}, Bootstrap p = {p_value:.4f}{tag}"
            )

    any_sig = any(entry["significant_at_5pct"] for entry in harmonics.values())
    sig_h = [int(h) for h, entry in harmonics.items() if entry["significant_at_5pct"]]
    # Content ledger: only harmonic 1 (f=1/7) can be E_b; h≥2 is unbound continuum/harmonic.
    from tav_shared.bound_unbound_energy import partition_energy

    powers = np.array(
        [float(harmonics[h]["observed_power"]) for h in range(1, max_harmonic + 1)],
        dtype=float,
    )
    discrete = np.array([h == 1 for h in range(1, max_harmonic + 1)], dtype=bool)
    pvals = np.array(
        [float(harmonics[h]["bootstrap_p_value"]) for h in range(1, max_harmonic + 1)],
        dtype=float,
    )
    s_proxy = np.where(pvals > 0, -np.log10(np.clip(pvals, 1e-300, 1.0)), 0.0)
    energy_split = partition_energy(
        powers,
        e_free=powers + np.maximum(powers - float(np.median(powers)), 0.0),
        e_th=np.inf,
        is_discrete_harmonic=discrete,
        tau_bind_ok=True,
        from_total_aggregator=False,
        significance=s_proxy,
        s_min=1.3,  # ~ p < 0.05 in -log10 p
        ledger="content",
    )
    return {
        "fundamental_freq": fundamental,
        "max_harmonic": int(max_harmonic),
        "n_bootstrap": int(n_bootstrap),
        "lomb_scargle_backend": backend,
        "harmonics": harmonics,
        "any_significant_at_5pct": bool(any_sig),
        "significant_harmonics": sig_h,
        "harmonic_1_significant": bool(harmonics.get(1, {}).get("significant_at_5pct", False)),
        "note": (
            "any_significant_at_5pct includes higher harmonics; it does not set "
            "the 1/7 detection bit (harmonic 2/7 alone is not a 1/7 ladder hit)"
        ),
        "E_bound_clean": float(energy_split["sum_E_b"]),
        "E_unbound_clean": float(energy_split["sum_E_u"]),
        "bound_flag_clean": bool(energy_split["n_bound"] > 0),
        "f_b": float(energy_split["sum_E_b"] / max(energy_split["sum_E_tot"], 1e-30)),
        "f_u": float(energy_split["sum_E_u"] / max(energy_split["sum_E_tot"], 1e-30)),
        "energy_ledger": "content",
    }


def _bootstrap_lomb_pvalue_at_freq(
    s: np.ndarray,
    resid: np.ndarray,
    freq: float,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Permutation-bootstrap Lomb-Scargle p-value at a single target frequency."""
    observed_power, _ = _lomb_scargle_power_at_frequency(s, resid, freq)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        y_perm = resid[rng.permutation(len(resid))]
        boot[i], _ = _lomb_scargle_power_at_frequency(s, y_perm, freq)
    return {
        "frequency": float(freq),
        "observed_power": float(observed_power),
        "bootstrap_p_value": float(np.mean(boot >= observed_power)),
    }


def search_whim_web_signatures(
    residuals: np.ndarray,
    s: np.ndarray,
    n_bootstrap: int = 2000,
    *,
    seed: int = 42,
    gamma: float | None = None,
    web_scale_mpc: float | None = None,
    s_reference: float = 50.0,
    tau_period: float = 7.0,
    verbose: bool = True,
    n_used: int | None = None,
    cycles: float = 0.0,
    periodicity_detection: bool = False,
    p_values_trusted: bool = True,
) -> dict[str, Any]:
    """
    Search for 1D/web-like correlations or scale features in residuals.

    Implements the sketch workflow (Pearson corr along ``s`` + targeted
    Lomb-Scargle at web scales) and extends it with bootstrap p-values,
    a :func:`~tau_sb_desi_scanner.whim_web_correction` template correlation,
    and data-driven BEC-well frequencies from the observed ``s`` span.

    The original placeholder ``web_freq = 1 / (web_scale / s_reference)`` is
    recorded in ``web_freq_placeholder``; prefer ``lomb_scargle["bec_well"]``
    (``f ≈ 1 / s_span``) on finite BAO vectors.
    """
    from scipy.stats import pearsonr, spearmanr

    s_arr = np.asarray(s, dtype=float)
    resid = np.asarray(residuals, dtype=float)
    if len(s_arr) != len(resid):
        raise ValueError(f"s length {len(s_arr)} != residuals length {len(resid)}")

    n = len(resid)
    result: dict[str, Any] = {
        "n_data": int(n),
        "n_bootstrap": int(n_bootstrap),
        "s_span": float(np.max(s_arr) - np.min(s_arr)) if n else 0.0,
    }

    if n < 3:
        result["note"] = f"n={n} too few for WHIM/web signature search (need n≥3)"
        if verbose:
            print(f"[WHIM/Web Signature] {result['note']}")
        return result

    try:
        from menus.astronomical.desi.scanner import (
            BEC_COHERENCE_LENGTH_MPC,
            TSB_RD_GAMMA_FALLBACK,
            whim_web_correction,
        )

        bec_mpc = float(web_scale_mpc if web_scale_mpc is not None else BEC_COHERENCE_LENGTH_MPC)
        gamma_used = float(gamma if gamma is not None else TSB_RD_GAMMA_FALLBACK)
    except ImportError:
        bec_mpc = float(web_scale_mpc if web_scale_mpc is not None else 5.5)
        gamma_used = float(gamma if gamma is not None else 8.8511)
        whim_web_correction = None  # type: ignore[assignment,misc]

    result["web_scale_mpc"] = bec_mpc
    result["gamma_used"] = gamma_used
    s_ref = float(s_reference)
    f_placeholder = (1.0 / (bec_mpc / s_ref)) if bec_mpc > 0 and s_ref > 0 else float("nan")
    result["web_freq_placeholder"] = {
        "web_scale_mpc": bec_mpc,
        "s_reference": s_ref,
        "frequency": float(f_placeholder),
        "note": "Sketch formula 1/(L_BEC/s_ref); use bec_well (1/s_span) on real data",
    }

    corr_s, p_s = pearsonr(s_arr, resid)
    result["pearson_s"] = {
        "correlation": float(corr_s),
        "p_value": float(p_s),
        "significant_at_5pct": float(p_s) < 0.05,
    }
    if verbose:
        print(
            f"[WHIM/Web Signature] Pearson corr(s, residuals) = {corr_s:.4f} (p={p_s:.4f})"
        )

    try:
        rho, p_rho = spearmanr(s_arr, resid)
        result["spearman_s"] = {
            "correlation": float(rho),
            "p_value": float(p_rho),
            "significant_at_5pct": float(p_rho) < 0.05,
        }
        if verbose:
            print(
                f"[WHIM/Web Signature] Spearman corr(s, residuals) = {rho:.4f} (p={p_rho:.4f})"
            )
    except Exception:
        result["spearman_s"] = None

    if whim_web_correction is not None:
        template = whim_web_correction(s_arr, gamma_used)
        corr_t, p_t = pearsonr(template, resid)
        result["pearson_whim_template"] = {
            "correlation": float(corr_t),
            "p_value": float(p_t),
            "significant_at_5pct": float(p_t) < 0.05,
        }
        if verbose:
            print(
                f"[WHIM/Web Signature] Pearson corr(whim_template, residuals) = "
                f"{corr_t:.4f} (p={p_t:.4f})"
            )

    s_span = float(result["s_span"])
    f_filament = 1.0 / (2.0 * float(tau_period))
    f_bec = (1.0 / s_span) if s_span > 0 else float("nan")
    f_bec_h2 = (2.0 / s_span) if s_span > 0 else float("nan")

    targets = {
        "filament_2tau": f_filament,
        "bec_well": f_bec,
        "bec_well_h2": f_bec_h2,
    }
    lomb: dict[str, dict[str, float]] = {}
    backend = "unknown"
    seed_offsets = {"filament_2tau": 0, "bec_well": 1, "bec_well_h2": 2}
    for name, freq in targets.items():
        if not np.isfinite(freq) or freq <= 0:
            continue
        boot = _bootstrap_lomb_pvalue_at_freq(
            s_arr,
            resid,
            freq,
            n_bootstrap=n_bootstrap,
            seed=seed + seed_offsets.get(name, 0),
        )
        boot["significant_at_5pct"] = boot["bootstrap_p_value"] < 0.05
        lomb[name] = boot
        _, backend = _lomb_scargle_power_at_frequency(s_arr, resid, freq)
        if verbose:
            print(
                f"[WHIM/Web Lomb] {name} (f={freq:.4f}): "
                f"Power = {boot['observed_power']:.2e}, "
                f"Bootstrap p = {boot['bootstrap_p_value']:.4f} "
                f"[diagnostic only; not a WHIM signature]"
            )

    result["target_frequencies"] = targets
    result["lomb_scargle"] = lomb
    result["lomb_scargle_backend"] = backend
    result["any_lomb_significant_at_5pct"] = any(
        v.get("significant_at_5pct") for v in lomb.values()
    )
    gated = evaluate_whim_signature(
        result,
        n_used=int(n_used if n_used is not None else n),
        cycles=float(cycles),
        periodicity_detection=bool(periodicity_detection),
        p_values_trusted=bool(p_values_trusted),
    )
    result.update(gated)

    if verbose:
        if not lomb:
            print("[WHIM/Web] No valid target frequencies for Lomb-Scargle search.")
        else:
            print(
                f"[WHIM/Web] Placeholder f (1/(L_BEC/s_ref)) = {f_placeholder:.4f} "
                f"(s_ref={s_ref:.1f}); data-driven bec_well f = {f_bec:.4f}"
            )
        print(f"[WHIM/Web] whim_signature_detected={result.get('whim_signature_detected')}")
        if result.get("whim_gate_reasons"):
            print(f"  gated: {'; '.join(result['whim_gate_reasons'])}")

    return result


def evaluate_whim_signature(
    whim: dict[str, Any] | None,
    *,
    n_used: int,
    cycles: float,
    periodicity_detection: bool,
    p_values_trusted: bool = True,
) -> dict[str, Any]:
    """
    WHIM / web Lomb is gated like the 1/7 flag.

    A lone harmonic p-value (e.g. bec_well_h2 p=0.002) is never a web
    signature when 1/7 is null, n_used<12, or cycles<2. Even a true 1/7
    sinusoid does not auto-set WHIM.
    """
    whim = dict(whim or {})
    reasons: list[str] = []
    if not p_values_trusted:
        reasons.append("p_values_untrusted")
    if not periodicity_detection:
        reasons.append("periodicity_detection=False (1/7 is null)")
    if int(n_used) < 12:
        reasons.append(f"n_used={int(n_used)}<12")
    if float(cycles) < 2.0:
        reasons.append(f"cycles={float(cycles):.2f}<2")
    # A 1/7 sinusoid must not auto-promote WHIM; require a WHIM-scale
    # dominant mode that is not harmonic 2 of 1/7 and not bec_well_h2 alone.
    lomb = whim.get("lomb_scargle") or {}
    h2 = lomb.get("bec_well_h2") or {}
    if h2.get("significant_at_5pct") and not periodicity_detection:
        reasons.append("bec_well_h2 p<0.05 is not a WHIM signature when 1/7 is null")
    detected = False
    whim["whim_signature_detected"] = detected
    whim["whim_signature_gated"] = bool(reasons)
    whim["whim_gate_reasons"] = reasons
    for entry in lomb.values():
        if isinstance(entry, dict):
            entry["signature"] = False
            if reasons:
                entry["diagnostic_only"] = True
    return whim


def estimate_tsb_percolation(
    n_data: int,
    *,
    tau_phases: int = 7,
    effective_dim: float = 2.5,
    p_c_2d: float = 0.59,
) -> dict[str, Any]:
    """
    Simplified 2.5D percolation estimate for Tau-cylinder bubble filling.

    Uses ``n_effective_sites ≈ n_data × tau_phases`` and a rough
    ``p_c ≈ p_c,2D / (d_eff - 1)`` scaling for a 2.5D-like lattice.
    """
    n = max(int(n_data), 0)
    n_sites = n * int(tau_phases)
    denom = float(effective_dim) - 1.0
    p_c_approx = float(p_c_2d / denom) if denom > 0 else float("nan")
    return {
        "estimated_p_c": p_c_approx,
        "n_effective_sites": int(n_sites),
        "n_data": int(n),
        "tau_phases": int(tau_phases),
        "effective_dim": float(effective_dim),
        "p_c_2d_reference": float(p_c_2d),
        "note": "Simplified estimate from 2.5D bubble percolation on Tau Cylinder",
    }


def fit_null_model(y: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """Weighted-mean null model χ² using full covariance."""
    y = np.asarray(y, dtype=float)
    cov = np.asarray(cov, dtype=float)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        from menus.astronomical.desi.analysis import apply_tikhonov_regularization

        cov, _ = apply_tikhonov_regularization(cov)
        cov_inv = np.linalg.inv(cov)
    weights = np.diag(cov_inv)
    null_value = float(np.average(y, weights=weights))
    resid = y - null_value
    chi2_null = float(resid @ cov_inv @ resid)
    return null_value, chi2_null


def compute_model_comparison(
    chi2_model: float,
    chi2_null: float,
    *,
    n_data: int,
    n_params_model: int,
) -> dict[str, float]:
    """ΔAIC/ΔBIC vs constant null; includes reduced χ²."""
    k_model = n_params_model
    k_null = 1
    aic_model = chi2_model + 2 * k_model
    aic_null = chi2_null + 2 * k_null
    bic_model = chi2_model + k_model * np.log(max(n_data, 1))
    bic_null = chi2_null + k_null * np.log(max(n_data, 1))
    dof = max(n_data - k_model, 1)
    return {
        "chi2_model": float(chi2_model),
        "chi2_null": float(chi2_null),
        "delta_chi2_vs_null": float(chi2_null - chi2_model),
        "delta_aic_vs_null": float(aic_model - aic_null),
        "delta_bic_vs_null": float(bic_model - bic_null),
        "reduced_chi2_model": float(chi2_model / dof),
        "n_data": float(n_data),
        "n_params_model": float(k_model),
        "dof": float(n_data - k_model),
    }


def gamma_jackknife_std(
    z: np.ndarray,
    *,
    n_hier: float,
    delta_n: float,
    period: float,
    calibrate_fn,
) -> dict[str, float]:
    """Leave-one-out spread on auto-calibrated γ."""
    z = np.asarray(z, dtype=float)
    if len(z) < 3:
        return {"gamma_median": float(calibrate_fn(z)), "gamma_std": 0.0, "n_jackknife": 0}

    gammas = []
    for i in range(len(z)):
        mask = np.ones(len(z), dtype=bool)
        mask[i] = False
        z_sub = z[mask]
        gammas.append(float(calibrate_fn(z_sub)))
    g = np.array(gammas, dtype=float)
    return {
        "gamma_median": float(np.median(g)),
        "gamma_std": float(np.std(g, ddof=1)) if len(g) > 1 else 0.0,
        "n_jackknife": float(len(g)),
    }


def tav_harmonic_check_robust(
    s: np.ndarray,
    residuals: np.ndarray,
    *,
    period: float = 7.0,
    phase_tolerance: float = 0.05,
    min_peaks: int = 1,
    n_bootstrap: int = 2000,
    seed: int = 42,
    lomb_peak_1_7: bool | None = None,
    detection: bool | None = None,
) -> dict[str, Any]:
    """
    Tav-harmonic residual check, gated on the strict 1/7 Lomb peak predicate.

    ``lomb_detected`` is **not** ``p_boot < 0.05``. Harmonic 7 (or 2) alone is
    never a 1/7 detection. Skip Lomb/Tav for n<8.
    """
    s = np.asarray(s, dtype=float)
    resid = np.asarray(residuals, dtype=float)
    n = len(resid)
    diag = diagnose_frequency_grid(s, period=period)
    expected_f = 1.0 / period

    result: dict[str, Any] = {
        "method": "lomb_scargle_bootstrap",
        "n_data": n,
        "n_lomb_grid": int(n),
        "s_oversampled": False,
        "n_freq_bins": int(max(50, diag["recommended_n_bins"])),
        "cycles_possible": diag["cycles_possible"],
        "s_grid_warning": diag["warning"],
        "s_grid_warning_message": diag.get("warning_message"),
        "power_at_expected": 0.0,
        "bootstrap_p_value": 1.0,
        "lomb_detected": False,
        "lomb_peak_1_7": False,
        "harmonic_peaks": [],
        "tav_resonance_detected": False,
        "tav_resonance_skipped": n < MIN_N_LOMB,
        "tav_skip_reason": None,
    }

    if n < MIN_N_LOMB:
        result["tav_skip_reason"] = (
            f"Lomb-Scargle / tav_resonance skipped: n={n} < {MIN_N_LOMB}"
        )
        result["underpowered_warning"] = (
            f"n={n} BAO points: Lomb-Scargle and tav-resonance are skipped (need n≥{MIN_N_LOMB})"
        )
        return result

    boot = bootstrap_lomb_scargle_pvalue(
        s, resid, freq=expected_f, n_bootstrap=n_bootstrap, seed=seed
    )
    result["n_lomb_grid"] = int(n)
    result["power_at_expected"] = boot["observed_power"]
    result["bootstrap_p_value"] = boot["bootstrap_p_value"]

    peak_ok = bool(lomb_peak_1_7) if lomb_peak_1_7 is not None else False
    det_ok = bool(detection) if detection is not None else peak_ok
    result["lomb_peak_1_7"] = peak_ok
    result["lomb_detected"] = peak_ok
    result["detection"] = det_ok

    try:
        from tav_shared.tav_resonance_bootstrap import ensure_tav_resonance_importable

        ensure_tav_resonance_importable()
        from tav_resonance import analyze_tav_harmonics

        freqs_t, _power_t, peaks, detected = analyze_tav_harmonics(
            resid,
            phase_tolerance=phase_tolerance,
            min_peaks=min_peaks,
        )
        result["method"] = "lomb_scargle_bootstrap+tav_periodogram"
        result["tav_raw_detected"] = bool(detected)
        result["harmonic_peaks"] = peaks
        result["tav_n_freq_bins"] = int(freqs_t.size)
        result["tav_resonance_skipped"] = False
        result["tav_skip_reason"] = None
        if bool(detected) and not peak_ok:
            result["tav_resonance_detected"] = False
            result["tav_resonance_suppressed"] = (
                "analyze_tav_harmonics peak ignored: not a 1/7 Lomb-Scargle maximum "
                "(harmonic 2/7 or residual-flag hits are not detections)"
            )
        else:
            result["tav_resonance_detected"] = bool(detected) and peak_ok
    except ImportError:
        result["tav_skip_reason"] = "tav_resonance package not installed"
    except Exception as trans_exc:
        result["tav_skip_reason"] = f"tav_resonance error: {trans_exc}"

    return result


def power_stamp_for_scan(*, n_used: int, cycles: float) -> str:
    """
    LOW if n_used<12 or cycles<1.5.
    MEDIUM only if n≥12 and 1.5≤cycles<2.
    HIGH only if n≥12 and cycles≥2.
    """
    n = int(n_used)
    cyc = float(cycles)
    if n < 12 or cyc < 1.5:
        return "LOW"
    if n >= 12 and 1.5 <= cyc < 2.0:
        return "MEDIUM"
    return "HIGH"


def assess_scan_power(
    *,
    n_data: int,
    cycles_possible: float,
    auto_calibrate_gamma: bool,
    n_params: int | None = None,
    tav_skipped: bool = False,
    min_n_recommended: int = 12,
) -> dict[str, Any]:
    """
    Summarize statistical power limitations for a DESI Tau-SB scan.

    Returns a list of issues with severity: critical | high | medium.
    ``power_stamp`` is LOW / MEDIUM / HIGH from n_used and τ-cycles.
    """
    issues: list[dict[str, str]] = []
    stamp = power_stamp_for_scan(n_used=n_data, cycles=cycles_possible)

    if n_data < 8:
        issues.append(
            {
                "issue": "n_data",
                "severity": "critical",
                "message": (
                    f"Only n={n_data} data points — fundamentally underpowered; "
                    "stay on DH_only/DM_only (do not mix channels); use batch tracers"
                ),
            }
        )
    elif n_data < min_n_recommended:
        issues.append(
            {
                "issue": "n_data",
                "severity": "high",
                "message": (
                    f"n={n_data} is below recommended minimum n≥{min_n_recommended} "
                    "for periodicity / model comparison"
                ),
            }
        )

    if auto_calibrate_gamma and 0.9 <= cycles_possible <= 1.1:
        issues.append(
            {
                "issue": "s_space_cycles",
                "severity": "critical",
                "message": (
                    f"Δs/τ = {cycles_possible:.2f} with auto-calibrated γ — s-range spans "
                    "~1τ period by construction; Lomb 1/7 test on auto-γ coordinates is "
                    "circular (see fixed-γ diagnostic)"
                ),
            }
        )
    elif cycles_possible < 2.0:
        issues.append(
            {
                "issue": "s_space_cycles",
                "severity": "medium",
                "message": (
                    f"{cycles_possible:.2f} τ cycles in s-space at fixed γ; "
                    "Lomb 1/7 has limited resolving power (prefer ≥2 cycles)"
                ),
            }
        )

    if tav_skipped:
        issues.append(
            {
                "issue": "tav_resonance",
                "severity": "high",
                "message": "tav_resonance periodogram skipped (requires n≥8)",
            }
        )

    if n_params is not None:
        dof = n_data - n_params
        if dof <= 0:
            issues.append(
                {
                    "issue": "model_complexity",
                    "severity": "critical",
                    "message": (
                        f"Model has {n_params} free parameters for n={n_data} "
                        f"(dof={dof}) — over-fitted or saturated"
                    ),
                }
            )
        elif dof < 3:
            issues.append(
                {
                    "issue": "model_complexity",
                    "severity": "medium",
                    "message": (
                        f"Only dof={dof} ({n_params} params, n={n_data}) — "
                        "uncertainties on oscillation amplitude are poorly constrained"
                    ),
                },
            )

    rank = {"critical": 3, "high": 2, "medium": 1, "ok": 0}
    overall = "ok"
    if issues:
        overall = max(issues, key=lambda row: rank[row["severity"]])["severity"]
    if stamp == "LOW":
        overall = "critical" if overall == "ok" else overall

    if stamp == "LOW":
        rec = (
            "Statistical power LOW: n_used<12 or τ-cycles<1.5 — "
            "not adequate for an exploratory 1/7 scan. Keep DATA_MODE=DH_only; "
            "unique-z the stack; trust fixed-γ Lomb AND-rule only."
        )
    elif stamp == "MEDIUM":
        rec = (
            "Statistical power MEDIUM: n≥12 but τ-cycles still <2. "
            "Keep DATA_MODE=DH_only; do not mix channels; prefer fixed γ=10."
        )
    else:
        rec = (
            "Statistical power HIGH: n≥12 and ≥2 τ-cycles. "
            "Keep DATA_MODE=DH_only or DM_only (do not mix channels)."
        )

    return {
        "issues": issues,
        "overall_severity": overall,
        "power_stamp": stamp,
        "n_data": n_data,
        "n_used": n_data,
        "cycles_possible": cycles_possible,
        "recommendation": rec,
    }


def build_recommended_next_steps(
    *,
    power_assessment: dict[str, Any] | None,
    fit: dict[str, Any] | None = None,
    auto_diagnostics: dict[str, Any] | None = None,
    residual_diagnostics: dict[str, Any] | None = None,
    n_data: int = 0,
    include_legacy_bao: bool = False,
    fit_phase: bool = True,
    fit_hier: bool = True,
    min_n_target: int = 12,
) -> list[str]:
    """
    Actionable checklist after a DESI Tau-SB scan (Priorities 1 & 2).
    """
    steps: list[str] = []
    severity = (power_assessment or {}).get("overall_severity", "unknown")

    if n_data < min_n_target:
        if not include_legacy_bao:
            steps.append(
                f"Priority 1 — Increase n: enable include_legacy_bao=yes "
                f"(DESI+BOSS/eBOSS stack; current n={n_data}, target n≥{min_n_target})"
            )
        else:
            steps.append(
                f"Priority 1 — n={n_data} still below target {min_n_target}; "
                "add more legacy surveys or run batch tracers for per-tracer checks"
            )
        steps.append(
            "Priority 1 — Do not use Joint_DH_DM or quantity=all to inflate n "
            "(mixed channels break single-template fits)"
        )

    if fit and int(fit.get("dof", 0)) < 3:
        if fit_phase:
            steps.append(
                "Priority 1 — Reduce parameters: retry with fit_phase=no "
                f"(current dof={fit.get('dof')})"
            )
        if fit_hier and n_data < 12:
            steps.append(
                "Priority 1 — Reduce parameters: set fit_hier=no on small vectors "
                f"(n={n_data})"
            )

    if severity in {"critical", "high"}:
        steps.append(
            "Priority 1 — Trust fixed-γ non-circular p-values and γ jackknife; "
            "ignore auto-γ Lomb at ~1τ cycle (circular by construction)"
        )

    diag_done = bool(auto_diagnostics)
    inj_done = bool((auto_diagnostics or {}).get("inj_results"))
    resid_done = bool(residual_diagnostics)
    cycles = float((power_assessment or {}).get("cycles_possible") or 0.0)
    if not diag_done:
        if cycles < 2.0:
            steps.append(
                "Priority 2 — Keep γ fixed; do not auto-calibrate on ≲1.5 τ-cycles "
                "(run_auto_diagnostics with auto_calibrate_gamma=False)"
            )
        else:
            steps.append(
                "Priority 2 — Run auto diagnostics: run_auto_diagnostics(..., z=meta['z']) "
                "with auto_calibrate_gamma=True"
            )
    elif not inj_done:
        steps.append(
            "Priority 2 — Run injection-recovery preview inside auto diagnostics "
            "(run_injection=True)"
        )
    if not resid_done:
        steps.append(
            "Priority 2 — Run full residual diagnostics after Tau-SB fit "
            "(run_full_residual_diagnostics on fit residuals)"
        )
    else:
        shapiro = (residual_diagnostics or {}).get("shapiro_wilk") or {}
        dw = float((residual_diagnostics or {}).get("durbin_watson_stat", float("nan")))
        if not shapiro.get("normal", shapiro.get("is_normal_at_5pct")):
            steps.append(
                "Priority 2 — Residuals fail normality (Shapiro-Wilk); "
                "inspect Q-Q plot and consider fewer free parameters"
            )
        if np.isfinite(dw) and (dw < 1.5 or dw > 2.5):
            rd = residual_diagnostics or {}
            n_res = int(rd.get("n") or n_data)
            lb = rd.get("ljung_box") or {}
            bg = rd.get("breusch_godfrey") or {}
            lb_null = int(lb.get("significant_lag_count") or 0) == 0
            bg_null = not bool(bg.get("significant_at_5pct", False)) if bg else True
            if n_res < 12 and lb_null and bg_null:
                pass  # DW on n=9 is not an ACF action item when LB/BG are null
            else:
                steps.append(
                    f"Priority 2 — Durbin-Watson={dw:.2f} suggests residual autocorrelation; "
                    "check ACF / Breusch-Godfrey output"
                )

    if not steps:
        steps.append(
            "Power and core diagnostics look adequate — proceed to Model Compare "
            "and optional MCMC if publishing"
        )
    return steps