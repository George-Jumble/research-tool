"""Ringdown frequency vs the GR Kerr l=m=2, n=0 quasinormal mode.

GR baseline (Berti, Cardoso & Will 2006 fit):
    f_220 = c³/(2π G M_f) · [1.5251 − 1.1568 (1 − a_f)^0.1292],
with M_f the detector-frame final mass, (1+z)·M_f,source. Mass and redshift come from the cached
GWOSC allevents catalog; the final spin only exists in the per-event detail JSON (``jsonurl``),
inside ``parameters`` (GWTC-2.1/3: ``final_spin``; GWTC-1: ``a_final``), cached as
``events/<EVENT>/event_detail.json``. σ(f_220) propagates the mass, spin and redshift bounds.

Observed value (never uses f_220): per detector file,
  1. strain segments ≥ 4 s are whitened by a Welch median PSD estimated with ±0.5 s around the
     event excluded (shorter synthetic segments are not whitened);
  2. 35–512 Hz band-pass (zero-phase Butterworth);
  3. the Hilbert-envelope peak within ±0.10 s of the catalog GPS defines t_peak and SNR
     (peak / robust noise σ of the band-passed data away from the event);
  4. the post-peak window [t_peak, t_peak + 0.05 s] gets a one-sided (1→0) Hann taper, so the window
     end has no step; the zero-padded periodogram peak is f_obs, σ = Rayleigh resolution 1/T
     (kept as ``measured_peak_hz``).
  5. blind damped-sinusoid fit (:func:`fit_damped_ringdown`) on [t_peak + 2 ms, + 30 ms]: variable
     projection over (f, τ), bounded least squares, sandwich covariance with the off-source Toeplitz
     noise covariance → (f, σ_f), (τ, σ_τ), γ = 1/(2πτ) [Hz], Q = π f τ. No GR quantity enters.

GR (2,2,0) also gives Q_220 = 0.7000 + 1.4187 (1 − a_f)^−0.4990, τ_220 = Q/(π f_220) and
γ_220 = f_220/(2 Q_220) (same Berti, Cardoso & Will 2006 fit), with σ propagated from M, z and a_f.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from menus.gravitic.common.gwosc_api import entry_rank

T_SUN_S = 4.925490947e-6  # G M_sun / c³
RINGDOWN_WINDOW_S = (0.0, 0.05)  # relative to the whitened envelope peak
BANDPASS_HZ = (35.0, 512.0)
PEAK_SEARCH_S = 0.10
PSD_EXCLUDE_S = 0.5
PSD_SPAN_S = 64.0  # strain used around the event for the PSD (4096 s files are cropped)
PSD_SEG_S = 0.5  # Welch segment length (2 Hz resolution, ~120 median averages on 32 s files)
MIN_WHITEN_S = 4.0
DEFAULT_MIN_SNR = 4.0
SPIN_KEYS = ("final_spin", "a_final", "chi_f")


def kerr_f220_hz(m_final_detector_msun: float, a_final: float) -> float:
    a = min(max(float(a_final), 0.0), 0.9999)
    omega = 1.5251 - 1.1568 * (1.0 - a) ** 0.1292
    return omega / (2.0 * math.pi * T_SUN_S * float(m_final_detector_msun))


def kerr_q220(a_final: float) -> float:
    """Quality factor of the Kerr (2,2,0) mode, Berti, Cardoso & Will (2006): Q = 0.7 + 1.4187 (1−a)^−0.4990."""
    a = min(max(float(a_final), 0.0), 0.9999)
    return 0.7000 + 1.4187 * (1.0 - a) ** (-0.4990)


def kerr_220(m_final_detector_msun: float, a_final: float) -> dict[str, float]:
    """f_220 [Hz], Q_220, τ_220 = Q/(π f) [s] and damping rate γ_220 = 1/(2π τ) = f/(2Q) [Hz]."""
    f = kerr_f220_hz(m_final_detector_msun, a_final)
    q = kerr_q220(a_final)
    tau = q / (math.pi * f)
    return {"f": f, "q": q, "tau": tau, "gamma": f / (2.0 * q)}


def _num(meta: dict, *keys: str) -> float | None:
    for key in keys:
        val = meta.get(key)
        try:
            if val is not None and math.isfinite(float(val)):
                return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _pe_score(params: dict[str, Any]) -> tuple[int, int, int]:
    has_spin = any(params.get(k) is not None for k in SPIN_KEYS)
    has_mass = params.get("final_mass_source") is not None
    return (1 if params.get("is_preferred") else 0, 1 if str(params.get("pipeline_type", "")).lower() == "pe" else 0,
            int(has_spin) + int(has_mass))


def extract_pe_parameters(detail: dict[str, Any] | None) -> dict[str, Any]:
    """Best PE parameter set from a GWOSC event-detail JSON (``is_preferred`` PE first).

    Accepts the event dict (with ``parameters``) or the raw API payload (``{"events": {...}}``).
    """
    if not isinstance(detail, dict):
        return {}
    if "parameters" not in detail and isinstance(detail.get("events"), dict):
        events = [e for e in detail["events"].values() if isinstance(e, dict)]
        detail = events[0] if len(events) == 1 else (max(events, key=lambda e: int(e.get("version") or 0)) if events else {})
    params = detail.get("parameters") or {}
    sets = [p for p in params.values() if isinstance(p, dict)]
    usable = [p for p in sets if any(p.get(k) is not None for k in SPIN_KEYS + ("final_mass_source",))]
    if not usable:
        return {}
    return dict(max(usable, key=_pe_score))


def resolve_event_metadata(meta: dict[str, Any] | None, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    """Catalog entry merged with the PE parameters (PE values fill gaps; catalog mass/z kept if present).

    Mass, spin, redshift and their bounds are taken as one consistent set: from the PE set when it
    has them, else from the catalog entry.
    """
    out = dict(meta or {})
    pe = extract_pe_parameters(detail if detail is not None else (out if "parameters" in out else None))
    if not pe:
        return out
    out.pop("parameters", None)
    for base in ("final_mass_source", "redshift") + SPIN_KEYS:
        if out.get(base) is None and pe.get(base) is not None:
            for suffix in ("", "_lower", "_upper"):
                if pe.get(base + suffix) is not None:
                    out[base + suffix] = pe[base + suffix]
    out.setdefault("pe_set", pe.get("name") or pe.get("pipeline_type") or "pe")
    return out


def _spin_with_bounds(meta: dict[str, Any]) -> tuple[float | None, float]:
    for key in SPIN_KEYS:
        val = _num(meta, key)
        if val is not None:
            da = 0.5 * (abs(_num(meta, key + "_upper") or 0.0) + abs(_num(meta, key + "_lower") or 0.0))
            return val, da
    return None, 0.0


def gr_prediction(meta: dict[str, Any]) -> dict[str, Any]:
    """f_220 and its catalog uncertainty from one GWOSC catalog entry (PE ``parameters`` honoured)."""
    if isinstance(meta.get("parameters"), dict):
        meta = resolve_event_metadata(meta)
    m_src = _num(meta, "final_mass_source")
    spin, da = _spin_with_bounds(meta)
    z = _num(meta, "redshift")
    missing = [n for n, v in (("final_mass_source", m_src), ("final_spin", spin), ("redshift", z)) if v is None]
    if missing:
        return {"ok": False, "missing": missing}
    m_det = (1.0 + z) * m_src
    k = kerr_220(m_det, spin)
    dm = 0.5 * (abs(_num(meta, "final_mass_source_upper") or 0.0) + abs(_num(meta, "final_mass_source_lower") or 0.0))
    dz = 0.5 * (abs(_num(meta, "redshift_upper") or 0.0) + abs(_num(meta, "redshift_lower") or 0.0))
    dm_det = math.hypot((1.0 + z) * dm, m_src * dz)
    # Linear propagation of independent (M_det, a_f) errors; central differences in spin.
    eps = 1e-4
    hi_a, lo_a = min(spin + eps, 0.9999), max(spin - eps, 0.0)
    kp, km = kerr_220(m_det, hi_a), kerr_220(m_det, lo_a)
    sig: dict[str, float] = {}
    for key in ("f", "q", "tau", "gamma"):
        d_da = (kp[key] - km[key]) / (hi_a - lo_a)
        # f, γ ∝ 1/M_det; τ ∝ M_det; Q independent of M_det
        d_dm = {"f": -k["f"], "gamma": -k["gamma"], "tau": k["tau"], "q": 0.0}[key] / m_det if m_det else 0.0
        sig[key] = math.hypot(d_dm * dm_det, d_da * da)
    return {"ok": True, "f220_hz": k["f"], "sigma_hz": sig["f"], "m_final_detector": m_det,
            "a_final": spin, "sigma_m_detector": dm_det, "sigma_spin": da,
            "q220": k["q"], "sigma_q": sig["q"], "tau220_s": k["tau"], "sigma_tau_s": sig["tau"],
            "gamma220_hz": k["gamma"], "sigma_gamma_hz": sig["gamma"]}


def _welch_psd(x: np.ndarray, fs: float, off_s: float) -> tuple[np.ndarray, np.ndarray]:
    n = x.size
    nper = int(fs * min(PSD_SEG_S, n / fs / 8.0))  # many averages: a noisy PSD distorts the whitened signal
    ex0, ex1 = int((off_s - PSD_EXCLUDE_S) * fs), int((off_s + PSD_EXCLUDE_S) * fs)
    psds, weights, freqs = [], [], None
    for part in (x[: max(ex0, 0)], x[max(ex1, 0):]):
        if part.size >= nper:
            freqs, p = signal.welch(part, fs=fs, nperseg=nper, average="median")
            psds.append(p)
            weights.append(part.size)
    if not psds:
        raise ValueError("not enough off-source strain for a PSD")
    return freqs, np.average(psds, axis=0, weights=weights)


def prepare_whitened(
    strain: np.ndarray,
    sample_rate: float,
    offset_s: float,
    *,
    band_hz: tuple[float, float] = BANDPASS_HZ,
    search_s: float = PEAK_SEARCH_S,
) -> dict[str, Any]:
    """Whiten (segments ≥ 4 s) + band-pass, scaled so the off-source background has unit (robust) σ.

    Returns y, fs, off (event offset in y, s), ip (envelope-peak index within ±search_s), snr, whitened,
    bg (bool mask of off-source samples: ±0.5 s around the event and 0.5 s at each edge excluded).
    """
    fs = float(sample_rate)
    x = np.asarray(strain, dtype=float)
    i_off = int(round(offset_s * fs))
    if not (0 <= i_off < x.size):
        raise ValueError(f"event offset {offset_s:.3f} s outside the {x.size / fs:.1f} s segment")
    half = int(PSD_SPAN_S / 2 * fs)
    a, b = max(0, i_off - half), min(x.size, i_off + half)
    x = x[a:b]
    off = offset_s - a / fs
    if not np.all(np.isfinite(x)):
        raise ValueError("strain has NaN/inf near the event")
    x = x - np.mean(x)
    n = x.size
    dur = n / fs
    whitened = dur >= MIN_WHITEN_S
    if whitened:
        freqs, psd = _welch_psd(x, fs, off)
        taper = signal.windows.tukey(n, alpha=min(1.0, 1.0 / dur))
        spec = np.fft.rfft(x * taper)
        fr = np.fft.rfftfreq(n, 1.0 / fs)
        spec /= np.sqrt(np.maximum(np.interp(fr, freqs, psd), 1e-300) * fs / 2.0)
        y = np.fft.irfft(spec, n)
    else:
        y = x
    hi = min(band_hz[1], 0.45 * fs)
    sos = signal.butter(4, [band_hz[0], hi], btype="band", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, y)
    t = np.arange(n) / fs
    bg = (np.abs(t - off) > PSD_EXCLUDE_S)
    if whitened:
        bg &= (t > 0.5) & (t < dur - 0.5)
    if bg.sum() > 100:
        noise = 1.4826 * float(np.median(np.abs(y[bg] - np.median(y[bg]))))
    else:
        noise = float(np.std(y))
        bg = np.zeros(n, dtype=bool)
    if noise > 0:
        y = y / noise
    env = np.abs(signal.hilbert(y))
    s0, s1 = max(0, int((off - search_s) * fs)), min(n, int((off + search_s) * fs) + 1)
    if s1 - s0 < 2:
        raise ValueError("peak search window outside the segment")
    ip = s0 + int(np.argmax(env[s0:s1]))
    return {"y": y, "fs": fs, "off": float(off), "ip": ip, "snr": float(env[ip]), "whitened": bool(whitened),
            "bg": bg, "band_hz": (float(band_hz[0]), float(hi)), "crop_start": a}


def noise_acf(y: np.ndarray, bg: np.ndarray, maxlag: int) -> np.ndarray:
    """Empirical autocovariance of the off-source samples (contiguous runs only), lags 0..maxlag−1."""
    acc = np.zeros(maxlag)
    cnt = np.zeros(maxlag)
    if bg is None or not np.any(bg):
        bg = np.ones(y.size, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], bg.astype(int), [0]])))
    for lo, hi in zip(edges[::2], edges[1::2]):
        run = y[lo:hi] - np.mean(y[lo:hi])
        m = run.size
        if m <= maxlag:
            continue
        nfft = 1 << int(np.ceil(np.log2(2 * m)))
        F = np.fft.rfft(run, nfft)
        r = np.fft.irfft(F * np.conj(F), nfft)[:maxlag]
        acc += r
        cnt += m - np.arange(maxlag)
    if not np.any(cnt):
        out = np.zeros(maxlag)
        out[0] = float(np.var(y))
        return out
    return acc / np.maximum(cnt, 1)


def _vp_basis(t: np.ndarray, f: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    env = np.exp(-t / tau[..., None])
    ph = 2.0 * np.pi * f[..., None] * t
    return env * np.cos(ph), env * np.sin(ph)


def _vp_explained(d: np.ndarray, t: np.ndarray, f: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Energy captured by the exact linear LS (c1, c2) at each trial (f, τ) (variable projection)."""
    b1, b2 = _vp_basis(t, f, tau)
    g11, g22, g12 = (b1 * b1).sum(-1), (b2 * b2).sum(-1), (b1 * b2).sum(-1)
    r1, r2 = (b1 * d).sum(-1), (b2 * d).sum(-1)
    det = g11 * g22 - g12 * g12
    det = np.where(det > 1e-300, det, np.inf)
    return (g22 * r1 * r1 - 2 * g12 * r1 * r2 + g11 * r2 * r2) / det


FIT_DT_START_S = 0.002
FIT_T_S = 0.030
FIT_TAU_BOUNDS_S = (0.0008, 0.080)
FIT_TAU_GRID_S = (0.0015, 0.003, 0.006, 0.012, 0.025)


def fit_damped_ringdown(
    y: np.ndarray,
    fs: float,
    i0: int,
    *,
    n_fit: int,
    acf: np.ndarray | None = None,
    band_hz: tuple[float, float] = BANDPASS_HZ,
) -> dict[str, Any]:
    """Blind damped-sinusoid fit h = e^{−(t−t0)/τ}[c1 cos 2πf(t−t0) + c2 sin 2πf(t−t0)] on y[i0:i0+n_fit].

    Uses only the data: a coarse variable-projection grid over f (2 Hz steps in the band plus the top
    spectral peaks) × τ ∈ {1.5, 3, 6, 12, 25} ms, then bounded ``least_squares`` refinement in (f, τ) with
    (c1, c2) solved exactly at each step. The covariance of (f, τ, c1, c2) is the sandwich
    (JᵀJ)⁻¹ Jᵀ Σ_n J (JᵀJ)⁻¹ with Σ_n the Toeplitz matrix of the off-source noise autocovariance
    ``acf`` (so filter-induced sample correlation is accounted for). No GR quantity enters.
    """
    from scipy.linalg import toeplitz
    from scipy.optimize import least_squares

    d = np.asarray(y[i0 : i0 + n_fit], dtype=float)
    n = d.size
    if n < 32 or i0 < 0:
        raise ValueError("fit window outside the segment")
    t = np.arange(n) / fs
    lo, hi = float(band_hz[0]), float(min(band_hz[1], 0.45 * fs))
    nfft = max(8192, 1 << int(np.ceil(np.log2(n * 16))))
    power = np.abs(np.fft.rfft(d * np.hanning(n), nfft)) ** 2
    fq = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (fq >= lo) & (fq <= hi)
    pk, _ = signal.find_peaks(np.where(band, power, 0.0))
    top = fq[pk[np.argsort(power[pk])[::-1][:3]]] if pk.size else np.array([])
    f_grid = np.unique(np.concatenate([np.arange(lo, hi + 1e-9, 2.0), top]))
    tau_grid = np.asarray(FIT_TAU_GRID_S)
    F, T = np.meshgrid(f_grid, tau_grid, indexing="ij")
    ex = _vp_explained(d, t, F, T)
    k = np.unravel_index(int(np.argmax(ex)), ex.shape)
    tb = FIT_TAU_BOUNDS_S

    def coeffs(f: float, tau: float) -> np.ndarray:
        b1, b2 = _vp_basis(t, np.array(f), np.array(tau))
        A = np.stack([b1, b2], axis=1)
        return np.linalg.lstsq(A, d, rcond=None)[0], A

    def resid(p: np.ndarray) -> np.ndarray:
        c, A = coeffs(p[0], p[1])
        return d - A @ c

    x0 = np.array([F[k], T[k]])
    sol = least_squares(resid, x0, bounds=([lo, tb[0]], [hi, tb[1]]), x_scale=[10.0, 0.003])
    f, tau = float(sol.x[0]), float(sol.x[1])
    c, A = coeffs(f, tau)
    e = np.exp(-t / tau)
    w = 2 * np.pi * f
    model = A @ c
    J = np.stack([
        e * (-c[0] * np.sin(w * t) + c[1] * np.cos(w * t)) * 2 * np.pi * t,  # ∂/∂f
        model * t / tau**2,  # ∂/∂τ
        A[:, 0], A[:, 1],
    ], axis=1)
    if acf is None or acf.size < n:
        acf_n = np.zeros(n)
        acf_n[0] = float(np.var(d - model))
    else:
        acf_n = np.asarray(acf[:n], dtype=float)
    sigma_n = toeplitz(acf_n)
    jtj_inv = np.linalg.pinv(J.T @ J)
    cov = jtj_inv @ (J.T @ sigma_n @ J) @ jtj_inv
    sig_f = float(np.sqrt(max(cov[0, 0], 0.0)))
    sig_tau = float(np.sqrt(max(cov[1, 1], 0.0)))
    gamma = 1.0 / (2 * np.pi * tau)
    amp = float(np.hypot(c[0], c[1]))
    at_bound = tau <= tb[0] * 1.05 or tau >= tb[1] * 0.95 or f <= lo + 1.0 or f >= hi - 1.0
    snr_fit = float(np.sqrt(max(model @ model, 0.0) / max(acf_n[0], 1e-300)))
    return {
        "f_hz": f, "sigma_f_hz": sig_f, "tau_s": tau, "sigma_tau_s": sig_tau,
        "gamma_hz": gamma, "sigma_gamma_hz": sig_tau / (2 * np.pi * tau**2),
        "q": float(np.pi * f * tau), "amplitude": amp, "phase_rad": float(np.arctan2(-c[1], c[0])),
        "snr_fit": snr_fit, "tau_at_bound": bool(at_bound),
        "gamma_constrained": bool(not at_bound and np.isfinite(sig_tau) and sig_tau < tau),
        "n_fit": int(n), "fit_cost": float(sol.cost),
    }


def measure_ringdown(
    strain: np.ndarray,
    sample_rate: float,
    offset_s: float,
    *,
    window_s: float = RINGDOWN_WINDOW_S[1] - RINGDOWN_WINDOW_S[0],
    band_hz: tuple[float, float] = BANDPASS_HZ,
    search_s: float = PEAK_SEARCH_S,
    dt_start_s: float = FIT_DT_START_S,
    fit_s: float = FIT_T_S,
    fit: bool = True,
) -> dict[str, Any]:
    """Whitened ringdown measurement around ``offset_s`` (seconds into ``strain``); no GR input.

    Returns the periodogram f_hz / sigma_hz (1/T) of the tapered post-peak window, t_peak_offset_s,
    snr_peak, whitened, window_s and, when ``fit``, ``fit`` = :func:`fit_damped_ringdown` on
    [t_peak + dt_start_s, + fit_s].
    """
    prep = prepare_whitened(strain, sample_rate, offset_s, band_hz=band_hz, search_s=search_s)
    out = ringdown_from_prepared(prep, window_s=window_s, dt_start_s=dt_start_s, fit_s=fit_s, fit=fit)
    return out


def ringdown_from_prepared(
    prep: dict[str, Any],
    *,
    window_s: float = RINGDOWN_WINDOW_S[1] - RINGDOWN_WINDOW_S[0],
    dt_start_s: float = FIT_DT_START_S,
    fit_s: float = FIT_T_S,
    fit: bool = True,
) -> dict[str, Any]:
    y, fs, ip = prep["y"], prep["fs"], prep["ip"]
    lo, hi = prep["band_hz"]
    L = int(round(window_s * fs))
    seg = y[ip : ip + L]
    if seg.size < 64:
        raise ValueError("ringdown window too short")
    L = seg.size
    one_sided = 0.5 * (1.0 + np.cos(np.pi * np.arange(L) / (L - 1)))  # 1 at t_peak → 0 at window end
    nfft = max(8192, 1 << int(np.ceil(np.log2(L * 16))))
    power = np.abs(np.fft.rfft(seg * one_sided, nfft)) ** 2
    fq = np.fft.rfftfreq(nfft, 1.0 / fs)
    m = (fq >= lo) & (fq <= hi)
    f = float(fq[m][int(np.argmax(power[m]))])
    out = {"f_hz": f, "sigma_hz": float(fs / L), "t_peak_offset_s": float(ip / fs - prep["off"]),
           "snr_peak": float(prep["snr"]), "whitened": bool(prep["whitened"]), "window_s": float(L / fs)}
    if fit:
        n_fit = int(round(fit_s * fs))
        acf = noise_acf(y, prep["bg"], n_fit)
        out["fit"] = fit_damped_ringdown(y, fs, ip + int(round(dt_start_s * fs)), n_fit=n_fit, acf=acf,
                                         band_hz=(lo, hi))
        out["fit"]["dt_start_s"] = float(dt_start_s)
        out["fit"]["window_s"] = float(n_fit / fs)
    return out


def measured_peak_hz(strain: np.ndarray, sample_rate: float, offset_s: float) -> tuple[float, float]:
    """(f_obs, σ) of the whitened post-peak ringdown window; see :func:`measure_ringdown`."""
    res = measure_ringdown(strain, sample_rate, offset_s, fit=False)
    return res["f_hz"], res["sigma_hz"]


def load_catalog(path: Path | str) -> dict[str, dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def catalog_entry(catalog: dict[str, dict[str, Any]], event: str) -> dict[str, Any] | None:
    want = event.strip().upper()
    best = None
    for meta in catalog.values():
        if str(meta.get("commonName", "")).upper() == want:
            if best is None or entry_rank(meta) > entry_rank(best):
                best = meta
    return best
