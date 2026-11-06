"""Blind post-merger secondary-marker / echo scan (20–300 ms after the peak) with empirical FAP.

A bank of damped-sinusoid atoms, f ∈ [50, 400] Hz (10 Hz steps) × τ ∈ {4, 10, 25} ms, each an
orthonormal cosine/sine pair so the phase is maximised analytically, is matched-filtered against the
coherent stack and the null stream from :mod:`coherent` (same sky position as the merger, so the
merger's Δt_d / Δφ_d alignment is reused). The statistic for atom j at start time t is

    ρ_gated² = |ρ_stack|² − |ρ_null|²   (clipped at 0; single detector: ρ_gated = |ρ|),

each |ρ|² normalised by its own off-source variance, so a transient in one detector — which leaks
into the null stream — is penalised. The candidate is the maximum over the bank and over start
delays [t_peak + 20 ms, t_peak + 300 ms].

Background: the identical max-over-bank-and-window statistic on off-source windows (|t − t_peak| ≥ 1 s,
0.5 s from the edges, 25 ms steps) of the zero-lag streams and of time-slid streams (non-reference
detectors circularly shifted by ±0.1, ±0.2 s ≫ 50 ms, so no real coincidence survives). FAP =
(1 + #bg ≥ candidate)/(1 + N_bg); σ_equiv = Φ⁻¹(1 − FAP); FAR = #bg ≥ candidate / background live time.
Windows overlap, so N_bg overstates independent trials; FAP is still unbiased per trial.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from scipy.stats import norm

from menus.gravitic.ligo.coherent import coherent_combine

BANK_F_HZ = tuple(range(50, 401, 10))
BANK_TAU_S = (0.004, 0.010, 0.025)
DELAY_S = (0.020, 0.300)
BG_EXCLUDE_S = 1.0
BG_EDGE_S = 0.5
BG_STEP_S = 0.025
SLIDES_S = (-0.2, -0.1, 0.1, 0.2)
DETECT_SIGMA = 3.0
CANDIDATE_NULL_MAX = 0.5


def atom_bank(fs: float, f_hz=BANK_F_HZ, tau_s=BANK_TAU_S) -> list[dict[str, Any]]:
    bank = []
    for tau in tau_s:
        n = int(round(min(5 * tau, 0.12) * fs))
        t = np.arange(n) / fs
        e = np.exp(-t / tau)
        for f in f_hz:
            c = e * np.cos(2 * np.pi * f * t)
            s = e * np.sin(2 * np.pi * f * t)
            c /= np.linalg.norm(c)
            s -= (s @ c) * c
            s /= np.linalg.norm(s)
            bank.append({"f_hz": float(f), "tau_s": float(tau), "c": c, "s": s})
    return bank


def _mf_power(y: np.ndarray, bank: list[dict[str, Any]], bg: np.ndarray) -> np.ndarray:
    """|ρ|² (atoms × start index), each atom normalised to unit mean off-source power per quadrature."""
    n = y.size
    nfft = 1 << int(np.ceil(np.log2(n + max(a["c"].size for a in bank))))
    Y = np.fft.rfft(y, nfft)
    out = np.empty((len(bank), n))
    for j, a in enumerate(bank):
        rc = np.fft.irfft(Y * np.conj(np.fft.rfft(a["c"], nfft)), nfft)[:n]
        rs = np.fft.irfft(Y * np.conj(np.fft.rfft(a["s"], nfft)), nfft)[:n]
        p = rc * rc + rs * rs
        m = bg[:n].copy()
        m[n - a["c"].size :] = False
        norm_ = 0.5 * float(np.mean(p[m])) if m.any() else 1.0
        out[j] = p / (norm_ if norm_ > 0 else 1.0)  # E[|ρ|²] = 2 on noise (two unit quadratures)
    return out


def _gated(stack_p: np.ndarray, null_p: np.ndarray | None) -> np.ndarray:
    if null_p is None:
        return np.sqrt(stack_p)
    return np.sqrt(np.clip(stack_p - null_p, 0.0, None))


def _window_max(stat_t: np.ndarray, width: int) -> np.ndarray:
    """max over [t, t + width) for each t."""
    return ndimage.maximum_filter1d(stat_t, size=width, origin=-(width // 2), mode="nearest")


def scan_postmerger_markers(
    stack: np.ndarray,
    fs: float,
    ip: int,
    *,
    null: np.ndarray | None = None,
    bg: np.ndarray | None = None,
    slid_streams: list[tuple[np.ndarray, np.ndarray | None]] | None = None,
    null_veto_passed: bool = True,
    delay_s: tuple[float, float] = DELAY_S,
) -> dict[str, Any]:
    """Loudest post-merger damped-sinusoid candidate and its empirical significance (GR-blind)."""
    n = stack.size
    bank = atom_bank(fs)
    if bg is None:
        bg = np.ones(n, dtype=bool)
    t = np.arange(n) / fs
    t_peak = ip / fs
    width = int(round((delay_s[1] - delay_s[0]) * fs)) + 1

    def stat_series(st: np.ndarray, nu: np.ndarray | None):
        sp = _mf_power(st, bank, bg)
        npow = _mf_power(nu, bank, bg) if nu is not None else None
        g = _gated(sp, npow)
        return g, sp, npow

    g, sp, npow = stat_series(stack, null)
    i0, i1 = ip + int(round(delay_s[0] * fs)), min(n - 1, ip + int(round(delay_s[1] * fs)))
    if i1 <= i0:
        raise ValueError("post-merger window outside the segment")
    sub = g[:, i0 : i1 + 1]
    j, k = np.unravel_index(int(np.argmax(sub)), sub.shape)
    cand = float(sub[j, k])
    ic = i0 + int(k)
    null_ratio = float(npow[j, ic] / sp[j, ic]) if npow is not None and sp[j, ic] > 0 else 0.0
    # background: window starts well away from the event and from the edges
    starts_ok = (np.abs(t - t_peak) >= BG_EXCLUDE_S) & (np.abs(t + (delay_s[1] - delay_s[0]) - t_peak) >= BG_EXCLUDE_S)
    starts_ok &= (t >= BG_EDGE_S) & (t + (delay_s[1] - delay_s[0]) <= t[-1] - BG_EDGE_S)
    step = max(1, int(round(BG_STEP_S * fs)))
    starts = np.flatnonzero(starts_ok)[::step]
    bg_vals = [_window_max(g.max(axis=0), width)[starts]]
    for st, nu in slid_streams or []:
        gs, _, _ = stat_series(st, nu)
        bg_vals.append(_window_max(gs.max(axis=0), width)[starts])
    bgv = np.concatenate(bg_vals) if bg_vals else np.array([])
    n_bg = int(bgv.size)
    n_ge = int(np.sum(bgv >= cand))
    fap = (1 + n_ge) / (1 + n_bg)
    live_s = float(starts.size * BG_STEP_S * (1 + len(slid_streams or [])))
    sig = float(norm.isf(fap))
    cand_veto = null_ratio <= CANDIDATE_NULL_MAX
    passed = bool(null_veto_passed and cand_veto)
    return {
        "delay_ms": float((ic - ip) / fs * 1e3), "f_hz": bank[j]["f_hz"], "tau_ms": bank[j]["tau_s"] * 1e3,
        "network_snr": cand, "candidate_null_ratio": null_ratio, "null_veto_passed": passed,
        "fap": float(fap), "far_per_s": float(n_ge / live_s) if live_s else None, "sigma_equiv": sig,
        "n_background": n_bg, "n_slides": len(slid_streams or []), "background_live_s": live_s,
        "background_median": float(np.median(bgv)) if n_bg else None,
        "background_p99": float(np.quantile(bgv, 0.99)) if n_bg else None,
        "detected": bool(sig >= DETECT_SIGMA and passed),
        "bank": {"f_hz": [BANK_F_HZ[0], BANK_F_HZ[-1], 10], "tau_ms": [x * 1e3 for x in BANK_TAU_S],
                 "delay_ms": [delay_s[0] * 1e3, delay_s[1] * 1e3]},
    }


def scan_from_preps(preps: list[dict[str, Any]], comb: dict[str, Any], ip: int, null_veto_passed: bool = True) -> dict[str, Any]:
    """Run the scan on a coherent combination, with time slides when ≥ 2 detectors."""
    fs = comb["fs"]
    slides = []
    if len(preps) >= 2:
        others = [d for d in comb["detectors"] if d != comb["reference"]]
        for sh in SLIDES_S:
            k = int(round(sh * fs))
            c2 = coherent_combine(preps, slide_samples={d: k * (i + 1) for i, d in enumerate(others)})
            slides.append((c2["stack"], c2.get("null")))
    return scan_postmerger_markers(comb["stack"], fs, ip, null=comb.get("null"), bg=comb["bg"],
                                   slid_streams=slides, null_veto_passed=null_veto_passed)
