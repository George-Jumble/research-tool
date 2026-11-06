"""Coherent multi-detector ringdown stack and null-stream glitch veto (blind to GR).

Per detector the strain is whitened by its own off-source Welch median PSD, band-passed 35–512 Hz and
scaled to unit off-source σ (:func:`qnm.prepare_whitened`). The highest-SNR detector is the reference.
Every other detector d is aligned to it by the complex (Hilbert analytic) cross-correlation in the
merger–ringdown window [t_peak,ref − 0.06 s, t_peak,ref + 0.03 s]: integer lag + parabolic sub-sample
refinement within the light-travel bound (±12 ms H1–L1, ±30 ms with V1/K1), relative phase
Δφ_d = arg C, applied as an FFT time shift and phase rotation of the analytic signal.

Coherent stack: Σ w_d ỹ_d with w_d ∝ SNR_d and Σ w_d² = 1, which keeps unit background variance and
gives network SNR ≈ √Σ SNR_d².

Null stream (reference r and the next-loudest detector s, amplitude ratio a = |C|/E_ref clipped to
[NULL_RATIO_MIN, 1/NULL_RATIO_MIN]): h_null = (a h_r − h̃_s)/√(1 + a²). A real wave seen in both
detectors cancels; a transient in one detector does not (the clip stops the ratio from collapsing
onto a lone glitch). Over [t_peak, t_peak + 0.032 s]:
  χ²_null,red = ⟨h_null²⟩_window / ⟨h_null²⟩_off-source,
  null_fraction = excess null energy / excess coherent-stack energy.
The event is vetoed when χ²_null,red > NULL_CHI2_MAX (4) and null_fraction > NULL_FRACTION_MAX (0.15).
With the 0.5 clip a lone glitch leaves ≈ 0.25 of the stack energy in the null; a real signal with any
ratio between detectors leaves < 0.15, which is why 0.15 (not 0.5) is the fraction threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from menus.gravitic.ligo.qnm import prepare_whitened

ALIGN_WINDOW_S = (-0.06, 0.03)
NULL_WINDOW_S = 0.032
NULL_RATIO_MIN = 0.5
NULL_CHI2_MAX = 4.0
NULL_FRACTION_MAX = 0.15
MAX_LAG_S = {frozenset({"H1", "L1"}): 0.012}
MAX_LAG_DEFAULT_S = 0.030


def max_lag_s(a: str, b: str) -> float:
    return MAX_LAG_S.get(frozenset({a.upper(), b.upper()}), MAX_LAG_DEFAULT_S)


def fractional_shift(z: np.ndarray, shift_s: float, fs: float) -> np.ndarray:
    """z(t + shift) for a complex series via an FFT phase ramp (circular)."""
    n = z.size
    fr = np.fft.fftfreq(n, 1.0 / fs)
    return np.fft.ifft(np.fft.fft(z) * np.exp(2j * np.pi * fr * shift_s))


def align_to_reference(z_ref: np.ndarray, z_d: np.ndarray, ip: int, fs: float, max_lag: float) -> dict[str, float]:
    """Lag, phase and amplitude ratio that best map detector d onto the reference near the peak."""
    w0, w1 = ip + int(ALIGN_WINDOW_S[0] * fs), ip + int(ALIGN_WINDOW_S[1] * fs)
    L = int(np.ceil(max_lag * fs))
    w0, w1 = max(w0, L), min(w1, z_ref.size - L - 1)
    ref = z_ref[w0:w1]
    lags = np.arange(-L, L + 1)
    c = np.array([np.sum(ref * np.conj(z_d[w0 + s : w1 + s])) for s in lags])
    k = int(np.argmax(np.abs(c)))
    delta = 0.0
    if 0 < k < len(lags) - 1:
        y0, y1, y2 = np.abs(c[k - 1 : k + 2])
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    shift = (lags[k] + delta) / fs
    zd = fractional_shift(z_d, shift, fs)
    cc = np.sum(ref * np.conj(zd[w0:w1]))
    e_ref = float(np.sum(np.abs(ref) ** 2))
    e_d = float(np.sum(np.abs(zd[w0:w1]) ** 2))
    return {"dt_s": float(shift), "dphi_rad": float(np.angle(cc)), "ratio": float(np.abs(cc) / e_ref) if e_ref else 0.0,
            "coherence": float(np.abs(cc) / np.sqrt(e_ref * e_d)) if e_ref and e_d else 0.0}


def _robust_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size > 10 else float(np.std(x))


def common_grid(segments: list[Any], gps: float) -> list[dict[str, Any]]:
    """Whitened, unit-σ series for every detector, cut to a common GPS grid (same sample rate)."""
    preps = []
    for seg in segments:
        prep = prepare_whitened(seg.strain, seg.sample_rate, gps - seg.gps_start)
        prep["detector"] = seg.detector
        prep["gps0"] = seg.gps_start + prep["crop_start"] / prep["fs"]
        preps.append(prep)
    fs = preps[0]["fs"]
    preps = [p for p in preps if p["fs"] == fs]
    # index of the GPS time in each series; equal fractional part since gps_start are integers
    i_gps = [int(round((gps - p["gps0"]) * fs)) for p in preps]
    before = min(i_gps)
    after = min(p["y"].size - i for p, i in zip(preps, i_gps))
    out = []
    for p, i in zip(preps, i_gps):
        sl = slice(i - before, i + after)
        q = dict(p)
        q["y"], q["bg"] = p["y"][sl], p["bg"][sl]
        q["off"] = before / fs
        q["ip"] = p["ip"] - (i - before)
        out.append(q)
    return out


def coherent_combine(preps: list[dict[str, Any]], *, slide_samples: dict[str, int] | None = None) -> dict[str, Any]:
    """Align to the highest-SNR detector, form the SNR-weighted stack and the null stream."""
    fs = preps[0]["fs"]
    order = sorted(range(len(preps)), key=lambda i: preps[i]["snr"], reverse=True)
    ref = preps[order[0]]
    ip = ref["ip"]
    z_ref = signal.hilbert(ref["y"])
    aligned = {ref["detector"]: ref["y"]}
    align: dict[str, dict[str, float]] = {}
    for i in order[1:]:
        p = preps[i]
        z_d = signal.hilbert(p["y"])
        a = align_to_reference(z_ref, z_d, ip, fs, max_lag_s(ref["detector"], p["detector"]))
        zd = fractional_shift(z_d, a["dt_s"], fs) * np.exp(1j * a["dphi_rad"])
        aligned[p["detector"]] = np.real(zd)
        align[p["detector"]] = a
    if slide_samples:
        for det, k in slide_samples.items():
            if det in aligned and det != ref["detector"]:
                aligned[det] = np.roll(aligned[det], int(k))
    snrs = {preps[i]["detector"]: max(preps[i]["snr"], 1e-6) for i in order}
    wn = np.sqrt(sum(v * v for v in snrs.values()))
    weights = {d: snrs[d] / wn for d in snrs}
    stack = sum(weights[d] * aligned[d] for d in aligned)
    bg = np.logical_and.reduce([p["bg"] for p in preps])
    s = _robust_sigma(stack[bg]) if bg.sum() > 100 else 1.0
    stack = stack / (s or 1.0)
    out = {"fs": fs, "reference": ref["detector"], "ip_ref": ip, "aligned": aligned, "align": align,
           "weights": weights, "snr_individual": snrs, "stack": stack, "bg": bg, "off": ref["off"],
           "detectors": [preps[i]["detector"] for i in order]}
    if len(order) >= 2:
        second = preps[order[1]]["detector"]
        a = float(np.clip(align[second]["ratio"], NULL_RATIO_MIN, 1.0 / NULL_RATIO_MIN))
        null = (a * aligned[ref["detector"]] - aligned[second]) / np.sqrt(1.0 + a * a)
        sn = _robust_sigma(null[bg]) if bg.sum() > 100 else 1.0
        out.update(null=null / (sn or 1.0), null_pair=(ref["detector"], second), null_ratio=a)
    return out


def null_stream_veto(comb: dict[str, Any], ip: int | None = None) -> dict[str, Any]:
    """χ²_null,red and the glitch veto over [t_peak, t_peak + 0.032 s]."""
    if "null" not in comb:
        return {"null_stream_chi2_red": None, "null_veto_passed": True, "null_fraction": None,
                "note": "single detector: no null stream"}
    fs = comb["fs"]
    ip = comb["ip_ref"] if ip is None else ip
    w = slice(ip, ip + int(round(NULL_WINDOW_S * fs)))
    null, stack, bg = comb["null"], comb["stack"], comb["bg"]
    var_null = float(np.mean(null[bg] ** 2)) if bg.sum() > 100 else 1.0
    var_stack = float(np.mean(stack[bg] ** 2)) if bg.sum() > 100 else 1.0
    nw = null[w].size
    e_null = float(np.sum(null[w] ** 2))
    e_stack = float(np.sum(stack[w] ** 2))
    chi2 = e_null / (nw * var_null)
    excess_null = max(e_null - nw * var_null, 0.0)
    excess_stack = max(e_stack - nw * var_stack, 1e-12)
    frac = excess_null / excess_stack
    passed = not (chi2 > NULL_CHI2_MAX and frac > NULL_FRACTION_MAX)
    return {"null_stream_chi2_red": float(chi2), "null_fraction": float(frac), "null_veto_passed": bool(passed),
            "null_energy": e_null, "coherent_energy": e_stack, "null_pair": list(comb["null_pair"]),
            "null_ratio": comb["null_ratio"]}


def network_snr(comb: dict[str, Any], search_s: float = 0.10) -> tuple[int, float]:
    """Envelope peak of the coherent stack near the reference peak → (index, network SNR)."""
    fs = comb["fs"]
    env = np.abs(signal.hilbert(comb["stack"]))
    ip = comb["ip_ref"]
    s0, s1 = max(0, ip - int(search_s * fs)), min(env.size, ip + int(search_s * fs) + 1)
    k = s0 + int(np.argmax(env[s0:s1]))
    return k, float(env[k])
