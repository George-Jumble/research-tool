"""
Material-differentiation gate for multi-material Casimir batches.

After raw-series integrity hashes are unique, pairwise distances of full
signature feature vectors must exceed a relative threshold for every pair of
chemically distinct materials. Collision aborts the batch and blocks summary
tables.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

import numpy as np

DEFAULT_RELATIVE_THRESHOLD = 1e-6
SPECTRUM_BINS = 128
STEP_SLOTS = 10
CHANGEPOINT_SLOTS = 10

_MATERIAL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ben(_au)?|benzene", "Ben_Au"),
    (r"cyc(_au)?|cyclo", "Cyc_Au"),
    (r"oct(_au)?|octane", "Oct_Au"),
    (r"gold|dielptgold", "Au"),
    (r"\bag\b|silver|dielptag", "Ag"),
    (r"\bal\b|alumini|dielptal", "Al"),
    (r"\bli\b|lithium|dielptli", "Li"),
)


class MaterialSignatureCollisionError(RuntimeError):
    """Raised when chemically distinct materials share indistinguishable signatures."""


def material_distance_threshold() -> float:
    raw = os.getenv("TAV_CASIMIR_MATERIAL_DISTANCE_THRESHOLD", str(DEFAULT_RELATIVE_THRESHOLD))
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_RELATIVE_THRESHOLD
    return value if value > 0.0 else DEFAULT_RELATIVE_THRESHOLD


def infer_material_id(label: str, file_path: str | None = None) -> str:
    """Map a data label / filename to a chemical identity (Ag, Al, Au, …)."""
    blob = f"{label} {file_path or ''}".lower()
    for pattern, material in _MATERIAL_PATTERNS:
        if re.search(pattern, blob, flags=re.IGNORECASE):
            return material
    stem = re.sub(r"[^a-z0-9]+", "", (label or "").lower())
    return stem or "unknown"


def _feat(analysis_or_result: dict[str, Any], ftype: str) -> dict[str, Any]:
    for feat in analysis_or_result.get("detected_features") or []:
        if feat.get("type") == ftype:
            return feat
    return {}


def _resample_spectrum(freqs: list[float] | np.ndarray, power: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(power, dtype=np.float64).ravel()
    f = np.asarray(freqs, dtype=np.float64).ravel()
    if p.size == 0:
        return np.zeros(SPECTRUM_BINS, dtype=np.float64)
    if p.size == SPECTRUM_BINS:
        return np.ascontiguousarray(p)
    if f.size == p.size and f.size > 1:
        grid = np.linspace(float(f.min()), float(f.max()), SPECTRUM_BINS)
        return np.interp(grid, f, p)
    src = np.linspace(0.0, 1.0, p.size)
    dst = np.linspace(0.0, 1.0, SPECTRUM_BINS)
    return np.interp(dst, src, p)


def _pad_locations(values: Iterable[float] | None, slots: int) -> np.ndarray:
    arr = np.asarray(list(values or []), dtype=np.float64).ravel()
    out = np.zeros(slots, dtype=np.float64)
    n = min(arr.size, slots)
    if n:
        out[:n] = arr[:n]
    return out


def build_signature_feature_vector(
    *,
    detected_features: list[dict[str, Any]],
    diagnostics: dict[str, Any] | None = None,
    tav_harmonics: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Concatenate periods, dominant frequencies, residual power spectrum,
    step locations, hysteresis magnitude, and changepoint locations.
    """
    diag = diagnostics or {}
    clock = next((f for f in detected_features if f.get("type") == "seven_phase_clockwork"), {})
    steps = next((f for f in detected_features if f.get("type") == "discrete_steps_thresholds"), {})
    hyst = next((f for f in detected_features if f.get("type") == "hysteresis_asymmetry"), {})
    cps = next((f for f in detected_features if f.get("type") == "changepoints"), {})
    tav = tav_harmonics or {}

    periods = _pad_locations(clock.get("periods"), 5)
    if not np.any(periods) and clock.get("expected_freq_1_over_7"):
        freq = float(clock.get("observed_frequency") or clock.get("expected_freq_1_over_7") or 0.0)
        periods = _pad_locations([1.0 / freq if freq else 0.0], 5)
    dominant = _pad_locations(
        clock.get("dominant_freqs")
        or ([clock["observed_frequency"]] if clock.get("observed_frequency") is not None else []),
        5,
    )
    spectrum = _resample_spectrum(diag.get("freqs") or [], diag.get("power") or [])
    step_locs = _pad_locations(steps.get("locations"), STEP_SLOTS)
    hyst_mag = np.array([float(hyst.get("magnitude") or 0.0)], dtype=np.float64)
    cp_locs = _pad_locations(cps.get("locations"), CHANGEPOINT_SLOTS)
    scalars = np.array(
        [
            float(clock.get("power_at_one_seventh") or 0.0),
            float(clock.get("snr_local") or 0.0),
            float(clock.get("p_value") or 1.0),
            float(steps.get("n_steps") or 0.0),
            float(steps.get("p_value") or 1.0),
            float(hyst.get("p_value") or 1.0),
            float(cps.get("n_breaks") or 0.0),
            float(cps.get("p_value") or 1.0),
            float(cps.get("penalty") or 0.0),
            float(tav.get("p_value_holm") or tav.get("p_value") or 1.0),
        ],
        dtype=np.float64,
    )
    vec = np.concatenate([periods, dominant, spectrum, step_locs, hyst_mag, cp_locs, scalars])
    return np.ascontiguousarray(np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0))


def pairwise_distances(vec_a: np.ndarray, vec_b: np.ndarray) -> dict[str, float]:
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    diff = a - b
    euclid = float(np.linalg.norm(diff))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    denom = max(na, nb, 1e-30)
    rel_euclid = euclid / denom
    dot = float(np.dot(a, b))
    cos_sim = dot / max(na * nb, 1e-30)
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
    cosine_distance = 1.0 - cos_sim
    return {
        "euclidean": euclid,
        "relative_euclidean": rel_euclid,
        "cosine_distance": cosine_distance,
        "cosine_similarity": cos_sim,
        "norm_a": na,
        "norm_b": nb,
    }


def evaluate_material_gate(
    records: list[dict[str, Any]],
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Pairwise gate. ``records`` items need material_id, data_label, feature_vector.

    Returns a report dict. ``claim_allowed`` is True only when every chemically
    distinct pair is above threshold.
    """
    thresh = float(threshold) if threshold is not None else material_distance_threshold()
    pairs: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = records[i], records[j]
            if a.get("material_id") == b.get("material_id"):
                continue
            dist = pairwise_distances(a["feature_vector"], b["feature_vector"])
            row = {
                "material_a": a.get("material_id"),
                "material_b": b.get("material_id"),
                "label_a": a.get("data_label"),
                "label_b": b.get("data_label"),
                **dist,
            }
            too_close = (
                dist["relative_euclidean"] < thresh or dist["cosine_distance"] < thresh
            )
            row["below_threshold"] = too_close
            pairs.append(row)
            if too_close:
                collisions.append(row)

    distinct_materials = sorted({str(r.get("material_id")) for r in records})
    claim_allowed = len(collisions) == 0 and len(distinct_materials) >= 2
    report = {
        "threshold_relative": thresh,
        "metric": "relative_euclidean and cosine_distance",
        "n_records": n,
        "materials": distinct_materials,
        "n_distinct_pairs": len(pairs),
        "pairs": pairs,
        "collisions": collisions,
        "claim_allowed": claim_allowed,
        "integrity_confirmed": all(r.get("sha256") for r in records)
        and len({r.get("sha256") for r in records}) == n,
    }
    return report


def enforce_material_gate(
    records: list[dict[str, Any]],
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Confirm unique raw-series hashes, then require distinct signatures.

    Raises MaterialSignatureCollisionError and must not be followed by summary writes.
    """
    hashes = [r.get("sha256") for r in records]
    if hashes and len(set(hashes)) != len(hashes):
        msg = "identical signatures across materials — claim falsified"
        print(f"[CASIMIR MATERIAL GATE] CRITICAL: {msg} (duplicate raw-series SHA-256)")
        raise MaterialSignatureCollisionError(msg)

    report = evaluate_material_gate(records, threshold=threshold)
    if not report["integrity_confirmed"] and records:
        print("[CASIMIR MATERIAL GATE] Warning: missing SHA-256 on one or more records")
    if report["collisions"]:
        msg = "identical signatures across materials — claim falsified"
        print(f"[CASIMIR MATERIAL GATE] CRITICAL: {msg}")
        for hit in report["collisions"]:
            print(
                f"  {hit['material_a']} vs {hit['material_b']}: "
                f"rel_L2={hit['relative_euclidean']:.3e} "
                f"cosine_d={hit['cosine_distance']:.3e} "
                f"(threshold={report['threshold_relative']})"
            )
        raise MaterialSignatureCollisionError(msg)

    print(
        f"[CASIMIR MATERIAL GATE] Pass: {len(report['materials'])} materials, "
        f"{report['n_distinct_pairs']} pairs all above "
        f"relative threshold {report['threshold_relative']}"
    )
    return report
