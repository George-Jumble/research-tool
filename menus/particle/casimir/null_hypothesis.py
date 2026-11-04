"""
Null-hypothesis Monte Carlo for TSB Casimir signature flags.

Builds synthetic series on the real Matsubara grid and estimates false-positive
rates. Real-data claims are refused when a flag sits inside the 95 % null
envelope of any decay null model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from menus.particle.casimir.casimir_artifacts import casimir_output_path, ensure_casimir_series
from menus.particle.casimir.series_integrity import PARAM_NAME, VALUE_NAME
from tav_shared.artifact_paths import TestSlug, test_artifacts_dir

NULL_MODELS: tuple[str, ...] = ("inverse_xi", "exponential", "sinusoid_period7")
NULL_MODELS_FOR_GATE: tuple[str, ...] = ("inverse_xi", "exponential")
TRACKED_FLAGS: tuple[str, ...] = (
    "seven_phase_clockwork",
    "discrete_steps_thresholds",
    "hysteresis_asymmetry",
    "changepoints",
    "tav_resonance_detected",
    "tau_cycle_142857",
    "resonant_peaks",
)
SEVEN_CYCLE_FLAGS: tuple[str, ...] = (
    "seven_phase_clockwork",
    "tav_resonance_detected",
)
SNR_LEVELS_DB: tuple[float, ...] = (10.0, 0.0, -10.0)
N_TRIALS_DEFAULT = 100
ENVELOPE_ALPHA = 0.05
STABLE_REPORT_NAME = "null_hypothesis_false_positive_rates.json"

_ENVELOPE_CACHE: dict[str, Any] | None = None


def load_matsubara_template(file_path: str | Path | None = None) -> np.ndarray:
    """Return the real Matsubara-frequency sampling grid (cleaned, sorted)."""
    from menus.particle.casimir.scanner import DATASETS_DIR, load_data

    if file_path is None:
        fig2 = DATASETS_DIR / "github_magnetic_fluid" / "Fig2"
        candidates = sorted(
            p for p in fig2.iterdir() if p.is_file() and p.name.lower() != "readme"
        ) if fig2.is_dir() else []
        if not candidates:
            raise FileNotFoundError(
                f"No Fig2 material files under {fig2}. "
                "Download GitHub Magnetic-Fluid Data first."
            )
        file_path = candidates[0]
    df = load_data(file_path)
    xi = np.asarray(df[PARAM_NAME], dtype=np.float64).ravel()
    xi = xi[np.isfinite(xi) & (xi > 0)]
    if xi.size < 8:
        raise ValueError("Matsubara template too short after cleaning")
    return np.ascontiguousarray(np.sort(xi))


def _signal_inverse_xi(xi: np.ndarray, amplitude: float) -> np.ndarray:
    return 1.0 + float(amplitude) / xi


def _signal_exponential(xi: np.ndarray, amplitude: float, tau: float) -> np.ndarray:
    return float(amplitude) * np.exp(-xi / float(tau))


def _signal_sinusoid_period7(xi: np.ndarray, amplitude: float, offset: float) -> np.ndarray:
    return float(offset) + float(amplitude) * np.sin(2.0 * np.pi * xi / 7.0)


def add_gaussian_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at the requested SNR (dB, signal-power / noise-power)."""
    sig = np.asarray(signal, dtype=np.float64)
    power = float(np.var(sig))
    if power <= 0:
        power = float(np.mean(sig * sig)) or 1.0
    noise_var = power / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, np.sqrt(max(noise_var, 1e-30)), size=sig.shape)
    return sig + noise


def synthesize_null_series(
    xi: np.ndarray,
    model: str,
    *,
    snr_db: float,
    seed: int,
    amplitude: float,
    exp_tau: float,
    sinusoid_offset: float,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if model == "inverse_xi":
        clean = _signal_inverse_xi(xi, amplitude)
    elif model == "exponential":
        clean = _signal_exponential(xi, amplitude, exp_tau)
    elif model == "sinusoid_period7":
        clean = _signal_sinusoid_period7(xi, amplitude, sinusoid_offset)
    else:
        raise ValueError(f"Unknown null model: {model}")
    return add_gaussian_noise(clean, snr_db, rng)


def _extract_flags(detection: dict[str, Any]) -> dict[str, bool]:
    raw = dict(detection.get("raw_signature_summary") or detection.get("signature_summary") or {})
    tav = detection.get("tav_harmonics") or {}
    if tav.get("tav_resonance_detected"):
        raw["tav_resonance_detected"] = True
    return {flag: bool(raw.get(flag, False)) for flag in TRACKED_FLAGS}


def _extract_seven_cycle_secondary(detection: dict[str, Any]) -> dict[str, bool]:
    sec = detection.get("seven_cycle_secondary") or {}
    return {flag: bool(sec.get(flag, False)) for flag in SEVEN_CYCLE_FLAGS}


def _empty_model_block(flags: tuple[str, ...]) -> dict[str, Any]:
    return {"by_snr": {}, "pooled": {flag: 0 for flag in flags}}


def _finalize_model_cell(
    models: dict[str, dict[str, Any]],
    model: str,
    snr: float,
    counts: dict[str, int],
    flags: tuple[str, ...],
    n_trials: int,
) -> None:
    denom = float(n_trials)
    models[model]["by_snr"][str(snr)] = {
        flag: {
            "hits": counts[flag],
            "n_trials": n_trials,
            "rate": counts[flag] / denom,
        }
        for flag in flags
    }


def _finalize_pooled(
    models: dict[str, dict[str, Any]],
    flags: tuple[str, ...],
    n_trials: int,
    n_snr: int,
) -> None:
    pooled_denom = float(n_trials * n_snr)
    for model in models:
        models[model]["pooled_rates"] = {
            flag: models[model]["pooled"][flag] / pooled_denom for flag in flags
        }


def _template_scales(xi: np.ndarray) -> tuple[float, float, float]:
    # Keep 1/ξ and exponential on an O(1–few) dielectric-like scale.
    amplitude = float(np.median(xi))
    exp_tau = float(max(np.ptp(xi) / 3.0, np.min(xi) * 4.0, 1e-6))
    sinusoid_offset = 2.0
    return amplitude, exp_tau, sinusoid_offset


def run_null_hypothesis_suite(
    *,
    n_trials: int = N_TRIALS_DEFAULT,
    snr_levels_db: Iterable[float] = SNR_LEVELS_DB,
    template_path: str | Path | None = None,
    seed0: int = 20260819,
    write_report: bool = True,
    one_seventh_power_ratio: float | None = None,
    seven_cycle_mf_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Monte Carlo FPR for each signature flag on three synthetic families.

    The 7-period sinusoid is a planted-signal control (detection power).
    Inverse-1/ξ and exponential families are the nulls used for the 95 % gate.

    Primary flags use the strict 0.55 pipeline. The independent 7-cycle
    matched filter is scored in the same trials at its own threshold.
    """
    from menus.particle.casimir.scanner import detect_tau_resonances
    from menus.particle.casimir.significance import one_seventh_power_ratio_threshold
    from menus.particle.casimir.seven_cycle_detector import seven_cycle_mf_threshold as mf_default

    xi = load_matsubara_template(template_path)
    primary_ratio = (
        float(one_seventh_power_ratio)
        if one_seventh_power_ratio is not None
        else one_seventh_power_ratio_threshold()
    )
    mf_threshold = (
        float(seven_cycle_mf_threshold)
        if seven_cycle_mf_threshold is not None
        else mf_default()
    )
    amplitude, exp_tau, sinusoid_offset = _template_scales(xi)
    snr_levels = tuple(float(s) for s in snr_levels_db)
    n_trials = int(n_trials)

    models: dict[str, dict[str, Any]] = {
        model: _empty_model_block(TRACKED_FLAGS) for model in NULL_MODELS
    }
    secondary_models: dict[str, dict[str, Any]] = {
        model: _empty_model_block(SEVEN_CYCLE_FLAGS) for model in NULL_MODELS
    }
    for model in NULL_MODELS:
        for snr in snr_levels:
            counts = {flag: 0 for flag in TRACKED_FLAGS}
            sec_counts = {flag: 0 for flag in SEVEN_CYCLE_FLAGS}
            for trial in range(n_trials):
                model_off = {"inverse_xi": 100_000, "exponential": 200_000, "sinusoid_period7": 300_000}[model]
                seed = int(seed0) + int(trial) + model_off + int(round(snr * 1000.0))
                y = synthesize_null_series(
                    xi,
                    model,
                    snr_db=snr,
                    seed=seed,
                    amplitude=amplitude,
                    exp_tau=exp_tau,
                    sinusoid_offset=sinusoid_offset,
                )
                frame = pd.DataFrame({PARAM_NAME: xi, VALUE_NAME: y})
                detection = detect_tau_resonances(
                    frame,
                    param_col=PARAM_NAME,
                    value_col=VALUE_NAME,
                    verbose=False,
                    apply_null_gate=False,
                    fast=True,
                    one_seventh_power_ratio=primary_ratio,
                    seven_cycle_mf_threshold=mf_threshold,
                )
                flags = _extract_flags(detection)
                for flag, hit in flags.items():
                    if hit:
                        counts[flag] += 1
                        models[model]["pooled"][flag] += 1
                sec_flags = _extract_seven_cycle_secondary(detection)
                for flag, hit in sec_flags.items():
                    if hit:
                        sec_counts[flag] += 1
                        secondary_models[model]["pooled"][flag] += 1
            _finalize_model_cell(models, model, snr, counts, TRACKED_FLAGS, n_trials)
            _finalize_model_cell(
                secondary_models, model, snr, sec_counts, SEVEN_CYCLE_FLAGS, n_trials
            )
    _finalize_pooled(models, TRACKED_FLAGS, n_trials, len(snr_levels))
    _finalize_pooled(secondary_models, SEVEN_CYCLE_FLAGS, n_trials, len(snr_levels))

    envelope = _build_envelope(models, TRACKED_FLAGS)
    secondary_envelope = _build_envelope(secondary_models, SEVEN_CYCLE_FLAGS)
    report = {
        "module": "tsb_casimir_null_hypothesis",
        "detector_version": "residual_significance_v4",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "one_seventh_power_ratio_threshold": primary_ratio,
        "seven_cycle_mf_threshold": mf_threshold,
        "n_trials_per_cell": n_trials,
        "snr_levels_db": list(snr_levels),
        "n_points": int(xi.size),
        "matsubara_min": float(xi.min()),
        "matsubara_max": float(xi.max()),
        "template_path": str(Path(template_path).resolve()) if template_path else None,
        "generator": {
            "inverse_xi": "y = 1 + A/ξ + noise",
            "exponential": "y = A·exp(-ξ/τ) + noise",
            "sinusoid_period7": "y = C + A·sin(2πξ/7) + noise (planted-signal control)",
            "amplitude": amplitude,
            "exp_tau": exp_tau,
            "sinusoid_offset": sinusoid_offset,
            "snr_definition": "10 log10(Var(signal)/Var(noise))",
        },
        "tracked_flags": list(TRACKED_FLAGS),
        "null_models_for_gate": list(NULL_MODELS_FOR_GATE),
        "envelope_alpha": ENVELOPE_ALPHA,
        "models": models,
        "false_positive_envelope": envelope,
        "seven_cycle_secondary": {
            "analysis": "seven_cycle_matched_filter",
            "detector": "quadrature_template_correlator",
            "independent_of": "primary_signature_summary",
            "period": 7.0,
            "threshold": mf_threshold,
            "primary_power_ratio_threshold": primary_ratio,
            "tracked_flags": list(SEVEN_CYCLE_FLAGS),
            "models": secondary_models,
            "false_positive_envelope": secondary_envelope,
            "note": (
                "Independent period-7 matched filter. Primary claims stay at "
                f"I(f)/max(I)>={primary_ratio}. This detector never writes "
                "primary flags. Its own decay-null max FPR must stay < 0.05."
            ),
        },
        "note": (
            "Primary false-positive envelope uses inverse_xi and exponential only. "
            "sinusoid_period7 is a planted 7-period signal (sensitivity, not a null). "
            "seven_cycle_secondary is a separate matched-filter module with its own FPR."
        ),
    }

    if write_report:
        series_path = _write_null_report(report)
        report["report_path"] = str(series_path)
        print(f"[CASIMIR NULL] Wrote {series_path}")
        print_null_summary_table(report)
    return report


def _build_envelope(
    models: dict[str, dict[str, Any]],
    flags: tuple[str, ...] = TRACKED_FLAGS,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {}
    for flag in flags:
        rates = []
        by_model: dict[str, float] = {}
        for model in NULL_MODELS_FOR_GATE:
            rate = float(models[model]["pooled_rates"][flag])
            by_model[model] = rate
            rates.append(rate)
            for snr, cell in models[model]["by_snr"].items():
                rates.append(float(cell[flag]["rate"]))
        max_fpr = max(rates) if rates else 0.0
        envelope[flag] = {
            "max_false_positive_rate": max_fpr,
            "by_null_model": by_model,
            "inside_95pct_envelope_if_true": max_fpr >= ENVELOPE_ALPHA,
            "claim_allowed": max_fpr < ENVELOPE_ALPHA,
        }
    return envelope


def _write_null_report(report: dict[str, Any]) -> Path:
    global _ENVELOPE_CACHE
    _ENVELOPE_CACHE = None
    ensure_casimir_series("tsb_casimir")
    series_path = casimir_output_path(
        "null_hypothesis",
        "false_positive_rates",
        "json",
        output_prefix="tsb_casimir",
    )
    # Keep the exact requested filename as a stable copy for the live gate.
    stable = test_artifacts_dir(TestSlug.CASIMIR) / STABLE_REPORT_NAME
    text = json.dumps(report, indent=2)
    series_path.write_text(text, encoding="utf-8")
    stable.write_text(text, encoding="utf-8")
    report["stable_report_path"] = str(stable)
    return series_path


def print_null_summary_table(report: dict[str, Any]) -> None:
    flags = report["tracked_flags"]
    print("\n=== Casimir null-hypothesis false-positive rates ===")
    print(f"n_trials/cell={report['n_trials_per_cell']}  "
          f"SNR(dB)={report['snr_levels_db']}  n={report['n_points']}")
    ratio = report.get("one_seventh_power_ratio_threshold")
    if ratio is not None:
        print(f"primary 1/7 power-ratio gate  I(f)/max(I) >= {ratio}")
    header = f"{'model':<18} {'snr':>6} " + " ".join(f"{f[:16]:>16}" for f in flags)
    print(header)
    print("-" * len(header))
    for model, block in report["models"].items():
        for snr, cell in block["by_snr"].items():
            rates = " ".join(f"{cell[f]['rate']:>16.3f}" for f in flags)
            print(f"{model:<18} {snr:>6} {rates}")
        pooled = " ".join(f"{block['pooled_rates'][f]:>16.3f}" for f in flags)
        print(f"{model:<18} {'pool':>6} {pooled}")
        print("-" * len(header))
    print("95% envelope (decay nulls only): claim allowed if max FPR < 0.05")
    for flag, env in report["false_positive_envelope"].items():
        status = "ALLOW" if env["claim_allowed"] else "REFUSE"
        print(
            f"  {flag:<28} maxFPR={env['max_false_positive_rate']:.3f}  {status}"
        )
    secondary = report.get("seven_cycle_secondary") or {}
    sec_env = secondary.get("false_positive_envelope") or {}
    if sec_env:
        sec_thr = secondary.get("threshold")
        print(
            f"secondary 7-cycle matched filter  period=7  score >= {sec_thr}  "
            "(independent; not a primary claim)"
        )
        sec_models = secondary.get("models") or {}
        sec_flags = secondary.get("tracked_flags") or list(SEVEN_CYCLE_FLAGS)
        header = f"{'model':<18} {'snr':>6} " + " ".join(
            f"{f[:16]:>16}" for f in sec_flags
        )
        print(header)
        print("-" * len(header))
        for model, block in sec_models.items():
            for snr, cell in block.get("by_snr", {}).items():
                rates = " ".join(f"{cell[f]['rate']:>16.3f}" for f in sec_flags)
                print(f"{model:<18} {snr:>6} {rates}")
            pooled = " ".join(
                f"{block['pooled_rates'][f]:>16.3f}" for f in sec_flags
            )
            print(f"{model:<18} {'pool':>6} {pooled}")
            print("-" * len(header))
        for flag, env in sec_env.items():
            status = "ALLOW" if env.get("claim_allowed") else "REFUSE"
            print(
                f"  secondary {flag:<18} maxFPR={env['max_false_positive_rate']:.3f}  {status}"
            )
    print()


def one_seventh_flag_max_fpr(report: dict[str, Any]) -> dict[str, float]:
    """Max decay-null FPR for the two 1/7 claim flags."""
    envelope = report.get("false_positive_envelope") or {}
    out: dict[str, float] = {}
    for flag in ("seven_phase_clockwork", "tav_resonance_detected"):
        env = envelope.get(flag) or {}
        out[flag] = float(env.get("max_false_positive_rate") or 0.0)
    return out


def _load_stable_null_report() -> dict[str, Any] | None:
    stable = test_artifacts_dir(TestSlug.CASIMIR, create=False) / STABLE_REPORT_NAME
    if not stable.is_file():
        return None
    try:
        payload = json.loads(stable.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_false_positive_envelope() -> dict[str, Any] | None:
    global _ENVELOPE_CACHE
    if _ENVELOPE_CACHE is not None:
        return _ENVELOPE_CACHE
    payload = _load_stable_null_report()
    if payload is None:
        return None
    _ENVELOPE_CACHE = payload.get("false_positive_envelope")
    return _ENVELOPE_CACHE


def load_seven_cycle_secondary_envelope() -> dict[str, Any] | None:
    payload = _load_stable_null_report()
    if payload is None:
        return None
    secondary = payload.get("seven_cycle_secondary") or {}
    env = secondary.get("false_positive_envelope")
    return env if isinstance(env, dict) else None


def attach_seven_cycle_documented_fpr(results: dict[str, Any]) -> None:
    """Copy the secondary envelope onto a detection without changing primary claims."""
    sec = results.get("seven_cycle_secondary")
    if not isinstance(sec, dict):
        return
    payload = _load_stable_null_report() or {}
    secondary = payload.get("seven_cycle_secondary") or {}
    env = secondary.get("false_positive_envelope") or {}
    documented: dict[str, float] = {}
    allowed: dict[str, bool] = {}
    for flag in SEVEN_CYCLE_FLAGS:
        item = env.get(flag) or {}
        if "max_false_positive_rate" in item:
            documented[flag] = float(item["max_false_positive_rate"])
        allowed[flag] = bool(item.get("claim_allowed", False)) if item else False
    if documented:
        sec["documented_false_positive_rate"] = documented
        sec["documented_claim_allowed"] = allowed
        sec["documented_threshold"] = secondary.get("threshold")


def apply_null_gate_to_detection(results: dict[str, Any], *, verbose: bool = True) -> dict[str, Any]:
    """
    Clear signature_summary claims that sit inside the 95 % null envelope.

    Raw detections stay in detected_features / raw_signature_summary.
    """
    envelope = load_false_positive_envelope()
    gate = {
        "applied": False,
        "alpha": ENVELOPE_ALPHA,
        "refused": [],
        "allowed": [],
        "missing_envelope": envelope is None,
        "skipped_stale_envelope": False,
    }
    raw = dict(results.get("raw_signature_summary") or {})
    gated: dict[str, bool] = {}
    # p-value detectors (residual_significance_v1) are claimed via Holm-adjusted
    # p < 0.01. A binary-flag envelope from the old peak-any detectors is stale.
    stable = test_artifacts_dir(TestSlug.CASIMIR, create=False) / STABLE_REPORT_NAME
    version = None
    if stable.is_file():
        try:
            version = json.loads(stable.read_text(encoding="utf-8")).get("detector_version")
        except (OSError, json.JSONDecodeError):
            version = None
    residual_ok = str(version or "").startswith("residual_significance")
    if envelope is not None and not residual_ok:
        gate["skipped_stale_envelope"] = True
        gate["missing_envelope"] = False
        results["null_gate"] = gate
        results.setdefault("diagnostics", {})["null_gate"] = gate
        attach_seven_cycle_documented_fpr(results)
        if verbose:
            print(
                "[CASIMIR NULL] Stale binary-flag envelope ignored; "
                "claims use residual p_Holm vs α=0.01. Re-run the null suite."
            )
        return results
    if envelope is None:
        if verbose:
            print(
                "[CASIMIR NULL] No null-hypothesis envelope on disk — "
                f"run the suite to write artifacts/casimir/{STABLE_REPORT_NAME}. "
                "Real-data claims are not gated."
            )
        results["null_gate"] = gate
        attach_seven_cycle_documented_fpr(results)
        return results

    for flag, hit in raw.items():
        env = envelope.get(flag) or {}
        allowed = bool(env.get("claim_allowed", True))
        if hit and not allowed:
            gated[flag] = False
            gate["refused"].append(
                {
                    "flag": flag,
                    "max_false_positive_rate": env.get("max_false_positive_rate"),
                    "reason": (
                        "Inside 95% false-positive envelope of a decay null "
                        f"(max FPR={env.get('max_false_positive_rate')})"
                    ),
                }
            )
            if verbose:
                print(
                    f"[CASIMIR NULL] Refusing claim {flag!r}: "
                    f"max null FPR={env.get('max_false_positive_rate')} ≥ {ENVELOPE_ALPHA}"
                )
        else:
            if hit:
                gated[flag] = True
                gate["allowed"].append(flag)
    results["signature_summary"] = gated
    results["null_gate"] = gate
    results.setdefault("diagnostics", {})["null_gate"] = gate
    attach_seven_cycle_documented_fpr(results)
    return results
