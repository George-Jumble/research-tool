"""Harmonic Binding Kit must not contradict Periodicity Scan on the same DH vector."""

from __future__ import annotations

import numpy as np
import pytest

_RETIRED_1_7 = "retired 2026-09-22: 1/7 is not a live Superblock search target"

from menus.astronomical.desi.harmonic_binding import (
    ladder_1_7_significant_from_periodicity,
    run_harmonic_binding_kit,
    search_harmonic_features,
)
from menus.astronomical.desi.scanner import (
    STANDARD_DEDUPE_Z_TOL,
    TauSBScanner,
    apply_analysis_z_dedupe,
    has_twin_z_pairs,
    s_from_z,
)
from menus.astronomical.desi.stats import (
    assess_scan_power,
    evaluate_1_7_detection,
    power_stamp_for_scan,
)


def _23_16_periodogram() -> tuple[np.ndarray, np.ndarray]:
    """Replay 27-08-2026 23:16: P(1/7)=0.077, max=0.521 at f≈0.30, ratio=0.15."""
    freqs = np.linspace(0.05, 0.5, 500)
    power = np.full_like(freqs, 1e-4)
    i_17 = int(np.argmin(np.abs(freqs - 1.0 / 7.0)))
    i_max = int(np.argmin(np.abs(freqs - 0.30)))
    power[i_17] = 0.077
    power[i_max] = 0.521
    return freqs, power


def test_23_16_kit_1_7_flag_matches_periodicity_false():
    freqs, power = _23_16_periodogram()
    det = evaluate_1_7_detection(
        freqs=freqs,
        power=power,
        p_boot=0.70,
        n_raw=20,
        n_deduped=9,
        cycles=1.26,
        residual_source="poly_baseline",
        gamma_source="fixed_10",
    )
    assert det["detection"] is False
    assert det["lomb_peak_1_7"] is False
    assert det["power_ratio"] < 0.5

    kit = run_harmonic_binding_kit(
        np.linspace(0.3, 2.3, 9),
        np.ones(9) * 20.0,
        np.zeros(9),
        s_from_z(np.linspace(0.3, 2.3, 9), gamma=10.0),
        verbose=False,
        periodicity_detection=bool(det["detection"]),
        p_values_trusted=True,
        n_bootstrap=20,
    )
    sm = kit["summary"]
    assert sm["ladder_1_7_significant"] is False
    assert sm["harmonic_ladder_significant"] is False
    assert sm["ladder_1_7_significant"] == det["detection"]
    assert sm["n_detected_transitions"] == 0
    assert sm["alignment"] is None


def test_harmonic_2_p_boot_does_not_set_1_7_flag():
    assert (
        ladder_1_7_significant_from_periodicity(
            periodicity_detection=False,
            p_values_trusted=True,
        )
        is False
    )
    z = np.linspace(0.2, 2.5, 16)
    s = s_from_z(z, gamma=10.0)
    # Residual with power at harmonic 2 (2/7), not 1/7.
    y = np.sin(2.0 * np.pi * s * (2.0 / 7.0))
    kit = run_harmonic_binding_kit(
        z,
        np.ones(16),
        y,
        s,
        verbose=False,
        periodicity_detection=False,
        p_values_trusted=True,
        n_bootstrap=80,
    )
    assert kit["summary"]["ladder_1_7_significant"] is False
    sig = kit["summary"]["significant_harmonics"]
    # Harmonic 2 may appear; it must not flip the 1/7 bit.
    assert 1 not in sig or kit["summary"]["ladder_1_7_significant"] is False


def test_untrusted_p_values_force_kit_1_7_false():
    assert (
        ladder_1_7_significant_from_periodicity(
            periodicity_detection=True,
            p_values_trusted=False,
        )
        is False
    )


@pytest.mark.skip(reason=_RETIRED_1_7)
def test_synthetic_1_7_periodicity_and_kit_agree_true():
    n = 16
    z = np.linspace(0.05, 4.0, n)
    s = s_from_z(z, gamma=10.0)
    y = 20.0 + 0.3 * z + 4.0 * np.sin(2.0 * np.pi * s / 7.0)
    err = np.full(n, 0.05)
    sc = TauSBScanner(gamma=10.0, auto_calibrate_gamma=False)
    per = sc.scan_periodicity(
        z,
        y,
        err,
        n_raw=n,
        n_deduped=n,
        residual_source="poly_baseline",
        gamma_source="fixed_10",
        n_bootstrap=400,
        n_freq=400,
    )
    assert per["detection"] is True, per.get("fail_reasons")
    kit = run_harmonic_binding_kit(
        z,
        y,
        per["residuals"],
        per["s"],
        verbose=False,
        periodicity_detection=bool(per["detection"]),
        p_values_trusted=True,
        n_bootstrap=80,
    )
    assert kit["summary"]["ladder_1_7_significant"] is True
    assert kit["summary"]["ladder_1_7_significant"] == per["detection"]


def test_white_noise_both_false_even_if_higher_harmonic_p_low():
    rng = np.random.default_rng(123)
    n = 16
    z = np.linspace(0.05, 4.0, n)
    y = rng.normal(20.0, 1.0, size=n)
    err = np.ones(n)
    sc = TauSBScanner(gamma=10.0, auto_calibrate_gamma=False)
    per = sc.scan_periodicity(
        z,
        y,
        err,
        n_raw=n,
        n_deduped=n,
        residual_source="poly_baseline",
        gamma_source="fixed_10",
        n_bootstrap=400,
        n_freq=400,
    )
    assert per["detection"] is False
    kit = run_harmonic_binding_kit(
        z,
        y,
        per["residuals"],
        per["s"],
        verbose=False,
        periodicity_detection=bool(per["detection"]),
        p_values_trusted=True,
        n_bootstrap=80,
    )
    assert kit["summary"]["ladder_1_7_significant"] is False
    assert kit["summary"]["ladder_1_7_significant"] == per["detection"]


def test_analysis_dedupe_n_used_equals_n_deduped_no_twins():
    z = np.array([0.51, 0.51, 0.71, 0.72, 1.32], dtype=float)
    y = np.array([19.0, 19.1, 19.5, 19.6, 21.0], dtype=float)
    err = np.ones(5) * 0.1
    data = {
        "z": z,
        "observable": y,
        "err": err,
        "quantity": ["DH_over_rs"] * 5,
        "cov": np.diag(err ** 2),
        "n_raw": 5,
        "label": "twin_z",
    }
    out = apply_analysis_z_dedupe(data, tol=STANDARD_DEDUPE_Z_TOL)
    assert int(out["n_used"]) == int(out["n_deduped"])
    assert int(out["n_used"]) == int(out["n_data"])
    assert int(out["n_deduped"]) < int(out["n_raw"])
    assert has_twin_z_pairs(out["z"], tol=STANDARD_DEDUPE_Z_TOL) is False


def test_power_stamp_low_medium_high():
    assert power_stamp_for_scan(n_used=9, cycles=1.26) == "LOW"
    assert power_stamp_for_scan(n_used=16, cycles=1.2) == "LOW"
    assert power_stamp_for_scan(n_used=12, cycles=1.6) == "MEDIUM"
    assert power_stamp_for_scan(n_used=12, cycles=2.0) == "HIGH"
    low = assess_scan_power(
        n_data=9,
        cycles_possible=1.26,
        auto_calibrate_gamma=False,
    )
    assert low["power_stamp"] == "LOW"
    assert "adequate for exploratory" not in low["recommendation"].lower()


def test_142857_probe_skipped_on_short_s_span():
    z = np.linspace(0.3, 2.3, 9)
    s = s_from_z(z, gamma=10.0)
    features = search_harmonic_features(np.zeros(9), s, z=z, n_bootstrap=10, verbose=False)
    probe = features["cycle_142857_probe"]
    assert probe.get("skipped") is True


def test_empty_detected_z_alignment_is_null():
    z = np.linspace(0.3, 2.3, 9)
    s = s_from_z(z, gamma=10.0)
    kit = run_harmonic_binding_kit(
        z,
        np.ones(9),
        np.zeros(9),
        s,
        verbose=False,
        periodicity_detection=False,
        n_bootstrap=10,
    )
    assert kit["summary"]["n_detected_transitions"] == 0
    assert kit["summary"]["alignment"] is None
    assert kit["summary"]["alignment_fraction"] is None


def _fat_duplicate_stack(n_raw: int = 20, n_unique: int = 9) -> dict:
    """Near-duplicate z stack with a fat, ill-conditioned covariance."""
    rng = np.random.default_rng(2329)
    z_u = np.linspace(0.3, 2.4, n_unique)
    z_parts = [float(zu) for zu in z_u]
    y_parts = [19.0 + 0.4 * float(zu) for zu in z_u]
    extra_i = 0
    while len(z_parts) < n_raw:
        zu = float(z_u[extra_i % n_unique])
        z_parts.append(zu + 0.008)
        y_parts.append(19.0 + 0.4 * zu + 0.01)
        extra_i += 1
    z = np.asarray(z_parts[:n_raw], dtype=float)
    y = np.asarray(y_parts[:n_raw], dtype=float)
    n = len(z)
    cov = 0.95 * np.ones((n, n)) + 0.05 * np.eye(n)
    cov += 1e-4 * rng.standard_normal((n, n))
    cov = 0.5 * (cov + cov.T)
    return {
        "z": z,
        "observable": y,
        "err": np.sqrt(np.clip(np.diag(cov), 1e-8, None)),
        "quantity": ["DH_over_rs"] * n,
        "cov": cov,
        "cov_native": cov.copy(),
        "n_raw": n,
        "label": "fat_dup",
        "cov_health": {"condition_number": float(np.linalg.cond(cov)), "n": n},
    }


def test_23_29_cov_rebuilt_n9_kappa_not_fat_stack():
    data = _fat_duplicate_stack(20, 9)
    data["cov_pre_unique_z"] = np.array(data["cov"], copy=True)
    out = apply_analysis_z_dedupe(data, tol=STANDARD_DEDUPE_Z_TOL)
    n_used = int(out["n_used"])
    assert n_used == 9
    assert int(out["cov_n"]) == 9
    assert out["cov"].shape == (9, 9)
    # Same native/fat → C is a principal minor of the 20×20: not a rebuild.
    assert out["cov_rebuilt_after_dedupe"] is False
    assert out["p_values_trusted"] is False


def test_rebuilt_true_when_native_not_fat_minor():
    """Native I_n slice is not the principal minor of an all-ones prepared C."""
    data = _fat_duplicate_stack(20, 9)
    n = int(data["n_raw"])
    data["cov_native"] = np.eye(n)
    data["cov_pre_unique_z"] = np.array(data["cov"], copy=True)
    out = apply_analysis_z_dedupe(data, tol=STANDARD_DEDUPE_Z_TOL)
    assert int(out["n_used"]) == 9
    assert out["cov"].shape == (9, 9)
    assert int(out["cov_n"]) == 9
    assert out["cov_rebuilt_after_dedupe"] is True


def test_rebuild_false_when_kappa_unchanged():
    """Identity C: 20→9 unique-z leaves κ=1. That is a no-op rebuild."""
    n_raw, n_unique = 20, 9
    z_u = np.linspace(0.3, 2.4, n_unique)
    z_parts = [float(zu) for zu in z_u]
    y_parts = [19.0] * n_unique
    i = 0
    while len(z_parts) < n_raw:
        z_parts.append(float(z_u[i % n_unique]) + 0.008)
        y_parts.append(19.0)
        i += 1
    z = np.asarray(z_parts, dtype=float)
    y = np.asarray(y_parts, dtype=float)
    cov = np.eye(n_raw)
    data = {
        "z": z,
        "observable": y,
        "err": np.ones(n_raw),
        "quantity": ["DH_over_rs"] * n_raw,
        "cov": cov,
        "cov_native": cov.copy(),
        "n_raw": n_raw,
        "label": "identity",
        "cov_health": {"condition_number": 1.0, "n": n_raw},
    }
    out = apply_analysis_z_dedupe(data, tol=STANDARD_DEDUPE_Z_TOL)
    assert int(out["n_used"]) == 9
    assert out["cov"].shape == (9, 9)
    assert np.isclose(float(out["cov_kappa"]), float(out["kappa_pre_dedupe"]), rtol=0.0, atol=1e-10)
    assert out["cov_rebuilt_after_dedupe"] is False
    assert out["p_values_trusted"] is False


def test_whim_not_a_signature_when_1_7_null():
    from menus.astronomical.desi.stats import evaluate_whim_signature, search_whim_web_signatures

    z = np.linspace(0.3, 2.3, 9)
    s = s_from_z(z, gamma=10.0)
    resid = np.sin(4.0 * np.pi * s / (s.max() - s.min() + 1e-9))
    whim = search_whim_web_signatures(
        resid,
        s,
        n_bootstrap=40,
        verbose=False,
        n_used=9,
        cycles=1.26,
        periodicity_detection=False,
        p_values_trusted=True,
    )
    assert whim["whim_signature_detected"] is False
    gated = evaluate_whim_signature(
        whim,
        n_used=9,
        cycles=1.26,
        periodicity_detection=False,
    )
    assert gated["whim_signature_detected"] is False


def test_acf_no_lag7_significant_on_n9():
    from menus.astronomical.desi.analysis import acf_lag_cap, analyze_residuals

    assert acf_lag_cap(9, requested=8) == 3
    rng = np.random.default_rng(9)
    n = 9
    y = rng.normal(size=n)
    model = np.zeros(n)
    s = np.linspace(0, 1, n)
    cov = np.eye(n)
    res = analyze_residuals(
        s, y, model, cov=cov, max_lag=8, print_summary=False, make_plots=False
    )
    assert 7 not in res["autocorrelation"]
    assert int(res["acf_max_lag_used"]) <= 3
    assert not any(int(k) >= 7 and v.get("significant") for k, v in res["autocorrelation"].items())


@pytest.mark.skip(reason=_RETIRED_1_7)
def test_synthetic_1_7_whim_not_auto_true():
    from menus.astronomical.desi.stats import search_whim_web_signatures

    n = 16
    z = np.linspace(0.05, 4.0, n)
    s = s_from_z(z, gamma=10.0)
    y = 20.0 + 0.3 * z + 4.0 * np.sin(2.0 * np.pi * s / 7.0)
    err = np.full(n, 0.05)
    sc = TauSBScanner(gamma=10.0, auto_calibrate_gamma=False)
    per = sc.scan_periodicity(
        z, y, err, n_raw=n, n_deduped=n,
        residual_source="poly_baseline", gamma_source="fixed_10",
        n_bootstrap=400, n_freq=400,
    )
    assert per["detection"] is True
    kit = run_harmonic_binding_kit(
        z, y, per["residuals"], per["s"],
        verbose=False, periodicity_detection=True, n_bootstrap=80,
    )
    assert kit["summary"]["ladder_1_7_significant"] is True
    whim = search_whim_web_signatures(
        per["residuals"], per["s"], n_bootstrap=40, verbose=False,
        n_used=n, cycles=float(per["cycles"]), periodicity_detection=True,
    )
    assert whim["whim_signature_detected"] is False


def test_tau_sb_underpowered_poly_primary_on_n9():
    from menus.astronomical.desi.scanner import DesiScanResult

    z = np.linspace(0.3, 2.4, 9)
    y = 20.0 + 0.5 * z + 0.02 * np.sin(z)
    err = np.ones(9) * 0.1
    cov = np.diag(err ** 2)
    sc = TauSBScanner(gamma=10.0, auto_calibrate_gamma=False)
    fit = sc.fit_tau_sb_model(z, y, err, cov=cov)
    assert fit["primary_model"] == "poly_baseline"
    assert fit["tau_sb_underpowered"] is True
    assert fit["lead_with_A_osc"] is False
    res = DesiScanResult(data_label="n9", tracer="t", gamma_used=10.0, fit=fit)
    text = "\n".join(res.summary_lines())
    assert "Primary (poly baseline)" in text
    assert "not a detection" in text
    assert "A_osc_frac≈" not in text
