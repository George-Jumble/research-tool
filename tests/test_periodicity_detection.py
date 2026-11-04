"""1/7 Lomb-Scargle detection predicate — 20:41 DH-only false-positive guards."""

from __future__ import annotations

import numpy as np
import pytest

_RETIRED_1_7 = "retired 2026-09-22: 1/7 is not a live Superblock search target"

from menus.astronomical.desi.scanner import (
    STANDARD_DEDUPE_Z_TOL,
    TauSBScanner,
    _stack_bao_blocks,
    count_unique_z,
    filter_dh_only_periodogram_vector,
    s_from_z,
)
from menus.astronomical.desi.stats import evaluate_1_7_detection, lomb_scargle_cyclic


def _nyquist_dominated_20_41_grid() -> tuple[np.ndarray, np.ndarray]:
    """Replay the 27-08-2026 20:41 power vector: P(1/7) ≪ P(0.5)."""
    freqs = np.linspace(0.05, 0.5, 500)
    power = np.full_like(freqs, 1e-12)
    i_17 = int(np.argmin(np.abs(freqs - 1.0 / 7.0)))
    i_ny = int(np.argmin(np.abs(freqs - 0.5)))
    power[i_17] = 5.8809e-08
    power[i_ny] = 4.4223e-04
    return freqs, power


def test_20_41_nyquist_false_positive_is_not_a_detection():
    freqs, power = _nyquist_dominated_20_41_grid()
    det = evaluate_1_7_detection(
        freqs=freqs,
        power=power,
        p_boot=0.021,
        n_raw=16,
        n_deduped=16,
        cycles=1.57,
        residual_source="poly_baseline",
        gamma_source="fixed",
    )
    assert det["lomb_peak_1_7"] is False
    assert det["detection"] is False
    assert det["f_max"] == 0.5 or abs(det["f_max"] - 0.5) < 1e-6
    assert det["power_ratio"] < 0.5

    # Even if cycles and n_deduped were inflated, Nyquist-max still fails.
    det_roomy = evaluate_1_7_detection(
        freqs=freqs,
        power=power,
        p_boot=0.021,
        n_raw=16,
        n_deduped=16,
        cycles=3.0,
        residual_source="poly_baseline",
        gamma_source="fixed",
    )
    assert det_roomy["lomb_peak_1_7"] is False
    assert det_roomy["detection"] is False


def test_p_boot_alone_does_not_set_peak():
    freqs, power = _nyquist_dominated_20_41_grid()
    det = evaluate_1_7_detection(
        freqs=freqs,
        power=power,
        p_boot=0.021,
        n_raw=16,
        n_deduped=16,
        cycles=3.0,
        residual_source="poly_baseline",
        gamma_source="fixed",
    )
    assert det["p_boot_ok"] is True
    assert det["lomb_peak_1_7"] is False


def test_geometry_mu_and_raw_dh_never_detect():
    freqs = np.linspace(0.05, 0.5, 200)
    power = np.zeros_like(freqs)
    i_17 = int(np.argmin(np.abs(freqs - 1.0 / 7.0)))
    power[i_17] = 1.0
    for src in ("geometric_mu", "raw_dh", "raw"):
        det = evaluate_1_7_detection(
            freqs=freqs,
            power=power,
            p_boot=0.001,
            n_raw=16,
            n_deduped=16,
            cycles=3.0,
            residual_source=src,
            gamma_source="fixed",
        )
        assert det["detection"] is False, src
        assert det["lomb_peak_1_7"] is False, src


@pytest.mark.skip(reason=_RETIRED_1_7)
def test_synthetic_sinusoid_at_1_7_detects():
    n = 16
    z = np.linspace(0.05, 4.0, n)
    gamma = 10.0
    s = s_from_z(z, gamma=gamma)
    cycles = float((s.max() - s.min()) / 7.0)
    assert cycles >= 2.0
    y = 20.0 + 0.3 * z + 4.0 * np.sin(2.0 * np.pi * s / 7.0)
    err = np.full(n, 0.05)
    sc = TauSBScanner(gamma=gamma, auto_calibrate_gamma=False)
    per = sc.scan_periodicity(
        z,
        y,
        err,
        n_raw=n,
        n_deduped=n,
        residual_source="poly_baseline",
        gamma_source="fixed",
        n_bootstrap=400,
        n_freq=400,
    )
    assert abs(per["f_max"] - 1.0 / 7.0) < 0.02, (
        f"f_max={per['f_max']}, ratio={per.get('power_ratio')}, "
        f"fail={per.get('fail_reasons')}"
    )
    assert per["lomb_peak_1_7"] is True, per.get("fail_reasons")
    assert per["detection"] is True, per.get("fail_reasons")
    assert per["gamma_source"] == "fixed"


def test_white_noise_n16_not_a_detection():
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
        gamma_source="fixed",
        n_bootstrap=400,
        n_freq=400,
    )
    assert per["detection"] is False
    assert per["lomb_peak_1_7"] is False


def test_skip_lomb_for_n_lt_8():
    z = np.linspace(0.2, 1.0, 7)
    y = np.ones(7)
    err = np.ones(7) * 0.1
    sc = TauSBScanner(gamma=10.0)
    per = sc.scan_periodicity(z, y, err)
    assert per["skipped"] is True
    assert per["detection"] is False
    assert per["lomb_peak_1_7"] is False


def test_duplicate_z_stack_n_deduped_lt_n_raw():
    def _block(z: float, y: float, label: str) -> dict:
        err = 0.1
        return {
            "z": np.array([z], dtype=float),
            "observable": np.array([y], dtype=float),
            "err": np.array([err], dtype=float),
            "quantity": ["DH_over_rs"],
            "cov": np.array([[err * err]], dtype=float),
            "source": label,
            "survey": label,
            "label": label,
        }

    stacked = _stack_bao_blocks(
        [
            _block(0.51, 19.0, "DESI_z0.51"),
            _block(0.51, 19.1, "BOSS_z0.51"),
            _block(0.71, 19.5, "DESI_z0.71"),
        ],
        label="dup_test",
        tracer="test",
        data_mode="DH_only",
        dedupe_z_tol=0.0,
        prefer_last=True,
    )
    assert int(stacked["n_raw"]) == 3
    assert int(stacked["n_deduped"]) < int(stacked["n_raw"])
    assert int(stacked["n_deduped"]) == count_unique_z(stacked["z"], tol=STANDARD_DEDUPE_Z_TOL)


def test_dv_converted_rows_absent_from_dh_only_periodogram():
    z = np.array([0.106, 0.51, 0.71], dtype=float)
    y = np.array([25.0, 19.0, 18.5], dtype=float)
    err = np.ones(3) * 0.2
    data = {
        "z": z,
        "observable": y,
        "err": err,
        "quantity": ["DH_over_rs", "DH_over_rs", "DH_over_rs"],
        "stack_sources": [
            "6dFGS_DV_to_DH fiducial-α DH",
            "DESI DR2 DH",
            "BOSS LRG DH",
        ],
        "dv_converted_rows": [True, False, False],
        "cov": np.diag(err ** 2),
        "n_raw": 3,
    }
    filtered = filter_dh_only_periodogram_vector(data)
    assert 0.106 not in set(np.round(filtered["z"], 4))
    assert filtered["z"].tolist() == [0.51, 0.71]
    assert int(filtered["dv_converted_rows_dropped"]) == 1


def test_json_fields_always_present():
    freqs, power = _nyquist_dominated_20_41_grid()
    det = evaluate_1_7_detection(
        freqs=freqs,
        power=power,
        p_boot=0.021,
        n_raw=16,
        n_deduped=12,
        cycles=1.57,
        residual_source="poly_baseline",
        gamma_source="fixed",
    )
    required = {
        "f_expected",
        "power_at_expected",
        "f_max",
        "power_max",
        "power_ratio",
        "p_boot",
        "n_raw",
        "n_deduped",
        "cycles",
        "gamma_source",
        "lomb_peak_1_7",
        "detection",
    }
    assert required.issubset(det.keys())


@pytest.mark.skip(reason=_RETIRED_1_7)
def test_lomb_scargle_cyclic_peaks_at_one_seventh():
    s = np.linspace(0.0, 21.0, 32)
    y = np.sin(2.0 * np.pi * s / 7.0)
    freqs = np.linspace(0.05, 0.5, 500)
    power = lomb_scargle_cyclic(s, y, freqs, normalize=True)
    f_max = float(freqs[int(np.argmax(power))])
    assert abs(f_max - 1.0 / 7.0) < 0.01
    ratio = float(power[int(np.argmin(np.abs(freqs - 1.0 / 7.0)))] / power.max())
    assert ratio >= 0.5
