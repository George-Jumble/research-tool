#!/usr/bin/env python3
"""Acceptance tests for TSB Casimir diagnostics, controls, and bound-energy residuals."""

from __future__ import annotations

import io
import tempfile
import uuid
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from menus.particle.casimir.bound_energy import (
    estimate_bound_energy,
    identity_subtraction,
)
from menus.particle.casimir.diagnostics import record_hash_ledger
from menus.particle.casimir.test_framework import (
    run_full_test_suite,
    test_anisotropy_channels as anisotropy_channels,
    test_log_modulations as log_modulations,
    test_periodicity as periodicity,
    test_powerlaw_index as powerlaw_index,
)


def test_log_modulations_permutation_distinguishes_sine_from_noise():
    rng = np.random.default_rng(11)
    x = np.logspace(-1.0, 2.0, 180)
    y_sine = 3.0 * np.sin(2.0 * np.pi * (1.0 / 7.0) * np.log10(x)) + 0.05 * rng.normal(
        size=x.size
    )
    y_noise = rng.normal(size=x.size)
    sine = log_modulations(x, y_sine, n_perm=79, seed=3)
    noise = log_modulations(x, y_noise, n_perm=79, seed=3)
    assert sine["p_permutation"] != noise["p_permutation"]
    assert sine["p_value"] < noise["p_value"]
    src = Path(__file__).resolve().parents[1].joinpath("test_framework.py").read_text(
        encoding="utf-8"
    )
    assert "residuals / resid_std" not in src
    assert "chi2_val = float(np.sum((residuals / resid_std) ** 2))" not in src
    assert sine.get("s_resid_note")


def test_anisotropy_identical_aux_is_null_and_control_ok():
    n = 80
    xi = np.linspace(0.1, 10.0, n)
    eps = 1.0 + 40.0 / (1.0 + xi**2)
    df = pd.DataFrame(
        {
            "matsubara_frequency": xi,
            "dielectric_response_primary": eps,
            "dielectric_response_1": eps.copy(),
            "dielectric_response_2": eps.copy(),
        }
    )
    out = anisotropy_channels(df, primary="dielectric_response_primary")
    assert out["evidence_for_TSB"] is False
    assert out["control_failed"] is False
    assert out["status"] == "ok"


def test_powerlaw_index_dielectric_vs_xi_not_applicable():
    x = np.linspace(0.1, 20.0, 50)
    y = 1.0 + 10.0 / (1.0 + x)
    out = powerlaw_index(
        x,
        y,
        param_col="matsubara_frequency",
        value_col="dielectric_response_primary",
    )
    assert out["status"] == "not_applicable"
    assert out["p_value"] is None
    assert out["evidence_for_TSB"] is False
    assert out["counted_in_S_TSB"] is False


def test_powerlaw_index_lifshitz_force_law_is_not_tsb():
    d = np.logspace(-8.0, -6.0, 40)
    force = 2.5e-12 * d ** (-4.0)
    out = powerlaw_index(
        d,
        force,
        param_col="separation_nm",
        value_col="force_pN",
    )
    assert out["status"] == "ok"
    assert out["evidence_for_TSB"] is False
    assert abs(out["m"] + 4.0) < 1e-6


def test_periodicity_sampling_comb_is_grid_locked():
    dx = 0.25
    x = np.arange(160, dtype=float) * dx
    y = np.cos(2.0 * np.pi * np.arange(160) / 4.0)
    out = periodicity(x, y)
    assert out["grid_locked"] is True
    assert out["evidence_for_TSB"] is False
    assert out["dominant_period"] is not None
    assert out["Delta_xi"] is not None


def test_identity_lifshitz_subtraction_has_no_e_bind():
    ident = identity_subtraction(np.array([-1.0e-12, -2.0e-12, -3.0e-12]))
    assert ident["identity_control_ok"] is True
    assert np.allclose(ident["Delta_E"], 0.0)

    xi = np.linspace(0.2, 8.0, 12)
    eps = 1.0 + 25.0 / (1.0 + xi**2)
    df = pd.DataFrame(
        {
            "matsubara_frequency": xi,
            "dielectric_response_primary": eps,
        }
    )
    be = estimate_bound_energy(
        df,
        param_col="matsubara_frequency",
        value_col="dielectric_response_primary",
        d=np.logspace(-8.0, -7.0, 6),
        n_k=24,
        data_label="synth_Ag",
    )
    assert be["identity_control_ok"] is True
    assert be["E_bind_claimed"] is False
    assert np.allclose(np.asarray(be["Delta_E"], dtype=float), 0.0)


def test_hash_ledger_stable_then_mismatch(tmp_path: Path | None = None):
    tmp_dir = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp(prefix="casimir_hash_"))
    ledger = tmp_dir / "casimir_hash_ledger.jsonl"
    csv_path = tmp_dir / "probe.csv"
    x = np.linspace(1.0, 10.0, 30)
    y = 1.0 / x
    pd.DataFrame({"separation_nm": x, "force_pN": y}).to_csv(csv_path, index=False)

    def _score(label: str) -> dict:
        df = pd.read_csv(csv_path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            report = run_full_test_suite(
                df,
                param_col="separation_nm",
                value_col="force_pN",
                output_prefix=f"pytest_{label}",
                plot=False,
                file_path=csv_path,
                data_label=label,
                n_bootstrap=20,
                n_perm=21,
            )
        return {"report": report, "stdout": buf.getvalue()}

    label = f"pytest_casimir_hash_{uuid.uuid4().hex[:8]}"
    first = _score(label)
    second = _score(label)
    assert "HASH MISMATCH" not in first["stdout"]
    assert "HASH MISMATCH" not in second["stdout"]
    assert first["report"]["raw_series_sha256"] == second["report"]["raw_series_sha256"]
    assert first["report"]["pool_allowed"] is True

    df = pd.read_csv(csv_path)
    df.loc[0, "force_pN"] = float(df.loc[0, "force_pN"]) + 0.5
    df.to_csv(csv_path, index=False)
    third = _score(label)
    assert "HASH MISMATCH" in third["stdout"]
    assert third["report"]["pool_allowed"] is False
    assert third["report"]["raw_series_sha256"] != first["report"]["raw_series_sha256"]

    # Direct ledger helper (temp file) still appends and never overwrites.
    H1 = first["report"]["raw_series_sha256"]
    H2 = third["report"]["raw_series_sha256"]
    record_hash_ledger(
        data_label="tmp_only",
        path=csv_path,
        H=H1,
        n_rows=30,
        n_cols=2,
        ledger_path=ledger,
    )
    record_hash_ledger(
        data_label="tmp_only",
        path=csv_path,
        H=H2,
        n_rows=30,
        n_cols=2,
        ledger_path=ledger,
    )
    lines = [ln for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2


if __name__ == "__main__":
    tests = [
        test_log_modulations_permutation_distinguishes_sine_from_noise,
        test_anisotropy_identical_aux_is_null_and_control_ok,
        test_powerlaw_index_dielectric_vs_xi_not_applicable,
        test_powerlaw_index_lifshitz_force_law_is_not_tsb,
        test_periodicity_sampling_comb_is_grid_locked,
        test_identity_lifshitz_subtraction_has_no_e_bind,
        test_hash_ledger_stable_then_mismatch,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
