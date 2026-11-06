"""Prime Past BBN: AlterBBN v2.2 baseline, computed G1–G5 gates, residual loader (ruling 2026-11-04).

Tests that need the AlterBBN binary skip when it is not installed. Every run writes its cache to
tmp_path (never into datasets/).
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

import superblock_residual as sr
from bbn_isolated import alterbbn, runner

# One lazy bootstrap per session (fetch pinned tarball + compile into datasets/prime_past/) unless
# SB_ALTERBBN_NO_AUTO_BUILD=1; offline / no compiler → skip.
_READY = alterbbn.ensure_available(verbose=True)
needs_alterbbn = pytest.mark.skipif(
    not (_READY or alterbbn.available()),
    reason="AlterBBN sb_bbn.x not available (offline, no C compiler, or SB_ALTERBBN_NO_AUTO_BUILD=1)",
)
needs_full_driver = pytest.mark.skipif(
    not _READY,
    reason="sb_bbn_full.x not available (python -m bbn_isolated.alterbbn ensure)",
)


@pytest.fixture
def no_alterbbn(monkeypatch):
    """Simulate a machine without AlterBBN: nothing found, auto-build disabled."""
    monkeypatch.setenv("SB_ALTERBBN_NO_AUTO_BUILD", "1")
    monkeypatch.setattr(alterbbn, "find_binary", lambda name=alterbbn.BINARY_NAME: None)


@pytest.fixture(autouse=True)
def _tmp_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "_ARTIFACTS", tmp_path / "reports")
    monkeypatch.setattr(alterbbn, "CACHE_DIR", tmp_path / "alterbbn_cache")


def _fake_run(values, *, closure_ok=True, err=0):
    """Stand-in for alterbbn.run: same result shape, chosen abundances."""

    def fake(eta=6.114e-10, n_eff=3.044, tau_n=877.75, failsafe=1, *, full=False, **_):
        out = {"eta": eta, "n_eff": n_eff, "tau_n": tau_n, "failsafe": failsafe, "err": err,
               "values": dict(values), "driver": "fake", "binary_sha256": None, "from_cache": False,
               "cache_path": None}
        if full:
            y_h = 0.7531581505731
            ratio = [0.0] * 27
            ratio[0], ratio[2], ratio[3], ratio[6] = eta, y_h, values["D_H"], values["Y_p"]
            ratio[5], ratio[8] = values["He3_H"], values["Li7_H"]
            A = [0.0, 1, 1, 2, 3, 3, 4, 6, 7, 7, 8, 8, 9, 10, 11, 11, 12, 12, 12, 13, 13, 14, 14, 14, 15, 15, 16]
            # close baryon number exactly unless told otherwise
            rest = sum(A[i] * ratio[i] * y_h for i in (3, 5, 8))
            ratio[2] = y_h if not closure_ok else (1.0 - values["Y_p"]) / (1.0 + rest / y_h)
            out.update(ratioH=ratio, A=A, Z=[0.0] * 27)
        return out

    return fake


GOOD = {"Y_p": 0.2468, "D_H": 2.43e-5, "He3_H": 1.03e-5, "Li7_H": 5.5e-10, "Be7_H": 5.2e-10}


# ----------------------------------------------------------------------------- AlterBBN wrapper
@needs_alterbbn
def test_alterbbn_planck_baseline_values_and_cache(tmp_path):
    first = alterbbn.run(cache_dir=tmp_path / "c")
    assert first["from_cache"] is False
    assert math.isclose(first["values"]["Y_p"], 0.2468, abs_tol=5e-4)
    assert math.isclose(first["values"]["D_H"], 2.43e-5, rel_tol=0.01)
    again = alterbbn.run(cache_dir=tmp_path / "c")
    assert again["from_cache"] is True and again["values"] == first["values"]
    files = sorted((tmp_path / "c").glob("alterbbn__sb_bbn__*.csv"))
    assert [f.name for f in files] == ["alterbbn__sb_bbn__eta6.1140e-10_neff3.0440_tau877.75_fs1.csv"]
    with open(files[0], newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {"observable", "value", "unc", "source", "year", "units", "reference"} <= set(rows[0])
    assert [r["observable"] for r in rows] == list(alterbbn.OBSERVABLES)
    assert rows[0]["binary_sha256"] and rows[0]["source"] == "AlterBBN v2.2"


def test_alterbbn_missing_raises_unavailable(tmp_path, no_alterbbn):
    with pytest.raises(alterbbn.AlterBBNUnavailable):
        alterbbn.run(cache_dir=tmp_path / "empty")


# ----------------------------------------------------------------------------- gates
@needs_alterbbn
@needs_full_driver
def test_gates_computed_from_alterbbn_all_green(tmp_path):
    rep = runner.report(cache_dir=tmp_path / "c")
    assert rep["status"] == "PASS_GATES" and rep["failed_gate"] == ""
    assert rep["gates"] == {g: True for g in runner.GATE_IDS}
    det = rep["gate_detail"]
    assert abs(det["G1"]["baryon_closure"]) < runner.G1_TOL
    assert abs(det["G4"]["delta_Y_p"]) < 0.005 and abs(det["G4"]["rel_delta_D_H"]) < 0.2
    assert set(det["G5"]["variants"]) == {"double_step", "half_step", "tighter_tolerance"}
    assert det["driver_agreement_rel"] < 1e-5  # sb_bbn.x and sb_bbn_full.x agree
    assert rep["eta"] == 6.114e-10 and rep["n_eff"] == 3.044 and rep["tau_n_used"] == 877.75


def test_gates_follow_the_output_not_hardcoded(monkeypatch):
    monkeypatch.setattr(alterbbn, "run", _fake_run(GOOD))
    assert runner.gates() == {g: True for g in runner.GATE_IDS}
    monkeypatch.setattr(alterbbn, "run", _fake_run({**GOOD, "D_H": 0.0}))
    g = runner.gates()
    assert g["G3"] is False and g["G4"] is False and g["G5"] is False
    monkeypatch.setattr(alterbbn, "run", _fake_run({**GOOD, "Li7_H": -1e-12}))
    assert runner.gates()["G2"] is False
    monkeypatch.setattr(alterbbn, "run", _fake_run({**GOOD, "Y_p": 0.2600}))
    assert runner.gates()["G4"] is False
    monkeypatch.setattr(alterbbn, "run", _fake_run(GOOD, closure_ok=False))
    assert runner.gates()["G1"] is False


def test_runner_without_alterbbn_reports_unavailable(tmp_path, no_alterbbn):
    rep = runner.report(cache_dir=tmp_path / "empty")
    assert rep["status"] == "ALTERBBN_UNAVAILABLE" and rep["failed_gate"] == "G1"
    assert rep["gates"] == {g: False for g in runner.GATE_IDS}


# ----------------------------------------------------------------------------- residual loader
@needs_alterbbn
@needs_full_driver
def test_prime_past_loader_scores_alterbbn_vs_cooke_and_aver(tmp_path):
    rep = sr.run_superblock_residual("PRIME_PAST_HARMONIC", "Verification Suite",
                                     {"alterbbn_cache_dir": str(tmp_path / "c")})
    assert rep["load_error"] is None and rep["N"] == 2
    assert rep["observables"] == ["D/H", "Y_p"]
    b = alterbbn.run(cache_dir=tmp_path / "c")["values"]
    D = np.array([2.527e-5, 0.2453])
    sig = np.array([0.030e-5, 0.0034])
    B = np.array([b["D_H"], b["Y_p"]])
    assert math.isclose(rep["chi2_Lambda"], float(np.sum(((D - B) / sig) ** 2)), rel_tol=1e-12)
    assert rep["chi2_SB"] is None and rep["verdict"] == "inconclusive"
    assert rep["sb_prediction"] == "pending" and rep["score"]["sb_prediction"] == "pending"
    assert "pending" in rep["score"]["kernel_note"] and rep["score"]["S_k"] is None
    assert rep["bbn_gates"] == {g: True for g in runner.GATE_IDS}
    assert math.isclose(rep["pulls"]["Y_p"], (0.2453 - b["Y_p"]) / 0.0034)


def test_prime_past_without_alterbbn_fails_cleanly(tmp_path, no_alterbbn):
    rep = sr.run_superblock_residual("PRIME_PAST_HARMONIC", "Verification Suite",
                                     {"alterbbn_cache_dir": str(tmp_path / "empty")})
    assert rep["N"] == 0 and rep["chi2_Lambda"] is None and rep["chi2_SB"] is None
    assert "not found" in rep["load_error"]


def test_prime_past_refuses_baseline_when_a_gate_fails(monkeypatch):
    monkeypatch.setattr(alterbbn, "run", _fake_run({**GOOD, "D_H": 0.0}))
    rep = sr.run_superblock_residual("PRIME_PAST_HARMONIC", "Verification Suite", {})
    assert rep["N"] == 0 and "gate G3" in rep["load_error"]
    assert rep["bbn_gates"]["G3"] is False


# ----------------------------------------------------------------------------- repo conventions
@needs_alterbbn
def test_alterbbn_baseline_is_a_curated_repo_source(tmp_path):
    import pandas as pd

    from menus.empirical_tests import tests as empirical

    repo = empirical.REPOS["alterbbn_planck_baseline"]
    assert repo["type"] == "curated" and repo["url"] is None
    frame = empirical.load_or_download("alterbbn_planck_baseline")
    assert list(frame["observable"]) == list(alterbbn.OBSERVABLES)
    assert len(getattr(empirical, repo["parser"])(frame)) == len(frame)
    dest = tmp_path / "alterbbn_planck_baseline.csv"
    assert empirical.materialize_curated_cache("alterbbn_planck_baseline", dest)
    assert list(pd.read_csv(dest)["observable"]) == list(alterbbn.OBSERVABLES)


def test_toy_bbn_network_is_deprecated_not_deleted():
    from pathlib import Path

    from menus.prime_past import bbn_network

    assert Path(bbn_network.__file__).is_file()
    assert bbn_network.DEPRECATED is True
    assert "DEPRECATED" in (bbn_network.__doc__ or "")
    with pytest.warns(DeprecationWarning, match="AlterBBN"):
        bbn_network.run_bbn(verbose=False)
