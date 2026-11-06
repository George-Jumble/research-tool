"""Template C (Mirron kernel) stays on LHCb only (author ruling 2026-11-04).

CERN dimuon and Casimir are template-C datasets, but the kernel mapping is not ruled, so they refuse
Template C with 'awaiting author ruling on mapping' even when q² is supplied.
"""

from __future__ import annotations

import numpy as np
import pytest

import superblock_residual as sr

MSG = "awaiting author ruling on mapping"


@pytest.fixture(autouse=True)
def _tmp_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "_ARTIFACTS", tmp_path / "reports")


def test_scope_constants():
    assert sr.MIRRON_C_SCOPE == frozenset({"LHCB_FLAVOR"})
    assert {"CERN_OPENDATA", "CASIMIR"} <= sr.TEMPLATE_C_REFUSED
    assert sr.TEMPLATE_C_REFUSAL == MSG
    for tag in ("CERN_OPENDATA", "CASIMIR", "LHCB_FLAVOR"):
        assert sr.DATASET_TEMPLATE[tag] == "C"


@pytest.mark.parametrize("dataset", ["CERN_OPENDATA", "CASIMIR"])
def test_template_c_refused_even_with_q2(dataset):
    r = np.array([0.1, -0.2, 0.05])
    out = sr.score_against_template(r, template="C", sigma=np.ones(3), q2=np.array([1.0, 2.0, 3.0]),
                                    dataset=dataset)
    assert out["kernel"] == "refused" and out["template_C_refused"] == MSG
    assert MSG in out["kernel_note"] and out["chi2_SB"] is None and out["verdict"] == "inconclusive"


@pytest.mark.parametrize("dataset", ["CERN_OPENDATA", "CASIMIR"])
def test_residual_report_refuses_template_c(dataset):
    D = np.array([10.0, 12.0, 9.0])
    B = np.array([10.5, 11.0, 9.5])
    rep = sr.residual_report(dataset=dataset, action="x", template="C", D=D, B=B, sigma=np.ones(3),
                             q2=np.array([1.0, 2.0, 3.0]))
    assert rep["chi2_SB"] is None and rep["chi2_Lambda"] is not None
    assert rep["score"]["template_C_refused"] == MSG


def test_cern_run_refuses_template_c(monkeypatch):
    x = np.linspace(61, 119, 5)
    monkeypatch.setattr(sr, "_load_cern", lambda options=None: (np.full(5, 100.0), np.full(5, 98.0),
                                                                 np.full(5, 10.0), x, ["synthetic"]))
    rep = sr.run_superblock_residual("CERN_OPENDATA", "Analyze", {})
    assert rep["N"] == 5 and rep["chi2_Lambda"] > 0 and rep["chi2_SB"] is None
    assert rep["score"]["kernel"] == "refused"
    assert any(MSG in n for n in rep["notes"])


def test_casimir_run_refuses_template_c(tmp_path):
    from menus.particle.casimir.force_table import ideal_force

    d = np.linspace(1e-7, 5e-7, 4)
    f = ideal_force(d, sphere_radius_m=1e-4)
    csv = tmp_path / "toy_force.csv"
    csv.write_text("separation_m,force_N,sigma_N\n" + "\n".join(f"{a},{b},{abs(b)*0.02}" for a, b in zip(d, f)))
    rep = sr.run_superblock_residual("CASIMIR", "Bound-Energy Residual Search",
                                     {"casimir_force_file": str(csv), "sphere_radius_m": 1e-4})
    assert rep["score"]["template_C_refused"] == MSG and rep["chi2_SB"] is None
    assert any(MSG in n for n in rep["notes"])


def test_lhcb_still_scored_by_mirron_kernel():
    rep = sr.run_superblock_residual("LHCB_FLAVOR", "Score Flavor Residuals", {})
    assert rep["score"].get("kernel") != "refused"
    assert rep["chi2_SB"] is not None and rep["verdict"] in {"SB_better", "baseline_better"}
