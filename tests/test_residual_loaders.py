"""Residual-engine loaders for Planck, LIGO, CERN, Casimir and Prime Past.

Real-data checks skip when the module cache is empty; format/physics checks use tiny
synthetic inputs written to tmp_path (never into datasets/).
"""

import gzip
import math

import numpy as np
import pytest

import superblock_residual as sr


@pytest.fixture(autouse=True)
def _tmp_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "_ARTIFACTS", tmp_path / "reports")


# --- Planck -----------------------------------------------------------------
def test_planck_binned_parser(tmp_path):
    from menus.planck_cmb.spectrum import load_tt_binned

    f = tmp_path / "tt.txt"
    f.write_text("# l Dl -dDl +dDl BestFit\n47.7 1500 50 52 1480\n30 1000 40 40 1010\n")
    d = load_tt_binned(f)
    assert list(d["ell"]) == [30, 47.7]
    assert d["sigma"][1] == 51


def test_planck_missing_fails_cleanly(monkeypatch, tmp_path):
    from menus.planck_cmb import spectrum

    if spectrum.find_cached_tt_spectrum() is not None:
        pytest.skip("Planck TT spectrum is cached")
    rep = sr.run_superblock_residual("PLANCK_CMB", "Compute Power Spectrum", {"auto_fetch": "no"})
    assert rep["N"] == 0 and rep["chi2_Lambda"] is None and "not cached" in rep["load_error"]


def test_planck_real_chi2_lambda():
    from menus.planck_cmb import spectrum

    if spectrum.find_cached_tt_spectrum() is None:
        pytest.skip("Planck TT spectrum not cached")
    rep = sr.run_superblock_residual("PLANCK_CMB", "Compute Power Spectrum", {"auto_fetch": "no"})
    assert rep["N"] > 50 and rep["chi2_Lambda"] > 0 and rep["chi2_SB"] is None


# --- LIGO -------------------------------------------------------------------
def test_kerr_f220_gw150914_scale():
    from menus.gravitic.ligo.qnm import gr_prediction

    gr = gr_prediction({"final_mass_source": 63.1, "final_spin": 0.69, "redshift": 0.09})
    assert gr["ok"] and 240 < gr["f220_hz"] < 260  # literature ≈ 251 Hz


def test_gr_prediction_reports_missing_spin():
    from menus.gravitic.ligo.qnm import gr_prediction

    assert gr_prediction({"final_mass_source": 63.1, "redshift": 0.09})["missing"] == ["final_spin"]


def test_ligo_loader_end_to_end_on_tmp_cache(tmp_path):
    import json

    fs, f0, dur = 4096, 250.0, 4
    t = np.arange(fs * dur) / fs
    merger = 2.0
    h = np.where(t > merger, np.exp(-(t - merger) / 0.004) * np.sin(2 * np.pi * f0 * (t - merger)), 0.0)
    h = h + 1e-3 * np.random.default_rng(0).normal(size=t.size)
    ev = tmp_path / "gwosc" / "events" / "GWTEST"
    ev.mkdir(parents=True)
    with gzip.open(ev / f"H-H1_TEST-1000-{dur}.txt.gz", "wt") as g:
        g.write("\n".join(f"{v:.6e}" for v in h))
    cat = tmp_path / "gwosc" / "event_catalog.json"
    cat.write_text(json.dumps({"x": {"commonName": "GWTEST", "GPS": 1000 + merger,
                                     "final_mass_source": 63.1, "final_spin": 0.69, "redshift": 0.09}}))
    D, B, S, X, notes = sr._load_ligo({"events_dir": str(ev.parent), "catalog_file": str(cat)})
    assert D.size == 2 and abs(D[0] - f0) < 3 * S[0] + 10  # rows: f [Hz], γ = 1/(2πτ) [Hz]
    assert 240 < B[0] < 260


def test_ligo_missing_fails_cleanly(tmp_path):
    rep = sr.run_superblock_residual("LIGO_GWOSC", "Full Ringdown Report", {"events_dir": str(tmp_path / "none")})
    assert rep["N"] == 0 and "No cached GWOSC strain" in rep["load_error"]


# --- CERN -------------------------------------------------------------------
def test_dimuon_mass_and_histogram():
    from menus.particle.cms.dimuon import dimuon_masses_from_arrays, histogram_comparison

    # two back-to-back 45.6 GeV muons at eta=0 -> m ≈ 91.2 GeV; same-sign pair dropped
    m = dimuon_masses_from_arrays([np.array([45.6, 45.6]), np.array([30.0, 20.0])],
                                  [np.zeros(2), np.zeros(2)],
                                  [np.array([0.0, math.pi]), np.zeros(2)],
                                  [np.array([1, -1]), np.array([1, 1])])
    assert m.size == 1 and abs(m[0] - 91.2) < 0.1
    rng = np.random.default_rng(1)
    res = histogram_comparison(rng.normal(91, 3, 5000), rng.normal(91, 3, 20000))
    assert math.isclose(res["B"].sum(), res["D"].sum())


def test_cern_missing_fails_cleanly(monkeypatch, tmp_path):
    from menus.particle.cms import dimuon

    if dimuon.cached_root_files(dimuon.DATA_TARGET) and dimuon.cached_root_files(dimuon.MC_TARGET):
        pytest.skip("CMS ROOT files are cached")
    rep = sr.run_superblock_residual("CERN_OPENDATA", "Analyze", {})
    assert rep["N"] == 0 and "ROOT files" in rep["load_error"]


# --- Casimir ----------------------------------------------------------------
def test_ideal_force_scaling():
    from menus.particle.casimir.force_table import ideal_force

    f1, f2 = ideal_force([1e-7, 2e-7], sphere_radius_m=1e-4)
    assert f1 < 0 and math.isclose(f1 / f2, 8.0)
    p1, p2 = ideal_force([1e-7, 2e-7], plate_area_m2=1e-8)
    assert math.isclose(p1 / p2, 16.0)


def test_casimir_loader_tmp_table_gives_real_chi2_lambda(tmp_path):
    from menus.particle.casimir.force_table import ideal_force

    d = np.linspace(1e-7, 5e-7, 6)
    f = ideal_force(d, sphere_radius_m=1e-4) * 1.01
    csv = tmp_path / "toy_force.csv"
    csv.write_text("separation_m,force_N,sigma_N\n" + "\n".join(f"{a},{b},{abs(b)*0.02}" for a, b in zip(d, f)))
    rep = sr.run_superblock_residual("CASIMIR", "Bound-Energy Residual Search",
                                     {"casimir_force_file": str(csv), "sphere_radius_m": 1e-4})
    assert rep["N"] == 6 and rep["chi2_Lambda"] is not None and rep["chi2_Lambda"] > 0
    assert rep["chi2_SB"] is None  # Template C Mirron kernel is LHCb-only


def test_casimir_without_geometry_fails_cleanly(tmp_path):
    csv = tmp_path / "x_force.csv"
    csv.write_text("separation_m,force_N\n1e-7,-1e-9\n2e-7,-2e-10\n")
    rep = sr.run_superblock_residual("CASIMIR", "Bound-Energy Residual Search", {"casimir_force_file": str(csv)})
    assert rep["N"] == 0 and "geometry" in rep["load_error"]


# --- Prime Past -------------------------------------------------------------
def test_prime_past_has_an_alterbbn_loader(tmp_path, monkeypatch):
    """Was 'missing baseline'. Prime Past now loads AlterBBN (full tests in test_prime_past_alterbbn.py)."""
    from bbn_isolated import alterbbn

    if not alterbbn.ensure_available():
        pytest.skip("AlterBBN not available (offline, no C compiler, or SB_ALTERBBN_NO_AUTO_BUILD=1)")
    rep = sr.run_superblock_residual("PRIME_PAST_HARMONIC", "Verification Suite",
                                     {"alterbbn_cache_dir": str(tmp_path / "alterbbn")})
    assert rep["N"] == 2 and rep["chi2_Lambda"] is not None and rep["chi2_SB"] is None
    assert not any("no residual loader" in n for n in rep["notes"])
