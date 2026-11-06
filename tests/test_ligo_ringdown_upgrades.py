"""Ringdown upgrades: blind (f, τ) fit + GR (f, τ, Q), coherent stack + null veto, post-merger scan,
HEAVY_BBH batching and 32 s strain selection."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from test_ligo_ringdown import META, _cache_tree

FS = 4096


def _rd(n: int, t0: float, f: float = 250.0, tau: float = 0.004, amp: float = 1.0, rise: float = 0.003,
        phase: float = 0.7) -> np.ndarray:
    tt = np.arange(n) / FS - t0
    env = np.where(tt >= 0, np.exp(-np.clip(tt, 0, None) / tau), np.exp(-(np.clip(tt, None, 0) / rise) ** 2))
    return amp * env * np.cos(2 * np.pi * f * tt + phase)


def _colored(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    sos = signal.butter(2, 60.0, btype="low", fs=FS, output="sos")
    return 30.0 * signal.sosfilt(sos, w) + 0.3 * w


# ---------------------------------------------------------------- GR (f, τ, Q)
def test_gr_tau_q_gamma():
    from menus.gravitic.ligo.qnm import gr_prediction, kerr_q220

    g = gr_prediction({"final_mass_source": 61.5, "final_spin": 0.68, "redshift": 0.1,
                       "final_mass_source_upper": 3.4, "final_mass_source_lower": -3.0,
                       "final_spin_upper": 0.05, "final_spin_lower": -0.05, "redshift_upper": 0.03,
                       "redshift_lower": -0.04})
    for key in ("ok", "f220_hz", "sigma_hz", "m_final_detector", "a_final", "sigma_m_detector", "sigma_spin"):
        assert key in g
    assert g["q220"] == pytest.approx(0.7 + 1.4187 * 0.32 ** -0.499)
    assert g["tau220_s"] == pytest.approx(g["q220"] / (np.pi * g["f220_hz"]))
    assert g["gamma220_hz"] == pytest.approx(1 / (2 * np.pi * g["tau220_s"]))
    assert 3.8e-3 < g["tau220_s"] < 4.4e-3 and 245 < g["f220_hz"] < 258
    assert all(g[k] > 0 for k in ("sigma_hz", "sigma_gamma_hz", "sigma_tau_s", "sigma_q"))
    assert kerr_q220(0.0) == pytest.approx(2.1187)


# ---------------------------------------------------------------- blind fit
def test_vp_fit_exact_on_clean_damped_sinusoid():
    from menus.gravitic.ligo.qnm import fit_damped_ringdown

    n = 400
    for f0, tau0, ph in ((250.0, 0.004, 0.3), (90.0, 0.015, 2.5), (400.0, 0.002, -1.2)):
        y = _rd(n, 0.0, f=f0, tau=tau0, phase=ph)
        r = fit_damped_ringdown(y, FS, 0, n_fit=123)
        assert r["f_hz"] == pytest.approx(f0, abs=0.05)
        assert r["tau_s"] == pytest.approx(tau0, rel=1e-3)
        assert r["gamma_hz"] == pytest.approx(1 / (2 * np.pi * tau0), rel=1e-3)


def test_fit_is_blind_to_gr(tmp_path, monkeypatch):
    from menus.gravitic.ligo import qnm
    import superblock_residual as sr

    for fn in (qnm.fit_damped_ringdown, qnm.measure_ringdown, qnm.ringdown_from_prepared, qnm.prepare_whitened):
        names = set(inspect.signature(fn).parameters)
        assert not names & {"f220_hz", "tau220_s", "gamma220_hz", "meta", "final_mass_source", "final_spin", "redshift"}
    events, catalog = _cache_tree(tmp_path, META)
    opts = {"events_dir": str(events), "catalog_file": str(catalog), "postmerger_scan": "no"}
    D1, B1, *_ = sr._load_ligo(opts)
    real = qnm.gr_prediction

    def shifted(meta):
        g = real(meta)
        return {**g, "f220_hz": g["f220_hz"] * 1.5, "gamma220_hz": g["gamma220_hz"] * 0.5}

    monkeypatch.setattr(qnm, "gr_prediction", shifted)
    D2, B2, *_ = sr._load_ligo(opts)
    assert np.array_equal(D1, D2) and not np.allclose(B1, B2)


def test_fit_uncertainty_monte_carlo_calibration():
    """Colored noise (red, ≈60 Hz knee) + damped sinusoid: sandwich σ gives unit-width pulls."""
    from menus.gravitic.ligo.qnm import measure_ringdown

    n, t0 = 32 * FS, 16.0
    pf, pt = [], []
    for k in range(60):
        r = measure_ringdown(_colored(n, 100 + k) + _rd(n, t0, amp=10.0), FS, t0)["fit"]
        pf.append((r["f_hz"] - 250.0) / r["sigma_f_hz"])
        pt.append((r["tau_s"] - 0.004) / r["sigma_tau_s"])
    pf, pt = np.array(pf), np.array(pt)
    assert 0.75 < np.std(pf) < 1.35 and abs(np.mean(pf)) < 0.5
    assert 0.75 < np.std(pt) < 1.35 and abs(np.mean(pt)) < 0.5


# ---------------------------------------------------------------- coherent + null
def _segments(h1: np.ndarray, l1: np.ndarray):
    from menus.gravitic.ligo.strain_io import StrainSegment

    return [StrainSegment(Path("H.h5"), "H1", 1000.0, h1.size / FS, FS, h1),
            StrainSegment(Path("L.h5"), "L1", 1000.0, l1.size / FS, FS, l1)]


def _two_det(seed: int, *, signal_h=0.0, signal_l=0.0, glitch_h=0.0, echo=0.0, n=32 * FS, t0=16.0, dt=0.007):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(n)
    l = rng.standard_normal(n)
    if signal_h:
        h = h + _rd(n, t0, amp=signal_h)
        l = l - _rd(n, t0 - dt, amp=signal_l)
        if echo:
            h = h + _rd(n, t0 + 0.1, f=180.0, tau=0.01, amp=echo)
            l = l - _rd(n, t0 + 0.1 - dt, f=180.0, tau=0.01, amp=echo * signal_l / signal_h)
    if glitch_h:
        tt = np.arange(n) / FS - t0
        h = h + glitch_h * np.exp(-(tt / 0.004) ** 2) * np.cos(2 * np.pi * 220 * tt)
    return _segments(h, l), 1000.0 + t0


def test_coherent_alignment_stack_and_null():
    from menus.gravitic.ligo.coherent import coherent_combine, common_grid, network_snr, null_stream_veto

    segs, gps = _two_det(1, signal_h=6.0, signal_l=4.4)
    preps = common_grid(segs, gps)
    comb = coherent_combine(preps)
    a = comb["align"]["L1"]
    assert comb["reference"] == "H1"
    # narrow-band ringdown: |C(lag)| follows the envelope, so Δt is only fixed to ~ms (phase absorbs the rest)
    assert abs(abs(a["dt_s"]) - 0.007) < 0.0025
    assert a["coherence"] > 0.5  # (Δt, Δφ) jointly map L1 onto H1 (sign flip absorbed in Δφ)
    assert sum(w * w for w in comb["weights"].values()) == pytest.approx(1.0)
    ip, net = network_snr(comb)
    assert net > max(p["snr"] for p in preps)
    v = null_stream_veto(comb, ip)
    assert v["null_veto_passed"] and v["null_stream_chi2_red"] < 4


def test_null_stream_vetoes_single_detector_glitch():
    from menus.gravitic.ligo.coherent import coherent_combine, common_grid, network_snr, null_stream_veto

    segs, gps = _two_det(2, glitch_h=60.0)
    comb = coherent_combine(common_grid(segs, gps))
    ip, _ = network_snr(comb)
    v = null_stream_veto(comb, ip)
    assert not v["null_veto_passed"]
    assert v["null_stream_chi2_red"] > 4 and v["null_fraction"] > 0.15


# ---------------------------------------------------------------- post-merger scan
def test_postmerger_scan_detects_injected_echo_and_not_noise():
    from menus.gravitic.ligo.coherent import coherent_combine, common_grid, network_snr
    from menus.gravitic.ligo.postmerger import scan_from_preps

    segs, gps = _two_det(3, signal_h=6.0, signal_l=4.4, echo=2.0)
    preps = common_grid(segs, gps)
    comb = coherent_combine(preps)
    ip, _ = network_snr(comb)
    res = scan_from_preps(preps, comb, ip)
    assert res["n_background"] >= 80
    assert res["detected"] and res["sigma_equiv"] >= 3.0
    assert abs(res["delay_ms"] - 100) < 15 and abs(res["f_hz"] - 180) <= 20

    segs, gps = _two_det(4, signal_h=3.0, signal_l=2.2)
    preps = common_grid(segs, gps)
    comb = coherent_combine(preps)
    ip, _ = network_snr(comb)
    null = scan_from_preps(preps, comb, ip)
    assert not null["detected"] and null["fap"] > 0.01


# ---------------------------------------------------------------- batching + 32 s selection
def test_heavy_bbh_and_comma_lists():
    from menus.gravitic.ligo.fetcher import HEAVY_BBH_EVENTS, expand_event_query
    from sb_shared.dataset_registry import _ligo_target_options

    assert expand_event_query("HEAVY_BBH") == list(HEAVY_BBH_EVENTS)
    assert 6 <= len(HEAVY_BBH_EVENTS) <= 10 and "GW150914" in HEAVY_BBH_EVENTS
    assert expand_event_query("gw150914, GW190521 ,GW200129_065458") == ["GW150914", "GW190521", "GW200129_065458"]
    both = expand_event_query("GW150914,HEAVY_BBH")
    assert both[0] == "GW150914" and len(both) == len(HEAVY_BBH_EVENTS)
    assert expand_event_query("H:H1_R:1126259447:1126259479") == ["H:H1_R:1126259447:1126259479"]
    assert _ligo_target_options(["gwosc:HEAVY_BBH"], {}) == {}


def test_pull_selected_targets_expands_preset(monkeypatch):
    from menus.gravitic.ligo import fetcher

    seen = []
    monkeypatch.setattr(fetcher, "pull_event_strain", lambda e, **k: seen.append(e) or fetcher.PullSummary(requested=1))
    fetcher.pull_selected_targets(["GW150914,GW170814"])
    fetcher.pull_selected_targets(["HEAVY_BBH"])
    assert seen[:2] == ["GW150914", "GW170814"] and seen[2:] == list(fetcher.HEAVY_BBH_EVENTS)


def _strain(det, dur, fmt="hdf5", rate=4096, ver="v4"):
    return {"detector": det, "duration": dur, "format": fmt, "sampling_rate": rate, "GPSstart": 1126259447,
            "url": f"https://example.invalid/{ver}/{det}-{dur}-{rate}.{fmt}"}


def test_32s_selection_falls_back_to_older_version(monkeypatch):
    from menus.gravitic.ligo import fetcher

    v4 = {"commonName": "GW150914", "jsonurl": "https://example.invalid/json/v4",
          "strain": [_strain(d, 4096, f, r) for d in ("H1", "L1") for f in ("hdf5", "gwf") for r in (4096, 16384)]}
    v3 = {"commonName": "GW150914", "strain": [_strain(d, 32, f, r, "v3") for d in ("H1", "L1")
                                               for f in ("hdf5", "gwf", "txt") for r in (4096, 16384)]}
    versions = {"GW150914-v4": {"commonName": "GW150914", "version": 4, "final_mass_source": 1, "redshift": 0.1,
                                "jsonurl": "https://example.invalid/json/v4"},
                "GW150914-v3": {"commonName": "GW150914", "version": 3, "final_mass_source": 1, "redshift": 0.1,
                                "jsonurl": "https://example.invalid/json/v3"}}
    monkeypatch.setattr(fetcher, "_event_versions", lambda force_refresh=False: versions)
    monkeypatch.setattr(fetcher, "fetch_event_detail_url", lambda url: v3 if url.endswith("v3") else v4)
    opts = fetcher._strain_options({})
    files = fetcher.select_event_strain("GW150914", v4, opts)
    assert sorted((f.detector, f.duration, f.fmt, f.sampling_rate) for f in files) == [
        ("H1", 32, "hdf5", 4096), ("L1", 32, "hdf5", 4096)]
    assert all(f.duration == 32 for f in fetcher.select_event_strain("GW150914", v3, opts))
    # no 32 s anywhere → shortest available (old auto-select) instead of failing
    monkeypatch.setattr(fetcher, "fetch_event_detail_url", lambda url: v4)
    assert {f.duration for f in fetcher.select_event_strain("GW150914", v4, opts)} == {4096}


def test_loader_prefers_32s_file_per_detector(tmp_path):
    from superblock_residual import _ligo_pick_files

    files = [tmp_path / n for n in ("H-H1_LOSC_4_V1-1126256640-4096.hdf5", "H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
                                    "L-L1_LOSC_4_V1-1126256640-4096.hdf5")]
    picked = _ligo_pick_files(files, 1126259462.4)
    assert [p.name for p in picked] == ["H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5", "L-L1_LOSC_4_V1-1126256640-4096.hdf5"]


# ---------------------------------------------------------------- joint constraints + plots
def test_joint_constraints():
    from superblock_residual import _ligo_joint

    pts = [{"event": f"E{i}", "label": f"E{i}", "f_obs_hz": 90.0, "f220_gr_hz": 100.0, "sigma_hz": 10.0,
            "gamma_constrained": True, "gamma_obs_hz": 12.0, "gamma220_gr_hz": 10.0, "sigma_gamma_hz": 2.0,
            "tau_ms": 5.0, "tau220_gr_ms": 4.0, "sigma_tau_ms": 1.0, "sigma_tau_gr_ms": 0.0, "snr_peak": 8.0}
           for i in range(4)]
    j = _ligo_joint(pts)
    assert j["n_events"] == 4
    assert j["delta_f_over_f"]["mean"] == pytest.approx(-0.1) and j["delta_f_over_f"]["sigma"] == pytest.approx(0.05)
    assert j["delta_gamma_over_gamma"]["mean"] == pytest.approx(0.2)
    assert j["delta_gamma_over_gamma"]["sigma"] == pytest.approx(0.1)
    assert j["delta_tau_over_tau"]["mean"] == pytest.approx(0.25)


def test_scan_action_writes_both_plots(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import sb_shared.project_paths as pp
    from menus.gravitic.ligo import extension

    monkeypatch.setattr(pp, "SUPERBLOCK_ROOT", tmp_path)
    events, catalog = _cache_tree(tmp_path, META, detectors=("H1", "L1"))
    assert "Scan Post-Merger Markers" in extension.MENU_ACTIONS
    path = extension.run_action("Scan Post-Merger Markers", {"events_dir": str(events), "catalog_file": str(catalog),
                                                             "show_graphics": "artifacts"})
    rep = json.loads(Path(path).read_text())
    assert rep["postmerger_scan"] and "joint_constraints" in rep
    assert rep["joint_constraints"]["chi2_Lambda_per_N"] == pytest.approx(rep["chi2_Lambda"] / rep["N"])
    for name in ("ligo_ringdown_residuals.png", "ligo_postmerger_scan.png"):
        assert (tmp_path / "artifacts" / "ligo_gwosc" / name).stat().st_size > 10_000
