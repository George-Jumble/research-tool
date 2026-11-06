"""LIGO GWOSC ringdown: catalog versions, PE spin extraction, backfill, whitened measurement, plot."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

FS = 4096


def _write_txt_gz(path: Path, strain: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for v in strain:
            fh.write(f"{v:.8e}\n")


def _ringdown(n: int, t0: float, f: float = 250.0, tau: float = 0.004, amp: float = 1.0) -> np.ndarray:
    t = np.arange(n) / FS
    out = np.zeros(n)
    m = t >= t0
    out[m] = amp * np.exp(-(t[m] - t0) / tau) * np.cos(2 * np.pi * f * (t[m] - t0))
    return out


META = {"commonName": "GWTEST", "GPS": 1002.0, "final_mass_source": 63.1, "final_spin": 0.69, "redshift": 0.09,
        "version": 3}

ALLEVENTS = {
    "GW150914-v1": {"commonName": "GW150914", "version": 1, "final_mass_source": None, "redshift": None,
                    "catalog.shortName": "O1_O2-Preliminary",
                    "jsonurl": "https://gwosc.org/eventapi/json/O1_O2-Preliminary/GW150914/v1"},
    "GW150914-v3": {"commonName": "GW150914", "version": 3, "final_mass_source": 63.1, "redshift": 0.09,
                    "catalog.shortName": "GWTC-1-confident",
                    "jsonurl": "https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3"},
    "GW150914-v4": {"commonName": "GW150914", "version": 4, "final_mass_source": 61.5, "redshift": 0.1,
                    "catalog.shortName": "GWTC-2.1-confident",
                    "jsonurl": "https://gwosc.org/eventapi/json/GWTC-2.1-confident/GW150914/v4"},
    "GW150914-v2": {"commonName": "GW150914", "version": 2, "final_mass_source": None, "redshift": None,
                    "catalog.shortName": "O1_O2-Preliminary",
                    "jsonurl": "https://gwosc.org/eventapi/json/O1_O2-Preliminary/GW150914/v2"},
}

DETAIL_GWTC1 = {"commonName": "GW150914", "version": 3, "GPS": 1126259462.4, "parameters": {
    "gwtc1_cwb_GW150914": {"is_preferred": False, "pipeline_type": "search", "final_spin": None},
    "gwtc1_pe_GW150914": {"is_preferred": True, "pipeline_type": "pe", "final_mass_source": 63.1,
                          "final_mass_source_lower": -3.0, "final_mass_source_upper": 3.4, "a_final": 0.69,
                          "a_final_lower": -0.04, "a_final_upper": 0.05, "redshift": 0.09,
                          "redshift_lower": -0.04, "redshift_upper": 0.03},
}}
DETAIL_GWTC21 = {"commonName": "GW150914", "version": 4, "GPS": 1126259462.4, "parameters": {
    "GWTC-1-confident_GW150914_R2_pe_combined": {"is_preferred": True, "pipeline_type": "pe",
                                                 "final_mass_source": 61.5, "final_spin": 0.68,
                                                 "final_spin_lower": -0.05, "final_spin_upper": 0.05,
                                                 "redshift": 0.1, "redshift_upper": 0.03, "redshift_lower": -0.04},
}}


# ---------------------------------------------------------------- catalog versions
def test_preliminary_versions_do_not_clobber():
    from menus.gravitic.common.gwosc_api import select_best_versions

    best = select_best_versions(ALLEVENTS)
    assert list(best) == ["GW150914"]
    assert best["GW150914"]["version"] == 4
    assert best["GW150914"]["final_mass_source"] == 61.5
    # a later preliminary-only version without mass/z still loses to a confident one
    entries = dict(ALLEVENTS)
    entries["GW150914-v5"] = {"commonName": "GW150914", "version": 5, "final_mass_source": None, "redshift": None}
    assert select_best_versions(entries)["GW150914"]["version"] == 4


def test_catalog_entry_numeric_version():
    from menus.gravitic.ligo.qnm import catalog_entry

    cat = {"a": {"commonName": "GWX", "version": 10, "final_mass_source": 1, "redshift": 0.1},
           "b": {"commonName": "GWX", "version": 9, "final_mass_source": 1, "redshift": 0.1}}
    assert catalog_entry(cat, "gwx")["version"] == 10


def test_stale_catalog_auto_refresh(tmp_path, monkeypatch):
    from menus.gravitic.common import gwosc_api

    cache = tmp_path / "event_catalog.json"
    cache.write_text(json.dumps({"GW150914": ALLEVENTS["GW150914-v2"]}))  # clobbered, no schema sidecar
    calls = []

    def fake_fetch(**_):
        calls.append(1)
        return gwosc_api.select_best_versions(ALLEVENTS)

    monkeypatch.setattr(gwosc_api, "fetch_all_events", fake_fetch)
    cat = gwosc_api.cache_event_catalog(cache)
    assert cat["GW150914"]["version"] == 4 and calls == [1]
    assert not gwosc_api.catalog_is_stale(cache)
    gwosc_api.cache_event_catalog(cache)  # fresh now: no second fetch
    assert calls == [1]


def test_stale_catalog_offline_keeps_cache(tmp_path, monkeypatch):
    from menus.gravitic.common import gwosc_api

    cache = tmp_path / "event_catalog.json"
    cache.write_text(json.dumps({"GW150914": ALLEVENTS["GW150914-v3"]}))

    def boom(**_):
        raise OSError("offline")

    monkeypatch.setattr(gwosc_api, "fetch_all_events", boom)
    assert gwosc_api.cache_event_catalog(cache)["GW150914"]["version"] == 3


# ---------------------------------------------------------------- PE spin
def test_extract_a_final_gwtc1_and_final_spin_gwtc21():
    from menus.gravitic.ligo.qnm import extract_pe_parameters, gr_prediction, resolve_event_metadata

    pe1 = extract_pe_parameters(DETAIL_GWTC1)
    assert pe1["a_final"] == 0.69 and pe1["is_preferred"]
    pe2 = extract_pe_parameters({"events": {"GW150914-v4": DETAIL_GWTC21}})
    assert pe2["final_spin"] == 0.68

    top = {k: v for k, v in ALLEVENTS["GW150914-v4"].items()}
    top.update(final_spin=None, GPS=1126259462.4)
    assert gr_prediction(top)["missing"] == ["final_spin"]
    g2 = gr_prediction(resolve_event_metadata(top, DETAIL_GWTC21))
    assert g2["ok"] and 245 < g2["f220_hz"] < 258 and g2["sigma_spin"] == pytest.approx(0.05)
    g1 = gr_prediction(dict(DETAIL_GWTC1))  # meta with a "parameters" dict is resolved directly
    assert g1["ok"] and g1["a_final"] == 0.69 and g1["sigma_spin"] == pytest.approx(0.045)


def test_redshift_bounds_propagate():
    from menus.gravitic.ligo.qnm import gr_prediction

    base = {"final_mass_source": 61.5, "final_spin": 0.68, "redshift": 0.1}
    s0 = gr_prediction(base)["sigma_hz"]
    s1 = gr_prediction({**base, "redshift_upper": 0.03, "redshift_lower": -0.04})["sigma_hz"]
    assert s0 == 0.0 and s1 > 5.0


# ---------------------------------------------------------------- backfill + notes
def _cache_tree(tmp_path: Path, meta: dict, event: str = "GWTEST", detectors=("H1",)):
    events = tmp_path / "events"
    n = 4 * FS
    rng = np.random.default_rng(0)
    for det in detectors:
        strain = _ringdown(n, 2.0) + 1e-3 * rng.standard_normal(n)
        _write_txt_gz(events / event / f"{det[0]}-{det}_TEST-1000-4.txt.gz", strain)
    catalog = tmp_path / "event_catalog.json"
    catalog.write_text(json.dumps({event: meta}))
    return events, catalog


def test_backfill_fetches_detail_when_spin_missing(tmp_path, monkeypatch):
    from menus.gravitic.ligo import fetcher
    from superblock_residual import _load_ligo

    meta = {k: v for k, v in META.items() if k != "final_spin"}
    meta["jsonurl"] = "https://example.invalid/GWTEST/v3"
    events, catalog = _cache_tree(tmp_path, meta)
    detail = {"commonName": "GWTEST", "parameters": {"pe": {"is_preferred": True, "pipeline_type": "pe",
                                                            "final_spin": 0.69}}}
    seen = []
    monkeypatch.setattr(fetcher, "fetch_event_detail_url", lambda url: seen.append(url) or detail)
    D, B, S, X, notes = _load_ligo({"events_dir": str(events), "catalog_file": str(catalog)})
    assert seen == ["https://example.invalid/GWTEST/v3"]
    assert (events / "GWTEST" / "event_detail.json").is_file()
    assert D.size == 2 and 240 < B[0] < 260  # rows: f, γ (both Hz)
    # second run uses the cached detail, no fetch
    _load_ligo({"events_dir": str(events), "catalog_file": str(catalog)})
    assert len(seen) == 1


def test_backfill_refreshes_preliminary_catalog(tmp_path, monkeypatch):
    from menus.gravitic.common import gwosc_api
    from menus.gravitic.ligo import fetcher
    from superblock_residual import _load_ligo

    prelim = {"commonName": "GWTEST", "GPS": 1002.0, "version": 1, "final_mass_source": None, "redshift": None,
              "jsonurl": "https://example.invalid/Preliminary/GWTEST/v1"}
    events, catalog = _cache_tree(tmp_path, prelim)
    good = {**META, "final_spin": None, "jsonurl": "https://example.invalid/confident/GWTEST/v3"}
    monkeypatch.setattr(gwosc_api, "fetch_all_events", lambda **_: {"GWTEST": good})
    urls = []
    detail = {"parameters": {"pe": {"is_preferred": True, "pipeline_type": "pe", "final_spin": 0.69}}}
    monkeypatch.setattr(fetcher, "fetch_event_detail_url", lambda url: urls.append(url) or detail)
    D, B, *_ = _load_ligo({"events_dir": str(events), "catalog_file": str(catalog)})
    assert urls == ["https://example.invalid/confident/GWTEST/v3"]
    assert json.loads(catalog.read_text())["GWTEST"]["version"] == 3
    assert D.size == 2


def test_no_fetch_when_meta_complete_or_disabled(tmp_path, monkeypatch):
    from menus.gravitic.ligo import fetcher
    from superblock_residual import _load_ligo

    def nope(url):
        raise AssertionError("must not fetch")

    monkeypatch.setattr(fetcher, "fetch_event_detail_url", nope)
    events, catalog = _cache_tree(tmp_path, META)
    assert _load_ligo({"events_dir": str(events), "catalog_file": str(catalog)})[0].size == 2
    meta = {k: v for k, v in META.items() if k != "final_spin"}
    catalog.write_text(json.dumps({"GWTEST": meta}))
    with pytest.raises(FileNotFoundError):
        _load_ligo({"events_dir": str(events), "catalog_file": str(catalog), "auto_fetch_detail": "no"})


def test_notes_preserved_when_loader_raises(tmp_path):
    from superblock_residual import run_superblock_residual

    meta = {k: v for k, v in META.items() if k != "final_spin"}
    events, catalog = _cache_tree(tmp_path, meta)
    rep = run_superblock_residual("LIGO_GWOSC", "Full Ringdown Report",
                                  {"events_dir": str(events), "catalog_file": str(catalog),
                                   "auto_fetch_detail": "no"})
    assert "No GWOSC event" in rep["load_error"]
    assert any("GWTEST" in n and "final_spin" in n for n in rep["notes"])


# ---------------------------------------------------------------- multi-detector + filtering
def test_multi_detector_points_and_event_filter(tmp_path):
    from superblock_residual import _LIGO_EXTRA, _load_ligo

    events, catalog = _cache_tree(tmp_path, META, detectors=("H1", "L1"))
    base = {"events_dir": str(events), "catalog_file": str(catalog), "postmerger_scan": "no"}
    D, B, S, X, notes = _load_ligo({**base, "event": "ALL", "detector_mode": "per_detector"})
    assert D.size == 4  # (f, γ) × 2 detectors
    assert [p["label"] for p in _LIGO_EXTRA["points"]] == ["GWTEST (H1)", "GWTEST (L1)"]
    assert [r["marker"] for r in _LIGO_EXTRA["rows"]] == ["f_hz", "gamma_hz", "f_hz", "gamma_hz"]
    Dc, *_ = _load_ligo({**base, "event": "gwtest"})  # coherent is the default
    assert Dc.size == 2 and [p["label"] for p in _LIGO_EXTRA["points"]] == ["GWTEST (H1+L1)"]
    with pytest.raises(FileNotFoundError, match="cached events: GWTEST"):
        _load_ligo({**base, "event": "GW999999"})
    one = events / "GWTEST" / "L-L1_TEST-1000-4.txt.gz"
    D1, *_ = _load_ligo({**base, "strain_path": str(one)})
    assert D1.size == 2 and _LIGO_EXTRA["points"][0]["detector"] == "L1"


def test_low_snr_segment_skipped(tmp_path):
    from superblock_residual import _load_ligo

    events = tmp_path / "events"
    rng = np.random.default_rng(1)
    _write_txt_gz(events / "GWTEST" / "H-H1_TEST-1000-4.txt.gz", rng.standard_normal(4 * FS))
    catalog = tmp_path / "event_catalog.json"
    catalog.write_text(json.dumps({"GWTEST": META}))
    with pytest.raises(FileNotFoundError) as ei:
        _load_ligo({"events_dir": str(events), "catalog_file": str(catalog), "min_snr": 50})
    assert any("SNR" in n for n in ei.value.loader_notes)


def test_registry_target_options():
    from menus.gravitic.ligo.fetcher import DATASETS_DIR
    from sb_shared.dataset_registry import _ligo_target_options

    assert _ligo_target_options(["gwosc:GW150914"], {"event": "ALL"}) == {"event": "GW150914"}
    out = _ligo_target_options(["gwosc:events/GW150914/H-H1_x.hdf5"], {})
    assert out == {"event": "GW150914", "strain_path": str(DATASETS_DIR / "events/GW150914/H-H1_x.hdf5")}
    assert _ligo_target_options(["gwosc:event_catalog.json"], {}) == {}
    assert _ligo_target_options(["gwosc:GW150914"], {"event": "GW170814"}) == {}


# ---------------------------------------------------------------- whitened measurement
def test_whitened_recovery_on_colored_noise():
    from scipy import signal

    from menus.gravitic.ligo.qnm import measure_ringdown, measured_peak_hz

    rng = np.random.default_rng(7)
    n = 32 * FS
    white = rng.standard_normal(n)
    # strongly red noise: low-frequency dominated, like the seismic/thermal wall
    sos = signal.butter(2, 60.0, btype="low", fs=FS, output="sos")
    colored = 30.0 * signal.sosfilt(sos, white) + 0.3 * white
    t0 = 16.0
    data = colored + _ringdown(n, t0, f=250.0, tau=0.004, amp=8.0)
    res = measure_ringdown(data, FS, t0 - 0.01)
    assert res["whitened"]
    assert abs(res["f_hz"] - 250.0) < res["sigma_hz"]
    assert abs(res["t_peak_offset_s"] - 0.01) < 0.01
    assert res["snr_peak"] > 5
    f, s = measured_peak_hz(data, FS, t0 - 0.01)
    assert f == res["f_hz"] and s == pytest.approx(20.0, rel=0.01)


def test_short_segment_fallback():
    from menus.gravitic.ligo.qnm import measure_ringdown

    n = 2 * FS
    res = measure_ringdown(_ringdown(n, 1.0) + 1e-4 * np.random.default_rng(3).standard_normal(n), FS, 1.0)
    assert not res["whitened"] and abs(res["f_hz"] - 250) < 20


# ---------------------------------------------------------------- plot
def test_full_report_plot(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from menus.gravitic.ligo import extension

    events, catalog = _cache_tree(tmp_path, META, detectors=("H1", "L1"))
    opts = {"events_dir": str(events), "catalog_file": str(catalog), "output_name": "test_ligo_plot.png"}
    import sb_shared.project_paths as pp

    monkeypatch.setattr(pp, "SUPERBLOCK_ROOT", tmp_path)
    report_path = extension.run_action("Full Ringdown Report", {**opts, "show_graphics": "artifacts"})
    assert report_path and Path(report_path).is_file()
    png = tmp_path / "artifacts" / "ligo_gwosc" / "test_ligo_plot.png"
    assert png.is_file() and png.stat().st_size > 10_000
    png.unlink()
    extension.run_action("Full Ringdown Report", {**opts, "show_graphics": "no"})
    assert not png.exists()
    fields = {f["key"]: f for f in extension.entry_fields("Full Ringdown Report")}
    assert fields["event"]["default"] == "ALL" and not fields["event"]["required"]


# ---------------------------------------------------------------- real data (skipped without the cache)
def test_real_gw150914_if_cached():
    from menus.gravitic.ligo.fetcher import CATALOG_CACHE, EVENTS_DIR, load_event_detail

    ev = EVENTS_DIR / "GW150914"
    if not list(ev.glob("*.hdf5")) or load_event_detail("GW150914") is None or not CATALOG_CACHE.is_file():
        pytest.skip("real GW150914 strain/detail not cached (run Pull Strain from GWOSC)")
    pytest.importorskip("h5py")
    from superblock_residual import _LIGO_EXTRA, _load_ligo

    D, B, S, X, notes = _load_ligo({"event": "GW150914", "auto_fetch_detail": "no", "postmerger_scan": "no",
                                    "detector_mode": "per_detector"})
    pts = _LIGO_EXTRA["points"]
    assert {p["detector"] for p in pts} >= {"H1", "L1"}
    assert all(abs(p["f_obs_hz"] - p["f220_gr_hz"]) < 2 * p["sigma_hz"] and 240 < p["f220_gr_hz"] < 262 for p in pts)
    assert all(p["snr_peak"] > 5 and 0 < p["t_peak_offset_s"] < 0.05 for p in pts)
    Dc, *_ = _load_ligo({"event": "GW150914", "auto_fetch_detail": "no", "postmerger_scan": "no"})
    p = _LIGO_EXTRA["points"][0]
    assert p["label"] == "GW150914 (H1+L1)" and p["null_veto_passed"] and p["network_snr"] > 8
    assert abs(abs(p["align"]["L1"]["dt_ms"]) - 6.9) < 1.5  # known H1–L1 arrival difference
    assert abs(p["f_obs_hz"] - p["f220_gr_hz"]) < 2 * p["sigma_hz"]
