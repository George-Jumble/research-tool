"""Curses line editor (multi-digit / numpad / backspace) and real LHCb 2020 data pipeline."""

from __future__ import annotations

import json
import os
import pty
import select
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------------
# PTY harness: run a curses snippet in a child with TERM=xterm-256color (keypad(True) via wrapper)
# --------------------------------------------------------------------------------------------
_CHILD = r'''
import curses, json, sys
sys.path.insert(0, {root!r})
from sb_research import curses_shell as cs
mode = {mode!r}
out = {{}}
def main(stdscr):
    curses.start_color(); curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    if mode == "form":
        fields = [{{"key": "n", "label": "Bins", "default": {default!r}, "required": False}}]
        orig = cs._read_field
        def spy(scr, row, col, width=56):
            out["col"] = col
            out["prompt_text"] = scr.instr(row, 4, col - 4).decode()
            return orig(scr, row, col, width)
        cs._read_field = spy
        out["value"] = cs.show_entry_form(stdscr, "T", "S", fields)
    elif mode == "n_select":
        out["value"] = cs.select_n_interactive(stdscr, default=25)
curses.wrapper(main)
open({outfile!r}, "w").write(json.dumps(out))
'''


def _run_pty(tmp_path, keys: bytes, *, mode="form", default="24") -> dict:
    outfile = str(tmp_path / "out.json")
    script = tmp_path / "child.py"
    script.write_text(_CHILD.format(root=str(ROOT), mode=mode, default=default, outfile=outfile))
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"] = "xterm-256color"
        os.environ["ESCDELAY"] = "25"
        os.execv(sys.executable, [sys.executable, str(script)])
    try:
        time.sleep(0.8)
        for chunk in keys.split(b"|"):
            os.write(fd, chunk)
            time.sleep(0.12)
        deadline = time.time() + 8
        while time.time() < deadline:
            r, _, _ = select.select([fd], [], [], 0.1)
            if r:
                try:
                    os.read(fd, 65536)
                except OSError:
                    break
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done:
                break
        else:
            os.kill(pid, 9)
    finally:
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        os.close(fd)
    return json.loads(Path(outfile).read_text())


@pytest.mark.parametrize("typed,expected", [(b"4|8|\r", "48"), (b"1|2|0|\r", "120")])
def test_top_row_multi_digit_not_enter(tmp_path, typed, expected):
    out = _run_pty(tmp_path, typed)
    assert out["value"] == {"n": expected}


def test_numpad_application_mode_digits(tmp_path):
    keys = b"|".join(b"\x1bO" + bytes([c]) for c in b"pqrstuvwxy") + b"|\x1bOM"
    out = _run_pty(tmp_path, keys)
    assert out["value"] == {"n": "0123456789"}


def test_backspace_and_cursor_column(tmp_path):
    out = _run_pty(tmp_path, b"9|9|\x7f|\x08|7|2|\r")
    assert out["value"] == {"n": "72"}
    assert out["col"] == 4 + len("[24] ")
    assert out["prompt_text"] == "[24] "


def test_enter_on_empty_keeps_default(tmp_path):
    assert _run_pty(tmp_path, b"\r", default="8")["value"] == {"n": "8"}


@pytest.mark.parametrize("down", [b"\x1bOB", b"\x1b[B"])
def test_select_n_custom_accepts_multi_digit(tmp_path, down):
    # digits in the preset list are ignored (never submit); move to "Custom..." (25 is index 4)
    # then type 120 with numpad 1 + top-row 20
    keys = b"7|" + b"|".join([down] * 4) + b"|\r|\x1bOq|2|0|\r"
    assert _run_pty(tmp_path, keys, mode="n_select")["value"] == 120


def test_translate_key_mapping():
    import curses

    from sb_research.curses_shell import translate_key

    assert translate_key(ord("7")) == "7"
    assert translate_key(10) == "ENTER" and translate_key(curses.KEY_ENTER) == "ENTER"
    assert translate_key(127) == "BACKSPACE"
    assert translate_key(curses.KEY_B2) == "5"
    assert translate_key(curses.KEY_A1) == "7"
    assert translate_key(curses.KEY_C3) == "3"


# --------------------------------------------------------------------------------------------
# LHCb 2020 real data
# --------------------------------------------------------------------------------------------
FLAVIO_SNIPPET = {
    f"LHCb B->K*mumu 2020 P {lo:g}-{hi:g}": {
        "values": [
            {"name": "<P5p>(B0->K*mumu)", "q2min": lo, "q2max": hi, "value": f"{v} ± 0.100 ± 0.030"},
            {"name": "<FL>(B0->K*mumu)", "q2min": lo, "q2max": hi, "value": "0.6 +0.04-0.02 ± 0.01"},
        ]
    }
    for (lo, hi), v in zip(
        [(0.1, 0.98), (1.1, 2.5), (2.5, 4.0), (4.0, 6.0), (6.0, 8.0), (11.0, 12.5), (15.0, 17.0), (17.0, 19.0)],
        [0.521, 0.365, -0.150, -0.439, -0.583, -0.622, -0.714, -0.590],
    )
}


def test_parse_flavio_value_formats():
    from menus.particle.lhcb.fetcher import parse_flavio_value

    assert parse_flavio_value("0.521 ± 0.095 ± 0.024") == (0.521, 0.095, 0.024)
    v, st, sy = parse_flavio_value(" 0.255 +0.032-0.030 ± 0.007")
    assert v == 0.255 and st == pytest.approx(0.031) and sy == 0.007
    assert parse_flavio_value("-0.439 ± 0.111 ± 0.036")[0] == -0.439


def test_parse_lhcb2020_bins_and_sm():
    from menus.particle.lhcb.fetcher import parse_lhcb2020

    rows = parse_lhcb2020(FLAVIO_SNIPPET, "P5p")
    assert len(rows) == 8
    r = rows[3]
    assert (r["q2_min_gev2"], r["q2_max_gev2"], r["q2_center_gev2"]) == (4.0, 6.0, 5.0)
    assert r["obs"] == -0.439 and r["obs_err"] == pytest.approx(np.hypot(0.1, 0.03))
    assert r["sm"] == pytest.approx(-0.7583)  # P5'_SM, never C9_SM
    assert all(abs(row["sm"]) <= 1 for row in rows)


def _write_synthetic(path: Path):
    from menus.particle.lhcb.flavor import save_default_dataset

    save_default_dataset(path, n_bins=24)
    (path.parent / "lhcb_bmeson_fetch.json").write_text(
        json.dumps({"source": "published_style_interpolated", "n_bins": 24})
    )


def test_fetcher_replaces_interpolated_cache_without_force(tmp_path, monkeypatch):
    from menus.particle.lhcb import fetcher

    monkeypatch.setattr(fetcher, "mark_processed", lambda *a, **k: None)
    dest = tmp_path / "lhcb_bmeson_data.npy"
    _write_synthetic(dest)
    meta = fetcher.download_lhcb_bmeson_dataset(dest=dest, force_refresh=False, fetch=lambda: FLAVIO_SNIPPET)
    assert meta["source"].startswith("flavio") and meta["real_data"] is True
    data = np.load(dest)
    assert len(data) == 8 and "obs" in data.dtype.names
    again = fetcher.download_lhcb_bmeson_dataset(dest=dest, fetch=lambda: pytest.fail("refetched"))
    assert again["source"] == "cached" and again["cached_source"] == meta["source"]
    # different observable invalidates the cache
    fl = fetcher.download_lhcb_bmeson_dataset(dest=dest, observable="FL", fetch=lambda: FLAVIO_SNIPPET)
    assert fl["observable"] == "FL" and fl["source"] != "cached"


def test_load_lhcb_provenance_and_template_c(tmp_path, monkeypatch):
    from menus.particle.lhcb import fetcher
    import superblock_residual as sr

    monkeypatch.setattr(fetcher, "mark_processed", lambda *a, **k: None)
    monkeypatch.setattr(sr, "_write_report", lambda rep: str(tmp_path / "rep.json"))
    dest = tmp_path / "lhcb_bmeson_data.npy"
    fetcher.download_lhcb_bmeson_dataset(dest=dest, fetch=lambda: FLAVIO_SNIPPET)
    rep = sr.run_superblock_residual("LHCB_FLAVOR", "Score Flavor Residuals", {"data_file": str(dest)})
    assert rep["N"] == 8 and rep["real_data"] is True
    assert rep["observable"] == "P5p" and rep["arxiv_id"] == "2003.04831" and rep["inspire_id"] == "1784890"
    assert "warning" not in rep
    sc = rep["score"]
    assert sc["M_mir_GeV"] == pytest.approx(0.3131)
    assert sc["A"] == pytest.approx(0.887, abs=1e-3)
    assert rep["chi2_SB"] is not None
    for tag in ("CERN_OPENDATA", "CASIMIR"):
        refused = sr.score_against_template(np.ones(3), template="C", q2=np.ones(3), dataset=tag)
        assert refused["chi2_SB"] is None and refused["kernel"] == "refused"


def test_load_lhcb_legacy_and_synthetic_warning(tmp_path):
    import superblock_residual as sr

    legacy = tmp_path / "legacy.npy"
    np.save(legacy, np.array([[4.5e5, -4.42, 0.38], [1.1e6, -4.55, 0.32], [2.5e6, -4.68, 0.29]]))
    D, B, sigma, q2, notes = sr._load_lhcb({"data_file": str(legacy)})
    assert np.allclose(B, -4.27) and np.allclose(q2, [0.45, 1.1, 2.5])
    assert sr._LHCB_EXTRA["real_data"] is False and "SYNTHETIC" in sr._LHCB_EXTRA["warning"]
    synth = tmp_path / "s" / "lhcb_bmeson_data.npy"
    synth.parent.mkdir()
    _write_synthetic(synth)
    sr._load_lhcb({"data_file": str(synth)})
    assert sr._LHCB_EXTRA["fetch_source"] == "published_style_interpolated"
    assert "warning" in sr._LHCB_EXTRA


def test_plot_axis_in_gev2(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from menus.particle.lhcb import extension, flavor

    monkeypatch.setattr(flavor, "ARTIFACTS_DIR", tmp_path)
    legacy = tmp_path / "legacy.npy"
    flavor.save_default_dataset(legacy, n_bins=12)
    captured = {}
    orig = plt.Figure.savefig

    def grab(self, *a, **k):
        captured["xlim"] = self.axes[0].get_xlim()
        captured["x"] = self.axes[0].lines[0].get_xdata() if self.axes[0].lines else None
        return orig(self, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", grab)
    out = extension._plot_residual_report({"verdict": "x", "N": 12}, {"data_file": str(legacy)}, False)
    assert out and Path(out).is_file()
    assert captured["xlim"][1] < 25

    bins = [{"q2_min_gev2": lo, "q2_max_gev2": hi, "q2_center_gev2": (lo + hi) / 2, "obs": 0.1,
             "obs_err": 0.1, "sm": 0.0, "sm_err": 0.01} for lo, hi in flavor.LHCB_2020_BINS]
    out = extension._plot_residual_report({"lhcb_bins": bins, "score": {"A": 0.887}}, {}, False)
    assert out and -2 < captured["xlim"][0] and captured["xlim"][1] < 25
