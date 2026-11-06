"""AlterBBN bootstrap (pinned fetch, safe unpack, portable build) and the Prime Past menu wiring."""

from __future__ import annotations

import io
import shutil
import tarfile
from pathlib import Path

import pytest

from bbn_isolated import alterbbn

ROOT = Path(__file__).resolve().parents[1]


def _src_tree() -> Path | None:
    for name in (alterbbn.BINARY_NAME, alterbbn.FULL_BINARY_NAME):
        exe = alterbbn.find_binary(name)
        if exe and (exe.parent / "src" / "include.h").is_file():
            return exe.parent
    if (alterbbn.LOCAL_TREE / "src" / "include.h").is_file():
        return alterbbn.LOCAL_TREE
    return None


# ----------------------------------------------------------------------------- bootstrap
def test_pinned_upstream_source():
    assert alterbbn.UPSTREAM_URL == "https://alterbbn.hepforge.org/downloads?f=alterbbn_v2.2.tgz"
    assert alterbbn.UPSTREAM_SHA256 == "46b94253cde6afce15ddb9367fa454bac91b25c9a1d3412cc19e512ec3515c17"
    assert alterbbn.LOCAL_TREE == ROOT / "datasets" / "prime_past" / "alterbbn_v2.2"
    assert alterbbn.TARBALL_PATH.parent == ROOT / "datasets" / "prime_past"
    for drv in alterbbn.DRIVER_SOURCES.values():
        assert drv.is_file()
    assert "-fopenmp" not in alterbbn._CFLAGS


def test_no_auto_build_env_disables_bootstrap(monkeypatch):
    monkeypatch.setenv("SB_ALTERBBN_NO_AUTO_BUILD", "1")
    monkeypatch.setattr(alterbbn, "find_binary", lambda name=alterbbn.BINARY_NAME: None)
    assert alterbbn.auto_build_enabled() is False
    with pytest.raises(alterbbn.AlterBBNUnavailable, match="auto-build is disabled"):
        alterbbn.ensure_alterbbn(verbose=False)


def test_fetch_rejects_sha256_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(alterbbn, "PRIME_PAST_DATA", tmp_path)
    monkeypatch.setattr(alterbbn, "TARBALL_PATH", tmp_path / "alterbbn_v2.2.tgz")
    monkeypatch.delenv("SB_ALTERBBN_TARBALL", raising=False)

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(alterbbn.urllib.request, "urlopen", lambda req, timeout=0: _Resp(b"not the tarball"))
    with pytest.raises(alterbbn.AlterBBNUnavailable, match="sha256 mismatch"):
        alterbbn.fetch_source(verbose=False)
    assert list(tmp_path.iterdir()) == []


def test_unpack_refuses_members_outside_the_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(alterbbn, "PRIME_PAST_DATA", tmp_path / "pp")
    bad = tmp_path / "bad.tgz"
    with tarfile.open(bad, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(alterbbn.AlterBBNUnavailable, match="unexpected tarball member"):
        alterbbn.unpack_source(bad, verbose=False)
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.skipif(alterbbn.find_compiler() is None or _src_tree() is None,
                    reason="needs a C compiler and an AlterBBN src/ tree")
def test_portable_build_without_openmp_matches(tmp_path):
    """Compile both drivers (no -fopenmp) into a scratch tree; identical output to the resolved binary."""
    import subprocess

    tree = tmp_path / "alterbbn_v2.2"
    tree.mkdir()
    (tree / "src").symlink_to(_src_tree() / "src", target_is_directory=True)
    built = alterbbn.build(tree, verbose=False)
    assert set(built) == {"sb_bbn.x", "sb_bbn_full.x"}
    argv = ["6.114e-10", "-0.002", "877.75", "none", "1"]
    ref = alterbbn.find_binary(alterbbn.BINARY_NAME)
    if ref is None:
        pytest.skip("no reference sb_bbn.x")
    out_new = subprocess.run([str(built["sb_bbn.x"]), *argv], capture_output=True, text=True).stdout
    out_ref = subprocess.run([str(ref), *argv], capture_output=True, text=True).stdout
    assert out_new.split() == out_ref.split()
    full = alterbbn.run(full=True, binary=built["sb_bbn_full.x"], use_cache=False)
    assert abs(alterbbn.baryon_closure(full)) < 1e-8


def test_cli_status_runs():
    from bbn_isolated.alterbbn.__main__ import main

    assert main(["status"]) == 0


def test_run_v5_tests_script_bootstraps_alterbbn():
    text = (ROOT / "scripts" / "run_v5_tests.sh").read_text()
    assert "-m bbn_isolated.alterbbn ensure" in text and "--no-alterbbn" in text


# ----------------------------------------------------------------------------- Prime Past menu
def test_prime_past_menu_actions_and_routing():
    import menus.prime_past.extension as pp
    from superblock_mode import classify_action

    assert pp.MENU_ACTIONS == [pp.FETCH_ACTION, pp.GATE_ACTION, pp.SCORE_ACTION, pp.PLOT_ACTION, pp.REPORT_ACTION]
    assert classify_action("PRIME_PAST_HARMONIC", pp.FETCH_ACTION) == "loader"
    assert classify_action("PRIME_PAST_HARMONIC", pp.GATE_ACTION) == "loader"
    for a in (pp.SCORE_ACTION, pp.PLOT_ACTION, pp.REPORT_ACTION):
        assert classify_action("PRIME_PAST_HARMONIC", a) == "residual"
        assert a in pp._ACTION_CONFIG
    assert pp._ACTION_CONFIG[pp.PLOT_ACTION]["plot"] and not pp._ACTION_CONFIG[pp.SCORE_ACTION]["plot"]
    assert pp.LEGACY_ACTIONS["Verification Suite"] == pp.SCORE_ACTION
    for a in pp.MENU_ACTIONS:
        assert isinstance(pp.entry_instructions(a), list) and isinstance(pp.entry_fields(a), list)


def test_prime_past_in_menu_tree_and_registry():
    from sb_research.menu_tree import build_menu_tree
    from sb_shared.dataset_registry import get_provider

    _, repos, _, action_map = build_menu_tree()
    import menus.prime_past.extension as pp

    for a in pp.MENU_ACTIONS:
        assert action_map[a] == "PRIME_PAST_HARMONIC"
    provider = get_provider("PRIME_PAST_HARMONIC")
    assert provider is not None
    ids = [t.id for t in provider.list_fetchable()]
    assert ids == ["prime_past:alterbbn_v2.2", "prime_past:alterbbn_planck_baseline"]
    assert provider.run_actions == pp.MENU_ACTIONS


def test_plot_handler_writes_png_and_skips_empty(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import menus.prime_past.extension as pp
    from sb_shared import project_paths

    monkeypatch.setattr(project_paths, "SUPERBLOCK_ROOT", tmp_path)
    rep = {"N": 2, "chi2_Lambda": 11.75, "table": [
        {"observable": "D/H", "D": 2.527e-5, "B": 2.425e-5, "sigma": 3e-7, "pull": 3.4},
        {"observable": "Y_p", "D": 0.2453, "B": 0.2468, "sigma": 0.0034, "pull": -0.44}]}
    out = pp._plot_residual_report(rep, {}, show_popup=False)
    assert out and Path(out).is_file() and Path(out).parent == tmp_path / "artifacts" / "prime_past"
    assert pp._plot_residual_report({"N": 0, "load_error": "x"}, {}, show_popup=False) is None


def test_gate_check_action_writes_reports(tmp_path, monkeypatch):
    import menus.prime_past.extension as pp
    from bbn_isolated import runner

    monkeypatch.setattr(runner, "HERE", tmp_path)
    monkeypatch.setattr(alterbbn, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SB_ALTERBBN_NO_AUTO_BUILD", "1")
    out = pp.run_action(pp.GATE_ACTION, show_plots=False, options={})
    assert out == str(tmp_path / "report.json")
    assert (tmp_path / "gates.json").is_file()


def test_fetch_action_without_alterbbn_reports_cleanly(monkeypatch, capsys):
    import menus.prime_past.extension as pp

    monkeypatch.setenv("SB_ALTERBBN_NO_AUTO_BUILD", "1")
    monkeypatch.setattr(alterbbn, "find_binary", lambda name=alterbbn.BINARY_NAME: None)
    assert pp.run_action(pp.FETCH_ACTION, show_plots=False, options={}) is None
    assert "AlterBBN not available" in capsys.readouterr().out
