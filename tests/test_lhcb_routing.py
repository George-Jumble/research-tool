"""LHCb menu actions: routing, config and residual-engine loading (v5 follow-up)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

from superblock_mode import classify_action
from menus.particle.lhcb import extension as lhcb


def test_lhcb_menu_actions_route_and_have_config():
    for action in lhcb.MENU_ACTIONS:
        route = classify_action(lhcb.MODULE_TAG, action)
        if action == lhcb.DOWNLOAD_ACTION:
            assert route == "loader"
            continue
        assert route == "residual", action
        assert action in lhcb._ACTION_CONFIG, action


def test_lhcb_plot_flags():
    assert lhcb._ACTION_CONFIG["Score Flavor Residuals"]["plot"] is False
    assert lhcb._ACTION_CONFIG["Plot Flavor Residuals"]["plot"] is True
    assert lhcb._ACTION_CONFIG["Full Flavor Residual Report"]["plot"] is True


def test_lhcb_old_action_names_removed():
    for old in ("Run Tav-Echo Correlation", "Plot Residuals vs Kernel", "Full Analysis (Stats + Plot)"):
        assert old not in lhcb.MENU_ACTIONS
        assert old not in lhcb._ACTION_CONFIG


def test_residual_engine_loads_lhcb(tmp_path):
    from menus.particle.lhcb.flavor import save_default_dataset
    from superblock_residual import _load_lhcb

    path = tmp_path / "lhcb.npy"
    save_default_dataset(path, n_bins=12)
    D, B, sigma, q2, notes = _load_lhcb({"data_file": str(path)})
    assert D is not None and B is not None and sigma is not None
    assert D.size == B.size == sigma.size == q2.size >= 6


def test_full_report_writes_plot(tmp_path, monkeypatch):
    from menus.particle.lhcb import flavor
    from menus.particle.lhcb.flavor import save_default_dataset

    monkeypatch.setattr(flavor, "ARTIFACTS_DIR", tmp_path)
    path = tmp_path / "lhcb.npy"
    save_default_dataset(path, n_bins=12)
    report = {"verdict": "x", "N": 12}
    out = lhcb._plot_residual_report(report, {"data_file": str(path)}, show_popup=False)
    assert out and os.path.isfile(out)
