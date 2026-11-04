"""Default path must not expose Tau/Tav/5D entry points."""

from __future__ import annotations

from pathlib import Path

from superblock_mode import (
    classify_action,
    load_boot_config,
    tau_templates_enabled,
)
from superblock_residual import refuse_tau_template, run_superblock_residual
from tav_research.menu_tree import build_menu_tree
from tav_research import registry

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = (
    "1/7",
    "7-Fold",
    "7-fold",
    "Tav",
    "Tau-SB",
    "Tau-Search",
    "EKK",
    "142857",
    "Berard",
    "RGC",
    "Cylinder",
    "Fractal Tau",
)


def test_boot_config_forces_superblock_only():
    cfg = load_boot_config()
    assert cfg["superblock_only"] is True
    assert cfg["tau_5d_templates"] is False
    assert tau_templates_enabled() is False


def test_registry_does_not_import_retired_modules():
    assert registry.tav_resonance_extension is None
    assert registry.rgc_mock_extension is None
    assert registry.berard_framework_extension is None
    assert registry.tsb_research_extension is None
    assert registry.integrator_extension is None
    assert "TAV_RESONANCE" not in registry.MODULE_EXTENSIONS
    assert "RGC_MOCK_CATALOG" not in registry.MODULE_EXTENSIONS
    assert "BERARD_FRAMEWORK" not in registry.MODULE_EXTENSIONS
    assert "TAV_BBN" not in registry.MODULE_EXTENSIONS
    assert "TSB_RESEARCH" not in registry.MODULE_EXTENSIONS
    text = (ROOT / "tav_research" / "registry.py").read_text(encoding="utf-8")
    assert "import menus.astronomical.tav_resonance" not in text
    assert "import menus.particle.rgc" not in text
    assert "import menus.astronomical.berard" not in text
    assert "import menus.prime_past.bbn_menu_extension" not in text
    assert "import menus.tsb_research" not in text
    boot = (ROOT / "research_tool.py").read_text(encoding="utf-8")
    assert "bootstrap_tav_resonance" not in boot


def test_default_menu_has_no_tau_labels():
    domains, repos, submenus, action_map = build_menu_tree()
    blob = "\n".join(
        list(domains)
        + [item for items in repos.values() for item in items]
        + [item for items in submenus.values() for item in items]
        + list(action_map)
    )
    for token in FORBIDDEN:
        assert token not in blob, token
    assert classify_action("TAU_SB_DESI", "Periodicity Scan (1/7 Tracks)") == "retired"
    assert classify_action("TAU_SB_DESI", "Real DESI DR2 Scan (Cobaya)") == "residual"
    assert classify_action("SPARC", "Pull Batch from Repository") == "loader"


def test_residual_refuses_tau_and_reports_negative_control():
    try:
        refuse_tau_template("T_tau")
    except RuntimeError as exc:
        assert "not applied" in str(exc).lower() or "retired" in str(exc).lower()
    else:
        raise AssertionError("T_tau should be refused")
    report = run_superblock_residual("LISA_PRE_RUNS", "List Cached Mock Data")
    assert report["negative_control"] == "Tau/5D templates were not applied"
    assert report["T_tau"] == "retired_not_evaluated"
    assert report["verdict"] == "inconclusive"
    assert report["chi2_SB"] is None
