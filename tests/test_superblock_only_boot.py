"""Default path: loader actions fetch data, everything else is the Superblock residual engine."""

from __future__ import annotations

import superblock_mode
from superblock_mode import classify_action, load_boot_config
from superblock_residual import run_superblock_residual
from sb_research.menu_tree import build_menu_tree
from sb_research import registry

FORBIDDEN = (
    "1/7",
    "7-Fold",
    "7-fold",
    "Tav",
    "Tau",
    "EKK",
    "142857",
    "Berard",
    "RGC",
    "Cylinder",
)


def _menu_blob() -> str:
    domains, repos, submenus, action_map = build_menu_tree()
    return "\n".join(
        list(domains)
        + [item for items in repos.values() for item in items]
        + [item for items in submenus.values() for item in items]
        + list(action_map)
    )


def test_boot_config_forces_superblock_only():
    cfg = load_boot_config()
    assert cfg["superblock_only"] is True
    assert cfg["alternative_under_test"] == "Superblock/Mirron"


def test_no_retired_routing_plumbing():
    for name in (
        "RETIRED_ACTIONS",
        "RETIRED_MODULE_TAGS",
        "RetiredModuleError",
        "live_menu_actions",
        "print_retired",
    ):
        assert not hasattr(superblock_mode, name), name
    for tag in registry.MODULE_EXTENSIONS:
        for action in registry.MODULE_EXTENSIONS[tag].MENU_ACTIONS:
            assert classify_action(tag, action) in {"loader", "keep", "residual"}


def test_default_menu_has_no_retired_labels():
    blob = _menu_blob()
    for token in FORBIDDEN:
        assert token not in blob, token
    assert classify_action("DESI_BAO", "Real DESI DR2 Scan (Cobaya)") == "residual"
    assert classify_action("SPARC", "Pull Batch from Repository") == "loader"


def test_residual_report_is_inconclusive_without_data():
    report = run_superblock_residual("LISA_PRE_RUNS", "List Cached Mock Data")
    assert report["verdict"] == "inconclusive"
    assert report["chi2_SB"] is None


def test_v5_retired_geometry_absent_from_default_menu():
    """v5: no Kaluza-Klein (incl. the dead Conformal KK Crossover Fork III), no 5D, no internal S^3."""
    blob = _menu_blob()
    for token in ("Kaluza", "Fork III", "Conformal KK", "5D", "S^3", "S³"):
        assert token not in blob, token
