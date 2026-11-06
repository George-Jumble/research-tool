"""
Planck CMB residuals ↔ research_tool.py

"Compute Power Spectrum" is scored by the Superblock residual engine.
"""

from __future__ import annotations

MODULE_TAG = "PLANCK_CMB"
SUBMENU_TITLE = "Planck CMB Residuals"

MENU_ACTIONS = [
    "Compute Power Spectrum",
]


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def entry_fields(action: str) -> list[dict]:
    _ = action
    return []


def entry_instructions(action: str) -> list[str]:
    _ = action
    return [
        "Scored by the Superblock residual engine (template E).",
        "Report: artifacts/superblock_residuals/PLANCK_CMB_*.json",
    ]


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> str | None:
    from superblock_mode import CONTINUE, begin_live_action

    _ = show_plots
    routed = begin_live_action(MODULE_TAG, selection, options)
    if routed is CONTINUE:
        print(f"[Superblock Residual Engine] Unknown Planck CMB action: {selection}")
        return None
    if isinstance(routed, dict):
        return routed.get("report_path")
    return routed
