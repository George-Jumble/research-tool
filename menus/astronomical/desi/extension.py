"""
DESI DR2 BAO residuals ↔ research_tool.py

Every live DESI action is scored by the Superblock residual engine
(``superblock_residual._load_desi`` → DESI DR2 tables cached under datasets/desi/).
"""

from __future__ import annotations

from menus.astronomical.desi.fetcher import DEFAULT_COBAYA_ROOT
from menus.astronomical.desi.loader import TRACER

MODULE_TAG = "DESI_BAO"
SUBMENU_TITLE = "DESI DR2 BAO Residuals"

MENU_ACTIONS = [
    "Real DESI DR2 Scan (Cobaya)",
    "Healpy Dipole Fit",
    "Change-Point Detection",
    "Joint DESI+SN+Planck Fit",
    "DESI Catalog → LSS Grid",
]

_TRACER_CHOICES = [
    "ALL_GCcomb",
    "BGS_BRIGHT-21.35_GCcomb",
    "LRG_GCcomb_z0.4-0.6",
    "LRG_GCcomb_z0.6-0.8",
    "ELG_LOPnotqso_GCcomb_z1.1-1.6",
    "QSO_GCcomb",
    "Lya_GCcomb",
]


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def entry_fields(action: str) -> list[dict]:
    """Form fields for DESI residual actions."""
    _ = action
    return [
        {
            "key": "tracer",
            "label": "DESI DR2 tracer",
            "default": TRACER,
            "required": False,
            "hint": "Live DR2 tables from datasets/desi/bao_data/desi_bao_dr2",
            "choices": _TRACER_CHOICES,
        },
    ]


def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the DESI BAO entry form."""
    _ = action
    return [
        "Live DR2 only: clone Cobaya bao_data → datasets/desi/bao_data (desi_bao_dr2/).",
        f"Default Cobaya path: {DEFAULT_COBAYA_ROOT}",
        "Scored by the Superblock residual engine against the published ΛCDM-ratio data.",
        "Reports: artifacts/superblock_residuals/DESI_BAO_*.json",
    ]


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> str | None:
    """Run a DESI action through the Superblock residual engine."""
    from superblock_mode import CONTINUE, begin_live_action

    _ = show_plots
    routed = begin_live_action(MODULE_TAG, selection, options)
    if routed is CONTINUE:
        print(f"[Superblock Residual Engine] Unknown DESI action: {selection}")
        return None
    if isinstance(routed, dict):
        return routed.get("report_path")
    return routed
