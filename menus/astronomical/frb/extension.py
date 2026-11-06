"""
FRB dispersion residuals ↔ research_tool.py

"Pull Datasets from Open Archives" runs the FRB/void fetcher. Analysis actions are scored
by the Superblock residual engine (``superblock_residual._load_frb``).
"""

from __future__ import annotations

from menus.astronomical.frb.fetcher import DATASETS_DIR, pull_open_archives

MODULE_TAG = "FRB_DM"
SUBMENU_TITLE = "FRB Dispersion Residuals"

PULL_ACTION = "Pull Datasets from Open Archives"

MENU_ACTIONS = [
    PULL_ACTION,
    "Analyze DM Residuals by Path",
]


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def entry_instructions(action: str) -> list[str]:
    from sb_shared.project_paths import FINISHED_A_DIR

    if action == PULL_ACTION:
        return [
            "Fetches CHIME FRB catalog (CSV mirrors) into datasets/frb/.",
            "Falls back to CDS VizieR J/ApJS/257/59 (536-burst Catalog 1) when mirrors fail.",
            "Uses cfod catalog API when installed; else 400-burst curated template.",
            "Fetches SDSS DR7 void catalog (Douglass+ 2023, VizieR J/ApJS/265/7).",
            "Skips re-download when cached or marked processed (datasets/processed.txt).",
            f"Processed CSVs move to {FINISHED_A_DIR}/ after analysis runs.",
        ]
    return [
        "DM minus Macquart IGM baseline, only when host z is independent of DM.",
        "Paths are classified void / sheet / filament / node from void proximity.",
        f"Uses the cached catalogs in {DATASETS_DIR}/ (run the pull action first).",
        "Report: artifacts/superblock_residuals/FRB_DM_*.json",
    ]


def entry_fields(action: str) -> list[dict]:
    from sb_shared.project_paths import FINISHED_A_DIR

    if action != PULL_ACTION:
        return []
    return [
        {
            "key": "force_refresh",
            "label": "Force re-download",
            "default": "no",
            "required": False,
            "hint": "Override processed ledger and cached CSVs",
            "choices": ["no", "yes"],
        },
        {
            "key": "restore_archived",
            "label": "Restore archived",
            "default": "yes",
            "required": False,
            "hint": f"Try {FINISHED_A_DIR}/ before remote pull",
            "choices": ["no", "yes"],
        },
    ]


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> str | None:
    from superblock_mode import CONTINUE, begin_live_action

    _ = show_plots
    routed = begin_live_action(MODULE_TAG, selection, options)
    if routed is not CONTINUE:
        if isinstance(routed, dict):
            return routed.get("report_path")
        return routed
    options = options or {}
    if selection != PULL_ACTION:
        print(f"[Superblock Residual Engine] Unknown FRB action: {selection}")
        return None
    summary = pull_open_archives(
        force_refresh=_yes(options.get("force_refresh", "no")),
        restore_archived=_yes(options.get("restore_archived", "no")),
    )
    print(f"[Superblock Residual Engine] FRB source: {summary.frb_source}")
    print(f"[Superblock Residual Engine] Void source: {summary.void_source}")
    print(f"[Superblock Residual Engine] Datasets in {DATASETS_DIR}/")
    return None
