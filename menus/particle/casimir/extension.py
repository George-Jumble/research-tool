"""
Casimir residuals ↔ research_tool.py

Download actions run the Casimir dataset fetchers (datasets/casimir/). Analysis actions are
scored by the Superblock residual engine (template C).
"""

from __future__ import annotations

from menus.particle.casimir.fetcher import DATASETS_DIR, EXAMPLE_DATASETS

MODULE_TAG = "CASIMIR"
SUBMENU_TITLE = "Casimir Residuals"

ZENODO_ACTION = "Download Zenodo Casimir Drums"
GITHUB_ACTION = "Download GitHub Magnetic-Fluid Data"

MENU_ACTIONS = [
    "Bound-Energy Residual Search",
    "Scan Local CSV",
    ZENODO_ACTION,
    GITHUB_ACTION,
    "Scan GitHub Magnetic-Fluid CSV",
    "Scan Extracted Dataset (batch)",
]


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for Casimir actions."""
    if action == ZENODO_ACTION:
        size_mb = EXAMPLE_DATASETS["zenodo_superconducting_casimir"].get("size_warning_mb", "780")
        return [
            {
                "key": "confirm_large_download",
                "label": "Confirm large download",
                "default": "no",
                "required": False,
                "hint": f"Zenodo zip is ~{size_mb} MB",
                "choices": ["no", "yes"],
            }
        ]
    if action == GITHUB_ACTION:
        return [
            {
                "key": "force_download",
                "label": "Force re-download",
                "default": "no",
                "required": False,
                "hint": "Override cached files",
                "choices": ["no", "yes"],
            }
        ]
    return []


def entry_instructions(action: str) -> list[str]:
    if action == ZENODO_ACTION:
        return [f"Downloads the Zenodo Casimir drums zip into {DATASETS_DIR}/."]
    if action == GITHUB_ACTION:
        return [f"Caches GitHub magnetic-fluid CSVs into {DATASETS_DIR}/."]
    return [
        "Scored by the Superblock residual engine (template C).",
        "Report: artifacts/superblock_residuals/CASIMIR_*.json",
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

    if selection == GITHUB_ACTION:
        from menus.particle.casimir.fetcher import download_github_magnetic_fluid

        paths = download_github_magnetic_fluid(force=_yes(options.get("force_download", "no")))
        print(f"[Superblock Residual Engine] Cached {len(paths)} GitHub magnetic-fluid files.")
        print("  Use 'Scan GitHub Magnetic-Fluid CSV' to analyze.")
        return None

    if selection == ZENODO_ACTION:
        meta = EXAMPLE_DATASETS["zenodo_superconducting_casimir"]
        size_mb = meta.get("size_warning_mb", "780")
        if not _yes(options.get("confirm_large_download", "no")):
            print(f"[Superblock Residual Engine] Zenodo Casimir drums zip is ~{size_mb} MB.")
            print("  Set confirm_large_download=yes to proceed.")
            print(f"  Or download manually to {DATASETS_DIR}/")
            return None
        from menus.particle.casimir.fetcher import download_file

        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DATASETS_DIR / meta["zip_name"]
        download_file(meta["url"], zip_path)
        print(f"[Superblock Residual Engine] Saved: {zip_path}")
        print("  Use 'Scan Extracted Dataset (batch)' after extraction.")
        return None

    print(f"[Superblock Residual Engine] Unknown Casimir action: {selection}")
    return None
