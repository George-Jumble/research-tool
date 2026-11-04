"""
LHCb Tav-Echo module ↔ research_tool.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from menus.particle.lhcb.echo import (
    DEFAULT_DATA_PATH,
    DEFAULT_N_BINS,
    clip_n_bins,
    run_pipeline,
)
from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis
from tav_shared.run_output import parse_show_graphics

MODULE_TAG = "LHCB_TAV_ECHO"
SUBMENU_TITLE = "LHCb Flavor Residuals"

DOWNLOAD_ACTION = "Download LHCb B-meson Dataset"

MENU_ACTIONS = [
    DOWNLOAD_ACTION,
    "Score Flavor Residuals",
    "Plot Flavor Residuals",
    "Full Flavor Residual Report",
]

_ACTION_CONFIG = {
    "Run Tav-Echo Correlation": {"plot": False},
    "Plot Residuals vs Kernel": {"plot": True},
    "Full Analysis (Stats + Plot)": {"plot": True},
}


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def _n_bins_field() -> dict:
    return {
        "key": "n_q2_bins",
        "label": "Number of q² bins (sample size)",
        "default": str(DEFAULT_N_BINS),
        "required": False,
        "hint": f"Default {DEFAULT_N_BINS} (was hard-wired to 6). Range 6–200.",
    }


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for LHCb Tav-Echo actions."""
    if action == DOWNLOAD_ACTION:
        return [
            _n_bins_field(),
            {
                "key": "force_refresh",
                "label": "Force refresh",
                "default": "no",
                "required": False,
                "hint": "yes = rebuild even if datasets/lhcb/ already has a series",
                "choices": ["no", "yes"],
            },
        ]
    return [
        {
            "key": "data_file",
            "label": "LHCb data file (.npy)",
            "default": str(DEFAULT_DATA_PATH),
            "required": False,
            "hint": "Blank uses datasets/lhcb/lhcb_bmeson_data.npy from Download",
        },
        _n_bins_field(),
        {
            "key": "mass_gap",
            "label": "Mass-gap anchor m₀ (MeV)",
            "default": "313.1",
            "required": False,
            "hint": "Resonance kernel uses 1/(m₀² + q²)",
        },
        {
            "key": "output_name",
            "label": "Plot filename (optional)",
            "default": "lhcb_tav_echo_correlation.png",
            "required": False,
            "hint": "Saved under artifacts/lhcb/ for plot actions",
        },
        {
            "key": "show_graphics",
            "label": "Graphics mode",
            "default": "artifacts",
            "required": False,
            "hint": "popup = Tk window | artifacts = PNG only",
            "choices": ["artifacts", "popup"],
        },
    ]


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the LHCb Tav-Echo entry form."""
    if action == DOWNLOAD_ACTION:
        return [
            "First step: materialize the B-meson C9(q²) series under datasets/lhcb/.",
            "Tries HEPData (LHCb PRL 125, 011802) then a published-style interpolation.",
            f"Sample size is the number of q² bins (default {DEFAULT_N_BINS}; was 6).",
            "Run an analysis action after the download finishes.",
        ]
    return [
        "Run 'Download LHCb B-meson Dataset' first if datasets/lhcb/ is empty.",
        "Loads structured NumPy file: q2_bin, C9_exp, C9_exp_err, C9_SM.",
        f"Default path: {DEFAULT_DATA_PATH}",
        f"q² bin count defaults to {DEFAULT_N_BINS} (overrides a stale 6-bin file).",
        "Correlates (C9_exp - C9_SM) with 1/(m₀² + q²); m₀ defaults to 313.1 MeV.",
        "Plot actions save PNG under artifacts/lhcb/; popup opens after curses exits.",
    ]


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> None:
    from superblock_mode import CONTINUE, begin_live_action

    routed = begin_live_action(MODULE_TAG, selection, options)
    if routed is not CONTINUE:
        return None
    options = options or {}
    n_bins = clip_n_bins(options.get("n_q2_bins") or DEFAULT_N_BINS)

    if selection == DOWNLOAD_ACTION:
        from menus.particle.lhcb.fetcher import download_lhcb_bmeson_dataset

        download_lhcb_bmeson_dataset(
            n_bins=n_bins,
            force_refresh=_yes(options.get("force_refresh", "no")),
        )
        print("[Superblock Residual Engine] Next: Run Tav-Echo Correlation or Full Analysis.")
        return

    config = _ACTION_CONFIG.get(selection)
    if config is None:
        print(f"[Superblock Residual Engine] Unknown LHCb Tav-Echo action: {selection}")
        return

    data_path = (options.get("data_file") or str(DEFAULT_DATA_PATH)).strip()
    try:
        mass_gap = float(options.get("mass_gap") or "313.1")
    except ValueError:
        mass_gap = 313.1

    output_name = (options.get("output_name") or "lhcb_tav_echo_correlation.png").strip()
    if not output_name.endswith(".png"):
        output_name += ".png"

    show_popup = parse_show_graphics(options, default="artifacts") if show_plots else False

    run_pipeline(
        data_path=Path(data_path),
        mass_gap_mev=mass_gap,
        plot=config["plot"],
        show_plot=show_popup,
        output_name=output_name,
        n_bins=n_bins,
    )

    if show_popup and config["plot"]:
        plt.show()
