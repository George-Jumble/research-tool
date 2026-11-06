"""
LHCb flavor residuals ↔ research_tool.py

Download materializes the C9(q²) series under datasets/lhcb/. Score / Plot / Full Report are
scored by the Superblock residual engine (template C); plot actions draw ΔC9 vs q².
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from menus.particle.lhcb.flavor import (
    DEFAULT_DATA_PATH,
    DEFAULT_N_BINS,
    clip_n_bins,
)
from sb_shared.run_output import parse_show_graphics

MODULE_TAG = "LHCB_FLAVOR"
SUBMENU_TITLE = "LHCb Flavor Residuals"

DOWNLOAD_ACTION = "Download LHCb B-meson Dataset"

MENU_ACTIONS = [
    DOWNLOAD_ACTION,
    "Score Flavor Residuals",
    "Plot Flavor Residuals",
    "Full Flavor Residual Report",
]

_ACTION_CONFIG = {
    "Score Flavor Residuals": {"plot": False},
    "Plot Flavor Residuals": {"plot": True},
    "Full Flavor Residual Report": {"plot": True},
}


def _plot_residual_report(report: dict, options: dict, show_popup: bool) -> str | None:
    """Plot C9_exp - C9_SM vs q² from the residual engine's LHCb load (Template C)."""
    from menus.particle.lhcb.flavor import ARTIFACTS_DIR, ensure_output_dirs, load_lhcb_data

    data_path = (options.get("data_file") or str(DEFAULT_DATA_PATH)).strip()
    n_bins = clip_n_bins(options.get("n_q2_bins") or DEFAULT_N_BINS)
    try:
        data, path = load_lhcb_data(Path(data_path), n_bins=n_bins)
    except Exception as exc:
        print(f"[Superblock Residual Engine] LHCb plot skipped: {exc}")
        return None
    ensure_output_dirs()
    output_name = (options.get("output_name") or "lhcb_flavor_residuals.png").strip()
    if not output_name.endswith(".png"):
        output_name += ".png"
    out = ARTIFACTS_DIR / output_name
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        data["q2_bin"], data["C9_exp"] - data["C9_SM"], yerr=data["C9_exp_err"],
        fmt="o", ms=4, capsize=2, label="C9_exp − C9_SM",
    )
    ax.axhline(0.0, color="k", lw=0.8, label="SM baseline")
    ax.set_xlabel("q² [GeV²]")
    ax.set_ylabel("ΔC9")
    ax.set_title(
        f"LHCb flavor residuals (Template C)  verdict={report.get('verdict')}  "
        f"N={report.get('N')}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[Superblock Residual Engine] Plot saved: {out} (data {path.name})")
    if show_popup:
        plt.show()
    else:
        plt.close(fig)
    return str(out)


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
    """Curses entry-form field specs for LHCb flavor actions."""
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
            "key": "output_name",
            "label": "Plot filename (optional)",
            "default": "lhcb_flavor_residuals.png",
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
    """Short help bullets shown above the LHCb flavor entry form."""
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
        "Residual D − B = C9_exp − C9_SM per q² bin (SM baseline).",
        "Plot actions save PNG under artifacts/lhcb/; popup opens after curses exits.",
    ]


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> str | None:
    from superblock_mode import CONTINUE, begin_live_action

    routed = begin_live_action(MODULE_TAG, selection, options)
    if routed is not CONTINUE:
        config = _ACTION_CONFIG.get(selection)
        if isinstance(routed, dict) and config and config["plot"]:
            show_popup = (
                parse_show_graphics(options or {}, default="artifacts") if show_plots else False
            )
            _plot_residual_report(routed, options or {}, show_popup)
        if isinstance(routed, dict):
            return routed.get("report_path")
        return routed
    options = options or {}
    n_bins = clip_n_bins(options.get("n_q2_bins") or DEFAULT_N_BINS)

    if selection == DOWNLOAD_ACTION:
        from menus.particle.lhcb.fetcher import download_lhcb_bmeson_dataset

        download_lhcb_bmeson_dataset(
            n_bins=n_bins,
            force_refresh=_yes(options.get("force_refresh", "no")),
        )
        print("[Superblock Residual Engine] Next: Score Flavor Residuals or Full Flavor Residual Report.")
        return None

    print(f"[Superblock Residual Engine] Unknown LHCb action: {selection}")
    return None
