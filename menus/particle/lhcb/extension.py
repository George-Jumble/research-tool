"""
LHCb flavor residuals ↔ research_tool.py

Download fetches the real LHCb 2020 angular bins under datasets/lhcb/. Score / Plot / Full Report
are scored by the Superblock residual engine (template C); plot actions draw Δobs vs q² [GeV²].
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from menus.particle.lhcb.flavor import (
    DEFAULT_DATA_PATH,
    DEFAULT_OBSERVABLE,
    LHCB_2020_BINS,
    SM_PREDICTIONS,
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


def _plot_arrays(report: dict, options: dict) -> dict | None:
    """Residual points for the plot: prefer the report's real bins, else load the cached file."""
    from menus.particle.lhcb.flavor import load_lhcb_data, residual_arrays

    bins = report.get("lhcb_bins")
    if bins:
        q2c = np.array([b["q2_center_gev2"] for b in bins], float)
        return {
            "q2": q2c,
            "lo": np.array([b["q2_min_gev2"] for b in bins], float),
            "hi": np.array([b["q2_max_gev2"] for b in bins], float),
            "r": np.array([b["obs"] - b["sm"] for b in bins], float),
            "sig": np.sqrt(np.array([b["obs_err"] ** 2 + b["sm_err"] ** 2 for b in bins], float)),
            "label": str((report.get("lhcb_provenance") or {}).get("observable_label") or "P5′"),
        }
    data_path = (options.get("data_file") or str(DEFAULT_DATA_PATH)).strip()
    try:
        data, _ = load_lhcb_data(Path(data_path))
        arr = residual_arrays(data)
    except Exception as exc:
        print(f"[Superblock Residual Engine] LHCb plot skipped: {exc}")
        return None
    return {
        # residual_arrays always returns q² in GeV² (q2_bin on disk is MeV²).
        "q2": arr["q2_gev2"],
        "lo": arr["q2_min_gev2"],
        "hi": arr["q2_max_gev2"],
        "r": arr["D"] - arr["B"],
        "sig": arr["sigma"],
        "label": "C9" if arr["schema"] == "legacy_c9" else "P5′",
    }


def _plot_residual_report(report: dict, options: dict, show_popup: bool) -> str | None:
    """Plot Δobs(q²) = obs − SM per q² bin [GeV²] with the Template C Mirron curve S_C(q²)."""
    from kernels.mirron_c import M_MIR_GEV, s_c
    from menus.particle.lhcb.flavor import ARTIFACTS_DIR, ensure_output_dirs

    pts = _plot_arrays(report, options)
    if pts is None:
        return None
    ensure_output_dirs()
    output_name = (options.get("output_name") or "lhcb_flavor_residuals.png").strip()
    if not output_name.endswith(".png"):
        output_name += ".png"
    out = ARTIFACTS_DIR / output_name
    label = pts["label"]
    q2 = pts["q2"]
    xerr = None
    if pts["lo"] is not None and pts["hi"] is not None:
        xerr = np.vstack([q2 - pts["lo"], pts["hi"] - q2])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(q2, pts["r"], xerr=xerr, yerr=pts["sig"], fmt="o", ms=4, capsize=2,
                label=f"Δ{label}(q²) = {label}_LHCb − {label}_SM")
    ax.axhline(0.0, color="k", lw=0.8, label="SM baseline")
    amp = report.get("A") if report.get("A") is not None else (report.get("score") or {}).get("A")
    if amp is not None:
        grid = np.linspace(0.05, max(19.5, float(np.max(q2)) + 0.5), 400)
        ax.plot(grid, s_c(grid, float(amp)), "r-",
                label=f"Mirron S_C(q²) = A·M²/(M²+q²), A={float(amp):.3f}, M={M_MIR_GEV * 1000:.1f} MeV")
    for lo, hi in ((8.0, 11.0), (12.5, 15.0)):
        ax.axvspan(lo, hi, color="0.9", lw=0)
    ax.set_xlabel("q² [GeV²]")
    ax.set_ylabel(f"Δ{label}")
    title = f"LHCb flavor residuals (Template C)  verdict={report.get('verdict')}  N={report.get('N')}"
    if report.get("fetch_source"):
        title += f"\n{report.get('fetch_source')}"
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[Superblock Residual Engine] Plot saved: {out}")
    if show_popup:
        plt.show()
    else:
        plt.close(fig)
    return str(out)


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def _observable_field() -> dict:
    return {
        "key": "observable",
        "label": "Observable (LHCb 2020, 8 narrow q² bins)",
        "default": DEFAULT_OBSERVABLE,
        "required": False,
        "hint": "P5p = P5′ (flagship anomaly) | FL | AFB. Real data has exactly 8 bins; no interpolation.",
        "choices": list(SM_PREDICTIONS),
    }


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for LHCb flavor actions."""
    if action == DOWNLOAD_ACTION:
        return [
            _observable_field(),
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
            "First step: fetch the real LHCb 2020 B0→K*μμ angular data into datasets/lhcb/.",
            "Source: LHCb PRL 125, 011802 (arXiv:2003.04831) via flavio measurements.yml (HTTPS).",
            f"{len(LHCB_2020_BINS)} narrow q² bins as published; synthetic caches are always replaced.",
            "Run an analysis action after the download finishes.",
        ]
    return [
        "Run 'Download LHCb B-meson Dataset' first if datasets/lhcb/ is empty.",
        "Loads LHCb 2020 bins: obs ± sqrt(stat²+syst²) and the SM value of the same observable.",
        f"Default path: {DEFAULT_DATA_PATH}",
        "Residual D − B = obs_LHCb − obs_SM per q² bin (e.g. ΔP5′), scored vs Mirron S_C(q²).",
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

    if selection == DOWNLOAD_ACTION:
        from menus.particle.lhcb.fetcher import download_lhcb_bmeson_dataset

        download_lhcb_bmeson_dataset(
            observable=(options.get("observable") or DEFAULT_OBSERVABLE).strip(),
            force_refresh=_yes(options.get("force_refresh", "no")),
        )
        print("[Superblock Residual Engine] Next: Score Flavor Residuals or Full Flavor Residual Report.")
        return None

    print(f"[Superblock Residual Engine] Unknown LHCb action: {selection}")
    return None
