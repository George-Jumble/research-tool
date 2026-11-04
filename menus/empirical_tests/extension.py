"""
Modular bridge: Superblock Empirical Tests ↔ research_tool.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from menus.empirical_tests import tests as empirical
from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis

MODULE_TAG = "EMPIRICAL_TESTS"

MENU_ACTIONS = [
    "Run All Tests (Curated)",
    "Light Quarks Confrontation",
    "Neutron Lifetime Test",
    "Yang-Mills Glueball Test",
    "BBN integrator status",
    "Full Pipeline (Plot + LaTeX)",
]

_ACTION_CONFIG: Dict[str, Dict] = {
    "Run All Tests (Curated)": {
        "tests": ["all"],
        "sources": ["pdg_light_quarks", "neutron_lifetime", "glueball_lattice"],
        "plot": False,
        "latex": False,
    },
    "Light Quarks Confrontation": {
        "tests": ["light_quarks"],
        "sources": ["pdg_light_quarks", "lattice_qcd"],
        "plot": True,
        "latex": False,
    },
    "Neutron Lifetime Test": {
        "tests": ["neutron"],
        "sources": ["neutron_lifetime"],
        "plot": True,
        "latex": False,
    },
    "Yang-Mills Glueball Test": {
        "tests": ["yang_mills"],
        "sources": ["glueball_lattice"],
        "plot": True,
        "latex": False,
    },
    "BBN integrator status": {
        "tests": ["bbn_standalone"],
        "sources": ["bbn_abundances"],
        "plot": True,
        "latex": False,
    },
    "Full Pipeline (Plot + LaTeX)": {
        "tests": ["all"],
        "sources": ["pdg_light_quarks", "neutron_lifetime", "glueball_lattice"],
        "plot": True,
        "latex": True,
    },
}


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for Superblock Empirical Tests actions."""
    return [
        {
            "key": "output_name",
            "label": "JSON report name (optional)",
            "default": "superblock_test_report.json",
            "required": False,
            "hint": "Saved under artifacts/",
        },
        {
            "key": "force_download",
            "label": "Force live download",
            "default": "no",
            "required": False,
            "hint": "Attempt remote CSV fetch before curated fallback",
            "choices": ["no", "yes"],
        },
    ]


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the empirical tests entry form."""
    base = [
        "Uses curated PDG/lattice/neutron/glueball tables from menus.empirical_tests.tests.",
        "Neutron lifetime + BBN: stable PDG/literature fallback via fetch_empirical_data().",
        "Output JSON name: .json file saved under artifacts/ (letters, numbers, _, -).",
        "Plots/LaTeX: .png and .tex written to artifacts/ when pipeline requests them.",
    ]
    if action == "BBN integrator status":
        base.append("Scorer refused. No abundance comparison is drawn.")
    return base


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> None:
    options = options or {}
    config = _ACTION_CONFIG.get(selection)
    if config is None:
        print(f"[Superblock Residual Engine] Unknown empirical test action: {selection}")
        return

    output_name = (options.get("output_name") or "").strip()
    dataset = Path(output_name).stem if output_name else selection
    empirical.begin_artifact_run(dataset)
    print(f"\n[{empirical.ENGINE_BANNER}] Superblock Empirical Tests — {selection}")
    print(f"[{empirical.ENGINE_BANNER}] Ledger: {empirical.LEDGER_DATE}")
    print(f"[{empirical.ENGINE_BANNER}] m0 = {empirical.DELTA_TEP_MEV} MeV frozen")
    print(f"[{empirical.ENGINE_BANNER}] Output directory: {empirical.ARTIFACTS_DIR}")

    output_path = None

    selected_raw = (options.get("selected_sources") or "").strip()
    sources = config["sources"]
    if selected_raw:
        picked = [part.strip() for part in selected_raw.split(",") if part.strip()]
        sources = [key for key in picked if key in empirical.REPOS] or sources

    if selection == "BBN integrator status":
        from menus.prime_past.bbn_interference import run_bbn_comparison

        comparison = run_bbn_comparison(options, verbose=True)
        comparison["status"] = "FAIL_INTEGRATOR"
        comparison["ledger"] = "FAIL_INTEGRATOR"
        comparison.setdefault("reason", "network did not form D / did not finish n->He4")
        report_path = empirical.run_artifact("bbn_status", "json")
        report_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        print(f"[Superblock Residual Engine] BBN integrator status: {report_path}")
        return report_path

    report = empirical.run_pipeline(
        tests=config["tests"],
        sources=sources,
        download=str(options.get("force_download", "")).lower() in {"1", "true", "yes", "y"},
        plot=config["plot"],
        latex=config["latex"],
        use_particle=empirical.HAS_PARTICLE,
        output=output_path,
        verbose=True,
    )

    if show_plots and config["plot"]:
        _show_saved_plots(report)

    print(
        f"\n[TOPOLOGICAL ANCHOR] {empirical.DELTA_TEP_MEV} MeV mass-gap signal is independent of "
        "empirical confrontation parameters."
    )


def _show_saved_plots(report: Dict) -> None:
    plot_paths: List[str] = []
    for result in report.get("tests", {}).values():
        plot_path = result.get("plot")
        if plot_path and os.path.isfile(plot_path):
            plot_paths.append(plot_path)

    if not plot_paths:
        return

    for plot_path in plot_paths:
        image = plt.imread(plot_path)
        plt.figure(figsize=(8, 5))
        plt.imshow(image)
        plt.axis("off")
        plt.title(os.path.basename(plot_path))
        plt.tight_layout()
    plt.show()