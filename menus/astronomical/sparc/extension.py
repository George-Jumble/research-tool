"""
Unified SPARC module ↔ research_tool.py

All analysis actions batch-process ./data/sparc/*.csv via sparc_fetcher.
Repository pulls populate that directory; graphics display after curses exits.
"""

from __future__ import annotations



from menus.astronomical.sparc.fetcher import (
    DEFAULT_DATA_DIR,
    DEFAULT_DONE_FILE,
    ensure_galaxy_csv,
    pull_sparc_batch,
)

_LAST_SHEAR_RESULTS: dict[str, dict[str, float]] = {}
_ACTIVE_PIPELINE_WEIGHTS: dict[str, float] = {}

SUBMENU_TITLE = "SPARC"
MODULE_TAG = "SPARC"
LOCAL_DATA_GLOB = "./datasets/sparc/*.csv"

MENU_ACTIONS = [
    "Pull Batch from Repository",
    "Pull Single Galaxy CSV",
    "Empirical Plot + MCMC Sweep",
    "Rotation Curve + Shadow Fit",
    "Mass Ratio Report (5.2 check)",
    "Run Domain Shear Diagnostic",
    "Apply Harmonic Damping Calibration",
    "Initiate Holdout Matrix Test",
    "Full SPARC Analysis",
    "Emergent DM Prediction",
    "Fit Emergent DM Parameters",
    "SPARC Test Suite (Systematic)",
    "Superblock SPARC Full Demo",
]

_ACTION_KEYS = {
    "Pull Batch from Repository": "pull_batch",
    "Pull Single Galaxy CSV": "pull_single",
    "Empirical Plot + MCMC Sweep": "empirical_mcmc",
    "Rotation Curve + Shadow Fit": "rotation_curve",
    "Mass Ratio Report (5.2 check)": "mass_ratio",
    "Run Domain Shear Diagnostic": "domain_shear_diagnostic",
    "Apply Harmonic Damping Calibration": "harmonic_damping_calibration",
    "Initiate Holdout Matrix Test": "holdout_matrix_test",
    "Full SPARC Analysis": "full_analysis",
    "Emergent DM Prediction": "predict",
    "Fit Emergent DM Parameters": "fit",
    "SPARC Test Suite (Systematic)": "suite",
    "Superblock SPARC Full Demo": "full_demo",
}

_BATCH_ACTIONS = {
    "empirical_mcmc",
    "rotation_curve",
    "mass_ratio",
    "full_analysis",
    "predict",
    "fit",
    "full_demo",
}

SPARC_RUN_FIELDS = [
    {
        "key": "batch_limit",
        "label": "Batch limit (0 = all pending)",
        "default": "10",
        "required": False,
        "hint": "Resumes via datasets/sparc/batch_done.txt; 0 = all pending CSVs",
    },
    {
        "key": "force_rescan",
        "label": "Force re-scan (ignore batch_done.txt)",
        "default": "no",
        "required": False,
        "hint": "Re-analyze from the first CSVs instead of resuming pending",
        "choices": ["no", "yes"],
    },
    {
        "key": "show_graphics",
        "label": "Graphics mode",
        "default": "artifacts",
        "required": False,
        "hint": "popup = Tk windows | artifacts = PNG only",
        "choices": ["artifacts", "popup"],
    },
]


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _parse_int(value: str, default: int) -> int:
    try:
        return int((value or "").strip())
    except ValueError:
        return default


def _yes(value: str) -> bool:
    return (value or "").strip().lower() in {"yes", "y", "true", "1"}


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for SPARC actions."""
    from sb_shared.project_paths import FINISHED_A_DIR

    if action == "Pull Batch from Repository":
        return [
            {
                "key": "pull_limit",
                "label": "Galaxies to pull",
                "default": "100",
                "required": False,
                "hint": "Max new targets (skips cached CSVs)",
            },
            {
                "key": "refresh_catalog",
                "label": "Refresh catalog",
                "default": "no",
                "required": False,
                "hint": "Re-download MassModels .mrt from SPARC",
                "choices": ["no", "yes"],
            },
            {
                "key": "force_repull",
                "label": "Force re-pull",
                "default": "no",
                "required": False,
                "hint": "Overwrite existing galaxy CSVs from catalog",
                "choices": ["no", "yes"],
            },
            {
                "key": "restore_archived",
                "label": "Restore archived",
                "default": "no",
                "required": False,
                "hint": f"Copy missing CSVs from {FINISHED_A_DIR}/",
                "choices": ["no", "yes"],
            },
        ]
    if action == "Pull Single Galaxy CSV":
        return [
            {
                "key": "query",
                "label": "Galaxy name",
                "default": "",
                "required": True,
                "hint": "e.g. NGC3198",
            },
        ]
    if action == "Run Domain Shear Diagnostic":
        return [
            *SPARC_RUN_FIELDS,
            {
                "key": "diagnostic_source",
                "label": "Diagnostic source",
                "default": "catalog",
                "required": False,
                "hint": "catalog = DataManager API | local_batch = pending CSV slice",
                "choices": ["catalog", "local_batch"],
            },
            {
                "key": "cross_match_cosmic_web",
                "label": "Cross-match cosmic web",
                "default": "yes",
                "required": False,
                "hint": "Adds environment tags when filament distance exists",
                "choices": ["no", "yes"],
            },
            {
                "key": "lambda_web",
                "label": "lambda_web",
                "default": "7.0",
                "required": False,
                "hint": "Scale (Mpc) for FILAMENT/FIELD/VOID tagging",
            },
            {
                "key": "solver_r_baseline",
                "label": "Solver baseline ratio",
                "default": "5.2",
                "required": False,
                "hint": "SuperblockSolver R_baseline",
            },
            {
                "key": "solver_f_align",
                "label": "Solver f_align",
                "default": "1.0",
                "required": False,
                "hint": "Alignment multiplier for R_eff",
            },
            {
                "key": "solver_epsilon_tep",
                "label": "Solver epsilon_tep",
                "default": "0.15",
                "required": False,
                "hint": "Phase-slip penalty coefficient",
            },
            {
                "key": "solver_omega",
                "label": "Solver omega",
                "default": "1.0",
                "required": False,
                "hint": "Quench-deficit heating exponent scale",
            },
        ]
    if action == "Apply Harmonic Damping Calibration":
        return [
            *SPARC_RUN_FIELDS,
            {
                "key": "diagnostic_source",
                "label": "Diagnostic source",
                "default": "catalog",
                "required": False,
                "hint": "catalog = DataManager API | local_batch = pending CSV slice",
                "choices": ["catalog", "local_batch"],
            },
            {
                "key": "cross_match_cosmic_web",
                "label": "Cross-match cosmic web",
                "default": "yes",
                "required": False,
                "hint": "Adds environment tags when filament distance exists",
                "choices": ["no", "yes"],
            },
            {
                "key": "lambda_web",
                "label": "lambda_web",
                "default": "7.0",
                "required": False,
                "hint": "Scale (Mpc) for FILAMENT/FIELD/VOID tagging",
            },
            {
                "key": "solver_r_baseline",
                "label": "Solver baseline ratio",
                "default": "5.2",
                "required": False,
                "hint": "SuperblockSolver R_baseline",
            },
            {
                "key": "solver_f_align",
                "label": "Solver f_align",
                "default": "1.0",
                "required": False,
                "hint": "Alignment multiplier for R_eff",
            },
            {
                "key": "solver_epsilon_tep",
                "label": "Solver epsilon_tep",
                "default": "0.15",
                "required": False,
                "hint": "Phase-slip penalty coefficient",
            },
            {
                "key": "solver_omega",
                "label": "Solver omega",
                "default": "1.0",
                "required": False,
                "hint": "Quench-deficit heating exponent scale",
            },
            {
                "key": "baseline_weight",
                "label": "Baseline weight",
                "default": "1.0",
                "required": False,
                "hint": "Initial gamma_i before damping",
            },
        ]
    if action == "Initiate Holdout Matrix Test":
        return [
            *SPARC_RUN_FIELDS,
            {
                "key": "diagnostic_source",
                "label": "Diagnostic source",
                "default": "catalog",
                "required": False,
                "hint": "catalog = DataManager API | local_batch = pending CSV slice",
                "choices": ["catalog", "local_batch"],
            },
            {
                "key": "cross_match_cosmic_web",
                "label": "Cross-match cosmic web",
                "default": "yes",
                "required": False,
                "hint": "Adds environment tags when filament distance exists",
                "choices": ["no", "yes"],
            },
            {
                "key": "lambda_web",
                "label": "lambda_web",
                "default": "7.0",
                "required": False,
                "hint": "Scale (Mpc) for FILAMENT/FIELD/VOID tagging",
            },
            {
                "key": "solver_r_baseline",
                "label": "Solver baseline ratio",
                "default": "5.2",
                "required": False,
                "hint": "SuperblockSolver R_baseline",
            },
            {
                "key": "solver_f_align",
                "label": "Solver f_align",
                "default": "1.0",
                "required": False,
                "hint": "Alignment multiplier for R_eff",
            },
            {
                "key": "solver_epsilon_tep",
                "label": "Solver epsilon_tep",
                "default": "0.15",
                "required": False,
                "hint": "Phase-slip penalty coefficient",
            },
            {
                "key": "solver_omega",
                "label": "Solver omega",
                "default": "1.0",
                "required": False,
                "hint": "Quench-deficit heating exponent scale",
            },
            {
                "key": "holdout_fraction",
                "label": "Holdout fraction",
                "default": "0.2",
                "required": False,
                "hint": "Fraction in test split (0.1-0.5)",
            },
            {
                "key": "holdout_repeats",
                "label": "Repeat count",
                "default": "25",
                "required": False,
                "hint": "Number of random holdout splits",
            },
            {
                "key": "holdout_seed",
                "label": "Random seed",
                "default": "7",
                "required": False,
                "hint": "Seed for reproducible split sampling",
            },
        ]
    return list(SPARC_RUN_FIELDS)


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the SPARC entry form."""
    from menus.astronomical.sparc.fetcher import DATASETS_DIR, FINISHED_A_DIR

    if action == "Pull Batch from Repository":
        return [
            "Fetches from astroweb.case.edu SPARC MassModels table.",
            f"Caches catalog + galaxy CSVs in {DATASETS_DIR}/.",
            f"Ledger: {DATASETS_DIR}/done.txt | processed CSVs -> {FINISHED_A_DIR}/.",
            "0 downloaded usually means all 175 galaxies are already cached locally.",
            f"force_repull=yes re-extracts CSVs; restore_archived=yes copies from {FINISHED_A_DIR}/.",
        ]
    if action == "Pull Single Galaxy CSV":
        return [
            f"Downloads one galaxy into {DATASETS_DIR}/{{name}}.csv.",
            "Examples: NGC3198, DDO154, CamB.",
        ]
    if action == "Run Domain Shear Diagnostic":
        return [
            "Computes per-galaxy observed mass ratio and shear from the 5.2 target.",
            "Uses batch_limit / force_rescan / selected_datasets controls.",
        ]
    if action == "Apply Harmonic Damping Calibration":
        return [
            "Runs domain shear sweep, then applies gamma_adj = gamma_i * exp(-shear/313.1).",
            "Set baseline_weight to control starting gamma_i.",
        ]
    if action == "Initiate Holdout Matrix Test":
        return [
            "Repeated train/test splits over local SPARC mass-ratio samples.",
            "Reports mean and p95 drift between holdout and train ratio means.",
        ]
    return [
        "Pull ledger: datasets/sparc/done.txt (repository fetch resume).",
        "Analysis ledger: datasets/sparc/batch_done.txt (batch analysis resume).",
        "batch_limit: 0 = all pending | force_rescan=yes ignores batch_done.txt.",
        "show_graphics: popup = Tk windows | artifacts = PNG only.",
        "Dataset Manager can pick specific galaxies to fetch/run.",
        "Text output saved to artifacts/run_*.txt automatically.",
    ]


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> None:
    from superblock_mode import CONTINUE, begin_live_action

    routed = begin_live_action(MODULE_TAG, selection, options)
    if routed is not CONTINUE:
        return None
    options = options or {}
    action = _ACTION_KEYS.get(selection)
    if action is None:
        print(f"[Superblock Residual Engine] Unknown SPARC action: {selection}")
        return
    _execute_action(selection, action, options, show_plots)


def _execute_action(selection: str, action: str, options: dict, show_plots: bool) -> None:
    if action == "pull_batch":
        summary = pull_sparc_batch(
            limit=_parse_int(options.get("pull_limit", ""), 100),
            data_dir=DEFAULT_DATA_DIR,
            done_file=DEFAULT_DONE_FILE,
            force_refresh_catalog=_yes(options.get("refresh_catalog", "")),
            force_repull=_yes(options.get("force_repull", "")),
            restore_archived=_yes(options.get("restore_archived", "")),
        )
        skipped = len(summary.skipped_done) + len(summary.skipped_existing)
        print(
            f"[Superblock Residual Engine] Batch pull complete: {summary.downloaded_count} downloaded, "
            f"{skipped} skipped, {len(summary.failed)} failed."
        )
        if summary.downloaded_count == 0 and skipped == 0 and not summary.failed:
            print(
                "[Superblock Residual Engine] Local cache is complete — run analysis actions "
                "(e.g. Full SPARC Analysis) on datasets/sparc/*.csv."
            )
        return

    if action == "pull_single":
        galaxy = (options.get("query") or "").strip()
        if not galaxy:
            print("[ERROR] Galaxy name is required for single pull.")
            return
        path = ensure_galaxy_csv(
            galaxy,
            data_dir=DEFAULT_DATA_DIR,
            done_file=DEFAULT_DONE_FILE,
            force_refresh_catalog=_yes(options.get("refresh_catalog", "")),
        )
        print(f"[Superblock Residual Engine] Single pull complete: {path}")
        return

    print(f"[Superblock Residual Engine] Unknown SPARC action: {selection}")
