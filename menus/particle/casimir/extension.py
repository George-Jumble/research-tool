"""
TSB Casimir tau resonance scanner ↔ research_tool.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis
from tav_shared.run_output import parse_show_graphics
from menus.particle.casimir.scanner import (
    ARTIFACTS_DIR,
    DATASETS_DIR,
    DEFAULT_DONE_FILE,
    EXAMPLE_DATASETS,
    run_batch_casimir_scan,
    run_casimir_scan,
)

MODULE_TAG = "TSB_CASIMIR"
SUBMENU_TITLE = "Casimir Residuals"

MENU_ACTIONS = [
    "Example Casimir Tau-Search",
    "TSB Falsification Test Suite (5 predictions)",
    "TSB Falsification Diagnostics (controls + audit)",
    "TSB Bound-Energy Residual Search",
    "Scan Local CSV",
    "Download Zenodo Casimir Drums",
    "Download GitHub Magnetic-Fluid Data",
    "Scan GitHub Magnetic-Fluid CSV",
    "Scan Extracted Dataset (batch)",
    "Full Casimir Tau-Search Pipeline",
    "Null-Hypothesis False-Positive Suite",
    "Blind Periodicity Scan",
    "Lock Blind Results",
    "Compare Blind vs Informed",
]

_ACTION_CONFIG: dict[str, dict] = {
    "Example Casimir Tau-Search": {
        "data_source": "mock",
        "plot": True,
    },
    "TSB Falsification Test Suite (5 predictions)": {
        "falsification": True,
        "data_source": "mock",
        "plot": True,
    },
    "TSB Falsification Diagnostics (controls + audit)": {
        "diagnostics": True,
        "data_source": "mock",
        "plot": False,
    },
    "TSB Bound-Energy Residual Search": {
        "bound_energy": True,
        "data_source": "mock",
        "plot": False,
    },
    "Scan Local CSV": {
        "data_source": "local",
        "plot": True,
    },
    "Download Zenodo Casimir Drums": {
        "download_only": True,
        "dataset": "zenodo",
    },
    "Download GitHub Magnetic-Fluid Data": {
        "download_only": True,
        "dataset": "github",
    },
    "Scan GitHub Magnetic-Fluid CSV": {
        "data_source": "github",
        "plot": True,
    },
    "Scan Extracted Dataset (batch)": {
        "batch": True,
        "plot": False,
    },
    "Full Casimir Tau-Search Pipeline": {
        "data_source": "github",
        "plot": True,
        "batch_if_extracted": True,
        "github_if_cached": True,
    },
    "Null-Hypothesis False-Positive Suite": {
        "null_hypothesis": True,
    },
    "Blind Periodicity Scan": {
        "data_source": "github",
        "plot": True,
        "blind": True,
    },
    "Lock Blind Results": {
        "lock_blind": True,
    },
    "Compare Blind vs Informed": {
        "compare_blind": True,
    },
}


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def handle_pre_form(stdscr, selection: str, params: dict[str, str]) -> str | None:
    """Skip the path form; tests and local scans pick up downloaded tables."""
    _ = stdscr
    if selection in {
        "TSB Falsification Test Suite (5 predictions)",
        "TSB Falsification Diagnostics (controls + audit)",
        "TSB Bound-Energy Residual Search",
    }:
        from menus.particle.casimir.test_framework import falsification_default_options

        params.update(falsification_default_options())
        if selection == "TSB Falsification Diagnostics (controls + audit)":
            params["output_prefix"] = "tsb_diagnostics"
        if selection == "TSB Bound-Energy Residual Search":
            params["output_prefix"] = "tsb_bound_energy"
        return "skip_form"
    if selection in {
        "Scan Local CSV",
        "Scan GitHub Magnetic-Fluid CSV",
        "Blind Periodicity Scan",
    }:
        from menus.particle.casimir.scanner import discover_local_casimir_files

        found = discover_local_casimir_files()
        if found:
            params["local_csv"] = str(found[0])
        params["auto_discover"] = "yes"
        return "skip_form"
    return None


def _show_saved_plot(plot_path: str | None) -> None:
    if not plot_path or not Path(plot_path).is_file():
        return
    image = plt.imread(plot_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(os.path.basename(plot_path))
    plt.tight_layout()
    plt.show()


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for TSB Casimir tau-resonance actions."""
    fields = [
        {
            "key": "local_csv",
            "label": "Local CSV path (optional)",
            "default": "",
            "required": False,
            "hint": "Blank = auto-discover downloaded tables under datasets/casimir/",
        },
        {
            "key": "param_col",
            "label": "Parameter column (x-axis)",
            "default": "",
            "required": False,
            "hint": "e.g. magnetic_field_mT, separation_nm — blank = auto-detect",
        },
        {
            "key": "value_col",
            "label": "Observable column (y-axis)",
            "default": "",
            "required": False,
            "hint": "e.g. force_pN — blank = auto-detect",
        },
        {
            "key": "extract_dir",
            "label": "Extracted dataset directory",
            "default": "",
            "required": False,
            "hint": "Blank = datasets/casimir/casimir_drums_extracted",
        },
        {
            "key": "max_files",
            "label": "Max CSV files (batch)",
            "default": "10",
            "required": False,
            "hint": "Pending files per run; 0 = all pending; resumes via datasets/casimir/done.txt",
        },
        {
            "key": "force_rescan",
            "label": "Force re-scan (ignore done.txt)",
            "default": "no",
            "required": False,
            "hint": "Re-scan from the first files instead of resuming pending",
            "choices": ["no", "yes"],
        },
        {
            "key": "confirm_large_download",
            "label": "Confirm large download (~780 MB)",
            "default": "no",
            "required": False,
            "hint": "Set yes to download Zenodo Casimir drums zip",
            "choices": ["no", "yes"],
        },
        {
            "key": "force_download",
            "label": "Force re-download",
            "default": "no",
            "required": False,
            "hint": "Re-fetch GitHub magnetic-fluid or Zenodo cache",
            "choices": ["no", "yes"],
        },
        {
            "key": "p_threshold",
            "label": "1/7 p-threshold (Holm-adjusted)",
            "default": "0.01",
            "required": False,
            "hint": "tav_resonance_detected only if residual 1/7 p_Holm < this (env TAV_CASIMIR_P_THRESHOLD)",
        },
        {
            "key": "one_seventh_power_ratio",
            "label": "Primary 1/7 power gate",
            "default": "0.55",
            "required": False,
            "hint": "Strict pipeline I(f_1/7)/max(I) for primary claims (env TAV_CASIMIR_ONE_SEVENTH_POWER_RATIO)",
        },
        {
            "key": "seven_cycle_mf_threshold",
            "label": "Secondary 7-cycle matched-filter threshold",
            "default": "0.105",
            "required": False,
            "hint": "Independent period-7 correlator score gate (env TAV_CASIMIR_SEVEN_CYCLE_MF_THRESHOLD); not a primary claim",
        },
        {
            "key": "output_prefix",
            "label": "Plot/report prefix",
            "default": "tsb_casimir",
            "required": False,
            "hint": "Artifacts saved under artifacts/",
        },
        {
            "key": "show_graphics",
            "label": "Graphics mode",
            "default": "popup",
            "required": False,
            "hint": "popup = Tk window | artifacts = PNG only",
            "choices": ["artifacts", "popup"],
        },
    ]
    if action == "Download Zenodo Casimir Drums":
        return [f for f in fields if f["key"] in {"confirm_large_download", "output_prefix"}]
    if action == "Download GitHub Magnetic-Fluid Data":
        return [f for f in fields if f["key"] in {"force_download", "output_prefix"}]
    if action == "Scan GitHub Magnetic-Fluid CSV":
        return [
            f
            for f in fields
            if f["key"] in {"local_csv", "param_col", "value_col", "output_prefix", "show_graphics"}
        ]
    if action == "Scan Extracted Dataset (batch)":
        return [
            f
            for f in fields
            if f["key"] in {"extract_dir", "max_files", "force_rescan", "output_prefix"}
        ] + [
            {
                "key": "material_distance_threshold",
                "label": "Material-distance threshold (relative)",
                "default": "1e-6",
                "required": False,
                "hint": "Abort batch if two distinct materials have relative signature distance below this",
            },
        ]
    if action == "Scan Local CSV":
        return [f for f in fields if f["key"] != "confirm_large_download"]
    if action in {
        "TSB Falsification Test Suite (5 predictions)",
        "TSB Falsification Diagnostics (controls + audit)",
        "TSB Bound-Energy Residual Search",
    }:
        return []
    if action in {"Lock Blind Results", "Compare Blind vs Informed"}:
        return []
    if action == "Blind Periodicity Scan":
        return [
            f
            for f in fields
            if f["key"] in {"local_csv", "param_col", "value_col", "output_prefix", "show_graphics"}
        ]
    if action == "Null-Hypothesis False-Positive Suite":
        return [
            {
                "key": "n_trials",
                "label": "Trials per model × SNR",
                "default": "100",
                "required": False,
                "hint": "100 seeds each; 3 models × 3 SNR = 900 detections",
            },
        ]
    return fields


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the Casimir scanner entry form."""
    if action == "Blind Periodicity Scan":
        return [
            "Blind path: no 1/7 prior, no seven-phase grid, no tav-resonance claim flag.",
            "Reports written with a blind_ prefix. Lock them before any comparison.",
            "See BLIND_ANALYSIS_PROTOCOL.md for the informed vs blind split.",
        ]
    if action == "Lock Blind Results":
        return [
            "Hashes current blind reports and writes artifacts/casimir/blind_lock.json.",
            "Required before Compare Blind vs Informed.",
        ]
    if action == "Compare Blind vs Informed":
        return [
            "Writes the side-by-side table only if the blind lock is intact.",
        ]
    if action == "Null-Hypothesis False-Positive Suite":
        return [
            "Synthetic series on the real Matsubara grid: 1/ξ, exponential, period-7 sinusoid.",
            "Gaussian noise at 10 / 0 / −10 dB; 100 seeds per cell.",
            "Primary envelope: all flags at the strict 0.55 power gate (max FPR ≤ 0.04).",
            "Secondary 7-cycle matched filter: independent period-7 correlator with its own FPR.",
            "Writes artifacts/casimir/null_hypothesis_false_positive_rates.json.",
            "Real-data primary claims are refused if they sit inside the primary 95% envelope.",
        ]
    if action == "TSB Falsification Test Suite (5 predictions)":
        return [
            "Five explicit TSB vs Lifshitz falsification tests with p-values.",
            "Predictions: discrete steps, hysteresis, log-Δn modulations, anisotropy, 1/7 periodicity.",
            "powerlaw_index is n/a on dielectric-vs-frequency tables; anisotropy_channels has a negative control.",
            "Auto-discovers downloaded tables under datasets/casimir/ (GitHub Fig2 first).",
            "Falls back to a generated mock only when the cache is empty.",
            "Outputs: artifacts/tsb_test_results/*_report.json, *_summary.txt, six-panel PNG.",
        ]
    if action == "TSB Falsification Diagnostics (controls + audit)":
        return [
            "Audit/control battery on auto-discovered datasets/casimir tables (7-file discovery preserved).",
            "log_modulations: permutation p + shuffle/surrogate audit (no self-normalized chi2).",
            "periodicity grid lock, hysteresis bootstrap CI, anisotropy identity control, hash ledger.",
            "Writes artifacts/tsb_test_results/tsb_diagnostics_*.",
        ]
    if action == "TSB Bound-Energy Residual Search":
        return [
            "ΔE = E_obs − E_Lifshitz (approximate planar Matsubara integrator).",
            "Identity subtraction control must pass before any E_bind claim.",
            "Shape tests are reused on (d, ΔE), not on ε(iξ) wiggles.",
            "Writes artifacts/tsb_test_results/tsb_bound_energy_*.",
        ]
    return [
        "Downloads: Zenodo Casimir drums (~780 MB) or GitHub magnetic-fluid Fig2/Fig34 spectra.",
        "Extracts ZIP→CSV/TSV; loads direct CSV, TSV, whitespace, or extensionless tables.",
        "Signatures: 7-phase/1/7 harmonics, 142857-cycle, steps, hysteresis, log-Δn, peaks.",
        "Methods: periodogram (Fourier), scipy peak/step detection, autocorrelation, ruptures CP.",
        "Plots overlay 7-phase markers, 1/7 template, periodogram, autocorrelation, summary.",
        "Reports: artifacts/tsb_casimir_*_report.json and *_summary.txt",
        f"Cache: {DATASETS_DIR}/ | batch ledger: {DATASETS_DIR}/done.txt (resume next pending).",
        "Batch scans skip files in done.txt; force_rescan=yes re-runs from the start.",
        "CLI: python tsb_casimir_scanner.py --data github --download-github",
    ]


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> None:
    from superblock_mode import CONTINUE, begin_live_action

    routed = begin_live_action(MODULE_TAG, selection, options)
    if routed is not CONTINUE:
        return None
    options = options or {}
    config = _ACTION_CONFIG.get(selection)
    if config is None:
        print(f"[Superblock Residual Engine] Unknown Casimir Tau-SB action: {selection}")
        return

    output_prefix = (options.get("output_prefix") or "tsb_casimir").strip()
    show_popup = parse_show_graphics(options, default="popup") if show_plots else False
    p_thresh = str(options.get("p_threshold") or "").strip()
    if p_thresh:
        os.environ["TAV_CASIMIR_P_THRESHOLD"] = p_thresh
    power_ratio = str(options.get("one_seventh_power_ratio") or "").strip()
    if power_ratio:
        os.environ["TAV_CASIMIR_ONE_SEVENTH_POWER_RATIO"] = power_ratio
    seven_cycle_mf = str(options.get("seven_cycle_mf_threshold") or "").strip()
    if seven_cycle_mf:
        os.environ["TAV_CASIMIR_SEVEN_CYCLE_MF_THRESHOLD"] = seven_cycle_mf
    mat_thresh = str(options.get("material_distance_threshold") or "").strip()
    if mat_thresh:
        os.environ["TAV_CASIMIR_MATERIAL_DISTANCE_THRESHOLD"] = mat_thresh

    if config.get("lock_blind"):
        from menus.particle.casimir.blind_mode import lock_blind_results

        lock_blind_results()
        return
    if config.get("compare_blind"):
        from menus.particle.casimir.blind_mode import write_blind_comparison_table

        try:
            path = write_blind_comparison_table()
        except RuntimeError as exc:
            print(f"[Superblock Residual Engine] {exc}")
            return
        print(f"[Superblock Residual Engine] Comparison: {path}")
        return
    if config.get("blind"):
        options = {**options, "blind": "yes"}

    if config.get("null_hypothesis"):
        from menus.particle.casimir.null_hypothesis import run_null_hypothesis_suite

        try:
            n_trials = int(options.get("n_trials") or 100)
        except ValueError:
            n_trials = 100
        report = run_null_hypothesis_suite(n_trials=n_trials)
        print(f"[Superblock Residual Engine] Null-hypothesis report: {report.get('report_path')}")
        print(f"[Superblock Residual Engine] Stable envelope: {report.get('stable_report_path')}")
        return

    if config.get("diagnostics"):
        from menus.particle.casimir.diagnostics import run_diagnostics_battery

        diag_prefix = output_prefix if output_prefix != "tsb_casimir" else "tsb_diagnostics"
        try:
            report = run_diagnostics_battery(
                param_col=(options.get("param_col") or "").strip() or None,
                value_col=(options.get("value_col") or "").strip() or None,
                output_prefix=diag_prefix,
                plot=False,
                show_popup=False,
                relock=_yes(options.get("relock", "no")),
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"[Superblock Residual Engine] TSB falsification diagnostics failed: {exc}")
            return
        print(f"[Superblock Residual Engine] Diagnostics report: {report.get('report_path')}")
        return

    if config.get("bound_energy"):
        from menus.particle.casimir.bound_energy import run_bound_energy_search

        be_prefix = output_prefix if output_prefix != "tsb_casimir" else "tsb_bound_energy"
        try:
            report = run_bound_energy_search(
                param_col=(options.get("param_col") or "").strip() or None,
                value_col=(options.get("value_col") or "").strip() or None,
                output_prefix=be_prefix,
                plot=False,
                show_popup=False,
                relock=_yes(options.get("relock", "no")),
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"[Superblock Residual Engine] TSB bound-energy search failed: {exc}")
            return
        print(f"[Superblock Residual Engine] Bound-energy report: {report.get('report_path')}")
        return

    if config.get("falsification"):
        from menus.particle.casimir.test_framework import (
            run_falsification_from_source,
            run_falsification_on_local_downloads,
        )

        local_csv = (options.get("local_csv") or "").strip() or None
        param_col = (options.get("param_col") or "").strip() or None
        value_col = (options.get("value_col") or "").strip() or None
        fals_prefix = output_prefix if output_prefix != "tsb_casimir" else "tsb_falsification"
        try:
            explicit = bool(local_csv) and Path(local_csv).is_file() and not _yes(
                options.get("auto_discover", "no")
            )
            if explicit:
                report = run_falsification_from_source(
                    data_source="local",
                    local_csv=local_csv,
                    param_col=param_col,
                    value_col=value_col,
                    output_prefix=fals_prefix,
                    plot=config.get("plot", True),
                    show_popup=show_popup,
                )
            else:
                report = run_falsification_on_local_downloads(
                    param_col=param_col,
                    value_col=value_col,
                    output_prefix=fals_prefix,
                    plot=config.get("plot", True),
                    show_popup=show_popup,
                )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"[Superblock Residual Engine] TSB falsification suite failed: {exc}")
            return
        if show_popup and report.get("plot_path"):
            _show_saved_plot(report["plot_path"])
        elif report.get("plot_path"):
            print(f"[Superblock Residual Engine] Falsification plot: {report['plot_path']}")
        print(f"[Superblock Residual Engine] Report: {report.get('report_path')}")
        return

    if config.get("download_only"):
        if config.get("dataset") == "github":
            from menus.particle.casimir.scanner import download_github_magnetic_fluid

            paths = download_github_magnetic_fluid(force=_yes(options.get("force_download", "no")))
            print(f"[Superblock Residual Engine] Cached {len(paths)} GitHub magnetic-fluid files.")
            print("  Use 'Scan GitHub Magnetic-Fluid CSV' to analyze.")
            return

        meta = EXAMPLE_DATASETS["zenodo_superconducting_casimir"]
        size_mb = meta.get("size_warning_mb", "780")
        if not _yes(options.get("confirm_large_download", "no")):
            print(f"[Superblock Residual Engine] Zenodo Casimir drums zip is ~{size_mb} MB.")
            print("  Set confirm_large_download=yes to proceed.")
            print(f"  Or download manually to {DATASETS_DIR}/")
            return
        from menus.particle.casimir.scanner import download_file

        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DATASETS_DIR / meta["zip_name"]
        download_file(meta["url"], zip_path)
        print(f"[Superblock Residual Engine] Saved: {zip_path}")
        print("  Use 'Scan Extracted Dataset (batch)' after extraction.")
        return

    if config.get("batch"):
        extract_dir = (options.get("extract_dir") or "").strip()
        if not extract_dir:
            extract_dir = str(DATASETS_DIR / "casimir_drums_extracted")
        if not Path(extract_dir).is_dir():
            print(f"[Superblock Residual Engine] Extract directory not found: {extract_dir}")
            print("  Run 'Download Zenodo Casimir Drums' first, or set extract_dir.")
            return
        try:
            max_files = int(options.get("max_files") or 10)
        except ValueError:
            max_files = 10
        try:
            batch = run_batch_casimir_scan(
                extract_dir,
                max_files=max_files,
                plot=_yes(options.get("batch_plots", "no")),
                output_prefix=output_prefix,
                done_file=DEFAULT_DONE_FILE,
                force_rescan=_yes(options.get("force_rescan", "no")),
            )
        except Exception as exc:
            from menus.particle.casimir.material_gate import MaterialSignatureCollisionError

            if isinstance(exc, MaterialSignatureCollisionError):
                print(f"[Superblock Residual Engine] Multi-material claim refused: {exc}")
                return
            raise
        for path, info in batch["summaries"].items():
            print(f"  {Path(path).name}: {info}")
        if batch.get("multi_material_claim_allowed"):
            print("[Superblock Residual Engine] Material-differentiation gate: PASS")
        return

    data_source = (options.get("data_source") or config.get("data_source", "mock")).strip().lower()
    local_csv = (options.get("local_csv") or "").strip() or None
    param_col = (options.get("param_col") or "").strip() or None
    value_col = (options.get("value_col") or "").strip() or None

    if data_source == "local" and _yes(options.get("auto_discover", "no")):
        from menus.particle.casimir.scanner import discover_local_casimir_files

        found = discover_local_casimir_files()
        if not found:
            print(
                f"[Superblock Residual Engine] No downloaded Casimir tables under {DATASETS_DIR}. "
                "Run 'Download GitHub Magnetic-Fluid Data' first."
            )
            return
        print(f"[Superblock Residual Engine] Auto-discovered {len(found)} Casimir table(s).")
        last_result = None
        try:
            for path in found:
                last_result = run_casimir_scan(
                    data_source="local",
                    local_csv=path,
                    param_col=param_col,
                    value_col=value_col,
                    download=_yes(options.get("force_download", "no")),
                    extract_dir=(options.get("extract_dir") or "").strip() or None,
                    plot=config.get("plot", True),
                    blind=bool(config.get("blind") or _yes(options.get("blind", "no"))),
                    output_prefix=output_prefix,
                )
                print("\n".join(last_result.summary_lines()))
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"[Superblock Residual Engine] Casimir scan failed: {exc}")
            return
        if last_result is None:
            return
        if config.get("plot") and show_popup:
            _show_saved_plot(last_result.plot_path)
        elif config.get("plot") and last_result.plot_path:
            print(f"[Superblock Residual Engine] Diagnostic plot: {last_result.plot_path}")
        return

    if data_source == "local" and not local_csv:
        from menus.particle.casimir.scanner import discover_local_casimir_files

        found = discover_local_casimir_files()
        if not found:
            print(
                f"[Superblock Residual Engine] No downloaded Casimir tables under {DATASETS_DIR}. "
                "Run 'Download GitHub Magnetic-Fluid Data' first."
            )
            return
        local_csv = str(found[0])
        print(f"[Superblock Residual Engine] Auto-discovered {found[0].name}")

    try:
        result = run_casimir_scan(
            data_source=data_source,
            local_csv=local_csv,
            param_col=param_col,
            value_col=value_col,
            download=_yes(options.get("force_download", "no")),
            extract_dir=(options.get("extract_dir") or "").strip() or None,
            plot=config.get("plot", True),
            blind=bool(config.get("blind") or _yes(options.get("blind", "no"))),
            output_prefix=output_prefix,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[Superblock Residual Engine] Casimir scan failed: {exc}")
        return

    print("\n".join(result.summary_lines()))

    if config.get("plot") and show_popup:
        _show_saved_plot(result.plot_path)
    elif config.get("plot") and result.plot_path:
        print(f"[Superblock Residual Engine] Diagnostic plot: {result.plot_path}")

    if config.get("batch_if_extracted"):
        extract_dir = DATASETS_DIR / "casimir_drums_extracted"
        if extract_dir.is_dir() and any(extract_dir.rglob("*")):
            print("\n[Superblock Residual Engine] Running batch scan on extracted Zenodo data...")
            batch = run_batch_casimir_scan(
                extract_dir,
                max_files=int(options.get("max_files") or 5),
                plot=False,
                output_prefix=f"{output_prefix}_batch",
                done_file=DEFAULT_DONE_FILE,
                force_rescan=_yes(options.get("force_rescan", "no")),
            )
            for path, info in batch["summaries"].items():
                print(f"  {Path(path).name}: {info}")
        else:
            print(f"[Superblock Residual Engine] No extracted data at {extract_dir} — Zenodo batch skipped.")

    if config.get("github_if_cached"):
        gh_dir = DATASETS_DIR / "github_magnetic_fluid"
        if gh_dir.is_dir() and any(gh_dir.rglob("*")):
            print("\n[Superblock Residual Engine] Running GitHub magnetic-fluid batch scan...")
            try:
                batch = run_batch_casimir_scan(
                    gh_dir,
                    max_files=int(options.get("max_files") or 3),
                    plot=False,
                    output_prefix=f"{output_prefix}_github",
                    done_file=DEFAULT_DONE_FILE,
                    force_rescan=_yes(options.get("force_rescan", "no")),
                )
            except Exception as exc:
                from menus.particle.casimir.material_gate import MaterialSignatureCollisionError

                if isinstance(exc, MaterialSignatureCollisionError):
                    print(f"[Superblock Residual Engine] Multi-material claim refused: {exc}")
                    return
                raise
            for path, info in batch["summaries"].items():
                print(f"  {Path(path).name}: {info}")
            if batch.get("multi_material_claim_allowed"):
                print("[Superblock Residual Engine] Material-differentiation gate: PASS")