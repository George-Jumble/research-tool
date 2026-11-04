"""
Unified SPARC module ↔ research_tool.py

All analysis actions batch-process ./data/sparc/*.csv via sparc_fetcher.
Repository pulls populate that directory; graphics display after curses exits.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from menus.astronomical.sparc.analyzer import (
    SPARCAnalyzer,
    TauSuperblockSolver,
    THEORY_RATIO_TARGET,
    calibrate_weighting_schema,
    run_diagnostic_sweep,
)
from tav_shared.batch_ledger import (
    SPARC_BATCH_DONE,
    append_done_entries,
    format_batch_banner,
    select_batch_items,
)
from menus.astronomical.sparc.fetcher import (
    DEFAULT_DATA_DIR,
    DEFAULT_DONE_FILE,
    archive_used_datasets,
    ensure_galaxy_csv,
    iter_local_csv_files,
    list_local_galaxies,
    pull_sparc_batch,
)
from tav_shared.run_output import parse_batch_limit, parse_show_graphics
from menus.astronomical.sparc.superblock import ARTIFACTS_DIR, SPARCData
from tav_shared.ingestion.data_manager import (
    DATA_MANAGER_LEGACY_PARAM_KEY,
    DATA_MANAGER_PARAM_KEY,
    get_runtime_data_manager,
)
from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis

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


def _parse_float(value: str, default: float) -> float:
    try:
        return float((value or "").strip())
    except ValueError:
        return default


def _parse_int(value: str, default: int) -> int:
    try:
        return int((value or "").strip())
    except ValueError:
        return default


def _yes(value: str) -> bool:
    return (value or "").strip().lower() in {"yes", "y", "true", "1"}


def _parse_selected_datasets(options: dict) -> Optional[set[str]]:
    raw = (options.get("selected_datasets") or "").strip()
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _iter_local_batch(
    limit: int = 0,
    selected: Optional[set[str]] = None,
    *,
    force_rescan: bool = False,
    done_file: Path | str = SPARC_BATCH_DONE,
) -> Iterator[Tuple[str, Path]]:
    """Yield (galaxy_name, csv_path) for pending ./datasets/sparc/*.csv files."""
    files = iter_local_csv_files(DEFAULT_DATA_DIR)
    if selected:
        files = [path for path in files if path.stem in selected]
    if not files:
        print(f"[Superblock Residual Engine] No CSV files found at {LOCAL_DATA_GLOB}")
        print("[Superblock Residual Engine] Run 'Pull Batch from Repository' or Dataset Manager fetch first.")
        return

    batch_files, status = select_batch_items(
        files,
        done_file,
        key_fn=lambda path: path.stem,
        limit=limit,
        force_rescan=force_rescan,
    )
    print(format_batch_banner("SPARC", status))
    if not batch_files:
        if status["done_count"] >= status["catalog_total"]:
            print(
                "[Superblock Residual Engine] All local SPARC CSVs already analyzed. "
                "Use force_rescan=yes to re-run from the start."
            )
        return
    print(f"[Superblock Residual Engine] Batch source: {LOCAL_DATA_GLOB}")
    for path in batch_files:
        yield path.stem, path


def _resolve_show_plots(options: dict, show_plots: bool) -> bool:
    if "show_graphics" in (options or {}):
        return parse_show_graphics(options, default="artifacts")
    return show_plots


def _display_plot(show_plots: bool) -> None:
    if show_plots:
        plt.show(block=True)
    else:
        plt.close("all")


def _load_theory(options: dict) -> Any:
    if not _yes(options.get("use_theory", "")):
        return None
    try:
        from menus.prime_past.harmonic import TavSuperblockPrimePastHarmonic

        return TavSuperblockPrimePastHarmonic()
    except Exception as exc:
        print(f"[Superblock Residual Engine] Theory object unavailable: {exc}")
        return None


def _print_fit_report(galaxy: str, fit: Dict[str, Any]) -> None:
    print(f"\n--- Emergent DM Fit: {galaxy} ---")
    if not fit.get("success"):
        print(f"[ERROR] Fit failed: {fit.get('message', 'unknown error')}")
        return
    print(f"beta2 = {fit['beta2']:.4f} ± {fit.get('beta2_err', 0):.4f}")
    print(f"phase_offset = {fit['phase_offset']:.4f} ± {fit.get('phase_err', 0):.4f}")
    print(f"chi2 = {fit['chi2']:.2f}  |  reduced chi2 = {fit['reduced_chi2']:.2f}")


def _run_analyzer_for_galaxy(
    selection: str,
    action: str,
    galaxy: str,
    csv_path: Path,
    run_output_dir: Path | None = None,
    show_plots: bool = True,
) -> None:
    analyzer = SPARCAnalyzer(str(csv_path))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[Superblock Residual Engine] SPARC — {selection}")
    print(f"[Superblock Residual Engine] Galaxy: {galaxy}")
    print(f"[Superblock Residual Engine] Data file: {csv_path}")

    if action in {"rotation_curve", "full_analysis", "empirical_mcmc"}:
        from tav_shared.artifact_paths import TestSlug, artifact_path, compose_dataset_slug

        if run_output_dir is not None:
            run_output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = run_output_dir / f"{galaxy}_rotation.png"
        else:
            plot_path = artifact_path(
                TestSlug.SPARC,
                compose_dataset_slug(galaxy, "rotation"),
                "plot",
                "png",
            )
        analyzer.plot_rotation_curve(galaxy, show=show_plots, save_path=str(plot_path))
        print(f"[Superblock Residual Engine] Rotation curve saved: {plot_path}")
        if show_plots:
            _display_plot(True)
        else:
            plt.close("all")

    if action in {"mass_ratio", "full_analysis"}:
        report = analyzer.analysis_report(galaxy)
        print("\n--- SPARC Mass Ratio Report ---")
        print(f"Mean shadow/baryon v^2 ratio: {report['mean_shadow_baryon_ratio']:.4f}")
        print(f"Theory target (Tav-Superblock): {report['theory_target_ratio']:.1f}")
        print(f"Delta from theory: {report['delta_from_theory']:.4f}")
        if report["within_10_percent"]:
            print("[STATISTICAL NOTE] Observed ratio within 10% of 5.2 expectation.")
        else:
            print("[STATISTICAL NOTE] Ratio deviates from 5.2 — inspect domain shear / phase alignment.")


def _run_empirical_mcmc_for_galaxy(
    galaxy: str,
    csv_path: Path,
    run_output_dir: Path | None = None,
    show_plots: bool = True,
) -> None:
    from tav_shared import mcmc_engine, require_topology

    analyzer = SPARCAnalyzer(str(csv_path))
    gal = analyzer.get_galaxy_frame(galaxy)
    x, y_total = gal["R"], gal["Vobs"]
    df = gal.rename(columns={"R": "Radius", "Vobs": "Velocity"})
    df["Kinematic_X"] = gal["R"]
    df["Observable_Y"] = gal["Vobs"]

    ratio = analyzer.calculate_mass_ratio(galaxy)
    print(f"[Superblock Residual Engine] SPARC shadow/baryon ratio: {ratio:.4f} (theory target {THEORY_RATIO_TARGET})")

    _run_analyzer_for_galaxy(
        "Empirical Plot + MCMC Sweep",
        "empirical_mcmc",
        galaxy,
        csv_path,
        run_output_dir,
        show_plots,
    )

    dev, is_anomalous = require_topology.calculate_topological_correlation(df, domain="SPARC")
    print(f"\n--- Analytical Report: {galaxy} ---")
    print(f"Mean Flux Intensity: {np.mean(y_total):.6f}")
    print(f"Max Deviation: {np.max(np.abs(dev)):.6f}")
    if is_anomalous:
        print("[CORRELATION FOUND] Anomalous phase-shift detected.")
    else:
        print("[SIGNAL STABLE] No TS dynamic deviations detected.")

    print("\n[Superblock Residual Engine] Initiating MCMC statistical sweep...")
    comparison = mcmc_engine.run_model_comparison(x, y_total)
    print(mcmc_engine.format_comparison_report(comparison))

    tav_row = next(row for row in comparison if row["model_type"] == "Tav")
    piso_row = next(row for row in comparison if row["model_type"] == "pISO")
    best = comparison[0]

    print(f"\n{mcmc_engine.describe_tav_fit(tav_row.get('params'))}")
    if np.isfinite(best["aic"]):
        print(f"Best global fit: {best['label']} (AIC = {best['aic']:.2f})")
    if np.isfinite(tav_row["aic"]):
        print(f"Tav-Superblock AIC: {tav_row['aic']:.2f}")
    if np.isfinite(piso_row["aic"]):
        print(f"pISO AIC: {piso_row['aic']:.2f}")


def _run_superblock_for_galaxy(
    action: str,
    galaxy: str,
    options: dict,
    run_output_dir: Path | None = None,
    show_plots: bool = True,
) -> None:
    from tav_shared.artifact_paths import TestSlug, artifact_path, compose_dataset_slug

    sparc = SPARCData(data_dir=DEFAULT_DATA_DIR)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if action == "predict":
        beta2 = _parse_float(options.get("beta2", ""), 0.3)
        phase = _parse_float(options.get("phase_offset", ""), 1.0)
        theory = _load_theory(options)
        theory_curve = sparc.predict_emergent_dm_velocity(
            galaxy, beta2=beta2, phase_offset=phase, theory=theory
        )
        if theory_curve.empty:
            print(f"[ERROR] No rotation curve data for {galaxy}")
            return
        if run_output_dir is not None:
            run_output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = run_output_dir / f"{galaxy}_emergent_dm.png"
        else:
            plot_path = artifact_path(
                TestSlug.SPARC,
                compose_dataset_slug(galaxy, "emergent_dm"),
                "plot",
                "png",
            )
        sparc.plot_rotation_curve(
            galaxy, theory_curve=theory_curve, show=show_plots, save_path=str(plot_path)
        )
        print(f"[Superblock Residual Engine] Plot saved: {plot_path}")
        if show_plots:
            _display_plot(True)
        else:
            plt.close("all")
        return

    if action == "fit":
        fit = sparc.fit_emergent_dm(
            galaxy,
            initial_beta2=_parse_float(options.get("beta2", ""), 0.3),
            initial_phase=_parse_float(options.get("phase_offset", ""), 1.0),
        )
        _print_fit_report(galaxy, fit)
        if fit.get("success"):
            theory_curve = sparc.predict_emergent_dm_velocity(
                galaxy, beta2=fit["beta2"], phase_offset=fit["phase_offset"]
            )
            if run_output_dir is not None:
                run_output_dir.mkdir(parents=True, exist_ok=True)
                plot_path = run_output_dir / f"{galaxy}_fit.png"
            else:
                plot_path = artifact_path(
                    TestSlug.SPARC,
                    compose_dataset_slug(galaxy, "fit"),
                    "plot",
                    "png",
                )
            sparc.plot_rotation_curve(
                galaxy, theory_curve=theory_curve, show=show_plots, save_path=str(plot_path)
            )
            print(f"[Superblock Residual Engine] Fit plot saved: {plot_path}")
            if show_plots:
                _display_plot(True)
            else:
                plt.close("all")
        return

    if action == "full_demo":
        fit = sparc.fit_emergent_dm(galaxy)
        _print_fit_report(galaxy, fit)
        beta2 = fit.get("beta2", 0.3) if fit.get("success") else 0.3
        phase = fit.get("phase_offset", 1.0) if fit.get("success") else 1.0
        theory = _load_theory(options)
        theory_curve = sparc.predict_emergent_dm_velocity(
            galaxy, beta2=beta2, phase_offset=phase, theory=theory
        )
        if run_output_dir is not None:
            run_output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = run_output_dir / f"{galaxy}_full_demo.png"
        else:
            plot_path = artifact_path(
                TestSlug.SPARC,
                compose_dataset_slug(galaxy, "full_demo"),
                "plot",
                "png",
            )
        sparc.plot_rotation_curve(
            galaxy, theory_curve=theory_curve, show=show_plots, save_path=str(plot_path)
        )
        print(f"[Superblock Residual Engine] Full demo plot saved: {plot_path}")
        if show_plots:
            _display_plot(True)
        else:
            plt.close("all")


def _run_batch(action: str, selection: str, options: dict, show_plots: bool = True) -> None:
    limit = parse_batch_limit(options, default=0)
    show_plots = _resolve_show_plots(options, show_plots)
    selected = _parse_selected_datasets(options)
    force_rescan = _yes(options.get("force_rescan", ""))
    print(
        f"[Superblock Residual Engine] Graphics: {'Tk popup' if show_plots else 'artifacts only (no popup)'}"
    )
    if limit > 0:
        print(f"[Superblock Residual Engine] Batch limit: {limit}")
    if selected:
        print(f"[Superblock Residual Engine] Selected datasets: {', '.join(sorted(selected))}")

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_output_dir = ARTIFACTS_DIR / f"sparc_run_{run_stamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Superblock Residual Engine] SPARC image output folder: {run_output_dir}")

    processed_paths: list[Path] = []
    completed_ids: list[str] = []
    mass_ratio_map: dict[str, float] = {}
    processed = 0
    for galaxy, csv_path in (
        _iter_local_batch(limit=limit, selected=selected, force_rescan=force_rescan) or []
    ):
        processed += 1
        print(f"\n{'=' * 60}")
        print(f"[Superblock Residual Engine] Batch item {processed}: {galaxy}")
        print(f"{'=' * 60}")

        if action == "empirical_mcmc":
            _run_empirical_mcmc_for_galaxy(galaxy, csv_path, run_output_dir, show_plots)
        elif action in {"rotation_curve", "mass_ratio", "full_analysis"}:
            _run_analyzer_for_galaxy(
                selection,
                action,
                galaxy,
                csv_path,
                run_output_dir,
                show_plots,
            )
        elif action in {"predict", "fit", "full_demo"}:
            print(f"\n[Superblock Residual Engine] SPARC Superblock — {selection}")
            _run_superblock_for_galaxy(action, galaxy, options, run_output_dir, show_plots)
        else:
            break

        if action in {"mass_ratio", "full_analysis", "empirical_mcmc"}:
            try:
                analyzer = SPARCAnalyzer(str(csv_path))
                mass_ratio_map[galaxy] = analyzer.calculate_mass_ratio(galaxy)
            except Exception as exc:
                print(f"[DIAGNOSTIC] Skipping {galaxy}: {exc}")

        processed_paths.append(csv_path)
        completed_ids.append(galaxy)

    if completed_ids:
        append_done_entries(SPARC_BATCH_DONE, completed_ids)
        print(f"[Superblock Residual Engine] Recorded {len(completed_ids)} galaxy(ies) in {SPARC_BATCH_DONE}")

    if processed:
        print(f"\n[Superblock Residual Engine] Batch complete: {processed} galaxies from {LOCAL_DATA_GLOB}")
        if mass_ratio_map:
            print("[DIAGNOSTIC] Running Tav-Superblock domain shear sweep...")
            shear_map = run_diagnostic_sweep(mass_ratio_map)
            baseline_weights = {name: 1.0 for name in mass_ratio_map}
            calibrated = calibrate_weighting_schema(baseline_weights, shear_map)
            avg_calibrated = float(np.mean(list(calibrated.values()))) if calibrated else 1.0
            print(
                "[DIAGNOSTIC] Harmonic damping applied "
                f"({len(calibrated)} galaxies, mean calibrated weight={avg_calibrated:.6f})"
            )
        if str(options.get("archive_after_run", "yes")).strip().lower() not in {
            "no",
            "n",
            "false",
            "0",
        }:
            archive_used_datasets(processed_paths)
            from tav_shared.dataset_ledger import mark_processed

            mark_processed(
                [f"sparc:{path.stem}" for path in processed_paths],
                note="sparc batch archive",
            )


def _run_test_suite(options: dict) -> None:
    sparc = SPARCData(data_dir=DEFAULT_DATA_DIR)
    galaxies = list_local_galaxies(DEFAULT_DATA_DIR)
    if not galaxies:
        print(f"[Superblock Residual Engine] No CSV files at {LOCAL_DATA_GLOB}. Run a repository pull first.")
        return

    limit = parse_batch_limit(options, default=0)
    force_rescan = _yes(options.get("force_rescan", ""))
    galaxies, status = select_batch_items(
        galaxies,
        SPARC_BATCH_DONE,
        key_fn=str,
        limit=limit,
        force_rescan=force_rescan,
    )
    print(format_batch_banner("SPARC test suite", status))
    if not galaxies:
        print("[Superblock Residual Engine] No pending galaxies for the test suite.")
        return
    max_galaxies = _parse_int(options.get("max_galaxies", ""), len(galaxies))
    max_galaxies = min(max_galaxies, len(galaxies))
    fit = _yes(options.get("run_fit", "yes"))
    run_galaxies = galaxies[:max_galaxies]
    results = sparc.run_sparc_test_suite(
        galaxies=run_galaxies, max_galaxies=max_galaxies, fit=fit
    )
    append_done_entries(SPARC_BATCH_DONE, run_galaxies)
    if results.empty:
        print("[Superblock Residual Engine] No galaxies met minimum data requirements.")
        return

    out_path = ARTIFACTS_DIR / "sparc_superblock_test_suite.csv"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\n--- SPARC Test Suite ({len(results)} galaxies) ---")
    display_cols = [
        c for c in ["galaxy", "success", "beta2", "reduced_chi2", "median_g_ratio"] if c in results.columns
    ]
    print(results[display_cols].to_string(index=False))
    print(f"[Superblock Residual Engine] Suite report saved: {out_path}")


def _diagnostic_artifact_path(stem: str) -> Path:
    diag_dir = ARTIFACTS_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return diag_dir / f"{stem}_{stamp}.json"


def save_shear_map(shear_results: dict[str, dict[str, float]]) -> Path:
    out_path = _diagnostic_artifact_path("sparc_shear_map")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_galaxies": len(shear_results),
        "shear_map": shear_results,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def save_calibrated_weights(calibrated_weights: dict[str, float]) -> Path:
    out_path = _diagnostic_artifact_path("sparc_calibrated_weights")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_galaxies": len(calibrated_weights),
        "weights": calibrated_weights,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _parse_outlier_galaxies(
    options: dict,
    shear_map: dict[str, dict[str, float]],
) -> list[str]:
    explicit = [
        item.strip()
        for item in str(options.get("outlier_galaxies", "")).split(",")
        if item.strip()
    ]
    threshold = _parse_float(options.get("outlier_shear_threshold", ""), 5.0)
    top_k = max(_parse_int(options.get("outlier_top_k", ""), 6), 1)

    ranked = sorted(
        (
            (name, float((payload or {}).get("shear", np.nan)))
            for name, payload in shear_map.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    threshold_hits = [name for name, shear in ranked if np.isfinite(shear) and shear >= threshold]
    ranked_top = [name for name, _ in ranked[:top_k]]

    merged: list[str] = []
    for item in [*explicit, *threshold_hits, *ranked_top]:
        if item not in merged:
            merged.append(item)
    return merged


def _cross_match_outlier_membership(
    options: dict,
    ratio_map: dict[str, float],
    shear_map: dict[str, dict[str, float]],
) -> None:
    if not ratio_map or not shear_map:
        return

    manager = options.get(DATA_MANAGER_PARAM_KEY) or options.get(DATA_MANAGER_LEGACY_PARAM_KEY)
    if manager is None:
        manager = get_runtime_data_manager(options)

    outlier_ids = _parse_outlier_galaxies(options, shear_map)
    if not outlier_ids:
        print("[DIAGNOSTIC] No outlier galaxies selected for cosmic-web cross-match.")
        return

    try:
        frame = manager.fetch_sparc_catalog()
    except Exception as exc:
        print(f"[DIAGNOSTIC] Outlier cosmic-web cross-match skipped: {exc}")
        return

    if frame is None or frame.empty or "galaxy_id" not in frame.columns:
        print("[DIAGNOSTIC] Outlier cosmic-web cross-match skipped: catalog unavailable.")
        return

    lambda_tav = _parse_float(options.get("lambda_tav", ""), 7.0)
    if "dist_to_filament_Mpc" in frame.columns:
        try:
            frame = manager.cross_match_cosmic_web(frame, lambda_tav=lambda_tav)
        except Exception as exc:
            print(f"[DIAGNOSTIC] Cosmic-web tagging failed for outliers: {exc}")

    subset = frame[frame["galaxy_id"].astype(str).isin(outlier_ids)].copy()
    if subset.empty:
        print("[DIAGNOSTIC] Outlier cosmic-web cross-match found no overlapping galaxies.")
        return

    rows: list[dict[str, Any]] = []
    print("[DIAGNOSTIC] Outlier cosmic-web membership (filament/void):")
    for _, row in subset.iterrows():
        galaxy = str(row.get("galaxy_id", "")).strip()
        env = str(row.get("environment", "FIELD") or "FIELD")
        dist = row.get("dist_to_filament_Mpc", np.nan)
        ratio = float(ratio_map.get(galaxy, np.nan))
        shear = float((shear_map.get(galaxy) or {}).get("shear", np.nan))

        dist_txt = f"{float(dist):.3f}" if np.isfinite(dist) else "nan"
        ratio_txt = f"{ratio:.4f}" if np.isfinite(ratio) else "nan"
        shear_txt = f"{shear:.4f}" if np.isfinite(shear) else "nan"
        print(
            "  "
            f"{galaxy}: env={env} | dist_to_filament_Mpc={dist_txt} | "
            f"ratio={ratio_txt} | shear={shear_txt}"
        )

        rows.append(
            {
                "galaxy_id": galaxy,
                "environment": env,
                "dist_to_filament_Mpc": None if not np.isfinite(dist) else float(dist),
                "r_shadow_baryon_obs": None if not np.isfinite(ratio) else ratio,
                "shear": None if not np.isfinite(shear) else shear,
            }
        )

    out_path = _diagnostic_artifact_path("sparc_outlier_cosmic_web_membership")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lambda_tav": lambda_tav,
        "n_outliers": len(rows),
        "outliers": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[DIAGNOSTIC] Outlier membership map saved: {out_path}")


def apply_to_pipeline(calibrated_weights: dict[str, float]) -> None:
    """Persist active weighting map in-memory for follow-up SPARC diagnostics."""
    global _ACTIVE_PIPELINE_WEIGHTS
    _ACTIVE_PIPELINE_WEIGHTS = dict(calibrated_weights)
    print(
        "[DIAGNOSTIC] Applied calibrated weights to active SPARC pipeline context "
        f"(n={len(_ACTIVE_PIPELINE_WEIGHTS)})."
    )


def _collect_mass_ratio_map(options: dict) -> tuple[dict[str, float], list[str]]:
    """Collect per-galaxy shadow/baryon ratios from catalog API or local batch slice."""
    source = str(options.get("diagnostic_source", "catalog")).strip().lower()
    if source == "catalog":
        manager = options.get(DATA_MANAGER_PARAM_KEY) or options.get(
            DATA_MANAGER_LEGACY_PARAM_KEY
        )
        if manager is None:
            manager = get_runtime_data_manager(options)
        try:
            frame = manager.fetch_sparc_catalog()
            if frame is None or frame.empty:
                raise ValueError("SPARC catalog returned no rows.")
            if _yes(options.get("cross_match_cosmic_web", "yes")):
                if "dist_to_filament_Mpc" in frame.columns:
                    lambda_tav = _parse_float(options.get("lambda_tav", ""), 7.0)
                    frame = manager.cross_match_cosmic_web(frame, lambda_tav=lambda_tav)

            if "galaxy_id" not in frame.columns or "r_shadow_baryon_obs" not in frame.columns:
                raise ValueError(
                    "SPARC catalog missing required columns: galaxy_id, r_shadow_baryon_obs"
                )

            if {
                "dist_to_filament_Mpc",
                "delta_phi_est",
            }.issubset(frame.columns):
                solver = TauSuperblockSolver(
                    lambda_tav=_parse_float(options.get("lambda_tav", ""), 7.0),
                    r_baseline=_parse_float(
                        options.get("solver_r_baseline", ""), THEORY_RATIO_TARGET
                    ),
                )
                f_align = _parse_float(options.get("solver_f_align", ""), 1.0)
                epsilon_tep = _parse_float(options.get("solver_epsilon_tep", ""), 0.15)
                omega = _parse_float(options.get("solver_omega", ""), 1.0)

                r_local = pd.to_numeric(frame["dist_to_filament_Mpc"], errors="coerce").to_numpy()
                delta_phi = pd.to_numeric(frame["delta_phi_est"], errors="coerce").fillna(0.0).to_numpy()
                if "shear_metric" in frame.columns:
                    shear_penalty = pd.to_numeric(frame["shear_metric"], errors="coerce").fillna(0.0).to_numpy()
                else:
                    ratio_vals = pd.to_numeric(
                        frame.get("r_shadow_baryon_obs", THEORY_RATIO_TARGET),
                        errors="coerce",
                    ).fillna(THEORY_RATIO_TARGET)
                    shear_penalty = np.abs(ratio_vals.to_numpy() - THEORY_RATIO_TARGET)

                # Explicitly exercise new solver choking path in menu-triggered diagnostics.
                kappa = solver.simulate_local_kappa(r_local, shear_penalty)
                frame["kappa_fil"] = kappa
                frame["r_eff_tsb"] = solver.compute_choked_shadow_ratio(kappa)
                frame["r_eff_tsb_phase"] = solver.compute_effective_ratio(
                    r_local,
                    delta_phi,
                    f_align=f_align,
                    epsilon_tep=epsilon_tep,
                )

                if "xray_excess_lum" in frame.columns:
                    e_kin = pd.to_numeric(frame["xray_excess_lum"], errors="coerce").fillna(0.0).to_numpy()
                else:
                    e_kin = pd.to_numeric(frame["r_shadow_baryon_obs"], errors="coerce").fillna(0.0).to_numpy()
                frame["delta_e_fluct"] = solver.compute_quench_deficit_heating(
                    e_kinetic=e_kin,
                    kappa=kappa,
                    omega=omega,
                )
                finite_kappa = np.isfinite(frame["kappa_fil"].to_numpy())
                finite_reff = np.isfinite(frame["r_eff_tsb"].to_numpy())
                if finite_kappa.any() and finite_reff.any():
                    print(
                        "[DIAGNOSTIC] TauSuperblockSolver summary: "
                        f"kappa_mean={float(np.nanmean(frame.loc[finite_kappa, 'kappa_fil'])):.4f} | "
                        f"r_eff_choked_mean={float(np.nanmean(frame.loc[finite_reff, 'r_eff_tsb'])):.4f}"
                    )

            ratio_map: dict[str, float] = {}
            for row in frame.itertuples(index=False):
                galaxy = str(getattr(row, "galaxy_id", "")).strip()
                ratio = getattr(row, "r_shadow_baryon_obs", np.nan)
                if not galaxy or not np.isfinite(ratio):
                    continue
                ratio_map[galaxy] = float(ratio)
                env = getattr(row, "environment", "")
                if env:
                    print(
                        "[DIAGNOSTIC] "
                        f"Galaxy: {galaxy} | Observed ratio: {ratio_map[galaxy]:.4f} | "
                        f"Env: {env}"
                    )
                else:
                    print(
                        "[DIAGNOSTIC] "
                        f"Galaxy: {galaxy} | Observed ratio: {ratio_map[galaxy]:.4f}"
                    )
            if ratio_map:
                return ratio_map, []
        except Exception as exc:
            print(f"[DIAGNOSTIC] Catalog mode failed ({exc}); falling back to local batch CSVs.")

    limit = parse_batch_limit(options, default=0)
    selected = _parse_selected_datasets(options)
    force_rescan = _yes(options.get("force_rescan", ""))
    ratio_map: dict[str, float] = {}
    completed_ids: list[str] = []

    for galaxy, csv_path in (
        _iter_local_batch(limit=limit, selected=selected, force_rescan=force_rescan) or []
    ):
        try:
            analyzer = SPARCAnalyzer(str(csv_path))
            ratio_map[galaxy] = analyzer.calculate_mass_ratio(galaxy)
            completed_ids.append(galaxy)
            print(
                "[DIAGNOSTIC] "
                f"Galaxy: {galaxy} | Observed ratio: {ratio_map[galaxy]:.4f}"
            )
        except Exception as exc:
            print(f"[DIAGNOSTIC] Skipping {galaxy}: {exc}")

    if completed_ids:
        append_done_entries(SPARC_BATCH_DONE, completed_ids)
    return ratio_map, completed_ids


def _run_domain_shear_diagnostic(options: dict) -> None:
    global _LAST_SHEAR_RESULTS
    ratio_map, completed_ids = _collect_mass_ratio_map(options)
    if not ratio_map:
        print("[DIAGNOSTIC] No valid galaxies available for domain shear diagnostic.")
        return
    shear_map = run_diagnostic_sweep(ratio_map)
    _LAST_SHEAR_RESULTS = dict(shear_map)
    shear_vals = [item["shear"] for item in shear_map.values()]
    shear_path = save_shear_map(shear_map)
    _cross_match_outlier_membership(options, ratio_map, shear_map)
    print(
        "[DIAGNOSTIC] Domain shear summary: "
        f"n={len(shear_vals)} | mean={float(np.mean(shear_vals)):.4f} | "
        f"max={float(np.max(shear_vals)):.4f}"
    )
    print(f"[DIAGNOSTIC] Shear map saved: {shear_path}")
    if completed_ids:
        print(f"[Superblock Residual Engine] Recorded {len(completed_ids)} galaxy(ies) in {SPARC_BATCH_DONE}")


def _run_harmonic_damping_calibration(options: dict) -> None:
    ratio_map, completed_ids = _collect_mass_ratio_map(options)
    if not ratio_map:
        print("[DIAGNOSTIC] No valid galaxies available for harmonic damping calibration.")
        return

    baseline_weight = _parse_float(options.get("baseline_weight", ""), 1.0)
    shear_map = {
        name: _LAST_SHEAR_RESULTS[name]
        for name in ratio_map
        if name in _LAST_SHEAR_RESULTS
    }
    if len(shear_map) != len(ratio_map):
        shear_map = run_diagnostic_sweep(ratio_map)

    baseline_weights = {
        name: _ACTIVE_PIPELINE_WEIGHTS.get(name, baseline_weight)
        for name in ratio_map
    }
    calibrated = calibrate_weighting_schema(baseline_weights, shear_map)
    if not calibrated:
        print("[DIAGNOSTIC] Calibration yielded no weights.")
        return

    values = np.array(list(calibrated.values()), dtype=float)
    weights_path = save_calibrated_weights(calibrated)
    apply_to_pipeline(calibrated)
    print(
        "[DIAGNOSTIC] Harmonic damping summary: "
        f"n={values.size} | mean={float(values.mean()):.6f} | "
        f"min={float(values.min()):.6f} | max={float(values.max()):.6f}"
    )
    print(f"[DIAGNOSTIC] Calibrated weights saved: {weights_path}")
    for galaxy in sorted(calibrated)[:5]:
        print(f"  {galaxy}: {calibrated[galaxy]:.6f}")
    if completed_ids:
        print(f"[Superblock Residual Engine] Recorded {len(completed_ids)} galaxy(ies) in {SPARC_BATCH_DONE}")


def _run_holdout_matrix_test(options: dict) -> None:
    ratio_map, completed_ids = _collect_mass_ratio_map(options)
    if len(ratio_map) < 4:
        print("[DIAGNOSTIC] Holdout matrix test requires at least 4 galaxies.")
        return

    holdout_fraction = min(max(_parse_float(options.get("holdout_fraction", ""), 0.2), 0.1), 0.5)
    n_repeats = max(_parse_int(options.get("holdout_repeats", ""), 25), 1)
    seed = _parse_int(options.get("holdout_seed", ""), 7)
    rng = np.random.default_rng(seed)

    galaxies = np.array(sorted(ratio_map.keys()))
    ratios = np.array([ratio_map[name] for name in galaxies], dtype=float)
    holdout_size = max(1, int(round(galaxies.size * holdout_fraction)))

    drift_vals: list[float] = []
    for _ in range(n_repeats):
        perm = rng.permutation(galaxies.size)
        test_idx = perm[:holdout_size]
        train_idx = perm[holdout_size:]
        if train_idx.size == 0:
            continue
        train_mean = float(np.mean(ratios[train_idx]))
        test_mean = float(np.mean(ratios[test_idx]))
        drift_vals.append(abs(train_mean - test_mean))

    if not drift_vals:
        print("[DIAGNOSTIC] Holdout matrix test could not compute train/test drift.")
        return

    drift = np.array(drift_vals, dtype=float)
    print(
        "[DIAGNOSTIC] Holdout matrix summary: "
        f"n={drift.size} | holdout_fraction={holdout_fraction:.2f} | "
        f"mean_drift={float(drift.mean()):.4f} | p95_drift={float(np.percentile(drift, 95)):.4f}"
    )
    if completed_ids:
        print(f"[Superblock Residual Engine] Recorded {len(completed_ids)} galaxy(ies) in {SPARC_BATCH_DONE}")


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for SPARC actions."""
    from tav_shared.tav_project_paths import FINISHED_A_DIR

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
                "key": "lambda_tav",
                "label": "lambda_tav",
                "default": "7.0",
                "required": False,
                "hint": "Scale (Mpc) for FILAMENT/FIELD/VOID tagging",
            },
            {
                "key": "solver_r_baseline",
                "label": "Solver baseline ratio",
                "default": "5.2",
                "required": False,
                "hint": "TauSuperblockSolver R_baseline",
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
                "key": "lambda_tav",
                "label": "lambda_tav",
                "default": "7.0",
                "required": False,
                "hint": "Scale (Mpc) for FILAMENT/FIELD/VOID tagging",
            },
            {
                "key": "solver_r_baseline",
                "label": "Solver baseline ratio",
                "default": "5.2",
                "required": False,
                "hint": "TauSuperblockSolver R_baseline",
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
                "key": "lambda_tav",
                "label": "lambda_tav",
                "default": "7.0",
                "required": False,
                "hint": "Scale (Mpc) for FILAMENT/FIELD/VOID tagging",
            },
            {
                "key": "solver_r_baseline",
                "label": "Solver baseline ratio",
                "default": "5.2",
                "required": False,
                "hint": "TauSuperblockSolver R_baseline",
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

    if action == "suite":
        _run_test_suite(options)
        return

    if action == "domain_shear_diagnostic":
        _run_domain_shear_diagnostic(options)
        return

    if action == "harmonic_damping_calibration":
        _run_harmonic_damping_calibration(options)
        return

    if action == "holdout_matrix_test":
        _run_holdout_matrix_test(options)
        return

    if action in _BATCH_ACTIONS:
        _run_batch(action, selection, options, show_plots=show_plots)