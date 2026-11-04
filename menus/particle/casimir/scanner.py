#!/usr/bin/env python3
"""
TSB Tau Resonance Signatures Search in Casimir Data
====================================================

Searches experimental/simulation Casimir datasets for features consistent with
Tau Cylinder / Topological Exclusion predictions:

- 7-phase / 1/7 harmonic periodicity or steps
- Discrete thresholds / hysteresis (magnetic tuning)
- Log-scale modulations (hierarchical Δn steps)
- Dynamic asymmetry or resonant peaks
- Non-local / synchronized features (if multi-channel data available)

Adapts to common public datasets (Zenodo, GitHub magnetic-fluid data, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


from tav_shared.batch_ledger import (
    CASIMIR_BATCH_DONE,
    append_done_entries,
    batch_status,
    format_batch_banner,
    read_done_list,
    select_batch_items,
)

from tav_shared.artifact_paths import (
    ARTIFACTS_DIR,
    TestSlug,
    compose_dataset_slug,
    test_artifacts_dir,
)
from menus.particle.casimir.casimir_artifacts import (
    casimir_output_path,
    current_casimir_series,
    end_casimir_artifact_series,
    start_casimir_artifact_series,
)
from menus.particle.casimir.series_integrity import (
    DuplicateRawSeriesError,
    VALUE_NAME,
    build_raw_series_integrity,
    clean_xy,
    register_series_hash,
    reset_seen_hashes,
    standardize_dielectric_columns,
)
from menus.particle.casimir.material_gate import (
    MaterialSignatureCollisionError,
    build_signature_feature_vector,
    enforce_material_gate,
    infer_material_id,
    material_distance_threshold,
)
from menus.particle.casimir.significance import (
    expected_one_seventh_frequency,
    fit_smooth_trend,
    holm_adjust,
    hysteresis_pvalue,
    one_seventh_power_ratio_threshold,
    p_threshold,
    pelt_changepoints_with_bootstrap,
    random_placement_pvalue,
    residual_periodogram_significance,
    residual_step_pvalue,
)
from menus.particle.casimir.seven_cycle_detector import (
    detect_seven_cycle_matched_filter,
)
from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT as PROJECT_ROOT
DATASETS_DIR = PROJECT_ROOT / "datasets" / "casimir"
DEFAULT_DONE_FILE = CASIMIR_BATCH_DONE

TAU_RESONANCE_PERIOD: float = 7.0
ONE_SEVENTH: float = 1.0 / 7.0
TAU_CYCLE_142857: float = 142857.0  # 1/7 repeating-decimal resonance anchor

GITHUB_MF_API = (
    "https://api.github.com/repos/malong201408/"
    "Dataset-of-Prediction-of-a-measurable-sign-change-in-the-Casimir-force-using-a-magnetic-fluid/contents"
)
GITHUB_MF_RAW = (
    "https://raw.githubusercontent.com/malong201408/"
    "Dataset-of-Prediction-of-a-measurable-sign-change-in-the-Casimir-force-using-a-magnetic-fluid/main"
)

EXAMPLE_DATASETS: dict[str, dict[str, str]] = {
    "zenodo_superconducting_casimir": {
        "url": "https://zenodo.org/records/14700381/files/Casimir%20drums%20data.zip?download=1",
        "zip_name": "casimir_drums_data.zip",
        "description": "Superconducting Casimir force measurements (~780 MB zip with CSVs/HDF5)",
        "size_warning_mb": "780",
    },
    "github_magnetic_fluid_prediction": {
        "description": "Magnetic-fluid Casimir sign-change dielectric spectra (Fig2/Fig34 on GitHub)",
        "subdirs": "Fig2,Fig34",
    },
}

COLUMN_ALIASES: dict[str, str] = {
    "B": "magnetic_field_mT",
    "B_mT": "magnetic_field_mT",
    "field": "magnetic_field_mT",
    "magnetic_field": "magnetic_field_mT",
    "F": "force_pN",
    "Force": "force_pN",
    "force": "force_pN",
    "d": "separation_nm",
    "distance": "separation_nm",
    "sep": "separation_nm",
    "time": "time_s",
    "t": "time_s",
}


def _ensure_artifacts() -> Path:
    return test_artifacts_dir(TestSlug.CASIMIR)


def _ensure_datasets() -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return DATASETS_DIR


def casimir_done_key(path: Path | str, *, data_root: Path | None = None) -> str:
    """Stable ledger key: path relative to datasets/casimir when possible."""
    root = (data_root or DATASETS_DIR).resolve()
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def casimir_batch_status(
    directory: str | Path,
    *,
    done_file: Path | str = DEFAULT_DONE_FILE,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Summarize how many tabular files are scanned, pending, or complete."""
    directory = Path(directory)
    root = data_root or DATASETS_DIR
    all_files = list_casimir_data_files(directory)
    return batch_status(
        all_files,
        done_file,
        key_fn=lambda file_path: casimir_done_key(file_path, data_root=root),
    )


def _try_tav_harmonic_check(residuals: np.ndarray) -> dict[str, Any] | None:
    try:
        from tav_shared.tav_resonance_bootstrap import ensure_tav_resonance_importable

        ensure_tav_resonance_importable()
        from tav_resonance import analyze_tav_harmonics
    except ImportError:
        return None
    freqs, power, peaks, detected = analyze_tav_harmonics(
        np.asarray(residuals, dtype=float),
        phase_tolerance=0.05,
        min_peaks=1,
    )
    return {
        "kind": "detrended_harmonic_residual",
        "note": "High-pass / trend-removed harmonic check; NOT Delta_E (Lifshitz residual).",
        "harmonic_peaks": peaks,
        "tav_resonance_detected": detected,
        "n_freq_bins": int(freqs.size),
    }


def download_file(url: str, save_path: str | Path, *, chunk_size: int = 8192) -> Path:
    """Download a file with progress."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {url} ...")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    downloaded = 0
    with open(save_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end="", flush=True)
    print(f"\nSaved to {save_path}")
    return save_path


def extract_zip(zip_path: str | Path, extract_to: str | Path) -> list[str]:
    """Extract ZIP and return list of tabular data files (CSV, TSV, TXT, DAT)."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    tabular_ext = {".csv", ".tsv", ".txt", ".dat", ".lst"}
    data_files: list[str] = []
    for root, _, files in os.walk(extract_to):
        for name in files:
            if Path(name).suffix.lower() in tabular_ext:
                data_files.append(os.path.join(root, name))
    print(f"Found {len(data_files)} tabular files.")
    return data_files


def _read_tabular_file(file_path: Path) -> pd.DataFrame:
    """Load CSV/TSV/whitespace tables; magnetic-fluid GitHub files are tab-separated."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".tsv", ".txt", ".dat", ".lst"}:
        try:
            return pd.read_csv(file_path, sep=r"\s+", engine="python")
        except pd.errors.ParserError:
            return pd.read_csv(file_path, sep="\t")
    # Extensionless numeric tables (GitHub Fig2/Fig34)
    try:
        return pd.read_csv(file_path, sep="\t", header=None)
    except pd.errors.ParserError:
        return pd.read_csv(file_path, sep=r"\s+", engine="python", header=None)


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load tabular Casimir data into DataFrame and standardize common column names."""
    file_path = Path(file_path).resolve()
    df = _read_tabular_file(file_path)
    df = standardize_dielectric_columns(df, file_path)

    print(f"Loaded {file_path} with shape {df.shape} and columns: {list(df.columns)}")
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    return df


def download_github_magnetic_fluid(
    *,
    subdirs: tuple[str, ...] = ("Fig2", "Fig34"),
    force: bool = False,
) -> list[Path]:
    """Download magnetic-fluid Casimir prediction spectra from GitHub."""
    out_root = _ensure_datasets() / "github_magnetic_fluid"
    out_root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for subdir in subdirs:
        api_url = f"{GITHUB_MF_API}/{subdir}?ref=main"
        print(f"Listing GitHub {subdir} ...")
        response = requests.get(api_url, timeout=60)
        response.raise_for_status()
        entries = response.json()
        if not isinstance(entries, list):
            continue

        target_dir = out_root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        skip_tokens = ("readme", "mathematica", "license", ".md", ".m", ".nb", ".pdf")
        for entry in entries:
            if entry.get("type") != "file":
                continue
            name = entry.get("name", "")
            lower = name.lower()
            if any(token in lower for token in skip_tokens):
                continue
            dest = target_dir / name
            if dest.is_file() and not force:
                saved.append(dest)
                continue
            raw_url = f"{GITHUB_MF_RAW}/{subdir}/{name}"
            download_file(raw_url, dest)
            saved.append(dest)

    print(f"GitHub magnetic-fluid cache: {len(saved)} files under {out_root}")
    return saved


def list_casimir_data_files(directory: str | Path) -> list[Path]:
    """Return sorted tabular files under a directory tree."""
    directory = Path(directory)
    tabular_ext = {".csv", ".tsv", ".txt", ".dat", ".lst", ""}
    skip_names = {"readme", "license", "done.txt"}
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file()
        and (p.suffix.lower() in tabular_ext or p.suffix == "")
        and p.name.lower() not in skip_names
        and p.suffix.lower() not in {".md", ".pdf", ".m", ".nb", ".zip"}
    ]
    return sorted(files)


_FIXTURE_FILENAMES = frozenset({"tsb_falsification_mock.csv"})


def discover_local_casimir_files(*, include_fixture: bool = False) -> list[Path]:
    """
    Find already-downloaded Casimir tables under ``datasets/casimir/``.

    Prefers GitHub magnetic-fluid Fig2/Fig34, then a Zenodo extract, then any
    other cached tables. The synthetic falsification fixture is omitted unless
    ``include_fixture`` is set. No user path is required.
    """
    _ensure_datasets()
    roots = (
        DATASETS_DIR / "github_magnetic_fluid" / "Fig2",
        DATASETS_DIR / "github_magnetic_fluid" / "Fig34",
        DATASETS_DIR / "github_magnetic_fluid",
        DATASETS_DIR / "casimir_drums_extracted",
        DATASETS_DIR,
    )
    seen: set[Path] = set()
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in list_casimir_data_files(root):
            resolved = path.resolve()
            if resolved in seen:
                continue
            if path.name in _FIXTURE_FILENAMES and not include_fixture:
                continue
            seen.add(resolved)
            found.append(path)
    return found


def resolve_local_casimir_file(local_csv: str | Path | None = None) -> Path:
    """Return an explicit path, or the first auto-discovered downloaded table."""
    if local_csv:
        path = Path(local_csv).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Casimir file not found: {path}")
        return path
    found = discover_local_casimir_files()
    if not found:
        raise FileNotFoundError(
            f"No downloaded Casimir tables under {DATASETS_DIR}. "
            "Run 'Download GitHub Magnetic-Fluid Data' first."
        )
    return found[0]


def _seven_phase_positions(x: np.ndarray) -> np.ndarray:
    """Expected 7-phase clockwork marker positions (k/7 of parameter span)."""
    if len(x) < 2:
        return np.array([])
    x_min, x_max = float(np.min(x)), float(np.max(x))
    span = x_max - x_min
    if span <= 0:
        return np.array([])
    return np.array([x_min + span * k / TAU_RESONANCE_PERIOD for k in range(1, int(TAU_RESONANCE_PERIOD))])


def _normalized_autocorrelation(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return lags and normalized autocorrelation of a 1D series."""
    y0 = np.asarray(y, dtype=float)
    y0 = y0 - np.mean(y0)
    if len(y0) < 4 or np.std(y0) < 1e-12:
        return np.arange(len(y0)), np.zeros(len(y0))
    ac = np.correlate(y0, y0, mode="full")
    ac = ac[ac.size // 2 :]
    if ac[0] > 0:
        ac = ac / ac[0]
    lags = np.arange(ac.size)
    return lags, ac


def _detect_changepoints(y: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    """Change-point detection on sorted sweep (ruptures with derivative fallback)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 8:
        return {"method": "none", "locations": [], "n_breaks": 0}

    try:
        import ruptures as rpt

        signal_2d = y.reshape(-1, 1)
        algo = rpt.Pelt(model="l2", min_size=3).fit(signal_2d)
        bkps = [b for b in algo.predict(pen=3.0) if b < len(y)]
        locations = [float(x[max(0, idx - 1)]) for idx in bkps if idx > 0]
        return {
            "method": "ruptures_pelt",
            "locations": locations[:10],
            "n_breaks": len(locations),
            "note": "Hierarchical level/slope transitions (ruptures PELT)",
        }
    except ImportError:
        dy = np.diff(y)
        threshold = np.std(dy) * 2.5
        idx = np.where(np.abs(dy) > threshold)[0]
        locations = [float(x[i + 1]) for i in idx[:10]]
        return {
            "method": "derivative_fallback",
            "locations": locations,
            "n_breaks": len(idx),
            "note": "Derivative-threshold change points (ruptures not installed)",
        }


def _auto_select_columns(df: pd.DataFrame) -> tuple[str, str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in dataset")

    param_col = next(
        (
            c
            for c in numeric_cols
            if any(
                token in c.lower()
                for token in (
                    "magnetic",
                    "matsubara",
                    "frequency",
                    "field_mt",
                    "field",
                    "separation",
                    "distance",
                    "time_s",
                    "time",
                )
            )
            or c.lower() in {"b", "d", "sep"}
        ),
        numeric_cols[0],
    )
    value_candidates = [c for c in numeric_cols if c != param_col]
    if VALUE_NAME in value_candidates:
        value_col = VALUE_NAME
    else:
        value_col = next(
            (
                c
                for c in value_candidates
                if any(
                    token in c.lower()
                    for token in ("force", "energy", "pressure", "signal", "dielectric", "response")
                )
            ),
            value_candidates[0] if value_candidates else numeric_cols[0],
        )
    return param_col, value_col


def detect_tau_resonances(
    df: pd.DataFrame,
    param_col: str | None = None,
    value_col: str | None = None,
    *,
    verbose: bool = True,
    apply_null_gate: bool = True,
    blind: bool | None = None,
    fast: bool = False,
    one_seventh_power_ratio: float | None = None,
    seven_cycle_power_ratio: float | None = None,
    seven_cycle_mf_threshold: float | None = None,
    search_bound_energy: bool | None = None,
    bound_energy_pass: bool = False,
) -> dict[str, Any]:
    """
    Core search for TSB tau resonance signatures.

    Looks for periodic 1/7 harmonics, discrete steps, hysteresis, and log-scale
    modulations. Dielectric-vs-frequency tables are not treated as
    force-vs-separation (no d^{-4} claim). Optional second pass on
    Delta_E = E_obs - E_Lifshitz is labeled observable="bound_energy_residual".
    """
    from menus.particle.casimir.blind_mode import is_blind

    if blind is None:
        blind = is_blind()
    if blind:
        from menus.particle.casimir.blind_detectors import detect_generic_signatures

        return detect_generic_signatures(
            df, param_col=param_col, value_col=value_col, verbose=verbose
        )

    results: dict[str, Any] = {
        "detected_features": [],
        "param_col": None,
        "value_col": None,
        "diagnostics": {},
        "analysis_mode": "informed",
    }

    if param_col is None or value_col is None:
        param_col, value_col = _auto_select_columns(df)
        if verbose:
            print(f"Auto-selected param={param_col}, value={value_col}")

    results["param_col"] = param_col
    results["value_col"] = value_col

    from menus.particle.casimir.diagnostics import (
        classify_casimir_table,
        print_casimir_load_context,
    )

    obs_meta = classify_casimir_table(df, param_col, value_col)
    results["observable_kind"] = obs_meta["observable_kind"]
    results["n_aux_dielectric_channels"] = obs_meta["n_aux_dielectric_channels"]
    results["applicable_tests"] = obs_meta["applicable_tests"]
    if bound_energy_pass:
        results["observable"] = "bound_energy_residual"
        results["observable_kind"] = "residual"
    if verbose:
        print_casimir_load_context(
            observable_kind_value=str(results["observable_kind"]),
            n_aux=int(obs_meta["n_aux_dielectric_channels"]),
            applicable=list(obs_meta["applicable_tests"]),
        )

    x = df[param_col].values.astype(float)
    y = df[value_col].values.astype(float)
    x_clean, y_clean = clean_xy(x, y)
    param_span = float(np.max(x_clean) - np.min(x_clean)) if len(x_clean) else 0.0
    seven_phase_x = _seven_phase_positions(x_clean)
    results["diagnostics"]["seven_phase_positions"] = seven_phase_x.tolist()

    trend_fit = fit_smooth_trend(x_clean, y_clean)
    residuals = np.asarray(trend_fit["residuals"], dtype=float)
    results["diagnostics"]["trend"] = {
        "model": trend_fit["model"],
        "rmse": trend_fit.get("rmse"),
        "params": trend_fit.get("params") or {},
        "trend": np.asarray(trend_fit["trend"], dtype=float).tolist(),
        "residuals": residuals.tolist(),
    }

    spec = residual_periodogram_significance(
        residuals, x_clean, n_bootstrap=0 if fast else 200
    )
    results["diagnostics"]["freqs"] = spec.get("freqs") or []
    results["diagnostics"]["power"] = spec.get("power") or []
    results["diagnostics"]["residual_periodogram"] = {
        k: spec[k]
        for k in spec
        if k not in {"freqs", "power"}
    }

    p_clock = float(spec.get("p_value") or 1.0)
    results["detected_features"].append(
        {
            "type": "seven_phase_clockwork",
            "p_value": p_clock,
            "p_local": spec.get("p_local"),
            "p_fisher_g": spec.get("p_fisher_g"),
            "p_bootstrap": spec.get("p_bootstrap"),
            "expected_freq_1_over_7": spec.get("expected_frequency"),
            "observed_frequency": spec.get("observed_frequency"),
            "power_at_one_seventh": spec.get("power_at_one_seventh"),
            "power_at_one_seventh_proxy": spec.get("power_at_one_seventh_proxy"),
            "local_continuum": spec.get("local_continuum"),
            "snr_local": spec.get("snr_local"),
            "trend_model": trend_fit["model"],
            "note": "1/7 residual periodogram power vs local continuum (after trend removal)",
        }
    )

    if len(residuals) > 14:
        lags, ac = _normalized_autocorrelation(residuals)
        results["diagnostics"]["autocorr_lags"] = lags[: min(80, len(lags))].tolist()
        results["diagnostics"]["autocorr"] = ac[: min(80, len(ac))].tolist()

    step = residual_step_pvalue(residuals)
    step_locations = (
        x_clean[step["indices"] + 1] if step["n_steps"] and len(x_clean) > 1 else np.array([])
    )
    tol = param_span / 14.0 if param_span > 0 else 0.0
    align = random_placement_pvalue(
        step_locations,
        seven_phase_x,
        x_min=float(np.min(x_clean)) if len(x_clean) else 0.0,
        x_max=float(np.max(x_clean)) if len(x_clean) else 1.0,
        tolerance=tol,
    )
    p_steps = float(step["p_value"])
    p_align = float(align["p_value"])
    results["detected_features"].append(
        {
            "type": "discrete_steps_thresholds",
            "p_value": p_steps,
            "p_alignment": p_align,
            "locations": step_locations[:10].tolist() if hasattr(step_locations, "tolist") else [],
            "n_steps": step["n_steps"],
            "aligned_with_seven_phase": align["k_obs"],
            "alignment_null": "uniform random placement",
            "note": "Residual |Δ| exceedances; alignment p-value vs random 7-phase placement",
        }
    )

    hyst = hysteresis_pvalue(residuals, n_perm=40 if fast else 400)
    p_hyst = float(hyst["p_value"])
    results["detected_features"].append(
        {
            "type": "hysteresis_asymmetry",
            "p_value": p_hyst,
            "magnitude": hyst.get("magnitude"),
            "n_perm": hyst.get("n_perm"),
            "note": "Permutation p-value of residual up/down asymmetry",
        }
    )

    cp = pelt_changepoints_with_bootstrap(
        residuals, x_clean, n_bootstrap=0 if fast else 40
    )
    p_cp = float(cp.get("p_value") or 1.0)
    results["diagnostics"]["changepoints"] = cp
    results["detected_features"].append(
        {
            "type": "changepoints",
            "p_value": p_cp,
            "locations": cp.get("locations") or [],
            "n_breaks": cp.get("n_breaks") or 0,
            "method": cp.get("method"),
            "penalty": cp.get("penalty"),
            "penalty_formula": cp.get("penalty_formula"),
            "bootstrap_intervals": cp.get("bootstrap_intervals") or [],
            "note": cp.get("note") or "PELT on residuals with bootstrap location CI",
        }
    )

    tav_lib = _try_tav_harmonic_check(residuals) if len(residuals) > 12 else None
    results["detrended_harmonic_residual"] = tav_lib
    results["diagnostics"]["detrended_harmonic_residual"] = tav_lib
    p_tav = p_clock
    alpha = p_threshold()
    primary_ratio = (
        float(one_seventh_power_ratio)
        if one_seventh_power_ratio is not None
        else one_seventh_power_ratio_threshold()
    )
    power_proxy = float(spec.get("power_at_one_seventh_proxy") or 0.0)
    primary_power_ok = power_proxy >= primary_ratio
    p_raw = [p_clock, p_steps, p_align, p_hyst, p_cp]
    p_holm = holm_adjust(p_raw)
    p_tav_adj = float(p_holm[0])
    holm_ok = bool(p_tav_adj < alpha)
    tav_detected = holm_ok and primary_power_ok
    clock_primary = holm_ok and primary_power_ok
    results["tav_harmonics"] = {
        **(tav_lib or {}),
        "tav_resonance_detected": tav_detected,
        "p_value": p_tav,
        "p_value_holm": p_tav_adj,
        "p_threshold": alpha,
        "power_at_one_seventh_proxy": power_proxy,
        "one_seventh_power_ratio_threshold": primary_ratio,
        "power_gate_passed": primary_power_ok,
        "multiple_testing": "holm_bonferroni",
        "n_tests": len(p_raw),
        "trend_model": trend_fit["model"],
        "residual_power_at_one_seventh": spec.get("power_at_one_seventh"),
        "local_continuum": spec.get("local_continuum"),
    }

    holm_by_type = {
        "seven_phase_clockwork": p_holm[0],
        "discrete_steps_thresholds": p_holm[1],
        "seven_phase_alignment": p_holm[2],
        "hysteresis_asymmetry": p_holm[3],
        "changepoints": p_holm[4],
    }
    for feat in results["detected_features"]:
        if feat["type"] in holm_by_type:
            feat["p_value_holm"] = holm_by_type[feat["type"]]

    results["diagnostics"]["p_values"] = {
        "raw": {
            "seven_phase_clockwork": p_clock,
            "discrete_steps_thresholds": p_steps,
            "seven_phase_alignment": p_align,
            "hysteresis_asymmetry": p_hyst,
            "changepoints": p_cp,
        },
        "holm": holm_by_type,
        "alpha": alpha,
    }

    raw_summary: dict[str, bool] = {}
    for feat in results["detected_features"]:
        p_adj = feat.get("p_value_holm", feat.get("p_value", 1.0))
        if feat["type"] == "seven_phase_clockwork":
            feat["power_at_one_seventh_proxy"] = power_proxy
            feat["one_seventh_power_ratio_threshold"] = primary_ratio
            feat["power_gate_passed"] = primary_power_ok
            raw_summary[feat["type"]] = clock_primary
        else:
            raw_summary[feat["type"]] = bool(float(p_adj) < alpha)
    raw_summary["tav_resonance_detected"] = tav_detected
    results["raw_signature_summary"] = raw_summary
    results["signature_summary"] = dict(raw_summary)
    results["diagnostics"]["raw_signature_summary"] = raw_summary

    # Independent 7-cycle detector. Runs only after primary flags are frozen.
    mf = detect_seven_cycle_matched_filter(
        x_clean,
        residuals,
        threshold=seven_cycle_mf_threshold,
    )
    results["seven_cycle_secondary"] = mf
    results["diagnostics"]["seven_cycle_secondary"] = mf
    if apply_null_gate:
        from menus.particle.casimir.null_hypothesis import apply_null_gate_to_detection

        apply_null_gate_to_detection(results, verbose=verbose)

    if (
        not bound_energy_pass
        and int(obs_meta.get("n_dielectric_columns") or 0) >= 2
    ):
        from menus.particle.casimir.test_framework import test_anisotropy_channels

        aniso = test_anisotropy_channels(df, primary=value_col, x=x_clean)
        results["anisotropy_channels"] = aniso
        results["diagnostics"]["anisotropy_channels"] = aniso
        results["detected_features"].append(
            {
                "type": "anisotropy_channels",
                "p_value": aniso.get("p_value"),
                "control_failed": aniso.get("control_failed"),
                "spread": aniso.get("spread"),
                "note": "Aux dielectric spread vs eps_pri; negative control broadcasts eps_pri",
            }
        )
        if aniso.get("status") != "not_applicable":
            results["signature_summary"]["anisotropy_channels"] = bool(
                aniso.get("evidence_for_TSB")
            )

    if search_bound_energy is None:
        search_bound_energy = (
            (not bound_energy_pass)
            and obs_meta["observable_kind"] in {"force", "energy"}
            and obs_meta.get("x_kind") == "separation"
        )
    if search_bound_energy and not bound_energy_pass:
        from menus.particle.casimir.bound_energy import estimate_bound_energy

        be = estimate_bound_energy(
            df,
            param_col=param_col,
            value_col=value_col,
            data_label=str(results.get("data_label") or ""),
        )
        results["bound_energy"] = {
            k: be.get(k)
            for k in (
                "identity_control_ok",
                "E_bind_claimed",
                "floor_p",
                "E_bind",
                "quadrature_rel",
                "lifshitz_approximations",
                "T",
            )
        }
        d_arr = np.asarray(be.get("d"), dtype=float) if be.get("d") is not None else np.array([])
        delta = (
            np.asarray(be.get("Delta_E"), dtype=float)
            if be.get("Delta_E") is not None
            else np.array([])
        )
        if d_arr.size and delta.size and np.any(np.isfinite(delta)):
            residual_df = pd.DataFrame({"d": d_arr, "Delta_E": delta})
            nested = detect_tau_resonances(
                residual_df,
                param_col="d",
                value_col="Delta_E",
                verbose=False,
                apply_null_gate=False,
                bound_energy_pass=True,
                search_bound_energy=False,
                fast=fast,
            )
            nested["observable"] = "bound_energy_residual"
            results["bound_energy_residual_scan"] = nested
        results["diagnostics"]["bound_energy"] = results.get("bound_energy")
        results["diagnostics"]["bound_energy_residual_scan"] = results.get(
            "bound_energy_residual_scan"
        )

    return results


def _expected_tsb_modulation(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Overlay: sinusoidal 1/7 clockwork modulation scaled to data."""
    if len(x) < 4:
        return None
    span = float(np.max(x) - np.min(x))
    if span <= 0:
        return None
    amp = 0.5 * float(np.std(y))
    offset = float(np.mean(y))
    phase = 2 * np.pi * (x - np.min(x)) / span * TAU_RESONANCE_PERIOD
    return offset + amp * np.sin(phase)


def plot_and_save(
    df: pd.DataFrame,
    param_col: str,
    value_col: str,
    results: dict[str, Any],
    prefix: str = "tau_search",
    *,
    output_dir: Path | None = None,
) -> str:
    """Generate diagnostic plots comparing data to expected TSB tau signatures."""
    out_dir = output_dir or _ensure_artifacts()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    diag = results.get("diagnostics", {})

    mask = df[param_col].notna() & df[value_col].notna()
    x = df.loc[mask, param_col].values.astype(float)
    y = df.loc[mask, value_col].values.astype(float)
    sort_idx = np.argsort(x) if len(x) else np.array([], dtype=int)
    x_s, y_s = x[sort_idx], y[sort_idx]

    # Raw data + fitted trend + 7-phase markers
    ax = axes[0, 0]
    ax.plot(x_s, y_s, "b.-", alpha=0.75, label="data")
    trend = np.asarray((diag.get("trend") or {}).get("trend") or [])
    if trend.size == x_s.size:
        ax.plot(x_s, trend, "k--", alpha=0.8, label=f"trend ({(diag.get('trend') or {}).get('model')})")
    for mark in diag.get("seven_phase_positions", []):
        ax.axvline(mark, color="orange", linestyle=":", alpha=0.6, linewidth=0.9)
    ax.set_xlabel(param_col)
    ax.set_ylabel(value_col)
    kind = str(results.get("observable_kind") or "")
    trend_title = (
        "Data + smooth trend (dielectric vs frequency; not a d^{-4} force law)"
        if kind == "dielectric"
        else "Data + smooth trend"
    )
    ax.set_title(trend_title)
    ax.legend(fontsize=8)
    ax.grid(True)

    # Steps / change-points on derivative
    ax = axes[0, 1]
    if len(y_s) > 1:
        dy = np.diff(y_s)
        ax.plot(x_s[1:], dy, "r.-", alpha=0.8, label="dy/dparam")
        for feat in results.get("detected_features", []):
            if feat["type"] in {
                "discrete_steps_thresholds",
                "residual_steps",
                "changepoints",
            }:
                for loc in feat.get("locations", [])[:6]:
                    ax.axvline(loc, color="purple", linestyle="--", alpha=0.5)
        ax.set_title("Steps & change-points")
        ax.grid(True)
    else:
        ax.set_axis_off()

    # Residuals after trend removal
    ax = axes[0, 2]
    resid = np.asarray((diag.get("trend") or {}).get("residuals") or [])
    if resid.size == x_s.size:
        ax.plot(x_s, resid, "g.-", alpha=0.7)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_title("Residuals (data − trend)")
        ax.set_xlabel(param_col)
        ax.grid(True)
    else:
        ax.plot(x_s, y_s, "b.-", alpha=0.6)
        ax.set_title("Series")
        ax.grid(True)

    # Residual periodogram
    ax = axes[1, 0]
    spec = diag.get("residual_periodogram") or {}
    freqs = np.asarray(diag.get("freqs") or [])
    power = np.asarray(diag.get("power") or [])
    blind = str(results.get("analysis_mode") or "") == "blind"
    if freqs.size and power.size:
        ax.semilogy(freqs, power, color="steelblue")
        if blind:
            f_peak = spec.get("strongest_frequency")
            if f_peak:
                ax.axvline(float(f_peak), color="purple", linestyle="--", label="strongest peak")
            ax.set_title(
                "Residual periodogram (blind)  "
                f"P={spec.get('strongest_period')}  p={spec.get('p_value'):.3g}"
            )
        else:
            f_exp = spec.get("expected_frequency") or expected_one_seventh_frequency(x_s)
            ax.axvline(f_exp, color="green", linestyle="--", alpha=0.9, label="expected 1/7")
            th = results.get("tav_harmonics") or {}
            ax.set_title(
                "Residual periodogram  "
                f"p={th.get('p_value', spec.get('p_value')):.3g}  "
                f"p_Holm={th.get('p_value_holm', float('nan')):.3g}"
            )
        ax.axhline(
            float(spec.get("local_continuum") or 0) or np.nanmedian(power),
            color="orange",
            linestyle=":",
            alpha=0.8,
            label="local continuum",
        )
        ax.set_xlabel("frequency")
        ax.legend(fontsize=8)
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()

    # Autocorrelation
    ax = axes[1, 1]
    ac_lags = np.asarray(diag.get("autocorr_lags") or [])
    ac_vals = np.asarray(diag.get("autocorr") or [])
    if ac_lags.size and ac_vals.size:
        ax.plot(ac_lags, ac_vals, "k-")
        if not blind:
            n = len(y_s)
            for k in range(1, int(TAU_RESONANCE_PERIOD)):
                lag = int(round(n * k / TAU_RESONANCE_PERIOD))
                if 0 < lag < len(ac_vals):
                    ax.axvline(lag, color="magenta", linestyle=":", alpha=0.5)
            ax.set_title("Autocorrelation (1/7 lag markers)")
        else:
            ax.set_title("Autocorrelation (no preferred lag)")
        ax.set_xlabel("lag")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()

    # Summary panel
    ax = axes[1, 2]
    summary_lines = [f"Param: {param_col} | Value: {value_col}", ""]
    if blind:
        summary_lines.append("Mode: BLIND (no preferred period)")
    trend_info = diag.get("trend") or {}
    summary_lines.append(f"Trend: {trend_info.get('model')} RMSE={trend_info.get('rmse')}")
    for feat in results.get("detected_features", []):
        pval = feat.get("p_value_holm", feat.get("p_value"))
        summary_lines.append(f"• {feat['type']}  p={pval}")
    if not blind:
        th = results.get("tav_harmonics") or {}
        sec = results.get("seven_cycle_secondary") or {}
        summary_lines.append(
            f"• primary 7-cycle @{th.get('one_seventh_power_ratio_threshold')}: "
            f"clock={results.get('raw_signature_summary', {}).get('seven_phase_clockwork')} "
            f"tav={th.get('tav_resonance_detected')}"
        )
        summary_lines.append(
            f"• secondary MF period-{sec.get('period')}: "
            f"score={sec.get('score')} thr={sec.get('threshold')} "
            f"hit={sec.get('detected')}"
        )
    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        transform=ax.transAxes,
    )
    ax.set_title("Signature summary")
    ax.set_axis_off()

    fig.suptitle("TSB Casimir Tau Resonance Search", fontsize=12)
    fig.tight_layout()
    plot_path = casimir_output_path(
        compose_dataset_slug(prefix),
        "plot",
        "png",
        output_prefix=prefix,
    )
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    results.setdefault("plots", []).append(str(plot_path))
    print(f"Plot saved: {plot_path}")
    return str(plot_path)


def generate_mock_casimir_data(n_points: int = 140, seed: int = 42) -> pd.DataFrame:
    """Synthetic magnetic-field Casimir sweep with 1/7 ripple and discrete steps."""
    rng = np.random.default_rng(seed)
    b_field = np.linspace(0.0, 70.0, n_points)
    base = -12.0 + 0.08 * b_field
    ripple = 0.35 * np.sin(2 * np.pi * b_field / TAU_RESONANCE_PERIOD)
    steps = np.zeros_like(b_field)
    for threshold in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]:
        steps += np.where(b_field >= threshold, 0.15, 0.0)
    noise = rng.normal(0.0, 0.05, size=n_points)
    force = base + ripple + steps + noise
    return pd.DataFrame({"magnetic_field_mT": b_field, "force_pN": force})


@dataclass
class CasimirScanResult:
    data_label: str
    file_path: str | None = None
    param_col: str = ""
    value_col: str = ""
    detected_features: list[dict[str, Any]] = field(default_factory=list)
    signature_summary: dict[str, bool] = field(default_factory=dict)
    tav_harmonics: dict[str, Any] | None = None
    seven_cycle_secondary: dict[str, Any] | None = None
    plot_path: str | None = None
    report_path: str | None = None
    text_report_path: str | None = None
    raw_series_integrity: dict[str, Any] = field(default_factory=dict)
    feature_vector: list[float] = field(default_factory=list)
    material_id: str = ""
    analysis_mode: str = "informed"
    observable_kind: str = ""
    n_aux_dielectric_channels: int = 0
    applicable_tests: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [
            "TSB Casimir tau resonance scan",
            f"Dataset: {self.data_label}",
        ]
        if self.file_path:
            lines.append(f"File: {self.file_path}")
        if self.param_col:
            lines.append(f"Columns: {self.param_col} → {self.value_col}")
        lines.append(f"Signatures tested: {len(self.detected_features)}")
        for feat in self.detected_features:
            pval = feat.get("p_value_holm", feat.get("p_value"))
            lines.append(f"  • {feat['type']}: p={pval}  {feat.get('note', '')}")
        claimed = [k for k, v in (self.signature_summary or {}).items() if v]
        lines.append(f"Claimed flags: {', '.join(claimed) if claimed else '(none)'}")
        if self.tav_harmonics:
            lines.append(
                "primary 7-cycle: "
                f"tav={self.tav_harmonics.get('tav_resonance_detected')} "
                f"p_Holm={self.tav_harmonics.get('p_value_holm')} "
                f"α={self.tav_harmonics.get('p_threshold')} "
                f"I(f)/max(I)={self.tav_harmonics.get('power_at_one_seventh_proxy')} "
                f"gate={self.tav_harmonics.get('one_seventh_power_ratio_threshold')}"
            )
        if self.seven_cycle_secondary:
            sec = self.seven_cycle_secondary
            lines.append(
                "secondary 7-cycle matched filter: "
                f"detected={sec.get('detected')} "
                f"score={sec.get('score')} "
                f"period={sec.get('period')} "
                f"thr={sec.get('threshold')} "
                f"(independent; not a primary claim)"
            )
            fpr = sec.get("documented_false_positive_rate") or {}
            if fpr:
                lines.append(
                    "secondary documented max FPR: "
                    f"clock={fpr.get('seven_phase_clockwork')} "
                    f"tav={fpr.get('tav_resonance_detected')}"
                )
        if self.plot_path:
            lines.append(f"Plot: {self.plot_path}")
        if self.report_path:
            lines.append(f"JSON report: {self.report_path}")
        if self.text_report_path:
            lines.append(f"Text report: {self.text_report_path}")
        return lines


def _write_report(result: CasimirScanResult, output_prefix: str, diagnostics: dict | None = None) -> tuple[str, str]:
    dataset = compose_dataset_slug(result.data_label or output_prefix)
    report_path = casimir_output_path(dataset, "report", "json", output_prefix=output_prefix)
    text_path = casimir_output_path(dataset, "summary", "txt", output_prefix=output_prefix)
    payload = {
        "data_label": result.data_label,
        "file_path": result.file_path,
        "param_col": result.param_col,
        "value_col": result.value_col,
        "observable_kind": result.observable_kind,
        "n_aux_dielectric_channels": result.n_aux_dielectric_channels,
        "applicable_tests": result.applicable_tests,
        "raw_series_sha256": (result.raw_series_integrity or {}).get("sha256"),
        "detected_features": result.detected_features,
        "signature_summary": result.signature_summary,
        "tav_harmonics": result.tav_harmonics,
        "seven_cycle_secondary": result.seven_cycle_secondary,
        "plot_path": result.plot_path,
        "diagnostics": diagnostics or {},
        "raw_series_integrity": result.raw_series_integrity or {},
        "detrended_harmonic_residual": (diagnostics or {}).get("detrended_harmonic_residual"),
        "null_gate": (diagnostics or {}).get("null_gate") or {},
        "material_id": result.material_id,
        "feature_vector": result.feature_vector,
        "analysis_mode": result.analysis_mode,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text_path.write_text("\n".join(result.summary_lines()) + "\n", encoding="utf-8")
    return str(report_path), str(text_path)


def run_casimir_scan(
    *,
    data_source: str = "mock",
    local_csv: str | Path | None = None,
    param_col: str | None = None,
    value_col: str | None = None,
    dataset_key: str = "zenodo_superconducting_casimir",
    download: bool = False,
    extract_dir: str | Path | None = None,
    plot: bool = True,
    output_prefix: str = "tsb_casimir",
    blind: bool | None = None,
) -> CasimirScanResult:
    """High-level Casimir tau-resonance scan entry point."""
    from menus.particle.casimir.blind_mode import apply_blind_prefix, is_blind, set_blind

    if blind is None:
        blind = is_blind()
    if blind:
        set_blind(True)
        output_prefix = apply_blind_prefix(output_prefix)
    owns_series = current_casimir_series() is None
    if owns_series:
        start_casimir_artifact_series(output_prefix)
    try:
        return _run_casimir_scan_body(
            data_source=data_source,
            local_csv=local_csv,
            param_col=param_col,
            value_col=value_col,
            dataset_key=dataset_key,
            download=download,
            extract_dir=extract_dir,
            plot=plot,
            output_prefix=output_prefix,
            blind=bool(blind),
        )
    finally:
        if owns_series:
            end_casimir_artifact_series()


def _run_casimir_scan_body(
    *,
    data_source: str,
    local_csv: str | Path | None,
    param_col: str | None,
    value_col: str | None,
    dataset_key: str,
    download: bool,
    extract_dir: str | Path | None,
    plot: bool,
    output_prefix: str,
    blind: bool = False,
) -> CasimirScanResult:
    data_source = (data_source or "mock").strip().lower()
    df: pd.DataFrame
    label: str
    file_path: str | None = None

    if data_source == "mock":
        df = generate_mock_casimir_data()
        label = "mock magnetic-field Casimir sweep"
    elif data_source in {"local", "csv", "file"}:
        resolved = resolve_local_casimir_file(local_csv)
        file_path = str(resolved)
        df = load_data(file_path)
        label = resolved.name
    elif data_source in {"zenodo", "download"}:
        meta = EXAMPLE_DATASETS.get(dataset_key, EXAMPLE_DATASETS["zenodo_superconducting_casimir"])
        ds_root = _ensure_datasets()
        zip_path = ds_root / meta["zip_name"]
        if download or not zip_path.is_file():
            download_file(meta["url"], zip_path)
        extract_to = Path(extract_dir) if extract_dir else ds_root / "casimir_drums_extracted"
        data_files = extract_zip(zip_path, extract_to)
        if not data_files:
            raise FileNotFoundError(f"No tabular files found under {extract_to}")
        file_path = data_files[0]
        df = load_data(file_path)
        label = f"zenodo extract ({Path(file_path).name})"
    elif data_source in {"github", "github_magnetic_fluid", "magnetic_fluid"}:
        cache = _ensure_datasets() / "github_magnetic_fluid"
        if download or not cache.is_dir() or not any(cache.rglob("*")):
            download_github_magnetic_fluid(force=download)
        candidates = list_casimir_data_files(cache)
        if not candidates:
            raise FileNotFoundError(f"No GitHub magnetic-fluid files under {cache}")
        if local_csv:
            file_path = str(Path(local_csv).resolve())
            if not Path(file_path).is_file():
                raise FileNotFoundError(file_path)
        else:
            file_path = str(candidates[0])
        df = load_data(file_path)
        label = f"github magnetic-fluid ({Path(file_path).name})"
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    analysis = detect_tau_resonances(
        df,
        param_col=param_col,
        value_col=value_col,
        blind=blind,
        apply_null_gate=not blind,
    )
    pcol = analysis["param_col"] or param_col or ""
    vcol = analysis["value_col"] or value_col or ""
    x_series = df[pcol].to_numpy(dtype=float) if pcol else np.asarray([])
    y_series = df[vcol].to_numpy(dtype=float) if vcol else np.asarray([])
    integrity = build_raw_series_integrity(
        x_series,
        y_series,
        file_path=file_path,
        data_label=label,
    )
    register_series_hash(integrity)
    from menus.particle.casimir.diagnostics import preregister_split, record_hash_ledger

    lock = preregister_split()
    ledger = record_hash_ledger(
        data_label=label,
        path=file_path,
        H=str(integrity.get("sha256") or ""),
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
    )
    analysis.setdefault("diagnostics", {})
    analysis["diagnostics"]["preregister"] = lock
    analysis["diagnostics"]["hash_ledger"] = ledger

    result = CasimirScanResult(
        data_label=label,
        file_path=file_path,
        param_col=pcol,
        value_col=vcol,
        detected_features=analysis.get("detected_features", []),
        signature_summary=analysis.get("signature_summary", {}),
        tav_harmonics=analysis.get("tav_harmonics"),
        seven_cycle_secondary=analysis.get("seven_cycle_secondary"),
        raw_series_integrity=integrity,
        material_id=infer_material_id(label, file_path),
        observable_kind=str(analysis.get("observable_kind") or ""),
        n_aux_dielectric_channels=int(analysis.get("n_aux_dielectric_channels") or 0),
        applicable_tests=list(analysis.get("applicable_tests") or []),
        feature_vector=build_signature_feature_vector(
            detected_features=analysis.get("detected_features") or [],
            diagnostics=analysis.get("diagnostics") or {},
            tav_harmonics=analysis.get("tav_harmonics"),
        ).tolist(),
        analysis_mode="blind" if blind else str(analysis.get("analysis_mode") or "informed"),
    )

    if plot and pcol and vcol:
        result.plot_path = plot_and_save(df, pcol, vcol, analysis, prefix=label)

    json_path, text_path = _write_report(result, output_prefix, diagnostics=analysis.get("diagnostics"))
    result.report_path = json_path
    result.text_report_path = text_path
    return result


def run_batch_casimir_scan(
    directory: str | Path,
    *,
    max_files: int = 10,
    plot: bool = False,
    output_prefix: str = "tsb_casimir_batch",
    done_file: Path | str = DEFAULT_DONE_FILE,
    force_rescan: bool = False,
    blind: bool | None = None,
) -> dict[str, Any]:
    """
    Scan tabular files under a directory, resuming from datasets/casimir/done.txt.

    Each run processes the next pending slice (up to ``max_files``); successful
    scans are appended to done.txt so the following run continues where the last
    one stopped. Set ``max_files`` to 0 to scan all pending files in one run.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    _ensure_datasets()
    all_files = list_casimir_data_files(directory)
    if not all_files:
        raise FileNotFoundError(f"No tabular files under {directory}")

    selected, status = select_batch_items(
        all_files,
        done_file,
        key_fn=lambda file_path: casimir_done_key(file_path),
        limit=max_files,
        force_rescan=force_rescan,
    )
    print(format_batch_banner("TSB Casimir", status))
    if not selected:
        if status["done_count"] >= status["catalog_total"]:
            print(
                f"[TSB Casimir] All {status['catalog_total']} files already scanned. "
                "Use force_rescan=yes to re-run from the start."
            )
        return {
            "summary_path": None,
            "summaries": {},
            "scanned": 0,
            "done_count": status["done_count"],
            "pending_count": 0,
            "catalog_total": status["catalog_total"],
        }

    from menus.particle.casimir.blind_mode import apply_blind_prefix, is_blind, set_blind

    if blind is None:
        blind = is_blind()
    if blind:
        set_blind(True)
        output_prefix = apply_blind_prefix(output_prefix)

    summaries: dict[str, str] = {}
    completed_keys: list[str] = []
    scan_records: list[dict[str, Any]] = []
    reset_seen_hashes()
    start_casimir_artifact_series(output_prefix)
    summary_path: str | None = None
    material_gate: dict[str, Any] | None = None
    try:
        for csv_path in selected:
            key = casimir_done_key(csv_path)
            try:
                result = run_casimir_scan(
                    data_source="local",
                    local_csv=csv_path,
                    plot=plot,
                    output_prefix=output_prefix,
                    blind=blind,
                )
                summaries[str(csv_path)] = (
                    f"{len(result.detected_features)} features"
                    + (
                        f", tav={result.tav_harmonics.get('tav_resonance_detected')}"
                        if result.tav_harmonics
                        else ""
                    )
                )
                completed_keys.append(key)
                scan_records.append(
                    {
                        "material_id": result.material_id,
                        "data_label": result.data_label,
                        "file_path": result.file_path,
                        "sha256": (result.raw_series_integrity or {}).get("sha256"),
                        "feature_vector": result.feature_vector,
                        "report_path": result.report_path,
                    }
                )
            except DuplicateRawSeriesError:
                raise
            except Exception as exc:
                summaries[str(csv_path)] = f"error: {exc}"

        if len(scan_records) >= 2:
            material_gate = enforce_material_gate(
                scan_records,
                threshold=material_distance_threshold(),
            )

        if completed_keys:
            append_done_entries(done_file, completed_keys)
            print(f"[TSB Casimir] Recorded {len(completed_keys)} file(s) in {done_file}")

        dest = casimir_output_path(
            "batch",
            "batch_summary",
            "json",
            output_prefix=output_prefix,
        )
        payload = {
            "directory": str(directory),
            "done_file": str(done_file),
            "catalog_total": status["catalog_total"],
            "done_count": status["done_count"] + len(completed_keys),
            "pending_before": status["pending_count"],
            "n_files": len(selected),
            "force_rescan": force_rescan,
            "summaries": summaries,
            "material_gate": material_gate,
            "multi_material_claim_allowed": bool(
                material_gate and material_gate.get("claim_allowed")
            ),
        }
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary_path = str(dest)
        print(f"Batch summary: {summary_path}")
    except MaterialSignatureCollisionError:
        print(
            "[CASIMIR MATERIAL GATE] Batch aborted — final summary tables not written."
        )
        raise
    finally:
        end_casimir_artifact_series()
    return {
        "summary_path": summary_path,
        "summaries": summaries,
        "scanned": len(completed_keys),
        "done_count": status["done_count"] + len(completed_keys),
        "pending_count": max(0, status["pending_count"] - len(completed_keys)),
        "catalog_total": status["catalog_total"],
        "material_gate": material_gate,
        "multi_material_claim_allowed": bool(
            material_gate and material_gate.get("claim_allowed")
        ),
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TSB Casimir tau resonance signature search")
    parser.add_argument(
        "--data",
        choices=["mock", "local", "zenodo", "github"],
        default="mock",
        help="Data source: mock, local file, Zenodo Casimir drums, or GitHub magnetic-fluid",
    )
    parser.add_argument(
        "--download-github",
        action="store_true",
        help="Fetch GitHub magnetic-fluid Fig2/Fig34 spectra and exit",
    )
    parser.add_argument("--local-csv", help="Path to local CSV when --data local")
    parser.add_argument("--param-col", help="Parameter column (x-axis)")
    parser.add_argument("--value-col", help="Observable column (y-axis)")
    parser.add_argument("--download", action="store_true", help="Force download Zenodo zip")
    parser.add_argument("--extract-dir", help="Extraction directory for Zenodo zip")
    parser.add_argument("--batch-dir", help="Scan all CSVs in directory and exit")
    parser.add_argument("--max-files", type=int, default=10, help="Max pending files for --batch-dir (0=all)")
    parser.add_argument(
        "--done-file",
        default=str(DEFAULT_DONE_FILE),
        help="Ledger of scanned files (default: datasets/casimir/done.txt)",
    )
    parser.add_argument(
        "--force-rescan",
        action="store_true",
        help="Ignore done.txt and scan from the first files again",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG output")
    parser.add_argument("--output-prefix", default="tsb_casimir", help="Artifact filename prefix")
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Blind analysis: no 1/7 prior; reports use a blind_ prefix",
    )
    parser.add_argument(
        "--lock-blind",
        action="store_true",
        help="Lock current blind reports so a comparison table may be written",
    )
    parser.add_argument(
        "--compare-blind",
        action="store_true",
        help="Write informed vs blind table (requires --lock-blind first)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)
    _ = args
    from superblock_mode import print_retired

    print_retired("TSB_CASIMIR", "scanner CLI")
    print(
        "[Superblock Residual Engine] Casimir Tau scoring CLI is retired. "
        "Tau/5D templates were not applied."
    )
    return 0

    if False and args.lock_blind:
        from menus.particle.casimir.blind_mode import lock_blind_results

        lock_blind_results()
        return 0
    if args.compare_blind:
        from menus.particle.casimir.blind_mode import write_blind_comparison_table

        write_blind_comparison_table()
        return 0

    if args.download_github:
        paths = download_github_magnetic_fluid(force=args.download)
        print(f"Downloaded {len(paths)} files.")
        return 0

    if args.batch_dir:
        run_batch_casimir_scan(
            args.batch_dir,
            max_files=args.max_files,
            plot=not args.no_plot,
            output_prefix=args.output_prefix,
            done_file=args.done_file,
            force_rescan=args.force_rescan,
            blind=bool(args.blind),
        )
        return 0

    result = run_casimir_scan(
        data_source=args.data,
        local_csv=args.local_csv,
        param_col=args.param_col,
        value_col=args.value_col,
        download=args.download,
        extract_dir=args.extract_dir,
        plot=not args.no_plot,
        output_prefix=args.output_prefix,
        blind=bool(args.blind),
    )
    print("\n".join(result.summary_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())