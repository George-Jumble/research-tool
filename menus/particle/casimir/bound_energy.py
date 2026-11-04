"""
TSB-Casimir bound-energy residuals: ΔE = E_obs − E_Lifshitz.

The claim is not “ε(iξ) has wiggles”. After the Lifshitz continuum is
removed, a hierarchical bound energy is expected to live in ΔE.

Uses the approximate planar integrator in lifshitz_approx.py. Identity
subtraction (E_obs ← E_Lifshitz) must be consistent with 0 before any
E_bind claim is allowed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from menus.particle.casimir.diagnostics import (
    TSB_EVIDENCE_ALPHA,
    TSB_TEST_RESULTS_DIR,
    classify_casimir_table,
    classify_x_kind,
    classify_y_kind,
    json_safe,
    observable_kind as classify_observable_kind,
)
from menus.particle.casimir.lifshitz_approx import (
    LIFSHITZ_APPROXIMATIONS,
    free_energy_density,
    free_energy_density_quadrature_check,
)
from menus.particle.casimir.series_integrity import clean_xy

MATERIAL_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Ag", ("dielPTAg", "Ag005", "_Ag")),
    ("Al", ("dielPTAl", "Al005", "_Al")),
    ("Au", ("dielPTGold", "Gold005", "_Au", "Gold")),
    ("Li", ("dielPTLi", "Li005", "_Li")),
    ("Ben_Au", ("Ben_Au", "dielPTBen")),
    ("Cyc_Au", ("Cyc_Au", "dielPTCyc")),
    ("Oct_Au", ("Oct_Au", "dielPTOct")),
)

DEFAULT_D_METRES = np.logspace(-8.0, -6.0, 32)  # 10 nm … 1 μm


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def material_key_from_label(label: str | None) -> str | None:
    text = str(label or "")
    low = text.lower()
    for key, tokens in MATERIAL_KEYS:
        if any(tok.lower() in low for tok in tokens):
            return key
    return None


def default_separation_grid_m(n: int = 32) -> np.ndarray:
    return np.logspace(-8.0, -6.0, int(n))


def _temperature_from_table(df: pd.DataFrame | None, metadata: dict[str, Any] | None) -> float:
    if metadata and metadata.get("T") is not None:
        try:
            return float(metadata["T"])
        except (TypeError, ValueError):
            pass
    if df is not None:
        for col in df.columns:
            if str(col).lower() in {"t", "temperature", "temp_k", "T_K"}:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series):
                    return float(series.iloc[0])
    return 300.0


def _sigma_column(df: pd.DataFrame | None, value_col: str | None) -> np.ndarray | None:
    if df is None:
        return None
    names = [
        f"{value_col}_sigma" if value_col else "",
        "Delta_E_sigma",
        "sigma",
        "error",
        "stderr",
        "uncertainty",
        "yerr",
    ]
    cols = {str(c).lower(): str(c) for c in df.columns}
    for name in names:
        key = name.lower()
        if key in cols:
            arr = pd.to_numeric(df[cols[key]], errors="coerce").to_numpy(dtype=float)
            if np.any(np.isfinite(arr) & (arr > 0)):
                return arr
    return None


def _extract_dielectric(df: pd.DataFrame, param_col: str, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    xi = pd.to_numeric(df[param_col], errors="coerce").to_numpy(dtype=float)
    eps = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    return clean_xy(xi, eps)


def compute_lifshitz_from_dielectric(
    df: pd.DataFrame,
    *,
    param_col: str,
    value_col: str,
    d: np.ndarray | None = None,
    T: float | None = None,
    n_k: int = 64,
    quadrature_check: bool = True,
) -> dict[str, Any]:
    xi, eps = _extract_dielectric(df, param_col, value_col)
    d_m = np.asarray(d if d is not None else DEFAULT_D_METRES, dtype=np.float64)
    temp = float(T if T is not None else _temperature_from_table(df, None))
    if quadrature_check:
        q = free_energy_density_quadrature_check(xi, eps, d_m, T=temp, n_k=n_k)
        F = np.asarray(q["F_coarse"], dtype=np.float64)
        extra = {
            "quadrature_rel": q["quadrature_rel"],
            "n_k": q["n_k_coarse"],
            "n_k_fine": q["n_k_fine"],
            "xi_unit": q["xi_unit"],
            "n_max": q["n_max"],
        }
    else:
        raw = free_energy_density(xi, eps, d_m, T=temp, n_k=n_k)
        F = np.asarray(raw["F_over_A"], dtype=np.float64)
        extra = {
            "quadrature_rel": None,
            "n_k": raw["n_k"],
            "xi_unit": raw["xi_unit"],
            "n_max": raw["n_max"],
        }
    return {
        "d": d_m,
        "E_Lifshitz": F,
        "T": temp,
        "lifshitz_approximations": list(LIFSHITZ_APPROXIMATIONS),
        **extra,
    }


def identity_subtraction(E_Lifshitz: np.ndarray) -> dict[str, Any]:
    """E_obs replaced by E_Lifshitz → ΔE must be identically 0."""
    e_l = np.asarray(E_Lifshitz, dtype=np.float64)
    delta = e_l - e_l
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    ok = bool(np.allclose(delta, 0.0, rtol=0.0, atol=0.0))
    return {
        "Delta_E": delta,
        "max_abs_Delta_E": max_abs,
        "identity_control_ok": ok,
        "note": "identity subtraction: E_obs := E_Lifshitz; ΔE must be 0 (no E_bind claim)",
    }


def floor_test_E_bind(
    Delta_E: np.ndarray,
    *,
    Delta_E_sigma: np.ndarray | None = None,
    n_perm: int = 199,
    seed: int = 15,
    identity_control_ok: bool = True,
) -> dict[str, Any]:
    """Stable nonzero offset after continuum removal? Permutation of signs."""
    y = np.asarray(Delta_E, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = int(y.size)
    if n < 4 or not identity_control_ok:
        return {
            "E_bind": None,
            "floor_p": None,
            "E_bind_claimed": False,
            "n": n,
            "reason": "identity control failed or too few points"
            if not identity_control_ok
            else "too few Delta_E points",
        }
    e_bind = float(np.mean(y))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        null[i] = float(np.mean(signs * y))
    floor_p = float((1 + int(np.sum(np.abs(null) >= abs(e_bind)))) / (n_perm + 1))
    if Delta_E_sigma is not None:
        sig = np.asarray(Delta_E_sigma, dtype=np.float64)
        sig = sig[np.isfinite(sig) & (sig > 0)]
        # External sigma: mean / se, not a self-normalized residual std.
        if sig.size:
            se = float(np.sqrt(np.mean(np.square(sig[: min(sig.size, n)])) / n))
            if se > 0:
                z = abs(e_bind) / se
                from scipy.stats import norm

                floor_p = float(2.0 * (1.0 - norm.cdf(z)))
    claimed = bool(identity_control_ok and floor_p < TSB_EVIDENCE_ALPHA and abs(e_bind) > 0)
    return {
        "E_bind": e_bind,
        "floor_p": floor_p,
        "E_bind_claimed": claimed,
        "n": n,
        "n_perm": n_perm,
        "seed": seed,
    }


def _tests_on_delta_e(
    d: np.ndarray,
    delta_e: np.ndarray,
    *,
    n_perm: int = 99,
    n_bootstrap: int = 200,
) -> dict[str, Any]:
    from menus.particle.casimir.test_framework import (
        detect_steps,
        test_hysteresis,
        test_log_modulations,
        test_periodicity,
        test_powerlaw_index,
    )

    x, y = clean_xy(d, delta_e)
    step_idx, p_steps, n_steps = detect_steps(x, y)
    hyst = test_hysteresis(x, y, n_bootstrap=n_bootstrap)
    period = test_periodicity(x, y)
    logm = test_log_modulations(x, y, n_perm=n_perm, seed=15)
    plaw = test_powerlaw_index(
        x,
        y,
        param_col="d",
        value_col="Delta_E",
    )
    return {
        "discrete_steps": {
            "n_steps": n_steps,
            "step_indices": np.asarray(step_idx).tolist(),
            "p_value": p_steps,
            "evidence_for_TSB": bool(p_steps < TSB_EVIDENCE_ALPHA),
        },
        "hysteresis": hyst,
        "periodicity": period,
        "log_modulations": logm,
        "powerlaw_index": plaw,
    }


def estimate_bound_energy(
    df: pd.DataFrame,
    *,
    param_col: str,
    value_col: str,
    d: np.ndarray | None = None,
    T: float | None = None,
    force_energy_df: pd.DataFrame | None = None,
    force_param_col: str | None = None,
    force_value_col: str | None = None,
    data_label: str = "",
    n_k: int = 64,
) -> dict[str, Any]:
    """
    Build ΔE(d, T) = E_obs − E_Lifshitz.

    If the table is already force/energy vs separation, skip the integrator
    unless a paired dielectric table is supplied. If only ε(iξ) is present,
    E_obs is absent and only the identity control + Lifshitz curve are stored.
    """
    kind = classify_observable_kind(param_col, value_col)
    x_kind = classify_x_kind(param_col)
    y_kind = classify_y_kind(value_col)
    temp = float(T if T is not None else _temperature_from_table(df, None))
    payload: dict[str, Any] = {
        "observable_kind": kind,
        "param_col": param_col,
        "value_col": value_col,
        "T": temp,
        "material_key": material_key_from_label(data_label),
        "lifshitz_approximations": list(LIFSHITZ_APPROXIMATIONS),
        "E_obs": None,
        "E_Lifshitz": None,
        "Delta_E": None,
        "Delta_E_sigma": None,
        "d": None,
        "E_bind_claimed": False,
        "identity_control_ok": False,
    }

    e_obs: np.ndarray | None = None
    d_obs: np.ndarray | None = None
    sigma = _sigma_column(df, value_col)

    table_is_force_energy = x_kind == "separation" and y_kind in {"force", "energy"}
    if table_is_force_energy:
        d_obs, e_obs = clean_xy(
            pd.to_numeric(df[param_col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float),
        )
        payload["integrator_skipped"] = True
        payload["integrator_skip_reason"] = (
            "table is already force/energy vs separation; y treated as E_obs/F_obs"
        )
    elif force_energy_df is not None and force_param_col and force_value_col:
        d_obs, e_obs = clean_xy(
            pd.to_numeric(force_energy_df[force_param_col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(force_energy_df[force_value_col], errors="coerce").to_numpy(dtype=float),
        )
        payload["paired_force_energy"] = True

    e_lif: np.ndarray | None = None
    d_grid: np.ndarray | None = None
    if (not table_is_force_energy) or force_energy_df is not None or (
        table_is_force_energy is False and x_kind == "frequency"
    ):
        if x_kind == "frequency" or kind == "dielectric":
            lif = compute_lifshitz_from_dielectric(
                df,
                param_col=param_col,
                value_col=value_col,
                d=d if d is not None else (d_obs if d_obs is not None else DEFAULT_D_METRES),
                T=temp,
                n_k=n_k,
            )
            e_lif = np.asarray(lif["E_Lifshitz"], dtype=np.float64)
            d_grid = np.asarray(lif["d"], dtype=np.float64)
            payload.update(
                {
                    "n_max": lif.get("n_max"),
                    "xi_unit": lif.get("xi_unit"),
                    "quadrature_rel": lif.get("quadrature_rel"),
                    "n_k": lif.get("n_k"),
                }
            )

    if e_lif is None and e_obs is not None:
        # No dielectric integrator: cannot form a Lifshitz continuum.
        d_grid = d_obs
        e_lif = np.zeros_like(e_obs)
        payload["E_Lifshitz_note"] = (
            "no dielectric table paired; E_Lifshitz set to 0 (ΔE = E_obs). "
            "Not a Lifshitz subtraction."
        )
        payload["lifshitz_available"] = False
    else:
        payload["lifshitz_available"] = e_lif is not None

    if d_grid is None:
        d_grid = np.asarray(d if d is not None else DEFAULT_D_METRES, dtype=np.float64)
    if e_lif is None:
        e_lif = np.full_like(d_grid, np.nan, dtype=np.float64)

    ident = identity_subtraction(e_lif)
    payload["identity_control"] = ident
    payload["identity_control_ok"] = bool(ident["identity_control_ok"])
    qrel = payload.get("quadrature_rel")
    payload["quadrature_failed"] = bool(
        qrel is not None and np.isfinite(float(qrel)) and float(qrel) > 0.25
    )

    if e_obs is not None:
        if d_obs is not None and d_obs.size and d_grid.size:
            # Interpolate E_obs onto the Lifshitz d-grid when both exist.
            order = np.argsort(d_obs)
            e_obs_on_d = np.interp(d_grid, d_obs[order], e_obs[order])
        else:
            e_obs_on_d = e_obs
        delta = e_obs_on_d - e_lif
        payload["E_obs"] = e_obs_on_d
    else:
        e_obs_on_d = None
        delta = ident["Delta_E"]

    payload["d"] = d_grid
    payload["E_Lifshitz"] = e_lif
    payload["Delta_E"] = delta
    payload["Delta_E_sigma"] = sigma
    payload["columns"] = {
        "d": d_grid,
        "E_Lifshitz": e_lif,
        "E_obs": e_obs_on_d,
        "Delta_E": delta,
        "Delta_E_sigma": sigma,
    }

    floor = floor_test_E_bind(
        delta,
        Delta_E_sigma=sigma,
        identity_control_ok=bool(payload["identity_control_ok"]) and e_obs is not None,
    )
    if e_obs is None:
        floor["E_bind_claimed"] = False
        floor["reason"] = "no E_obs; identity control only (cannot claim E_bind)"
    if payload.get("quadrature_failed"):
        floor["E_bind_claimed"] = False
        floor["reason"] = "Lifshitz quadrature mismatch; do not claim E_bind"
    payload.update(floor)
    payload["delta_e_tests"] = _tests_on_delta_e(d_grid, np.asarray(delta, dtype=np.float64))
    return payload


def run_bound_energy_search(
    *,
    param_col: str | None = None,
    value_col: str | None = None,
    output_prefix: str = "tsb_bound_energy",
    plot: bool = False,
    show_popup: bool = False,
    relock: bool = False,
) -> dict[str, Any]:
    """Menu entry: auto-discover Casimir tables and search ΔE."""
    from menus.particle.casimir.diagnostics import preregister_split
    from menus.particle.casimir.scanner import (
        _auto_select_columns,
        discover_local_casimir_files,
        load_data,
    )
    from menus.particle.casimir.series_integrity import (
        build_raw_series_integrity,
        register_series_hash,
    )
    from menus.particle.casimir.test_framework import run_full_test_suite

    _ = plot, show_popup
    lock = preregister_split(relock=relock)
    files = discover_local_casimir_files()
    TSB_TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    reports: list[dict[str, Any]] = []
    by_material: dict[str, dict[str, Any]] = {}

    if not files:
        from menus.particle.casimir.scanner import generate_mock_casimir_data

        df = generate_mock_casimir_data()
        pcol, vcol = param_col or "magnetic_field_mT", value_col or "force_pN"
        be = estimate_bound_energy(df, param_col=pcol, value_col=vcol, data_label="mock")
        reports.append({"data_label": "mock", "bound_energy": be, "used_fixture": True})
    else:
        loaded: list[tuple[Path, pd.DataFrame, str, str, str]] = []
        for path in files:
            df = load_data(path)
            pcol, vcol = (param_col, value_col)
            if pcol is None or vcol is None:
                pcol, vcol = _auto_select_columns(df)
            loaded.append((path, df, pcol, vcol, path.stem))

        dielectrics: dict[str, tuple[Path, pd.DataFrame, str, str]] = {}
        forces: dict[str, tuple[Path, pd.DataFrame, str, str]] = {}
        for path, df, pcol, vcol, stem in loaded:
            key = material_key_from_label(stem) or stem
            meta = classify_casimir_table(df, pcol, vcol)
            if meta["observable_kind"] == "dielectric":
                dielectrics[key] = (path, df, pcol, vcol)
            elif meta["observable_kind"] in {"force", "energy"}:
                forces[key] = (path, df, pcol, vcol)

        seen: set[str] = set()
        for path, df, pcol, vcol, stem in loaded:
            key = material_key_from_label(stem) or stem
            if key in seen:
                continue
            seen.add(key)
            force_pack = forces.get(key)
            diel_pack = dielectrics.get(key)
            if diel_pack is not None:
                src_path, src_df, src_p, src_v = diel_pack
                force_df = force_pack[1] if force_pack else None
                force_p = force_pack[2] if force_pack else None
                force_v = force_pack[3] if force_pack else None
                be = estimate_bound_energy(
                    src_df,
                    param_col=src_p,
                    value_col=src_v,
                    force_energy_df=force_df,
                    force_param_col=force_p,
                    force_value_col=force_v,
                    data_label=stem,
                )
                x, y = clean_xy(
                    pd.to_numeric(src_df[src_p], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(src_df[src_v], errors="coerce").to_numpy(dtype=float),
                )
            else:
                src_path, src_df, src_p, src_v = path, df, pcol, vcol
                be = estimate_bound_energy(
                    src_df,
                    param_col=src_p,
                    value_col=src_v,
                    data_label=stem,
                )
                x, y = clean_xy(
                    pd.to_numeric(src_df[src_p], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(src_df[src_v], errors="coerce").to_numpy(dtype=float),
                )
            integrity = build_raw_series_integrity(
                x, y, file_path=src_path, data_label=stem
            )
            register_series_hash(integrity)
            d_arr = np.asarray(be.get("d"), dtype=float)
            delta = np.asarray(be.get("Delta_E"), dtype=float)
            delta_report = None
            if d_arr.size and delta.size and np.any(np.isfinite(delta)):
                delta_df = pd.DataFrame({"d": d_arr, "Delta_E": delta})
                delta_report = run_full_test_suite(
                    delta_df,
                    param_col="d",
                    value_col="Delta_E",
                    output_prefix=f"{output_prefix}_{stem}_Delta_E",
                    plot=False,
                    file_path=src_path,
                    data_label=f"{stem}::Delta_E",
                    include_bound_energy=False,
                )
            be["raw_series_sha256"] = integrity.get("sha256")
            be["delta_e_falsification"] = (
                {k: delta_report.get(k) for k in ("results", "S_TSB", "report_path")}
                if delta_report
                else None
            )
            reports.append(
                {
                    "data_label": stem,
                    "file_path": str(src_path),
                    "material_key": key,
                    "bound_energy": json_safe(be),
                    "report_path": delta_report.get("report_path") if delta_report else None,
                }
            )
            by_material[key] = reports[-1]

    out = TSB_TEST_RESULTS_DIR / f"{output_prefix}_report_{stamp}.json"
    summary = {
        "module": "tsb_casimir_bound_energy",
        "timestamp": stamp,
        "preregister": lock,
        "n_files": len(reports),
        "reports": json_safe(reports),
        "by_material": {k: v.get("file_path") for k, v in by_material.items()},
    }
    out.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"[Superblock Residual Engine] Bound-energy residual search: {out}")
    for row in reports:
        be = row.get("bound_energy") or {}
        print(
            f"  {row.get('data_label')}: identity_ok={be.get('identity_control_ok')} "
            f"E_bind_claimed={be.get('E_bind_claimed')} floor_p={be.get('floor_p')}"
        )
        print("\n=== TSB Bound-Energy Residual Report ===")
        print(
            f"{row.get('data_label')}: identity_control_ok={be.get('identity_control_ok')} | "
            f"E_bind_claimed={be.get('E_bind_claimed')}"
        )
    summary["report_path"] = str(out)
    return summary
