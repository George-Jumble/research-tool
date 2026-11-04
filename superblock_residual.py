"""
3+1D residual path. The only default scoring path.

  D     observed data vector
  B     conventional 3+1D baseline (ΛCDM / SM / GR / Newtonian SPARC)
  r     residual, r = D − B
  S_k   Superblock / Mirron template k, k ∈ {A, B, C, D, E}
  σ     reported uncertainty on D
  χ²_SB goodness-of-fit of r against S_k (3+1D only; numeric kernel TODO)
  χ²_Λ  goodness-of-fit of D against B
  T_τ   Tau / Tav / KK / 5D template — retired; never evaluated
  N     number of residual bins
  Δ     dipole residual after CMB-frame subtraction
  V_orb domain orbital-velocity eigenvalue from an E₈ projection
  G_avg time-averaged gravitational coupling seen in our domain
  M_mir Mirron / Planck-Kerr geometric mass-gap scale (not a KK mass)
  a     FLRW scale factor (baseline only)
  z     redshift (observable; not a 5D coordinate)

Templates (geometry-derived; numeric kernels are TODO, not Tau stand-ins):

  A  multi-domain 3+1D kinematics (six domains, E₈-projected V_orb)
  B  conformal / twistor gravity (G_avg, emergent geometric mass)
  C  Mirron / Planck-Kerr sector (M_mir, non-evaporating Mirron-PBH, 4D resonance)
  D  fermion/boson assignment and emergent QCD (geometric Yang-Mills gap)
  E  observational residuals (dipole Δ, SPARC, DESI BAO, CMS/LHCb, LIGO/LISA, geometric sterile)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from superblock_mode import ENGINE_PREFIX, RETIRED_ON, tau_templates_enabled

TEMPLATES: dict[str, str] = {
    "A": "Multi-domain 3+1D kinematics (six co-located domains, E₈-projected V_orb, causal isolation)",
    "B": "Conformal/twistor gravity (G_avg; emergent geometric mass, not a 5D force and not a WIMP)",
    "C": "Mirron / Planck-Kerr sector (M_mir geometric mass gap; 4D resonance, not a KK tower)",
    "D": "Fermion/boson assignment and emergent QCD (geometric Yang-Mills gap, not a 5D Wilson line)",
    "E": "Observational residual (dipole Δ, SPARC, DESI BAO, flavor, stochastic background, geometric sterile)",
}

# Which Superblock template a live dataset is allowed to be compared with.
DATASET_TEMPLATE: dict[str, str] = {
    "TAU_SB_DESI": "E",
    "SPARC": "B",
    "LHCB_TAV_ECHO": "C",
    "CERN_OPENDATA": "C",
    "LIGO_GWOSC": "E",
    "FRB_COSMIC_WEB_TAV": "E",
    "PLANCK_CMB_TAV": "E",
    "TSB_CASIMIR": "C",
    "PRIME_PAST_HARMONIC": "D",
    "LISA_PRE_RUNS": "E",
}

_ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "superblock_residuals"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def refuse_tau_template(name: str = "T_tau") -> None:
    """T_τ is retired. Calling this is a programming error, not a fit."""
    if tau_templates_enabled():
        raise RuntimeError("tau_5d_templates cannot be enabled")
    raise RuntimeError(
        f"{name} is retired ({RETIRED_ON}). Superblock/Mirron only. "
        "Tau/5D templates were not applied."
    )


def _as_float_array(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return arr


def _chi2(y: np.ndarray, model: np.ndarray, sigma: np.ndarray | None) -> tuple[float, bool]:
    resid = np.asarray(y, dtype=float) - np.asarray(model, dtype=float)
    if (
        sigma is not None
        and np.asarray(sigma).size == resid.size
        and np.all(np.asarray(sigma, dtype=float) > 0)
    ):
        sig = np.asarray(sigma, dtype=float)
        return float(np.sum(np.square(resid / sig))), True
    return float(np.sum(np.square(resid))), False


def score_against_template(
    r: np.ndarray | None,
    *,
    template: str,
    sigma: np.ndarray | None = None,
    v_obs: np.ndarray | None = None,
    v_baryon: np.ndarray | None = None,
) -> dict[str, Any]:
    """χ² of the loaded curve. Template B uses the frozen G_avg factor."""
    if tau_templates_enabled():
        refuse_tau_template("T_tau")
    if (
        template == "B"
        and v_obs is not None
        and v_baryon is not None
        and v_obs.size
        and v_obs.size == v_baryon.size
    ):
        from kernels.g_avg import g_avg_over_gn, v_sb

        k = float(g_avg_over_gn())
        v_model = v_sb(v_baryon)
        chi2_lambda, sigma_used = _chi2(v_obs, v_baryon, sigma)
        chi2_sb, sigma_used_sb = _chi2(v_obs, v_model, sigma)
        return {
            "template": template,
            "template_description": TEMPLATES.get(template, TEMPLATES["B"]),
            "S_k": k,
            "kernel": "G_avg_equal_tick_gamma_inv2",
            "kernel_note": "R-independent; not a halo; k frozen",
            "chi2_Lambda": chi2_lambda,
            "chi2_SB": chi2_sb,
            "N": int(v_obs.size),
            "sigma_used": bool(sigma_used and sigma_used_sb),
            "verdict": "scale_only",
            "inherits_m0": False,
        }
    if template == "E":
        n = 0 if r is None else int(r.size)
        return {
            "template": template,
            "template_description": TEMPLATES.get(template, TEMPLATES["E"]),
            "S_k": None,
            "kernel": "observational_residual_no_S_k",
            "kernel_note": (
                "Template E is data minus a conventional baseline. "
                "No Superblock FRB kernel is evaluated. "
                "1/7 Tav harmonics were not applied."
            ),
            "chi2_SB": None,
            "N": n,
            "sigma_used": sigma is not None,
            "verdict": "inconclusive",
            "inherits_m0": False,
        }
    n = 0 if r is None else int(r.size)
    return {
        "template": template,
        "template_description": TEMPLATES.get(template, TEMPLATES["E"]),
        "S_k": None,
        "kernel": "TODO",
        "kernel_note": (
            "3+1D Superblock/Mirron kernel is not numeric yet. "
            "No Tau, Tav, KK, 142857, or 7 Mpc template was evaluated."
        ),
        "chi2_SB": None,
        "N": n,
        "sigma_used": sigma is not None,
        "verdict": "inconclusive",
    }


def residual_report(
    *,
    dataset: str,
    action: str,
    template: str,
    D: np.ndarray | None,
    B: np.ndarray | None,
    sigma: np.ndarray | None = None,
    z: np.ndarray | None = None,
    notes: list[str] | None = None,
    load_error: str | None = None,
) -> dict[str, Any]:
    """Steps 2–4. r = D − B when both exist and share a length."""
    if tau_templates_enabled():
        refuse_tau_template()
    r: np.ndarray | None = None
    chi2_lambda = None
    if D is not None and B is not None and D.size == B.size and D.size > 0:
        r = D - B
        if sigma is not None and sigma.size == r.size and np.all(sigma > 0):
            chi2_lambda = float(np.sum(np.square((D - B) / sigma)))
        else:
            chi2_lambda = float(np.sum(np.square(r)))
    elif D is not None and B is None:
        r = None
    scored = score_against_template(
        r,
        template=template,
        sigma=sigma,
        v_obs=D if template == "B" else None,
        v_baryon=B if template == "B" else None,
    )
    if scored.get("chi2_Lambda") is not None:
        chi2_lambda = scored["chi2_Lambda"]
    if load_error:
        scored["verdict"] = "inconclusive"
        scored["chi2_SB"] = None
        chi2_lambda = None
    amplitude = None if r is None else float(np.sqrt(np.mean(np.square(r))))
    scale = None
    if z is not None and z.size >= 2:
        scale = float(np.nanmax(z) - np.nanmin(z))
    elif D is not None and D.size >= 2:
        scale = float(np.nanmax(D) - np.nanmin(D))
    report = {
        "engine": ENGINE_PREFIX,
        "mode": "superblock_mirron_3p1d",
        "timestamp": _utc(),
        "dataset": dataset,
        "action": action,
        "template": template,
        "template_description": TEMPLATES[template],
        "N": int(D.size) if D is not None else 0,
        "residual_amplitude": amplitude,
        "residual_scale": scale,
        "chi2_Lambda": chi2_lambda,
        "chi2_SB": scored["chi2_SB"],
        "verdict": scored["verdict"],
        "negative_control": "Tau/5D templates were not applied",
        "T_tau": "retired_not_evaluated",
        "load_error": load_error,
        "notes": notes or [],
        "score": scored,
    }
    return report


def _write_report(report: dict[str, Any]) -> str:
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() else "_" for ch in report["dataset"])[:40]
    path = _ARTIFACTS / f"{safe}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _print_report(report: dict[str, Any], path: str) -> None:
    print(f"\n[{ENGINE_PREFIX}] === Superblock residual report ===")
    print(f"[{ENGINE_PREFIX}] dataset={report['dataset']} action={report['action']}")
    print(f"[{ENGINE_PREFIX}] template {report['template']}: {report['template_description']}")
    print(
        f"[{ENGINE_PREFIX}] N={report['N']} "
        f"residual_amplitude={report['residual_amplitude']} "
        f"residual_scale={report['residual_scale']}"
    )
    print(
        f"[{ENGINE_PREFIX}] verdict={report['verdict']} "
        f"chi2_SB={report['chi2_SB']} chi2_Lambda={report['chi2_Lambda']}"
    )
    print(f"[{ENGINE_PREFIX}] {report['negative_control']}")
    if report.get("load_error"):
        print(f"[{ENGINE_PREFIX}] load: {report['load_error']}")
    for note in report.get("notes") or []:
        print(f"[{ENGINE_PREFIX}] {note}")
    print(f"[{ENGINE_PREFIX}] wrote {path}")


def _load_desi() -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    notes = [
        "Step 1 baseline: DESI DR2 observable as published (ΛCDM-ratio data). "
        "A separate ΛCDM theory vector is not subtracted unless the table already is a residual."
    ]
    try:
        from menus.astronomical.desi.scanner import load_desi_from_cobaya_repo

        data = load_desi_from_cobaya_repo()
    except Exception as exc:
        return None, None, None, None, notes + [f"DESI loader did not return a table: {exc}"]
    D = _as_float_array(data.get("observable"))
    sigma = _as_float_array(data.get("err"))
    z = _as_float_array(data.get("z"))
    # Conventional baseline for a BAO ratio is the fiducial prediction when present.
    B = _as_float_array(data.get("lcdm") or data.get("theory") or data.get("baseline"))
    notes.append("Tau/5D periodicity was not run. Comparison model on the table is ΛCDM, not Tau-SB.")
    return D, B, sigma, z, notes


def _sparc_columns(frame) -> dict[str, str]:
    return {str(c).lower(): str(c) for c in frame.columns}


def _named_column(frame, *names: str) -> np.ndarray | None:
    cols = _sparc_columns(frame)
    for name in names:
        key = name.lower()
        if key in cols:
            return _as_float_array(frame[cols[key]])
    return None


def _load_sparc_one() -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """Every local SPARC galaxy that has gas, disk, and bulge columns. Not CamB alone."""
    notes = [
        "Step 1 baseline: Newtonian baryonic speed from V_gas, V_disk, V_bul.",
    ]
    try:
        from menus.astronomical.sparc.fetcher import iter_local_csv_files
        import pandas as pd
    except Exception as exc:
        raise FileNotFoundError("full SPARC table not found") from exc
    paths = list(iter_local_csv_files())
    v_obs_parts: list[np.ndarray] = []
    v_bar_parts: list[np.ndarray] = []
    sigma_parts: list[np.ndarray] = []
    sigma_complete = True
    n_galaxies = 0
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        v_obs = _named_column(frame, "Vobs", "V_obs", "vobs")
        gas = _named_column(frame, "Vgas", "V_gas")
        disk = _named_column(frame, "Vdisk", "V_disk")
        bul = _named_column(frame, "Vbul", "V_bul", "Vbulge", "V_bulge")
        if v_obs is None or gas is None or disk is None or bul is None:
            continue
        n = min(v_obs.size, gas.size, disk.size, bul.size)
        if n < 1:
            continue
        v_obs = v_obs[:n]
        v_bar = np.sqrt(np.square(gas[:n]) + np.square(disk[:n]) + np.square(bul[:n]))
        sig = _named_column(frame, "e_Vobs", "errV", "e_vobs", "e_Vobs")
        if sig is None or sig.size < n or not np.all(sig[:n] > 0):
            sigma_complete = False
        else:
            sigma_parts.append(sig[:n])
        v_obs_parts.append(v_obs)
        v_bar_parts.append(v_bar)
        n_galaxies += 1
    if n_galaxies == 0:
        raise FileNotFoundError("full SPARC table not found")
    D = np.concatenate(v_obs_parts)
    B = np.concatenate(v_bar_parts)
    sigma = np.concatenate(sigma_parts) if sigma_complete and sigma_parts else None
    notes.append(f"n_galaxies={n_galaxies}")
    notes.append(f"n_points={int(D.size)}")
    return D, B, sigma, None, notes


def _frb_independent_z(frame) -> np.ndarray | None:
    import pandas as pd

    for name in ("z_host", "redshift", "z_spec", "z_cmb", "z"):
        if name not in frame.columns:
            continue
        z = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        finite = z[np.isfinite(z) & (z > 0)]
        if finite.size >= 8:
            return z
    return None


def _load_frb() -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """CHIME-style DM minus Macquart IGM only when host z is independent of DM."""
    from menus.astronomical.frb.cosmic_web_tav import (
        IGM_DM_PER_Z,
        classify_paths,
        expected_dm_igm,
        load_frb_catalog,
        load_void_catalog,
    )
    from menus.astronomical.frb.fetcher import resolve_dataset_paths

    notes = [
        "Step 1 baseline: Macquart-style IGM DM = 855 z + 50 pc cm^-3, only if host z is independent.",
        "Redshift inferred from DM is not used (that residual is identically ~0).",
        "1/7 Tav path classification and periodograms were not applied.",
    ]
    frb_path, void_path = resolve_dataset_paths(auto_fetch=False, restore_archived=True)
    frame = load_frb_catalog(frb_path)
    notes.append(f"frb_file={frb_path.name}")
    notes.append(f"n_bursts={len(frame)}")
    D = _as_float_array(frame["DM"])
    if D is None:
        raise FileNotFoundError("FRB DM column empty after quality cuts")
    sigma = None
    for name in ("dm_fitb_err", "dm_err", "e_dm"):
        if name in frame.columns:
            sigma = _as_float_array(frame[name])
            break
    if sigma is not None and sigma.size != D.size:
        sigma = None
    z_ind = _frb_independent_z(frame)
    B = None
    z_out = None
    if z_ind is not None:
        mask = np.isfinite(z_ind) & (z_ind > 0) & np.isfinite(D)
        if int(np.sum(mask)) >= 8:
            D = D[mask]
            z_out = z_ind[mask]
            B = expected_dm_igm(z_out)
            if sigma is not None:
                sigma = sigma[mask]
            notes.append(f"independent_host_z=true n_with_z={int(D.size)} IGM_DM_PER_Z={IGM_DM_PER_Z}")
        else:
            notes.append("independent_host_z=false (too few finite host z)")
    else:
        notes.append("independent_host_z=false (no host-z column; Macquart residual not formed)")
    try:
        voids = load_void_catalog(void_path)
        labels, _sep = classify_paths(
            frame["ra"].to_numpy(),
            frame["dec"].to_numpy(),
            voids,
        )
        uniq, cnt = np.unique(np.asarray(labels, dtype=object), return_counts=True)
        mix = {str(k): int(v) for k, v in zip(uniq, cnt)}
        notes.append(f"void_file={void_path.name}")
        notes.append(f"path_mix={mix}")
    except Exception as exc:
        notes.append(f"path_classification_skipped={exc}")
    return D, B, sigma, z_out, notes


def _load_generic_note(kind: str) -> tuple[None, None, None, None, list[str]]:
    return (
        None,
        None,
        None,
        None,
        [
            f"Step 0 loader for {kind} is unchanged on disk but was not asked to score.",
            "Step 1 conventional baseline is the dataset's ΛCDM / SM / GR / Newtonian reduction.",
            "Step 3 numeric S_k kernel is TODO. Tau/5D templates were not applied.",
        ],
    )


def run_superblock_residual(
    module_tag: str,
    action: str,
    options: dict | None = None,
) -> dict[str, Any]:
    """Steps 0–4 for one live menu action. Never evaluates T_τ."""
    _ = options
    if tau_templates_enabled():
        refuse_tau_template()
    template = DATASET_TEMPLATE.get(module_tag, "E")
    load_error = None
    try:
        if module_tag == "TAU_SB_DESI":
            D, B, sigma, z, notes = _load_desi()
        elif module_tag == "SPARC":
            D, B, sigma, z, notes = _load_sparc_one()
        elif module_tag == "FRB_COSMIC_WEB_TAV":
            D, B, sigma, z, notes = _load_frb()
        else:
            D, B, sigma, z, notes = _load_generic_note(module_tag)
    except Exception as exc:
        D = B = sigma = z = None
        notes = []
        load_error = str(exc)
    report = residual_report(
        dataset=module_tag,
        action=action,
        template=template,
        D=D,
        B=B,
        sigma=sigma,
        z=z,
        notes=notes,
        load_error=load_error,
    )
    if module_tag == "TAU_SB_DESI":
        from sb_engine.sound_horizon import calibrate_gamma, sound_horizon

        report["sound_horizon"] = sound_horizon(calibrate_gamma())
        report["notes"] = list(report.get("notes") or []) + [
            report["sound_horizon"]["note"]
        ]
    path = _write_report(report)
    report["report_path"] = path
    _print_report(report, path)
    return report
