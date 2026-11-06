"""
3+1D residual path. The only default scoring path.

  D     observed data vector
  B     conventional 3+1D baseline (ΛCDM / SM / GR / Newtonian SPARC)
  r     residual, r = D − B
  S_k   Superblock / Mirron template k, k ∈ {A, B, C, D, E}
  σ     reported uncertainty on D
  χ²_SB goodness-of-fit of r against S_k (3+1D only; numeric kernel TODO)
  χ²_Λ  goodness-of-fit of D against B
  N     number of residual bins
  Δ     dipole residual after CMB-frame subtraction
  V_orb domain orbital-velocity eigenvalue from an E₈ projection
  G_avg time-averaged coupling <1/γ²> — RETIRED 2026-10-31 (T1) / N15; not evaluated
  V_RL  hierarchical L_res rotation kernel V_flat·sqrt(r/(r+R_core)) (v5 §6)
  M_mir Mirron / Planck-Kerr geometric mass-gap scale (not a KK mass)
  a     FLRW scale factor (baseline only)
  z     redshift (observable; not a 5D coordinate)

Templates (geometry-derived; numeric kernels are TODO):

  A  multi-domain 3+1D kinematics (six domains, E₈-projected V_orb)
  B  hierarchical L_res gravity (six-domain sum, dominant-glob rule; V_RL kernel)
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

from superblock_mode import ENGINE_PREFIX

TEMPLATES: dict[str, str] = {
    "A": "Multi-domain 3+1D kinematics (six co-located domains, E₈-projected V_orb, causal isolation)",
    "B": "Hierarchical L_res gravity (six-domain sum; V_RL kernel, one R_core per galaxy; not a 5D force, not a WIMP)",
    "C": "Mirron / Planck-Kerr sector (M_mir geometric mass gap; 4D resonance, not a KK tower)",
    "D": "Fermion/boson assignment and emergent QCD (geometric Yang-Mills gap, not a 5D Wilson line)",
    "E": "Observational residual (dipole Δ, SPARC, DESI BAO, flavor, stochastic background, geometric sterile)",
}

# Which Superblock template a live dataset is allowed to be compared with.
DATASET_TEMPLATE: dict[str, str] = {
    "DESI_BAO": "E",
    "SPARC": "B",
    "LHCB_FLAVOR": "C",
    "CERN_OPENDATA": "C",
    "LIGO_GWOSC": "E",
    "FRB_DM": "E",
    "PLANCK_CMB": "E",
    "CASIMIR": "C",
    "PRIME_PAST_HARMONIC": "D",
    "LISA_PRE_RUNS": "E",
}

# Template C (Mirron kernel) scope, author ruling 2026-11-04: LHCb only. CERN dimuon and Casimir are
# template-C datasets but the kernel is not mapped to them; they refuse it until the author rules.
MIRRON_C_SCOPE: frozenset[str] = frozenset({"LHCB_FLAVOR"})
TEMPLATE_C_REFUSED: frozenset[str] = frozenset({"CERN_OPENDATA", "CASIMIR"})
TEMPLATE_C_REFUSAL = "awaiting author ruling on mapping"

# Template D (fermion/boson assignment, emergent QCD) has no formula yet: SB prediction pending.
SB_PREDICTION_PENDING = "pending"

_ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "superblock_residuals"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    radius: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    q2: np.ndarray | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """χ² of the loaded curve. Template B uses the hierarchical L_res V_RL kernel (v5 §6).

    The retired G_avg = <1/γ²> factor (Weyl/conformal averaging kernel with Scheme A weights) is never
    evaluated. V_RL needs radii; without them Template B is inconclusive.
    """
    if (
        template == "B"
        and v_obs is not None
        and v_baryon is not None
        and v_obs.size
        and v_obs.size == v_baryon.size
    ):
        from kernels.l_res_rl import KERNEL_NAME, fit_galaxy

        base = {
            "template": template,
            "template_description": TEMPLATES.get(template, TEMPLATES["B"]),
            "kernel": KERNEL_NAME,
            "N": int(v_obs.size),
            "inherits_m0": False,
            "retired_kernel_not_evaluated": "G_avg_equal_tick_gamma_inv2",
        }
        if radius is None or radius.size != v_obs.size:
            chi2_lambda, sigma_used = _chi2(v_obs, v_baryon, sigma)
            return {
                **base,
                "S_k": None,
                "kernel_note": "V_RL needs radii; none supplied",
                "chi2_Lambda": chi2_lambda,
                "chi2_SB": None,
                "sigma_used": bool(sigma_used),
                "verdict": "inconclusive",
            }
        gids = np.zeros(v_obs.size, dtype=int) if groups is None else np.asarray(groups)
        sig = sigma if sigma is not None and sigma.size == v_obs.size and np.all(sigma > 0) else None
        fits = []
        for gid in dict.fromkeys(gids.tolist()):
            m = gids == gid
            if int(np.sum(m)) < 2:
                continue
            fits.append(
                fit_galaxy(radius[m], v_obs[m], v_baryon[m], None if sig is None else sig[m])
            )
        chi2_sb = float(sum(f["chi2_SB"] for f in fits))
        chi2_lambda = float(sum(f["chi2_baryon"] for f in fits))
        wins = int(sum(f["chi2_SB"] < f["chi2_baryon"] for f in fits))
        return {
            **base,
            "S_k": [f["r_core_kpc"] for f in fits],
            "kernel_note": "V_flat from outer 30 %; one free R_core per galaxy; M/L = 1; eta = 0",
            "chi2_Lambda": chi2_lambda,
            "chi2_SB": chi2_sb,
            "n_galaxies": len(fits),
            "n_galaxies_SB_better": wins,
            "sigma_used": sig is not None,
            "verdict": "fit_with_free_core" if fits else "inconclusive",
        }
    if template == "C" and dataset is not None and dataset not in MIRRON_C_SCOPE:
        n = 0 if r is None else int(r.size)
        return {
            "template": template,
            "template_description": TEMPLATES.get(template, TEMPLATES["E"]),
            "S_k": None,
            "kernel": "refused",
            "kernel_note": (
                f"Template C Mirron kernel is LHCb-only; {dataset}: {TEMPLATE_C_REFUSAL}"
            ),
            "template_C_refused": TEMPLATE_C_REFUSAL,
            "chi2_SB": None,
            "N": n,
            "sigma_used": sigma is not None,
            "verdict": "inconclusive",
        }
    if template == "C" and r is not None and q2 is not None and r.size and q2.size == r.size:
        from kernels.mirron_c import score as score_c

        sig = sigma if sigma is not None and sigma.size == r.size and np.all(sigma > 0) else None
        out = score_c(r, q2, sig)
        return {
            "template": template,
            "template_description": TEMPLATES.get(template, TEMPLATES["E"]),
            "kernel_note": (
                "S_C(q²) = A·M²/(M²+q²), M = 313.1 MeV, A = α_s(313.1 MeV) not fitted; q² in GeV²"
            ),
            "sigma_used": sig is not None,
            **out,
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
                "No Superblock FRB kernel is evaluated."
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
            f"3+1D Superblock/Mirron kernel is not numeric yet: Template {template} has no formula, "
            "so the SB prediction is pending (no S_k is invented)."
        ),
        "sb_prediction": SB_PREDICTION_PENDING,
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
    radius: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    q2: np.ndarray | None = None,
) -> dict[str, Any]:
    """Steps 2–4. r = D − B when both exist and share a length."""
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
        radius=radius if template == "B" else None,
        groups=groups if template == "B" else None,
        q2=q2,
        dataset=dataset,
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
        from menus.astronomical.desi.loader import load_desi_from_cobaya_repo

        data = load_desi_from_cobaya_repo()
    except Exception as exc:
        return None, None, None, None, notes + [f"DESI loader did not return a table: {exc}"]
    D = _as_float_array(data.get("observable"))
    sigma = _as_float_array(data.get("err"))
    z = _as_float_array(data.get("z"))
    # Conventional baseline for a BAO ratio is the fiducial prediction when present.
    B = _as_float_array(data.get("lcdm") or data.get("theory") or data.get("baseline"))
    notes.append("Comparison model on the table is ΛCDM.")
    return D, B, sigma, z, notes


_SPARC_GEOMETRY: dict[str, np.ndarray] = {}


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
    radius_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
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
        rad = _named_column(frame, "R", "Rad", "r_kpc")
        if rad is not None and rad.size >= n:
            radius_parts.append(rad[:n])
        v_obs_parts.append(v_obs)
        v_bar_parts.append(v_bar)
        group_parts.append(np.full(n, n_galaxies, dtype=int))
        n_galaxies += 1
    if n_galaxies == 0:
        raise FileNotFoundError("full SPARC table not found")
    D = np.concatenate(v_obs_parts)
    B = np.concatenate(v_bar_parts)
    sigma = np.concatenate(sigma_parts) if sigma_complete and sigma_parts else None
    _SPARC_GEOMETRY.clear()
    if len(radius_parts) == n_galaxies:
        _SPARC_GEOMETRY["radius"] = np.concatenate(radius_parts)
        _SPARC_GEOMETRY["groups"] = np.concatenate(group_parts)
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
    from menus.astronomical.frb.catalog import (
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


_LHCB_EXTRA: dict[str, Any] = {}


def _load_lhcb(options: dict | None = None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """LHCb B⁰→K*⁰μμ per q² bin.

    Real LHCb 2020 schema: D = ⟨obs⟩_LHCb, B = ⟨obs⟩_SM (same observable, flavio SM),
    σ = sqrt(stat² + syst² + σ_SM²). Legacy C9 schema: D = C9_exp, B = C9_SM, σ = C9_exp_err.
    Provenance from lhcb_bmeson_fetch.json is kept in _LHCB_EXTRA for the report.
    """
    from menus.particle.lhcb.fetcher import is_real_provenance, read_fetch_metadata
    from menus.particle.lhcb.flavor import DEFAULT_DATA_PATH, OBSERVABLE_LABELS, load_lhcb_data, residual_arrays

    _LHCB_EXTRA.clear()
    opts = options or {}
    data_path = str(opts.get("data_file") or DEFAULT_DATA_PATH).strip()
    n_bins = opts.get("n_q2_bins")
    data, path = load_lhcb_data(data_path, n_bins=int(n_bins) if n_bins else None)
    arr = residual_arrays(data)
    meta = read_fetch_metadata(path)
    real = arr["schema"] == "lhcb2020_real" and is_real_provenance(meta)
    observable = meta.get("observable") if arr["schema"] == "lhcb2020_real" else "C9"
    prov = {
        "fetch_source": meta.get("source") or "missing_sidecar",
        "real_data": bool(real),
        "schema": arr["schema"],
        "observable": observable,
        "observable_label": OBSERVABLE_LABELS.get(str(observable), str(observable)),
        "inspire_id": meta.get("inspire_id") or meta.get("inspire"),
        "arxiv_id": meta.get("arxiv_id"),
        "citation": meta.get("citation"),
        "sm_source": meta.get("sm_source"),
        "data_file": str(path),
    }
    if not real:
        prov["warning"] = (
            "SYNTHETIC / FALLBACK LHCb data loaded (source="
            f"{prov['fetch_source']!r}, schema={arr['schema']}): not a measurement. "
            "Run 'Download LHCb B-meson Dataset' to fetch the real LHCb 2020 bins."
        )
    if arr["schema"] == "lhcb2020_real":
        prov["bins"] = [
            {
                "q2_min_gev2": float(r["q2_min_gev2"]),
                "q2_max_gev2": float(r["q2_max_gev2"]),
                "q2_center_gev2": float(r["q2_center_gev2"]),
                "obs": float(r["obs"]),
                "stat_err": float(r["stat_err"]),
                "syst_err": float(r["syst_err"]),
                "obs_err": float(r["obs_err"]),
                "sm": float(r["sm"]),
                "sm_err": float(r["sm_err"]),
                "residual": float(r["obs"] - r["sm"]),
            }
            for r in data
        ]
    _LHCB_EXTRA.update(prov)
    if arr["schema"] == "lhcb2020_real":
        notes = [
            f"Step 1 baseline: SM ⟨{observable}⟩ per q² bin ({meta.get('sm_source') or 'tabulated'}).",
            "σ = sqrt(stat² + syst² + σ_SM²) per bin.",
        ]
    else:
        notes = ["Step 1 baseline: SM Wilson coefficient C9_SM per q² bin."]
    notes += [
        f"lhcb_file={path.name}",
        f"n_q2_bins={len(data)}",
        f"fetch_source={prov['fetch_source']}",
    ]
    if prov.get("warning"):
        notes.append(prov["warning"])
    notes.append("q² in GeV² for Template C")
    return arr["D"], arr["B"], arr["sigma"], arr["q2_gev2"], notes


def _opt_flag(options: dict | None, key: str, default: bool) -> bool:
    val = (options or {}).get(key)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_planck(options: dict | None = None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """Planck PR3 binned TT: D = D_l, B = ΛCDM best-fit D_l (same file), σ = mean(±dD_l), x = ℓ."""
    from menus.planck_cmb.spectrum import TT_BINNED_FILE, ensure_tt_spectrum, load_tt_binned

    path = ensure_tt_spectrum(auto_fetch=_opt_flag(options, "auto_fetch", True))
    data = load_tt_binned(path)
    notes = [
        "Step 1 baseline: Planck 2018 ΛCDM best-fit TT spectrum (BestFit column of the PLA binned file).",
        f"planck_file={TT_BINNED_FILE}",
        f"ell_range={int(data['ell'][0])}-{int(data['ell'][-1])}",
        "Bins are treated as independent (no covariance); σ = mean of the asymmetric errors.",
    ]
    return data["Dl"], data["bestfit"], data["sigma"], data["ell"], notes


class LoaderError(FileNotFoundError):
    """Loader failure that keeps the notes gathered so far (per-event skips etc.)."""

    def __init__(self, message: str, notes: list[str] | None = None):
        super().__init__(message)
        self.loader_notes = list(notes or [])


_LIGO_EXTRA: dict[str, Any] = {}

_LIGO_EVENT_ALL = {"", "ALL", "*"}


def _ligo_event_filter(opts: dict) -> str | None:
    raw = str(opts.get("event") or "").strip().upper()
    return None if raw in _LIGO_EVENT_ALL else raw


def _ligo_pick_files(files: list, gps: float) -> list:
    """One file per detector: the shortest segment that contains the event GPS (32 s over 4096 s)."""
    import re

    best: dict[str, tuple[float, Path]] = {}
    for path in files:
        name = Path(path).name
        m = re.match(r"^([A-Z])-([A-Z]\d)_.*-(\d+)-(\d+)\.", name)
        if m:
            det, start, dur = m.group(2), float(m.group(3)), float(m.group(4))
            if not (start <= gps < start + dur):
                continue
        else:
            det, dur = name, float("inf")
        if det not in best or dur < best[det][0]:
            best[det] = (dur, Path(path))
    return [best[d][1] for d in sorted(best)]


def _ligo_point(label: str, event: str, detectors: str, res: dict, gr: dict, merged: dict, extra: dict) -> dict:
    fit = res.get("fit") or {}
    f_obs = fit.get("f_hz", res["f_hz"])
    s_f_meas = fit.get("sigma_f_hz", res["sigma_hz"])
    s_f = float(np.hypot(s_f_meas, gr["sigma_hz"]))
    pt = {
        "label": label, "event": event, "detector": detectors,
        "f_obs_hz": f_obs, "f220_gr_hz": gr["f220_hz"], "sigma_hz": s_f, "sigma_meas_hz": s_f_meas,
        "sigma_gr_hz": gr["sigma_hz"], "pull": (f_obs - gr["f220_hz"]) / s_f,
        "f_peak_periodogram_hz": res["f_hz"], "snr_peak": res["snr_peak"],
        "t_peak_offset_s": res["t_peak_offset_s"], "whitened": res["whitened"],
        "m_final_detector": gr["m_final_detector"], "a_final": gr["a_final"],
        "catalog_version": merged.get("version"),
        "tau220_gr_ms": gr["tau220_s"] * 1e3, "gamma220_gr_hz": gr["gamma220_hz"], "q220_gr": gr["q220"],
        "sigma_gamma_gr_hz": gr["sigma_gamma_hz"], "sigma_tau_gr_ms": gr["sigma_tau_s"] * 1e3,
    }
    if fit:
        s_g = float(np.hypot(fit["sigma_gamma_hz"], gr["sigma_gamma_hz"]))
        pt.update({
            "tau_ms": fit["tau_s"] * 1e3, "sigma_tau_ms": fit["sigma_tau_s"] * 1e3, "q_obs": fit["q"],
            "gamma_obs_hz": fit["gamma_hz"], "sigma_gamma_meas_hz": fit["sigma_gamma_hz"], "sigma_gamma_hz": s_g,
            "pull_gamma": (fit["gamma_hz"] - gr["gamma220_hz"]) / s_g if s_g > 0 else None,
            "gamma_constrained": fit["gamma_constrained"], "fit_snr": fit["snr_fit"],
            "fit_dt_start_ms": fit["dt_start_s"] * 1e3, "fit_window_ms": fit.get("window_s", 0.0) * 1e3,
        })
    pt.update(extra)
    return pt


def _ligo_joint(points: list[dict]) -> dict:
    """Inverse-variance weighted fractional deviations across scored events."""
    def wmean(vals: list[tuple[float, float]]) -> dict | None:
        vals = [(v, s) for v, s in vals if s and np.isfinite(v) and np.isfinite(s) and s > 0]
        if not vals:
            return None
        w = np.array([1 / s**2 for _, s in vals])
        v = np.array([x for x, _ in vals])
        mean = float(np.sum(w * v) / np.sum(w))
        chi2 = float(np.sum(w * (v - mean) ** 2))
        return {"mean": mean, "sigma": float(1 / np.sqrt(np.sum(w))), "n": len(vals),
                "chi2_dof": chi2 / (len(vals) - 1) if len(vals) > 1 else None}

    df = [((p["f_obs_hz"] - p["f220_gr_hz"]) / p["f220_gr_hz"], p["sigma_hz"] / p["f220_gr_hz"]) for p in points]
    g_pts = [p for p in points if p.get("gamma_constrained")]
    dg = [((p["gamma_obs_hz"] - p["gamma220_gr_hz"]) / p["gamma220_gr_hz"], p["sigma_gamma_hz"] / p["gamma220_gr_hz"])
          for p in g_pts]
    dtau = [((p["tau_ms"] - p["tau220_gr_ms"]) / p["tau220_gr_ms"],
             float(np.hypot(p["sigma_tau_ms"], p["sigma_tau_gr_ms"])) / p["tau220_gr_ms"]) for p in g_pts]
    pm = [p.get("postmerger_scan") for p in points if p.get("postmerger_scan")]
    return {
        "n_events": len({p["event"] for p in points}),
        "delta_f_over_f": wmean(df),
        "delta_gamma_over_gamma": wmean(dg),
        "delta_tau_over_tau": wmean(dtau),
        "network_snr": {p["label"]: p.get("network_snr", p["snr_peak"]) for p in points},
        "postmerger": {
            "n_scanned": len(pm),
            "n_detected": sum(1 for s in pm if s.get("detected")),
            "max_sigma_equiv": max((s["sigma_equiv"] for s in pm), default=None),
            "min_fap": min((s["fap"] for s in pm), default=None),
        },
    }


def _load_ligo(options: dict | None = None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """Ringdown markers per cached GWOSC event, in Hz.

    For each event (coherent H1+L1+V1 stack by default; ``detector_mode=per_detector`` for one row per
    detector file) two residual entries: oscillation frequency f vs GR Kerr f_220, and damping rate
    γ = 1/(2πτ) vs γ_220 (only when the blind fit's τ is inside its bounds and constrained). Both come
    from a variable-projection damped-sinusoid fit to whitened data that never sees GR; σ = hypot(fit
    sandwich σ, catalog σ_GR). x = detector-frame M_f.

    Options: ``events_dir``, ``catalog_file``, ``event`` (ALL/*/blank or one ID), ``strain_path`` (one file;
    forces per_detector), ``detector_mode`` (coherent | per_detector), ``auto_fetch_detail`` (default True),
    ``min_snr`` (4), ``postmerger_scan`` (default yes in coherent mode), ``fit_dt_start_ms`` (2),
    ``fit_window_ms`` (30).
    """
    from menus.gravitic.ligo.coherent import coherent_combine, common_grid, network_snr, null_stream_veto
    from menus.gravitic.ligo.fetcher import CATALOG_CACHE, EVENTS_DIR, ensure_event_detail, load_event_detail
    from menus.gravitic.ligo.postmerger import scan_from_preps
    from menus.gravitic.ligo.qnm import (
        DEFAULT_MIN_SNR,
        FIT_DT_START_S,
        FIT_T_S,
        catalog_entry,
        gr_prediction,
        load_catalog,
        measure_ringdown,
        resolve_event_metadata,
        ringdown_from_prepared,
    )
    from menus.gravitic.ligo.strain_io import list_event_strain_files, load_strain_file

    _LIGO_EXTRA.clear()
    opts = options or {}
    events_dir = Path(opts.get("events_dir") or EVENTS_DIR)
    catalog_path = Path(opts.get("catalog_file") or CATALOG_CACHE)
    auto_fetch = _opt_flag(opts, "auto_fetch_detail", True)

    def _f(key: str, default: float) -> float:
        try:
            v = opts.get(key)
            return float(v) if v not in (None, "") else default
        except (TypeError, ValueError):
            return default

    min_snr = _f("min_snr", DEFAULT_MIN_SNR)
    dt_start = _f("fit_dt_start_ms", FIT_DT_START_S * 1e3) / 1e3
    fit_s = _f("fit_window_ms", FIT_T_S * 1e3) / 1e3
    want = _ligo_event_filter(opts)
    strain_path = str(opts.get("strain_path") or "").strip()
    mode = str(opts.get("detector_mode") or "").strip().lower().replace("-", "_")
    if mode not in {"coherent", "per_detector"}:
        mode = "per_detector" if strain_path else "coherent"
    do_scan = _opt_flag(opts, "postmerger_scan", True)

    if strain_path:
        sp = Path(strain_path)
        if not sp.is_absolute() and not sp.is_file():
            sp = events_dir.parent / strain_path
        if not sp.is_file():
            raise FileNotFoundError(f"strain_path not found: {strain_path}")
        jobs = [((want or sp.parent.name.upper()), [sp])]
        events_dir = sp.parent.parent if not opts.get("events_dir") else events_dir
    else:
        if not events_dir.is_dir() or not any(events_dir.iterdir()):
            raise FileNotFoundError(f"No cached GWOSC strain under {events_dir} (run 'Pull Strain from GWOSC')")
        cached = sorted(p.name for p in events_dir.iterdir() if p.is_dir() and list_event_strain_files(p.name, datasets_root=events_dir.parent))
        if want:
            if want not in {c.upper() for c in cached}:
                raise FileNotFoundError(
                    f"Event {want} has no cached strain under {events_dir}; cached events: {', '.join(cached) or 'none'}"
                )
            cached = [c for c in cached if c.upper() == want]
        jobs = [(ev, list_event_strain_files(ev, datasets_root=events_dir.parent)) for ev in cached]
    if not catalog_path.is_file():
        raise FileNotFoundError(f"No cached GWOSC event catalog at {catalog_path}")
    catalog = load_catalog(catalog_path)
    notes = [
        "Step 1 baseline: GR Kerr (2,2,0) f_220 and γ_220 = f_220/(2Q_220) (Berti, Cardoso & Will 2006) from catalog "
        "final mass + redshift and PE final spin.",
        "D (blind to GR): whitened (Welch median PSD, ±0.5 s excluded), 35–512 Hz, Hilbert-envelope peak ±0.10 s; "
        f"variable-projection damped-sinusoid fit on [t_peak + {dt_start * 1e3:g} ms, + {fit_s * 1e3:g} ms]; "
        "σ = hypot(sandwich σ with off-source Toeplitz noise covariance, σ_GR). Residual rows are f and γ, both in Hz.",
        f"detector_mode={mode} min_snr={min_snr:g} postmerger_scan={'yes' if do_scan and mode == 'coherent' else 'no'}",
    ]
    D, B, S, X, points, rows = [], [], [], [], [], []

    def _score(pt: dict) -> None:
        D.append(pt["f_obs_hz"]); B.append(pt["f220_gr_hz"]); S.append(pt["sigma_hz"]); X.append(pt["m_final_detector"])
        rows.append({"label": pt["label"], "marker": "f_hz"})
        if pt.get("gamma_constrained"):
            D.append(pt["gamma_obs_hz"]); B.append(pt["gamma220_gr_hz"]); S.append(pt["sigma_gamma_hz"])
            X.append(pt["m_final_detector"]); rows.append({"label": pt["label"], "marker": "gamma_hz"})
        points.append(pt)
        g_txt = (f"γ={pt['gamma_obs_hz']:.1f}±{pt['sigma_gamma_meas_hz']:.1f} Hz vs γ_220={pt['gamma220_gr_hz']:.1f} Hz "
                 f"(τ={pt['tau_ms']:.2f} vs {pt['tau220_gr_ms']:.2f} ms, Q={pt['q_obs']:.2f} vs {pt['q220_gr']:.2f})"
                 if "gamma_obs_hz" in pt else "")
        if "gamma_obs_hz" in pt and not pt.get("gamma_constrained"):
            g_txt += " [γ not constrained: not scored]"
        extra = ""
        if pt.get("null_stream_chi2_red") is not None:
            extra = f" net SNR={pt['network_snr']:.1f} null χ²_red={pt['null_stream_chi2_red']:.2f}"
        notes.append(
            f"{pt['label']}: f={pt['f_obs_hz']:.1f}±{pt['sigma_meas_hz']:.1f} Hz vs f_220^GR={pt['f220_gr_hz']:.1f} Hz "
            f"pull={pt['pull']:+.2f}; {g_txt}; SNR={pt['snr_peak']:.1f}{extra}"
        )

    for event, files in jobs:
        meta = catalog_entry(catalog, event)
        detail = load_event_detail(event, events_dir)
        merged = resolve_event_metadata(meta, detail)
        gr = gr_prediction(merged) if merged else {"ok": False, "missing": ["catalog entry"]}
        if not gr["ok"] and auto_fetch and detail is None:
            try:
                detail = ensure_event_detail(event, events_dir=events_dir, catalog=catalog, catalog_path=catalog_path)
                catalog = load_catalog(catalog_path)
                meta = catalog_entry(catalog, event)
                merged = resolve_event_metadata(meta, detail)
                gr = gr_prediction(merged)
                notes.append(f"{event}: fetched and cached event_detail.json (PE parameters)")
            except Exception as exc:
                notes.append(f"{event}: could not fetch event detail JSON ({exc})")
        if not merged:
            notes.append(f"{event}: not in cached catalog; skipped")
            continue
        if not gr["ok"]:
            notes.append(f"{event}: catalog/PE lacks {', '.join(gr['missing'])}; skipped")
            continue
        try:
            gps = float(merged.get("GPS"))
        except (TypeError, ValueError):
            notes.append(f"{event}: catalog has no GPS time; skipped")
            continue
        files = files if strain_path else _ligo_pick_files(files, gps)
        segs = []
        for path in files:
            try:
                segs.append(load_strain_file(path))
            except Exception as exc:
                notes.append(f"{event} ({Path(path).name}): strain not usable ({exc}); skipped")

        if mode == "coherent" and len(segs) >= 2:
            try:
                preps = common_grid(segs, gps)
                usable = [p for p in preps if p["snr"] >= min_snr]
                for p in preps:
                    if p["snr"] < min_snr:
                        notes.append(f"{event} ({p['detector']}): envelope SNR {p['snr']:.1f} < {min_snr:g}; left out of the stack")
                if not usable:
                    notes.append(f"{event}: no detector above SNR {min_snr:g}; skipped")
                    continue
                comb = coherent_combine(usable)
                ip, net = network_snr(comb)
                veto = null_stream_veto(comb, ip)
                label = f"{event} ({'+'.join(sorted(comb['detectors']))})"
                prep = {"y": comb["stack"], "fs": comb["fs"], "off": comb["off"], "ip": ip, "snr": net,
                        "whitened": all(p["whitened"] for p in usable), "bg": comb["bg"], "band_hz": usable[0]["band_hz"]}
                res = ringdown_from_prepared(prep, dt_start_s=dt_start, fit_s=fit_s)
                extra = {"network_snr": net, "reference": comb["reference"],
                         "align": {d: {"dt_ms": a["dt_s"] * 1e3, "dphi_rad": a["dphi_rad"], "ratio": a["ratio"],
                                       "coherence": a["coherence"]} for d, a in comb["align"].items()},
                         "weights": comb["weights"], "snr_individual": comb["snr_individual"], **veto}
                if do_scan:
                    try:
                        extra["postmerger_scan"] = scan_from_preps(usable, comb, ip, veto["null_veto_passed"])
                    except Exception as exc:
                        notes.append(f"{label}: post-merger scan failed ({exc})")
                pt = _ligo_point(label, event, "+".join(sorted(comb["detectors"])), res, gr, merged, extra)
                if not veto["null_veto_passed"]:
                    points.append({**pt, "vetoed": True})
                    notes.append(f"{label}: VETOED by null stream (χ²_null,red={veto['null_stream_chi2_red']:.2f}, "
                                 f"null fraction={veto['null_fraction']:.2f}); not scored")
                    continue
                _score(pt)
                pm = extra.get("postmerger_scan")
                if pm:
                    notes.append(
                        f"{label}: post-merger scan loudest delay={pm['delay_ms']:.0f} ms f={pm['f_hz']:.0f} Hz "
                        f"τ={pm['tau_ms']:.0f} ms ρ={pm['network_snr']:.2f} FAP={pm['fap']:.3g} "
                        f"({pm['sigma_equiv']:.2f}σ, N_bg={pm['n_background']}) detected={pm['detected']}"
                    )
            except Exception as exc:
                notes.append(f"{event}: coherent analysis failed ({exc}); skipped")
            continue

        for seg in segs:
            try:
                res = measure_ringdown(seg.strain, seg.sample_rate, gps - seg.gps_start, dt_start_s=dt_start, fit_s=fit_s)
            except Exception as exc:  # window outside segment etc.
                notes.append(f"{event} ({Path(seg.path).name}): strain not usable ({exc}); skipped")
                continue
            label = f"{event} ({seg.detector})"
            if res["snr_peak"] < min_snr:
                notes.append(f"{label}: envelope SNR {res['snr_peak']:.1f} < {min_snr:g}; skipped")
                continue
            _score(_ligo_point(label, event, seg.detector, res, gr, merged, {"file": Path(seg.path).name}))
    if not D:
        raise LoaderError("No GWOSC event had both usable strain and catalog final mass/spin/redshift", notes)
    _LIGO_EXTRA["points"] = points
    _LIGO_EXTRA["rows"] = rows
    _LIGO_EXTRA["joint_constraints"] = _ligo_joint([p for p in points if not p.get("vetoed")])
    j = _LIGO_EXTRA["joint_constraints"]
    if j["delta_f_over_f"]:
        notes.append(f"joint ⟨δf/f⟩={j['delta_f_over_f']['mean']:+.3f}±{j['delta_f_over_f']['sigma']:.3f} "
                     f"(N={j['delta_f_over_f']['n']})")
    if j["delta_gamma_over_gamma"]:
        notes.append(f"joint ⟨δγ/γ⟩={j['delta_gamma_over_gamma']['mean']:+.3f}±{j['delta_gamma_over_gamma']['sigma']:.3f} "
                     f"(N={j['delta_gamma_over_gamma']['n']})")
    return np.asarray(D), np.asarray(B), np.asarray(S), np.asarray(X), notes


def _load_cern(options: dict | None = None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """CMS opposite-sign dimuon mass histogram (data) vs Drell-Yan MC normalised to data."""
    from menus.particle.cms.dimuon import (
        DATA_TARGET,
        MASS_RANGE_GEV,
        MC_TARGET,
        N_BINS,
        cached_root_files,
        histogram_comparison,
        read_dimuon_masses,
    )

    opts = options or {}
    data_files = cached_root_files(DATA_TARGET)
    mc_files = cached_root_files(MC_TARGET)
    if not data_files or not mc_files:
        raise FileNotFoundError(
            f"Need cached ROOT files for {DATA_TARGET} and {MC_TARGET} under datasets/cern/ "
            f"(found {len(data_files)} data, {len(mc_files)} MC); run 'Pull' first"
        )
    max_ev = int(opts.get("max_events") or 2_000_000)
    lo, hi = MASS_RANGE_GEV
    res = histogram_comparison(read_dimuon_masses(data_files, max_events=max_ev), read_dimuon_masses(mc_files, max_events=max_ev))
    notes = [
        "Step 1 baseline: Drell-Yan (DYJetsToLL) MC, normalised to the data integral (no lumi/xsec weights).",
        f"mass_range_GeV={lo}-{hi} n_bins={N_BINS} max_events_per_sample={max_ev}",
        f"n_data_pairs={res['n_data']} n_mc_pairs={res['n_mc']} mc_scale={res['mc_scale']:.4g}",
        "σ = sqrt(N_data + scale²·N_MC) per bin.",
    ]
    return res["D"], res["B"], res["sigma"], res["x"], notes


def _load_casimir(options: dict | None = None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """Measured force vs separation (user-selected CSV) against the ideal-conductor PFA force."""
    from menus.particle.casimir.force_table import find_force_tables, ideal_force, load_force_table

    opts = options or {}
    path = opts.get("casimir_force_file")
    if not path:
        tables = find_force_tables()
        if not tables:
            raise FileNotFoundError(
                "No Casimir force-vs-separation table (datasets/casimir/**/*force*.csv with "
                "separation_m, force_N[, sigma_N]); the cached downloads are not force tables"
            )
        path = tables[0]
    radius = opts.get("sphere_radius_m")
    area = opts.get("plate_area_m2")
    if not radius and not area:
        raise ValueError("Casimir geometry not given (sphere_radius_m or plate_area_m2)")
    t = load_force_table(path)
    B = ideal_force(t["d"], sphere_radius_m=float(radius) if radius else None, plate_area_m2=float(area) if area else None)
    notes = [
        "Step 1 baseline: ideal-conductor T=0 Casimir force (PFA sphere–plate or parallel plates); "
        "Lifshitz finite-conductivity/temperature corrections not applied.",
        f"casimir_file={Path(path).name}",
        f"geometry={'sphere_radius_m=' + str(radius) if radius else 'plate_area_m2=' + str(area)}",
    ]
    if t["sigma"] is None:
        notes.append("No sigma_N column: chi2_Lambda is an unweighted sum of squares.")
    return t["F"], B, t["sigma"], t["d"], notes


_PRIME_PAST_OBSERVABLES = (
    # (residual-row label, curated bbn_abundances observable, AlterBBN observable)
    ("D/H", "D_H", "D_H"),
    ("Y_p", "Y_p_mass_He4", "Y_p"),
)

_PRIME_PAST_EXTRA: dict[str, Any] = {}


def _load_prime_past(options: dict | None = None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """Prime Past BBN: D = Cooke 2018 D/H and Aver 2021 Y_p, B = AlterBBN v2.2 at Planck 2018 inputs.

    Author ruling 2026-11-04: AlterBBN is the standard-BBN baseline (the toy network
    menus/prime_past/bbn_network.py is deprecated). The run is cached under
    datasets/prime_past/alterbbn/. B is used only when bbn_isolated gates G1–G5 are green.
    Template D has no formula yet, so the SB prediction is pending and chi2_SB stays None.
    """
    from bbn_isolated import alterbbn
    from bbn_isolated.runner import evaluate, first_failed
    from menus.empirical_tests.tests import load_or_download

    opts = options or {}
    cache_dir = opts.get("alterbbn_cache_dir")
    _PRIME_PAST_EXTRA.clear()
    ev = evaluate(cache_dir=cache_dir)
    failed = first_failed(ev["gates"])
    _PRIME_PAST_EXTRA["bbn_gates"] = ev["gates"]
    if failed:
        raise RuntimeError(f"AlterBBN baseline fails bbn_isolated gate {failed}; not scored")
    obs = load_or_download("bbn_abundances")
    if obs is None or obs.empty:
        raise FileNotFoundError("curated bbn_abundances table did not materialize")
    obs = obs.set_index("observable")
    vals = ev["nominal"]["values"]
    D = np.array([float(obs.loc[k, "value"]) for _, k, _ in _PRIME_PAST_OBSERVABLES])
    sigma = np.array([float(obs.loc[k, "unc"]) for _, k, _ in _PRIME_PAST_OBSERVABLES])
    B = np.array([float(vals[a]) for _, _, a in _PRIME_PAST_OBSERVABLES])
    pulls = (D - B) / sigma
    inp = ev["inputs"]
    notes = [
        f"Step 1 baseline: {alterbbn.ALTERBBN_VERSION} standard BBN at Planck 2018 inputs "
        f"(eta={inp['eta']:.4g}, tau_n={inp['tau_n']} s, N_eff={inp['n_eff']}).",
        "bbn_isolated gates: " + " ".join(f"{g}={'pass' if v else 'FAIL'}" for g, v in ev["gates"].items()),
        f"alterbbn_cache={'hit' if ev['nominal'].get('from_cache') else 'written'}: "
        f"{Path(ev['nominal']['cache_path']).name if ev['nominal'].get('cache_path') else 'none'}",
        "Data: Cooke, Pettini & Steidel 2018 D/H; Aver et al. 2021 Y_p (curated bbn_abundances).",
        "σ is the observational error only (no AlterBBN rate-uncertainty term).",
        "Y_p baseline is AlterBBN 4·n_He4/n_b (Y_p^BBN convention).",
    ]
    for (label, _, _), d, b, s_, p in zip(_PRIME_PAST_OBSERVABLES, D, B, sigma, pulls):
        notes.append(f"{label}: D={d:.5g} ± {s_:.2g}, B={b:.5g}, r={d - b:+.4g}, pull={p:+.2f}σ")
    notes.append(
        "Template D has no formula yet: SB prediction pending; chi2_SB not computed."
    )
    _PRIME_PAST_EXTRA["observables"] = [label for label, _, _ in _PRIME_PAST_OBSERVABLES]
    _PRIME_PAST_EXTRA["pulls"] = {label: float(p) for (label, _, _), p in zip(_PRIME_PAST_OBSERVABLES, pulls)}
    _PRIME_PAST_EXTRA["table"] = [
        {"observable": label, "D": float(d), "B": float(b), "sigma": float(s_), "pull": float(p)}
        for (label, _, _), d, b, s_, p in zip(_PRIME_PAST_OBSERVABLES, D, B, sigma, pulls)
    ]
    return D, B, sigma, None, notes


def _load_generic_note(kind: str) -> tuple[None, None, None, None, list[str]]:
    return (
        None,
        None,
        None,
        None,
        [
            f"Step 0 loader for {kind} is unchanged on disk but was not asked to score.",
            "Step 1 conventional baseline is the dataset's ΛCDM / SM / GR / Newtonian reduction.",
            "Step 3 numeric S_k kernel is TODO.",
        ],
    )


def run_superblock_residual(
    module_tag: str,
    action: str,
    options: dict | None = None,
) -> dict[str, Any]:
    """Steps 0–4 for one live menu action."""
    from sb_shared.legacy_names import canonical_module_tag
    module_tag = canonical_module_tag(module_tag)
    template = DATASET_TEMPLATE.get(module_tag, "E")
    load_error = None
    try:
        if module_tag == "DESI_BAO":
            D, B, sigma, z, notes = _load_desi()
        elif module_tag == "SPARC":
            D, B, sigma, z, notes = _load_sparc_one()
        elif module_tag == "FRB_DM":
            D, B, sigma, z, notes = _load_frb()
        elif module_tag == "LHCB_FLAVOR":
            D, B, sigma, z, notes = _load_lhcb(options)
        elif module_tag == "PLANCK_CMB":
            D, B, sigma, z, notes = _load_planck(options)
        elif module_tag == "LIGO_GWOSC":
            D, B, sigma, z, notes = _load_ligo(options)
        elif module_tag == "CERN_OPENDATA":
            D, B, sigma, z, notes = _load_cern(options)
        elif module_tag == "CASIMIR":
            D, B, sigma, z, notes = _load_casimir(options)
        elif module_tag == "PRIME_PAST_HARMONIC":
            D, B, sigma, z, notes = _load_prime_past(options)
        else:
            D, B, sigma, z, notes = _load_generic_note(module_tag)
    except Exception as exc:
        D = B = sigma = z = None
        notes = list(getattr(exc, "loader_notes", None) or [])
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
        radius=_SPARC_GEOMETRY.get("radius") if module_tag == "SPARC" else None,
        groups=_SPARC_GEOMETRY.get("groups") if module_tag == "SPARC" else None,
        q2=z if module_tag == "LHCB_FLAVOR" else None,
    )
    if module_tag in TEMPLATE_C_REFUSED and template == "C":
        report["notes"] = list(report.get("notes") or []) + [
            f"Template C (Mirron kernel) refused for {module_tag}: {TEMPLATE_C_REFUSAL}."
        ]
    if module_tag == "PRIME_PAST_HARMONIC":
        report["sb_prediction"] = SB_PREDICTION_PENDING
        for key in ("bbn_gates", "observables", "pulls", "table"):
            if key in _PRIME_PAST_EXTRA:
                report[key] = _PRIME_PAST_EXTRA[key]
    if module_tag == "LIGO_GWOSC" and not load_error and _LIGO_EXTRA.get("points"):
        report["points"] = list(_LIGO_EXTRA["points"])
        report["rows"] = list(_LIGO_EXTRA.get("rows") or [])
        joint = dict(_LIGO_EXTRA.get("joint_constraints") or {})
        if report.get("chi2_Lambda") is not None and report.get("N"):
            joint["chi2_Lambda"] = report["chi2_Lambda"]
            joint["chi2_Lambda_per_N"] = report["chi2_Lambda"] / report["N"]
        report["joint_constraints"] = joint
        report["postmerger_scan"] = {p["label"]: p["postmerger_scan"] for p in report["points"] if p.get("postmerger_scan")}
    if module_tag == "LHCB_FLAVOR" and _LHCB_EXTRA:
        report["lhcb_provenance"] = {k: v for k, v in _LHCB_EXTRA.items() if k != "bins"}
        if _LHCB_EXTRA.get("bins"):
            report["lhcb_bins"] = list(_LHCB_EXTRA["bins"])
        for key in ("fetch_source", "observable", "inspire_id", "arxiv_id", "citation", "real_data"):
            report[key] = _LHCB_EXTRA.get(key)
        if _LHCB_EXTRA.get("warning"):
            report["warning"] = _LHCB_EXTRA["warning"]
    if module_tag == "DESI_BAO":
        from sb_engine.sound_horizon import sound_horizon

        report["sound_horizon"] = sound_horizon()
        report["notes"] = list(report.get("notes") or []) + [
            report["sound_horizon"]["note"]
        ]
    path = _write_report(report)
    report["report_path"] = path
    _print_report(report, path)
    return report
