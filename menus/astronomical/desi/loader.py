"""
DESI DR2 BAO table loader (CobayaSampler/bao_data cache under datasets/desi/).

Step 0 loader for the Superblock residual engine. Reads the Gaussian BAO mean/cov tables,
inventories per-tracer quantities and resolves the data mode.
"""

from __future__ import annotations
import re
from pathlib import Path
from collections.abc import Sequence
from typing import Any
import numpy as np
from menus.astronomical.desi.fetcher import DATASETS_DIR, DEFAULT_COBAYA_ROOT


QUANTITY_ALIASES: dict[str, str] = {
    "DM_over_rs_only": "DM_over_rs",
    "DH_over_rs_only": "DH_over_rs",
    "DV_over_rs_only": "DV_over_rs",
}


JOINT_DH_DM: str = "joint_DH_DM"


DATA_MODE_DH_ONLY: str = "DH_only"


DATA_MODE_DM_ONLY: str = "DM_only"


DATA_MODE_DV_ONLY: str = "DV_only"


DATA_MODE_ANISOTROPIC: str = "anisotropic"


DATA_MODE_MIXED: str = "mixed"


DATA_MODE_JOINT: str = "Joint_DH_DM"


DATA_MODE_DEFAULT: str = DATA_MODE_DH_ONLY


DATA_MODE: str = "DH_only"     # default production scan (DH_only); Method 10 uses DM_only


TRACER: str = "ALL_GCcomb"


TRACER_DEFAULT_MODE: dict[str, str] = {
    "BGS_BRIGHT-21.35_GCcomb": DATA_MODE_DV_ONLY,
    "QSO_GCcomb": DATA_MODE_ANISOTROPIC,
    "Lya_GCcomb": DATA_MODE_ANISOTROPIC,
    "LRG_GCcomb_z0.4-0.6": DATA_MODE_ANISOTROPIC,
    "LRG_GCcomb_z0.6-0.8": DATA_MODE_ANISOTROPIC,
    "LRG+ELG_LOPnotqso_GCcomb": DATA_MODE_ANISOTROPIC,
    "ELG_LOPnotqso_GCcomb_z1.1-1.6": DATA_MODE_ANISOTROPIC,
    "ALL_GCcomb": DATA_MODE_MIXED,
}


DATA_MODE_TO_QUANTITY: dict[str, str | None] = {
    DATA_MODE_DH_ONLY: "DH_over_rs",
    DATA_MODE_DM_ONLY: "DM_over_rs",
    DATA_MODE_DV_ONLY: "DV_over_rs",
    DATA_MODE_ANISOTROPIC: JOINT_DH_DM,
    DATA_MODE_JOINT: JOINT_DH_DM,
    DATA_MODE_MIXED: None,
}


_DATA_MODE_INPUT_ALIASES: dict[str, str] = {
    "DH_only": DATA_MODE_DH_ONLY,
    "DH_over_rs": DATA_MODE_DH_ONLY,
    "DH_over_rs_only": DATA_MODE_DH_ONLY,
    "DM_only": DATA_MODE_DM_ONLY,
    "DM_over_rs": DATA_MODE_DM_ONLY,
    "DM_over_rs_only": DATA_MODE_DM_ONLY,
    "DV_only": DATA_MODE_DV_ONLY,
    "DV_over_rs": DATA_MODE_DV_ONLY,
    "DV_over_rs_only": DATA_MODE_DV_ONLY,
    "anisotropic": DATA_MODE_ANISOTROPIC,
    "mixed": DATA_MODE_MIXED,
    "auto": DATA_MODE_MIXED,
    "Joint_DH_DM": DATA_MODE_JOINT,
    "joint_DH_DM": DATA_MODE_JOINT,
    JOINT_DH_DM: DATA_MODE_JOINT,
}


def normalize_data_mode(mode: str | None) -> str:
    """Map UI / legacy quantity labels to a canonical DATA_MODE value."""
    if mode is None:
        return DATA_MODE_DEFAULT
    key = str(mode).strip()
    if not key:
        return DATA_MODE_DEFAULT
    resolved = _DATA_MODE_INPUT_ALIASES.get(key)
    if resolved is None:
        resolved = _DATA_MODE_INPUT_ALIASES.get(QUANTITY_ALIASES.get(key, ""))
    if resolved is None:
        raise ValueError(f"Unknown DATA_MODE: {mode}")
    return resolved


def data_mode_to_quantity_filter(mode: str | None) -> str | None:
    """Translate DATA_MODE (or alias) to the row filter used in Cobaya tables."""
    return DATA_MODE_TO_QUANTITY[normalize_data_mode(mode)]


def normalize_quantity_filter(quantity_filter: str | None) -> str | None:
    """
    Map UI presets to loader filters.

    Returns ``None`` for the full published vector (all channels), ``joint_DH_DM``
    for DM+DH only, or a single quantity label e.g. ``DH_over_rs``.
    """
    if quantity_filter is None:
        return None
    q = str(quantity_filter).strip()
    if not q or q.lower() in {"all", "*", "none"}:
        return None
    if q in _DATA_MODE_INPUT_ALIASES or q in QUANTITY_ALIASES:
        try:
            return data_mode_to_quantity_filter(q)
        except ValueError:
            pass
    if q == JOINT_DH_DM:
        return JOINT_DH_DM
    return QUANTITY_ALIASES.get(q, q)


def _quantity_selection_mask(quantities: list[str], quantity_filter: str | None) -> np.ndarray:
    """Boolean mask for rows to keep given a normalized quantity filter."""
    if quantity_filter is None:
        return np.ones(len(quantities), dtype=bool)
    if quantity_filter == JOINT_DH_DM:
        return np.array([q in {"DM_over_rs", "DH_over_rs"} for q in quantities], dtype=bool)
    return np.array([q == quantity_filter for q in quantities], dtype=bool)


def _parse_cobaya_mean_table(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        z_val, obs_val, quantity = float(parts[0]), float(parts[1]), parts[2]
        rows.append((z_val, obs_val, quantity))
    if not rows:
        raise ValueError(f"No data rows parsed from {path}")
    z = np.array([r[0] for r in rows], dtype=float)
    observable = np.array([r[1] for r in rows], dtype=float)
    quantities = [r[2] for r in rows]
    return z, observable, quantities


def _parse_cobaya_cov_table(path: Path) -> np.ndarray:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.extend(float(x) for x in line.split())
    if not values:
        raise ValueError(f"No covariance entries in {path}")
    n = int(round(np.sqrt(len(values))))
    if n * n != len(values):
        raise ValueError(f"Covariance file {path} is not square ({len(values)} entries)")
    return np.array(values, dtype=float).reshape(n, n)


def resolve_cobaya_dr2_dir(local_path: str | Path | None = None) -> Path:
    """Resolve path to ``desi_bao_dr2`` inside a cloned bao_data repo."""
    if local_path is None:
        if DEFAULT_COBAYA_ROOT.is_dir():
            return DEFAULT_COBAYA_ROOT
        raise FileNotFoundError(
            "DESI DR2 tables not found. Clone Cobaya bao_data:\n"
            "  git clone https://github.com/CobayaSampler/bao_data.git "
            f"{DATASETS_DIR / 'bao_data'}"
        )
    root = Path(local_path)
    if (root / "desi_bao_dr2").is_dir():
        return root / "desi_bao_dr2"
    if root.name == "desi_bao_dr2":
        return root
    if root.name == "bao_data":
        candidate = root / "desi_bao_dr2"
        if candidate.is_dir():
            return candidate
    if root.is_dir() and any(root.glob("desi_gaussian_bao_*_mean.txt")):
        return root
    raise FileNotFoundError(f"Could not locate desi_bao_dr2 tables under {root}")


def tracer_available_quantities(
    local_path: str | Path | None, tracer: str
) -> list[str]:
    """Inventory quantity tags from the third column of a tracer mean table."""
    dr2_dir = resolve_cobaya_dr2_dir(local_path)
    mean_path = dr2_dir / f"desi_gaussian_bao_{tracer}_mean.txt"
    if not mean_path.is_file():
        return []
    _z, _obs, quantities = _parse_cobaya_mean_table(mean_path)
    return sorted(set(quantities))


def inventory_tracer_quantities(
    local_path: str | Path | None, tracer: str
) -> list[str]:
    """Parse ``desi_gaussian_bao_<TRACER>_mean.txt`` quantity column; do not assume DH."""
    return tracer_available_quantities(local_path, tracer)


def tracer_supports_mode(available: Sequence[str], mode: str) -> bool:
    """True if the tracer mean file can satisfy ``mode``."""
    av = set(available)
    key = normalize_data_mode(mode)
    if key == DATA_MODE_DH_ONLY:
        return "DH_over_rs" in av
    if key == DATA_MODE_DM_ONLY:
        return "DM_over_rs" in av
    if key == DATA_MODE_DV_ONLY:
        return "DV_over_rs" in av
    if key in {DATA_MODE_ANISOTROPIC, DATA_MODE_JOINT}:
        return {"DM_over_rs", "DH_over_rs"} <= av
    if key == DATA_MODE_MIXED:
        return bool(av & {"DH_over_rs", "DM_over_rs", "DV_over_rs"})
    return False


def resolve_data_mode_for_tracer(
    requested: str | None,
    tracer: str,
    available: Sequence[str],
    *,
    log: bool = True,
) -> str | None:
    """
    Pick a load mode the mean file can actually satisfy.

    Requested mode if that quantity exists; else fall back:
    DH_only missing → anisotropic if both DM+DH exist;
    anisotropic missing → DV_only if DV exists (DESI DR2 BGS = isotropic D_V only).
    Nothing usable → None (caller skips or hard-errors if this tracer is required).
    """
    avail_list = list(available)
    requested_key: str | None = None
    if requested is not None and str(requested).strip():
        try:
            requested_key = normalize_data_mode(requested)
        except ValueError:
            requested_key = None

    chosen: str | None = None
    if requested_key and tracer_supports_mode(avail_list, requested_key):
        chosen = requested_key
    else:
        if requested_key in {None, DATA_MODE_DH_ONLY, DATA_MODE_DM_ONLY, DATA_MODE_JOINT}:
            if tracer_supports_mode(avail_list, DATA_MODE_ANISOTROPIC):
                chosen = DATA_MODE_ANISOTROPIC
            elif tracer_supports_mode(avail_list, DATA_MODE_DV_ONLY):
                chosen = DATA_MODE_DV_ONLY
        if chosen is None:
            default = TRACER_DEFAULT_MODE.get(tracer)
            if default and tracer_supports_mode(avail_list, default):
                chosen = normalize_data_mode(default)
        if chosen is None and tracer_supports_mode(avail_list, DATA_MODE_MIXED):
            chosen = DATA_MODE_MIXED

    if log:
        req_s = requested_key or requested or "unset"
        if chosen is None:
            print(
                f"[DESI LOAD] tracer={tracer} requested={req_s} "
                f"available={avail_list} → skip (no usable quantities)"
            )
        elif chosen != requested_key:
            iso = " (isotropic)" if chosen == DATA_MODE_DV_ONLY else ""
            print(
                f"[DESI LOAD] tracer={tracer} requested={req_s} "
                f"available={avail_list} → using {chosen}{iso}"
            )
    return chosen


def dv_over_rd_from_dm_dh(
    z: float,
    dm_over_rd: float,
    dh_over_rd: float,
) -> float:
    """
    Isotropic D_V/r_d from D_M/r_d and D_H/r_d.

    D_V = (z D_M^2 D_H)^{1/3}, so
    D_V/r_d = (z (D_M/r_d)^2 (D_H/r_d))^{1/3}.
    DESI DR2 BGS at z=0.295 is published as this combination only — do not
    invent a D_H point.
    """
    z_pos = max(float(z), 1e-12)
    return float((z_pos * float(dm_over_rd) ** 2 * float(dh_over_rd)) ** (1.0 / 3.0))


def list_desi_tracers(local_path: str | Path | None = None) -> list[str]:
    """Return tracer keys available in a Cobaya DR2 directory."""
    dr2_dir = resolve_cobaya_dr2_dir(local_path)
    tracers: list[str] = []
    pattern = re.compile(r"desi_gaussian_bao_(.+)_mean\.txt$")
    for mean_file in sorted(dr2_dir.glob("desi_gaussian_bao_*_mean.txt")):
        match = pattern.match(mean_file.name)
        if match:
            tracers.append(match.group(1))
    return tracers


def load_desi_from_cobaya_repo(
    local_path: str | Path | None = None,
    *,
    tracer: str = "ALL_GCcomb",
    quantity_filter: str | None = "DM_over_rs",
) -> dict[str, Any]:
    """
    Load DESI DR2 Gaussian BAO tables from CobayaSampler/bao_data.

    Parameters
    ----------
    local_path:
        Path to cloned ``bao_data`` repo, ``desi_bao_dr2/`` folder, or None
        for ``datasets/desi/bao_data/desi_bao_dr2``.
    tracer:
        Tracer key, e.g. ``ALL_GCcomb``, ``LRG_GCcomb_z0.4-0.6``, ``Lya_GCcomb``.
    quantity_filter:
        Row filter preset:

        - ``None`` / ``all`` — full 13-point ALL_GCcomb vector (mixed channels)
        - ``DH_over_rs`` / ``DH_over_rs_only`` — 6 D_H/r_d points
        - ``DM_over_rs`` / ``DM_over_rs_only`` — 6 D_M/r_d points
        - ``joint_DH_DM`` — 12 DM+DH points (no DV; still mixed unless joint model)
        - ``DV_over_rs`` — 1 point only (underpowered)
    """
    dr2_dir = resolve_cobaya_dr2_dir(local_path)
    mean_path = dr2_dir / f"desi_gaussian_bao_{tracer}_mean.txt"
    cov_path = dr2_dir / f"desi_gaussian_bao_{tracer}_cov.txt"
    if not mean_path.is_file():
        available = ", ".join(list_desi_tracers(dr2_dir))
        raise FileNotFoundError(
            f"Missing {mean_path.name}. Available tracers: {available}"
        )
    if not cov_path.is_file():
        raise FileNotFoundError(f"Missing covariance file: {cov_path}")

    z, observable, quantities = _parse_cobaya_mean_table(mean_path)
    cov = _parse_cobaya_cov_table(cov_path)
    if cov.shape[0] != len(observable):
        raise ValueError(
            f"Mean vector ({len(observable)}) and covariance ({cov.shape[0]}) "
            f"size mismatch for tracer {tracer}"
        )

    norm_filter = normalize_quantity_filter(quantity_filter)
    mask = _quantity_selection_mask(quantities, norm_filter)
    if not np.any(mask):
        raise ValueError(
            f"Tracer {tracer} has no rows for quantity={quantity_filter!r}. "
            f"Available: {sorted(set(quantities))}"
        )
    z = z[mask]
    observable = observable[mask]
    quantities = [q for q, keep in zip(quantities, mask, strict=True) if keep]
    idx = np.where(mask)[0]
    cov = cov[np.ix_(idx, idx)]

    label = f"DESI_DR2_{tracer}"
    if norm_filter:
        label += f"_{norm_filter}"
    mixed_channels = len(set(quantities)) > 1

    from menus.astronomical.desi.covariance import prepare_covariance

    cov, cov_health = prepare_covariance(cov, name=label)
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    return {
        "z": z,
        "observable": observable,
        "err": err,
        "quantity": quantities,
        "quantity_filter": norm_filter,
        "quantity_types": sorted(set(quantities)),
        "mixed_channels": mixed_channels,
        "cov": cov,
        "cov_health": cov_health,
        "label": label,
        "tracer": tracer,
        "source": str(mean_path),
        "n_data": len(z),
        "has_full_covariance": True,
    }


def get_data_by_mode(
    mode: str | None = None,
    tracer: str = TRACER,
    *,
    local_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load DESI DR2 rows for a tracer using the quantities the mean file actually has.

    Modes: ``DH_only``, ``DM_only``, ``DV_only``, ``anisotropic``, ``mixed``,
    ``Joint_DH_DM``. If ``mode`` is None, use :data:`TRACER_DEFAULT_MODE`.
    Missing DH on BGS falls back to isotropic ``DV_only`` (no invented D_H).
    """
    available = inventory_tracer_quantities(local_path, tracer)
    if mode is None:
        mode = TRACER_DEFAULT_MODE.get(tracer, DATA_MODE)
    requested = mode
    mode_key = resolve_data_mode_for_tracer(requested, tracer, available, log=True)
    if mode_key is None:
        raise ValueError(
            f"Tracer {tracer!r} has zero usable BAO quantities "
            f"(available={available}). Cannot load."
        )

    data = load_desi_from_cobaya_repo(
        local_path,
        tracer=tracer,
        quantity_filter=None,
    )
    quantities = np.asarray(data["quantity"], dtype=str)
    q_filter = data_mode_to_quantity_filter(mode_key)
    mask = _quantity_selection_mask(quantities.tolist(), q_filter)
    if not np.any(mask):
        raise ValueError(
            f"Tracer {tracer!r} resolved to {mode_key!r} but no matching rows remain "
            f"(available={sorted(set(quantities.tolist()))})."
        )

    y = np.asarray(data["observable"], dtype=float)[mask]
    cov = np.asarray(data["cov"], dtype=float)
    idx = np.where(mask)[0]
    cov = cov[np.ix_(idx, idx)]
    z = np.asarray(data["z"], dtype=float)[mask]
    err = np.asarray(data["err"], dtype=float)[mask]
    filtered_quantities = quantities[mask]

    label = f"DESI_DR2_{tracer}_{mode_key}"
    from menus.astronomical.desi.covariance import prepare_covariance

    cov, cov_health = prepare_covariance(cov, name=label)
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    filtered_metadata: dict[str, Any] = {
        "quantity": filtered_quantities,
        "z": z,
        "err": err,
        "tracer": tracer,
        "label": label,
        "source": data.get("source"),
        "data_mode": mode_key,
        "quantity_filter": q_filter,
        "quantity_types": sorted(set(filtered_quantities.tolist())),
        "mixed_channels": len(set(filtered_quantities.tolist())) > 1,
        "cov_health": cov_health,
        "n_data": len(y),
        "requested_mode": requested,
        "requested_tracer": tracer,
        "available_quantities": list(available),
    }

    print(f"\n[DATA MODE] {mode_key} | tracer={tracer} → {len(y)} points selected")
    if mode_key == DATA_MODE_DV_ONLY:
        print(
            "  (DESI DR2 BGS = isotropic D_V only; quadrupole S/N too low for (D_M, D_H). "
            "Do not invent a D_H point.)"
        )
    if mode_key in {DATA_MODE_JOINT, DATA_MODE_MIXED} and len(set(filtered_quantities.tolist())) > 1:
        print("  ⚠ Mixed-quantity vector (not recommended for a single oscillation template)")

    return y, cov, z, filtered_metadata


def load_scan_data_by_data_mode(
    mode: str = DATA_MODE,
    *,
    local_path: str | Path | None = None,
    tracer: str = TRACER,
) -> dict[str, Any]:
    """Return a scanner-compatible data dict for the requested DATA_MODE."""
    y, cov, _z, metadata = get_data_by_mode(
        mode,
        local_path=local_path,
        tracer=tracer,
    )
    quantities = metadata["quantity"]
    if isinstance(quantities, np.ndarray):
        quantities = quantities.tolist()
    return {
        "z": metadata["z"],
        "observable": y,
        "err": metadata["err"],
        "quantity": quantities,
        "quantity_filter": metadata["quantity_filter"],
        "data_mode": metadata["data_mode"],
        "quantity_types": metadata["quantity_types"],
        "mixed_channels": metadata["mixed_channels"],
        "cov": cov,
        "cov_health": metadata["cov_health"],
        "label": metadata["label"],
        "tracer": tracer,
        "source": metadata.get("source"),
        "n_data": metadata["n_data"],
        "has_full_covariance": True,
    }

