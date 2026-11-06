"""Measured Casimir force vs separation, against the ideal-conductor proximity-force baseline.

No bundled Casimir file is a force-vs-separation table in a fixed layout (the Zenodo drums zip is
organised per paper figure; the GitHub magnetic-fluid files are dielectric spectra). The loader reads
a user-selected CSV under ``datasets/casimir/`` with columns

    separation_m, force_N[, sigma_N]

and needs the geometry: ``sphere_radius_m`` (sphere–plate, PFA) or ``plate_area_m2`` (parallel plates).

Ideal (perfect conductor, T = 0) baselines:
    plate–plate  F = −π² ħ c A / (240 d⁴)
    sphere–plate F = −π³ ħ c R / (360 d³)
Finite conductivity / temperature (Lifshitz) corrections are not applied here.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from menus.particle.casimir.fetcher import DATASETS_DIR

HBAR = 1.054571817e-34
C_LIGHT = 299792458.0


def ideal_force(d_m, *, sphere_radius_m: float | None = None, plate_area_m2: float | None = None) -> np.ndarray:
    d = np.asarray(d_m, dtype=float)
    if sphere_radius_m:
        return -(math.pi**3) * HBAR * C_LIGHT * float(sphere_radius_m) / (360.0 * d**3)
    if plate_area_m2:
        return -(math.pi**2) * HBAR * C_LIGHT * float(plate_area_m2) / (240.0 * d**4)
    raise ValueError("Casimir geometry needed: sphere_radius_m or plate_area_m2")


def find_force_tables(root: Path = DATASETS_DIR) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.csv") if "force" in p.name.lower())


def load_force_table(path: Path | str) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in frame.columns}
    for need in ("separation_m", "force_n"):
        if need not in cols:
            raise ValueError(f"{path}: need columns separation_m, force_N[, sigma_N]")
    frame = frame.sort_values(cols["separation_m"])
    out = {
        "d": frame[cols["separation_m"]].to_numpy(float),
        "F": frame[cols["force_n"]].to_numpy(float),
    }
    out["sigma"] = frame[cols["sigma_n"]].to_numpy(float) if "sigma_n" in cols else None
    return out
