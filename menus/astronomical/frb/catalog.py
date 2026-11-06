"""
FRB and cosmic-void catalog loaders (CHIME-style FRB table + void catalog under datasets/frb/).

Step 0 loader helpers for the Superblock residual engine: catalog I/O, Macquart IGM DM
baseline, and void/filament path classification.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree


IGM_DM_PER_Z = 855.0  # pc/cm^3 per unit z (Macquart-style mean relation)


def load_frb_catalog(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    lower = {col: col.lower() for col in frame.columns}
    frame = frame.rename(columns=lower)

    if "ra" not in frame.columns or "dec" not in frame.columns:
        raise ValueError(f"{path} must contain ra and dec columns.")

    dm_col = None
    for candidate in ("dm_exc_ne2001", "bonsai_dm", "dm_fitb", "dm"):
        if candidate in frame.columns:
            dm_col = candidate
            break
    if dm_col is None:
        raise ValueError(f"{path} must contain a DM column.")

    frame = frame.copy()
    frame["DM"] = pd.to_numeric(frame[dm_col], errors="coerce")
    frame["ra"] = pd.to_numeric(frame["ra"], errors="coerce")
    frame["dec"] = pd.to_numeric(frame["dec"], errors="coerce")

    if "excluded_flag" in frame.columns:
        flag = pd.to_numeric(frame["excluded_flag"], errors="coerce").fillna(0)
        frame = frame[flag == 0]

    frame = frame.dropna(subset=["ra", "dec", "DM"])
    frame = frame[(frame["DM"] > 0) & (frame["DM"] < 5000)]
    frame = frame[(frame["dec"] >= -90.0) & (frame["dec"] <= 90.0)]
    frame["ra"] = frame["ra"] % 360.0
    return frame.reset_index(drop=True)


def load_void_catalog(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    lower = {col: col.lower() for col in frame.columns}
    frame = frame.rename(columns=lower)

    ra_col = "ra" if "ra" in frame.columns else "radeg"
    dec_col = "dec" if "dec" in frame.columns else "dedeg"
    if ra_col not in frame.columns or dec_col not in frame.columns:
        raise ValueError(f"{path} must contain RA/Dec columns.")

    radius_col = None
    for candidate in ("reff_mpc", "reff", "radius_mpc", "radius"):
        if candidate in frame.columns:
            radius_col = candidate
            break
    if radius_col is None:
        frame["reff_mpc"] = 15.0
        radius_col = "reff_mpc"

    z_col = "z_void" if "z_void" in frame.columns else "z"
    if z_col not in frame.columns:
        frame["z_void"] = 0.05

    frame = frame.copy()
    frame["ra"] = pd.to_numeric(frame[ra_col], errors="coerce")
    frame["dec"] = pd.to_numeric(frame[dec_col], errors="coerce")
    frame["reff_mpc"] = pd.to_numeric(frame[radius_col], errors="coerce")
    frame["z_void"] = pd.to_numeric(frame[z_col], errors="coerce")
    frame = frame.dropna(subset=["ra", "dec", "reff_mpc"])
    frame = frame[frame["reff_mpc"] > 0]
    return frame.reset_index(drop=True)


def expected_dm_igm(z: np.ndarray) -> np.ndarray:
    return IGM_DM_PER_Z * np.asarray(z, dtype=float) + 50.0


def _cartesian_unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return coord.cartesian.xyz.value.T


def _angular_diameter_distance_mpc(z: np.ndarray) -> np.ndarray:
    """Low-z analytic approximation (flat LambdaCDM, H0=70)."""
    z = np.asarray(z, dtype=float)
    h0 = 70.0
    c_km_s = 299792.458
    return (c_km_s / h0) * z / (1.0 + z)


def classify_paths(
    frb_ra: np.ndarray,
    frb_dec: np.ndarray,
    voids: pd.DataFrame,
    *,
    void_center_fraction: float = 0.35,
    boundary_shell_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify each FRB sightline relative to the nearest cosmic void.

    Returns path_type labels and angular separation to nearest void (deg).
    """
    void_coords = SkyCoord(
        ra=voids["ra"].to_numpy() * u.deg,
        dec=voids["dec"].to_numpy() * u.deg,
    )
    frb_coords = SkyCoord(ra=frb_ra * u.deg, dec=frb_dec * u.deg)

    void_xyz = _cartesian_unit_vectors(voids["ra"].to_numpy(), voids["dec"].to_numpy())
    frb_xyz = _cartesian_unit_vectors(frb_ra, frb_dec)
    tree = cKDTree(void_xyz)
    dist, idx = tree.query(frb_xyz, k=1)

    sep = frb_coords.separation(void_coords[idx]).deg
    z_void = voids["z_void"].to_numpy()[idx]
    reff_mpc = voids["reff_mpc"].to_numpy()[idx]
    da_mpc = _angular_diameter_distance_mpc(z_void)
    theta_void_deg = np.degrees(np.arctan2(reff_mpc, da_mpc))

    labels = np.full(len(frb_ra), "filament", dtype=object)
    center_mask = sep <= void_center_fraction * theta_void_deg
    boundary_mask = (~center_mask) & (
        sep <= (1.0 + boundary_shell_fraction) * theta_void_deg
    )
    labels[center_mask] = "void_center"
    labels[boundary_mask] = "boundary_cross"
    return labels, sep
