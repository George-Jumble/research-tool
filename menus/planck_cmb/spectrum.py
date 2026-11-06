"""Planck 2018 (PR3) binned TT power spectrum with its ΛCDM best-fit column.

The PLA file ``COM_PowerSpect_CMB-TT-binned_R3.01.txt`` has columns
``l, D_l, -dD_l, +dD_l, BestFit`` (μK²). It is cached next to the module's FITS files
(``menus/planck_cmb/fits/``) and restored from the finished-data archives like the masks.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from menus.planck_cmb.fetcher import FITS_DIR, ensure_output_dirs
from sb_shared.project_paths import finished_fits_search_dirs

TT_BINNED_FILE = "COM_PowerSpect_CMB-TT-binned_R3.01.txt"
TT_BINNED_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/"
    + TT_BINNED_FILE
)


def find_cached_tt_spectrum(cache_dir: Path | str = FITS_DIR) -> Path | None:
    for directory in (Path(cache_dir), *finished_fits_search_dirs()):
        candidate = directory / TT_BINNED_FILE
        if candidate.is_file() and candidate.stat().st_size > 500:
            return candidate
    return None


def ensure_tt_spectrum(*, cache_dir: Path | str = FITS_DIR, auto_fetch: bool = True) -> Path:
    """Cached file, else restore from archive, else wget from IRSA (when auto_fetch)."""
    cache_dir = Path(cache_dir)
    dest = cache_dir / TT_BINNED_FILE
    found = find_cached_tt_spectrum(cache_dir)
    if found is not None:
        if found != dest:
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found, dest)
        return dest
    if not auto_fetch:
        raise FileNotFoundError(
            f"Planck TT spectrum not cached: {dest} (fetch: wget -O {dest} '{TT_BINNED_URL}')"
        )
    ensure_output_dirs()
    partial = dest.with_suffix(dest.suffix + ".partial")
    result = subprocess.run(
        ["wget", "-q", "--timeout=60", "--tries=1", "-O", str(partial), TT_BINNED_URL],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not partial.is_file() or partial.stat().st_size < 500:
        if partial.exists():
            partial.unlink()
        raise FileNotFoundError(
            f"Could not download Planck TT spectrum ({TT_BINNED_URL}); "
            f"wget rc={result.returncode}. Manual: wget -O {dest} '{TT_BINNED_URL}'"
        )
    partial.replace(dest)
    return dest


def load_tt_binned(path: Path | str) -> dict[str, np.ndarray]:
    """Parse the binned file; σ = mean of the asymmetric errors."""
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"{path}: expected 5 columns (l, Dl, -dDl, +dDl, BestFit)")
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return {
        "ell": arr[:, 0],
        "Dl": arr[:, 1],
        "sigma": 0.5 * (np.abs(arr[:, 2]) + np.abs(arr[:, 3])),
        "bestfit": arr[:, 4],
    }
