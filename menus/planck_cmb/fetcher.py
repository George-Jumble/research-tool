"""
Planck CMB map / mask fetcher. FITS files are cached under menus/planck_cmb/fits/ and archived
to the finished-data directories after use.
"""

from __future__ import annotations
import re
import shutil
import subprocess
from pathlib import Path
from sb_shared.project_paths import (
    FINISHED_DIR,
    LEGACY_FINISHED2_DIR,
    LEGACY_FINISHED_DIR,
    ensure_project_dirs,
    finished_fits_search_dirs,
    move_to_finished_archive,
)


PROJECT_ROOT = Path(__file__).resolve().parent


FITS_DIR = PROJECT_ROOT / "fits"


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


EXAMPLE_CMB_MAP = "COM_CMB_IQU-commander_2048_R3.00_full.fits"


EXAMPLE_CMB_MASK = "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"


CMB_MAP_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/"
    "component-maps/cmb/COM_CMB_IQU-commander_2048_R3.00_full.fits"
)


CMB_MASK_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/masks/"
    "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
)


PLANCK_MASK_URL_BASE = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_{release}/ancillary-data/masks/"
)


def ensure_output_dirs() -> None:
    FITS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_project_dirs()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def archive_used_fits(cmb_path: Path | str, mask_path: Path | str) -> list[Path]:
    """
    Move processed FITS files from ./fits/ into ../finished/.

    Only files under FITS_DIR are moved (explicit outside paths are left in place).
    Prior archives of the same FITS name are deleted before saving the current run.
    """
    ensure_output_dirs()
    moved: list[Path] = []
    seen: set[Path] = set()

    for src in (Path(cmb_path), Path(mask_path)):
        src = src.resolve()
        if src in seen or not src.is_file():
            continue
        seen.add(src)

        try:
            src.relative_to(FITS_DIR.resolve())
        except ValueError:
            print(f"[Superblock Residual Engine] Leaving in place (not under ./fits/): {src}")
            continue

        dest = move_to_finished_archive(
            src,
            FINISHED_DIR,
            also_search=(LEGACY_FINISHED_DIR, LEGACY_FINISHED2_DIR),
        )
        moved.append(dest)
        print(f"[Superblock Residual Engine] Archived FITS: {src.name} -> {dest}")

    if moved:
        print(f"[Superblock Residual Engine] {len(moved)} file(s) moved to {FINISHED_DIR}/")
    return moved


def _parse_planck_map_meta(map_name: str) -> tuple[int, str]:
    """Extract HEALPix nside and Planck release tag from a CMB map filename."""
    match = re.search(r"_(\d{3,5})_(R\d+\.\d+)", map_name, re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2)
    return 2048, "R3.00"


def mask_filename_for_cmb_map(cmb_path: Path | str) -> str:
    """Planck common intensity mask matching the map nside/release."""
    nside, release = _parse_planck_map_meta(Path(cmb_path).name)
    return f"COM_Mask_CMB-common-Mask-Int_{nside}_{release}.fits"


def mask_download_url(cmb_path: Path | str) -> str:
    mask_name = mask_filename_for_cmb_map(cmb_path)
    _nside, release = _parse_planck_map_meta(Path(cmb_path).name)
    release_dir = release.lower().replace("r", "").split(".")[0]
    return f"{PLANCK_MASK_URL_BASE.format(release=release_dir)}{mask_name}"


def download_planck_mask_for_map(
    cmb_path: Path | str,
    *,
    dest_dir: Path | str = FITS_DIR,
) -> Path:
    """
    Fetch the Planck common CMB mask that matches a CMB map in ./fits/.

    Uses wget. Restores from ../finished/ if present; otherwise downloads from IRSA.
    """
    ensure_output_dirs()
    cmb_path = Path(cmb_path)
    dest_dir = Path(dest_dir)
    mask_name = mask_filename_for_cmb_map(cmb_path)
    dest = dest_dir / mask_name

    if dest.is_file() and dest.stat().st_size > 1000:
        print(f"[Superblock Residual Engine] Mask already in ./fits/: {mask_name}")
        return dest

    for archive_dir in finished_fits_search_dirs():
        finished_mask = archive_dir / mask_name
        if finished_mask.is_file() and finished_mask.stat().st_size > 1000:
            shutil.copy2(finished_mask, dest)
            print(f"[Superblock Residual Engine] Restored mask from {archive_dir}/: {mask_name}")
            return dest
        matches = sorted(archive_dir.glob(f"{Path(mask_name).stem}*{Path(mask_name).suffix}"))
        for candidate in matches:
            if candidate.is_file() and candidate.stat().st_size > 1000:
                shutil.copy2(candidate, dest)
                print(f"[Superblock Residual Engine] Restored mask from {archive_dir}/: {candidate.name}")
                return dest

    url = mask_download_url(cmb_path)
    print(f"[Superblock Residual Engine] No mask in ./fits/ for {cmb_path.name}")
    print(f"[Superblock Residual Engine] Downloading via wget: {mask_name}")
    print(f"[Superblock Residual Engine] URL: {url}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    result = subprocess.run(
        ["wget", "-q", "--timeout=120", "-O", str(partial), url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not partial.is_file() or partial.stat().st_size < 1000:
        if partial.exists():
            partial.unlink()
        stderr = result.stderr.strip() or result.stdout.strip() or "(no wget output)"
        raise FileNotFoundError(
            f"wget could not download Planck mask for {cmb_path.name}.\n"
            f"  URL: {url}\n"
            f"  wget: {stderr}\n"
            f"Manual: wget -O {dest} '{url}'"
        )

    partial.replace(dest)
    print(f"[Superblock Residual Engine] Mask ready: {dest}")
    return dest


def list_fits_files(directory: Path | str = FITS_DIR) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = list(directory.glob("*.fits")) + list(directory.glob("*.FITS"))
    return sorted({path.resolve() for path in files})
