"""
Casimir dataset fetchers: Zenodo superconducting Casimir drums and GitHub magnetic-fluid CSVs,
cached under datasets/casimir/.
"""

from __future__ import annotations
from pathlib import Path
import requests
from sb_shared.project_paths import SUPERBLOCK_ROOT as PROJECT_ROOT


DATASETS_DIR = PROJECT_ROOT / "datasets" / "casimir"


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


def _ensure_datasets() -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return DATASETS_DIR


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
