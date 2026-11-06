"""
LHCb B → K*μμ Wilson-coefficient dataset (C9 per q² bin): curated series, cache I/O and loader.

Cache: datasets/lhcb/lhcb_bmeson_data.npy (bundled template under menus/particle/lhcb/data/).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from sb_shared.project_paths import ARTIFACTS_ROOT, DATASETS_ROOT


PROJECT_ROOT = Path(__file__).resolve().parent


DATA_DIR = PROJECT_ROOT / "data"


CACHE_DIR = DATASETS_ROOT / "lhcb"


ARTIFACTS_DIR = ARTIFACTS_ROOT / "lhcb"


BUNDLE_DATA_PATH = DATA_DIR / "lhcb_bmeson_data.npy"


DEFAULT_DATA_PATH = CACHE_DIR / "lhcb_bmeson_data.npy"




DEFAULT_N_BINS = 24


MIN_N_BINS = 6


MAX_N_BINS = 200


DTYPE = np.dtype(
    [
        ("q2_bin", "f8"),
        ("C9_exp", "f8"),
        ("C9_exp_err", "f8"),
        ("C9_SM", "f8"),
    ]
)


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def clip_n_bins(n_bins: int | None) -> int:
    try:
        value = int(n_bins) if n_bins is not None else DEFAULT_N_BINS
    except (TypeError, ValueError):
        value = DEFAULT_N_BINS
    return max(MIN_N_BINS, min(MAX_N_BINS, value))


def published_c9_anchors() -> np.ndarray:
    """
    Six published-style C9(q²) anchors (q² in MeV²).

    These are the original skeleton points; denser series are interpolated
    from this curve with errors scaled by √(n/6).
    """
    rows = [
        (4.5e5, -4.42, 0.38, -4.27),
        (1.1e6, -4.55, 0.32, -4.27),
        (2.5e6, -4.68, 0.29, -4.27),
        (4.0e6, -4.79, 0.34, -4.27),
        (5.8e6, -4.88, 0.41, -4.27),
        (7.2e6, -4.95, 0.45, -4.27),
    ]
    return np.array(rows, dtype=DTYPE)


def build_curated_lhcb_data(n_bins: int | None = None) -> np.ndarray:
    """LHCb-style C9 series with ``n_bins`` q² points (default 24, not 6)."""
    return resample_c9_series(published_c9_anchors(), clip_n_bins(n_bins))


def resample_c9_series(data: np.ndarray, n_bins: int) -> np.ndarray:
    """Interpolate a C9 series onto ``n_bins`` q² centers."""
    n_bins = clip_n_bins(n_bins)
    order = np.argsort(data["q2_bin"])
    q2 = np.asarray(data["q2_bin"][order], dtype=float)
    if q2.size < 2:
        return data
    q2_new = np.linspace(float(q2[0]), float(q2[-1]), n_bins)
    scale = float(np.sqrt(n_bins / max(q2.size, 1)))
    rows = []
    for q in q2_new:
        c9 = float(np.interp(q, q2, data["C9_exp"][order]))
        err = float(np.interp(q, q2, data["C9_exp_err"][order])) * scale
        sm = float(np.interp(q, q2, data["C9_SM"][order]))
        rows.append((q, c9, err, sm))
    return np.array(rows, dtype=DTYPE)


def save_default_dataset(
    path: Path | str | None = None,
    *,
    n_bins: int | None = None,
) -> Path:
    path = Path(path or DEFAULT_DATA_PATH)
    ensure_output_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, build_curated_lhcb_data(n_bins))
    print(f"[Superblock Residual Engine] Wrote LHCb C9 series ({clip_n_bins(n_bins)} bins): {path}")
    return path


def load_lhcb_data(
    path: Path | str | None = None,
    *,
    n_bins: int | None = None,
) -> tuple[np.ndarray, Path]:
    path = Path(path or DEFAULT_DATA_PATH)
    ensure_output_dirs()
    if not path.is_file():
        fallback = BUNDLE_DATA_PATH if BUNDLE_DATA_PATH.is_file() else None
        if fallback is not None and path != fallback:
            print(f"[Superblock Residual Engine] Missing {path}; copying bundled LHCb template.")
            data = np.load(fallback, allow_pickle=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, data)
        else:
            print(
                f"[Superblock Residual Engine] Missing {path}; writing curated LHCb series "
                f"({clip_n_bins(n_bins)} bins). Run 'Download LHCb B-meson Dataset' "
                "to choose the sample size first."
            )
            save_default_dataset(path, n_bins=n_bins)
    data = np.load(path, allow_pickle=False)
    if data.dtype.names is None:
        raise ValueError(
            f"{path} must be a structured array with fields "
            "q2_bin, C9_exp, C9_exp_err, C9_SM"
        )
    required = {"q2_bin", "C9_exp", "C9_exp_err", "C9_SM"}
    missing = required - set(data.dtype.names)
    if missing:
        raise ValueError(f"{path} missing required fields: {sorted(missing)}")
    want = clip_n_bins(n_bins) if n_bins is not None else None
    if want is not None and want > len(data):
        print(
            f"[Superblock Residual Engine] Resampling C9 series {len(data)} → {want} q² bins"
        )
        data = resample_c9_series(data, want)
    elif want is not None and want < len(data):
        print(
            f"[Superblock Residual Engine] Keeping downloaded {len(data)} q² bins "
            f"(requested {want} would downsample)."
        )
    return data, path
