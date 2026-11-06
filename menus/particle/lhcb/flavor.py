"""
LHCb B⁰ → K*⁰μ⁺μ⁻ dataset: cache I/O and loader.

Primary schema (``REAL_DTYPE``): the published LHCb 2020 angular measurement
(PRL 125, 011802; arXiv:2003.04831; INSPIRE 1784890), one row per narrow q² bin, with a
per-bin SM prediction of the *same* observable (e.g. ⟨P5′⟩_SM), never C9_SM.

Legacy schema (``DTYPE``): the old C9(q²) template series (q2_bin, C9_exp, C9_exp_err, C9_SM).
It is still readable for backward compatibility but is synthetic, not a measurement.

Cache: datasets/lhcb/lhcb_bmeson_data.npy + lhcb_bmeson_fetch.json (provenance sidecar).
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


# Real LHCb 2020 rows. q2_bin keeps the cache convention (bin centre in MeV²); the explicit
# bin edges / centre in GeV² sit next to it so nothing has to guess units.
REAL_DTYPE = np.dtype(
    [
        ("q2_bin", "f8"),
        ("q2_min_gev2", "f8"),
        ("q2_max_gev2", "f8"),
        ("q2_center_gev2", "f8"),
        ("obs", "f8"),
        ("stat_err", "f8"),
        ("syst_err", "f8"),
        ("obs_err", "f8"),
        ("sm", "f8"),
        ("sm_err", "f8"),
    ]
)

# The 8 narrow q² bins [GeV²] of LHCb PRL 125, 011802 (charmonium regions excluded).
LHCB_2020_BINS: tuple[tuple[float, float], ...] = (
    (0.1, 0.98), (1.1, 2.5), (2.5, 4.0), (4.0, 6.0),
    (6.0, 8.0), (11.0, 12.5), (15.0, 17.0), (17.0, 19.0),
)

DEFAULT_OBSERVABLE = "P5p"

# SM bin averages (value, theory σ) in the same 8 bins, from flavio 2.7.1 with default SM
# nuisance parameters: flavio.sm_prediction / sm_uncertainty(N=300) of "<obs>(B0->K*mumu)".
# flavio is the package LHCb itself used for its C9 fit in arXiv:2003.04831. Computed once,
# offline (flavio is not a research-tool dependency). Above 8 GeV² flavio warns that QCDF
# corrections are not trusted, so the high-q² SM values carry extra model dependence.
SM_SOURCE = "flavio 2.7.1 default SM (sm_prediction / sm_uncertainty N=300), computed offline"
SM_PREDICTIONS: dict[str, tuple[tuple[float, float], ...]] = {
    "P5p": ((0.6602, 0.0264), (0.1404, 0.0851), (-0.5003, 0.1027), (-0.7583, 0.0709),
            (-0.8342, 0.0574), (-0.8224, 0.0268), (-0.6705, 0.0423), (-0.4831, 0.0447)),
    "FL": ((0.2941, 0.0471), (0.7594, 0.0396), (0.7967, 0.0340), (0.7120, 0.0471),
           (0.6076, 0.0585), (0.4353, 0.0392), (0.3485, 0.0328), (0.3281, 0.0289)),
    "AFB": ((-0.0966, 0.0083), (-0.1385, 0.0281), (-0.0180, 0.0329), (0.1219, 0.0412),
            (0.2399, 0.0459), (0.3908, 0.0248), (0.4022, 0.0326), (0.3187, 0.0321)),
}

OBSERVABLE_LABELS = {"P5p": "P5′", "FL": "F_L", "AFB": "A_FB"}


def sm_prediction(observable: str, q2_min: float, q2_max: float) -> tuple[float, float] | None:
    table = SM_PREDICTIONS.get(observable)
    if table is None:
        return None
    for (lo, hi), val in zip(LHCB_2020_BINS, table):
        if abs(lo - q2_min) < 1e-6 and abs(hi - q2_max) < 1e-6:
            return val
    return None


def is_real_schema(data: np.ndarray) -> bool:
    return bool(data.dtype.names) and {"obs", "sm", "obs_err"} <= set(data.dtype.names)


def is_legacy_schema(data: np.ndarray) -> bool:
    return bool(data.dtype.names) and {"C9_exp", "C9_SM", "C9_exp_err"} <= set(data.dtype.names)


def residual_arrays(data: np.ndarray, *, include_theory: bool = True) -> dict:
    """D, B, σ, q² [GeV²] for either schema (q2_bin is always stored in MeV²)."""
    if is_real_schema(data):
        sig = np.asarray(data["obs_err"], float)
        if include_theory:
            sig = np.sqrt(sig**2 + np.asarray(data["sm_err"], float) ** 2)
        return {
            "D": np.asarray(data["obs"], float),
            "B": np.asarray(data["sm"], float),
            "sigma": sig,
            "q2_gev2": np.asarray(data["q2_center_gev2"], float),
            "q2_min_gev2": np.asarray(data["q2_min_gev2"], float),
            "q2_max_gev2": np.asarray(data["q2_max_gev2"], float),
            "schema": "lhcb2020_real",
        }
    if is_legacy_schema(data):
        return {
            "D": np.asarray(data["C9_exp"], float),
            "B": np.asarray(data["C9_SM"], float),
            "sigma": np.asarray(data["C9_exp_err"], float),
            "q2_gev2": np.asarray(data["q2_bin"], float) / 1.0e6,
            "q2_min_gev2": None,
            "q2_max_gev2": None,
            "schema": "legacy_c9",
        }
    raise ValueError("unrecognised LHCb array schema")


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
    if data.dtype.names is None and data.ndim == 2 and data.shape[1] >= 3:
        # Legacy plain (q2 [MeV²], C9, err) array: promote to the C9 schema, C9_SM = -4.27.
        rows = [(float(r[0]), float(r[1]), float(r[2]), -4.27) for r in data]
        data = np.array(rows, dtype=DTYPE)
    if data.dtype.names is not None and is_real_schema(data):
        # Real measurement: never interpolate or resample.
        return data, path
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
