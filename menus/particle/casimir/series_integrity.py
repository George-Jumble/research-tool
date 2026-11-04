"""
Cryptographic integrity for Casimir scanner numerical series.

Hashes the exact (matsubara_frequency, dielectric_response_primary) pair
that is analysed after cleaning (NaN drop + sort). Duplicate hashes across
material files are treated as a critical loader bug.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PARAM_NAME = "matsubara_frequency"
VALUE_NAME = "dielectric_response_primary"


class DuplicateRawSeriesError(RuntimeError):
    """Raised when two material files hash to the same analysed series."""


def clean_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite pairs and sort by the independent axis (no interpolation)."""
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    n = min(xx.size, yy.size)
    xx, yy = xx[:n], yy[:n]
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[mask], yy[mask]
    if xx.size == 0:
        return xx, yy
    order = np.argsort(xx, kind="mergesort")
    return np.ascontiguousarray(xx[order]), np.ascontiguousarray(yy[order])


def sha256_raw_series(x: np.ndarray, y: np.ndarray) -> str:
    """SHA-256 of cleaned float64 little-endian bytes: frequency then response."""
    xx, yy = clean_xy(x, y)
    digest = hashlib.sha256()
    digest.update(xx.tobytes())
    digest.update(yy.tobytes())
    return digest.hexdigest()


def build_raw_series_integrity(
    x: np.ndarray,
    y: np.ndarray,
    *,
    file_path: str | Path | None,
    data_label: str,
) -> dict[str, Any]:
    xx, yy = clean_xy(x, y)
    digest = sha256_raw_series(xx, yy)
    abs_path = str(Path(file_path).resolve()) if file_path else None
    payload = {
        "sha256": digest,
        "file_path": abs_path,
        "data_label": str(data_label),
        "n_matsubara_frequency": int(xx.size),
        "n_dielectric_response_primary": int(yy.size),
    }
    print(
        f"[CASIMIR LOAD] raw_series_integrity sha256={digest} "
        f"file={abs_path or '<in-memory>'} "
        f"data_label={data_label!r} "
        f"n_{PARAM_NAME}={xx.size} n_{VALUE_NAME}={yy.size}"
    )
    return payload


_SEEN_HASHES: dict[str, str] = {}


def reset_seen_hashes() -> None:
    _SEEN_HASHES.clear()


def register_series_hash(integrity: dict[str, Any], *, halt_on_collision: bool = True) -> None:
    """Record a material hash; halt if another file already produced it."""
    digest = str(integrity.get("sha256") or "")
    label = str(integrity.get("data_label") or integrity.get("file_path") or "?")
    if not digest:
        return
    prior = _SEEN_HASHES.get(digest)
    if prior is not None and prior != label:
        msg = (
            "CRITICAL loader bug: identical analysed-series SHA-256 for two materials. "
            f"hash={digest} first={prior!r} second={label!r}. "
            "Halting further analysis."
        )
        print(f"[CASIMIR LOAD] {msg}")
        if halt_on_collision:
            raise DuplicateRawSeriesError(msg)
    _SEEN_HASHES[digest] = label


def _is_material_sidecar(path: Path) -> bool:
    name = path.name.lower()
    if name in {"readme", "license", "done.txt"}:
        return True
    if path.suffix.lower() in {".md", ".txt", ".pdf", ".m", ".nb"}:
        return True
    return False


def distinctive_dielectric_index(raw: pd.DataFrame, file_path: Path) -> int:
    """
    Choose the dielectric column (1..) that distinguishes this file from siblings.

    Fig2 tables share Matsubara frequency (col 0) and several common-layer
    dielectrics. The file name is a material; the distinctive ε(iξ) lives in a
    later column. Scoring: maximize how many siblings disagree, then max |Δ|.
    """
    values = raw.to_numpy(dtype=np.float64, copy=False)
    n_cols = values.shape[1]
    if n_cols <= 2:
        return min(1, n_cols - 1)

    siblings: list[np.ndarray] = []
    parent = file_path.parent
    if parent.is_dir():
        for sib in sorted(parent.iterdir()):
            if not sib.is_file() or sib.resolve() == file_path.resolve():
                continue
            if _is_material_sidecar(sib):
                continue
            try:
                from menus.particle.casimir.scanner import _read_tabular_file

                other = _read_tabular_file(sib).to_numpy(dtype=np.float64)
            except Exception:
                continue
            if other.shape == values.shape:
                siblings.append(other)

    best_idx = 1
    best_key = (-1, -1.0)
    for idx in range(1, n_cols):
        col = values[:, idx]
        if not siblings:
            score = float(np.nanmedian(np.abs(col)))
            key = (0, score)
        else:
            n_disagree = 0
            max_delta = 0.0
            for other in siblings:
                delta = float(np.nanmax(np.abs(col - other[:, idx])))
                max_delta = max(max_delta, delta)
                if not np.allclose(col, other[:, idx], equal_nan=True, rtol=0.0, atol=0.0):
                    n_disagree += 1
            key = (n_disagree, max_delta)
        if key > best_key:
            best_key = key
            best_idx = idx
    return best_idx


def standardize_dielectric_columns(df: pd.DataFrame, file_path: Path | None) -> pd.DataFrame:
    """
    Name Matsubara + dielectric columns; set dielectric_response_primary to the
    distinctive material series (not a shared host-layer column).
    """
    out = df.copy()
    n_cols = out.shape[1]
    if n_cols < 2:
        return out

    unnamed = all(str(c).isdigit() or isinstance(c, (int, np.integer)) for c in out.columns)
    if unnamed:
        rename: dict[Any, str] = {out.columns[0]: PARAM_NAME}
        for i, col in enumerate(out.columns[1:], start=1):
            rename[col] = f"dielectric_response_{i}"
        out = out.rename(columns=rename)

    if PARAM_NAME not in out.columns:
        return out

    diel_cols = [c for c in out.columns if str(c).startswith("dielectric_response")]
    if not diel_cols:
        return out

    primary_src = diel_cols[0]
    if file_path is not None and unnamed:
        raw_idx = distinctive_dielectric_index(df, Path(file_path))
        if 1 <= raw_idx < n_cols:
            mapped = f"dielectric_response_{raw_idx}"
            if mapped in out.columns:
                primary_src = mapped

    if VALUE_NAME in out.columns and VALUE_NAME != primary_src:
        # Keep the chosen series under the canonical analysis name.
        out = out.drop(columns=[VALUE_NAME])
    if primary_src != VALUE_NAME:
        out = out.rename(columns={primary_src: VALUE_NAME})
    return out
