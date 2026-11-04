"""
Download / materialize the LHCb B-meson C9(q²) series used by Tav-Echo.

HEPData is attempted first (LHCb PRL 125, 011802 / ins1784890). Cloudflare
often blocks that host; the fallback writes a denser interpolated series
from the published-style C9 anchors (default 24 q² bins, not 6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import requests

from menus.particle.lhcb.echo import (
    CACHE_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_N_BINS,
    DTYPE,
    clip_n_bins,
    resample_c9_series,
    save_default_dataset,
)
from tav_shared.dataset_ledger import mark_processed

HEPDATA_INSPIRE = "1784890"
HEPDATA_JSON_URL = f"https://www.hepdata.net/record/ins{HEPDATA_INSPIRE}?format=json"
METADATA_NAME = "lhcb_bmeson_fetch.json"


def cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def cached_series_path() -> Path:
    return DEFAULT_DATA_PATH


def _try_hepdata_c9_table() -> np.ndarray | None:
    """Best-effort HEPData pull. Returns None if the host is blocked or unusable."""
    try:
        response = requests.get(
            HEPDATA_JSON_URL,
            timeout=30,
            headers={"User-Agent": "tav-research-tool/1.0"},
        )
        response.raise_for_status()
        if "application/json" not in (response.headers.get("content-type") or "") and not (
            response.text.lstrip().startswith("{") or response.text.lstrip().startswith("[")
        ):
            return None
        payload = response.json()
    except (OSError, ValueError, requests.RequestException):
        return None

    tables = payload.get("data_tables") or payload.get("dataTables") or []
    if not isinstance(tables, list):
        return None
    for table in tables:
        parsed = _table_to_c9(table)
        if parsed is not None and len(parsed) >= 6:
            return parsed
    return None


def _table_to_c9(table: dict[str, Any]) -> np.ndarray | None:
    """Map a HEPData table onto q2_bin, C9_exp, C9_exp_err, C9_SM when possible."""
    independent = table.get("independent_variables") or table.get("independentVariables") or []
    dependent = table.get("dependent_variables") or table.get("dependentVariables") or []
    if not independent or not dependent:
        return None
    x_header = str((independent[0] or {}).get("header", {}).get("name") or "").lower()
    if "q" not in x_header and "q2" not in x_header.replace("^", ""):
        # still accept first independent axis
        pass
    x_values = (independent[0] or {}).get("values") or []
    y_var = None
    for cand in dependent:
        name = str((cand or {}).get("header", {}).get("name") or "").lower()
        if "c9" in name.replace("_", "") or "c_9" in name:
            y_var = cand
            break
    if y_var is None:
        return None
    y_values = y_var.get("values") or []
    n = min(len(x_values), len(y_values))
    if n < 3:
        return None
    rows = []
    for i in range(n):
        xv = x_values[i] or {}
        yv = y_values[i] or {}
        low = xv.get("low")
        high = xv.get("high")
        if low is not None and high is not None:
            q2 = 0.5 * (float(low) + float(high))
        else:
            q2 = float(xv.get("value") or 0.0)
        # HEPData q² is usually GeV²; stored series uses MeV².
        if q2 < 50.0:
            q2 *= 1.0e6
        y = yv.get("value")
        if y is None:
            continue
        errs = yv.get("errors") or []
        err = 0.0
        if errs:
            err = float(errs[0].get("symerror") or errs[0].get("asymerror", {}).get("plus") or 0.0)
        rows.append((q2, float(y), abs(err) or 0.3, -4.27))
    if len(rows) < 3:
        return None
    return np.array(rows, dtype=DTYPE)


def download_lhcb_bmeson_dataset(
    *,
    n_bins: int | None = None,
    force_refresh: bool = False,
    dest: Path | str | None = None,
) -> dict[str, Any]:
    """
    Materialize the C9 series under ``datasets/lhcb/``.

    Tries HEPData, then writes an n-bin interpolation of the published-style
    anchors. ``n_bins`` defaults to 24 (the old template was hard-wired at 6).
    """
    dest_path = Path(dest or cached_series_path())
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    n_bins = clip_n_bins(n_bins)
    source = "template"
    data: np.ndarray | None = None

    if dest_path.is_file() and not force_refresh:
        existing = np.load(dest_path, allow_pickle=False)
        if existing.dtype.names and len(existing) == n_bins:
            print(f"[LHCb] Cached C9 series already has {n_bins} bins: {dest_path}")
            return {
                "path": str(dest_path),
                "n_bins": int(len(existing)),
                "source": "cached",
            }

    print(f"[LHCb] Fetching B-meson C9 series (n_bins={n_bins})…")
    hep = _try_hepdata_c9_table()
    if hep is not None:
        source = f"hepdata:ins{HEPDATA_INSPIRE}"
        data = resample_c9_series(hep, n_bins) if len(hep) != n_bins else hep
        print(f"[LHCb] HEPData ins{HEPDATA_INSPIRE}: {len(hep)} published bins")
    else:
        print(
            "[LHCb] HEPData unavailable (blocked or no C9 table); "
            f"writing interpolated published-style series with {n_bins} q² bins."
        )
        data = None

    if data is None:
        save_default_dataset(dest_path, n_bins=n_bins)
        source = "published_style_interpolated"
    else:
        np.save(dest_path, data)

    meta = {
        "path": str(dest_path),
        "n_bins": n_bins,
        "source": source,
        "inspire": HEPDATA_INSPIRE,
        "note": (
            "q² in MeV². Errors on interpolated bins are scaled by √(n/n_anchor) "
            "so extra points do not fake extra precision."
        ),
    }
    (dest_path.parent / METADATA_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    mark_processed(["lhcb:lhcb_bmeson_data"], note=f"lhcb fetch n_bins={n_bins}")
    print(f"[LHCb] Ready: {dest_path}  bins={n_bins}  source={source}")
    return meta
