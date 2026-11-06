"""
Download the real LHCb B⁰ → K*⁰μ⁺μ⁻ 2020 angular measurement scored by the Superblock residual engine.

Source: LHCb, PRL 125, 011802 (2020), arXiv:2003.04831, INSPIRE 1784890 — 8 narrow q² bins.
The numbers are read live over HTTPS from flavio's public measurement database
(``flavio/data/measurements.yml``, entries ``LHCb B->K*mumu 2020 P <bin>`` / ``... S <bin>``),
which transcribes the paper's supplemental tables (stat ± syst per bin).

HEPData (``www.hepdata.net/record/ins1784890``) is not used: its record endpoint answers HTTP 403
(Cloudflare) to non-browser clients, and the paper publishes angular observables per q² bin,
never a per-bin C9 column. No synthetic interpolation is ever written by this fetcher.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml

from menus.particle.lhcb.flavor import (
    CACHE_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_OBSERVABLE,
    LHCB_2020_BINS,
    REAL_DTYPE,
    SM_PREDICTIONS,
    SM_SOURCE,
    is_real_schema,
    sm_prediction,
)
from sb_shared.dataset_ledger import mark_processed

INSPIRE_ID = "1784890"
ARXIV_ID = "2003.04831"
CITATION = "LHCb collaboration, R. Aaij et al., Phys. Rev. Lett. 125 (2020) 011802, arXiv:2003.04831"
FLAVIO_MEASUREMENTS_URL = (
    "https://raw.githubusercontent.com/flav-io/flavio/master/flavio/data/measurements.yml"
)
FETCH_SOURCE = "flavio_measurements_yml:LHCb B->K*mumu 2020"
SYNTHETIC_SOURCES = {"published_style_interpolated", "template", ""}
METADATA_NAME = "lhcb_bmeson_fetch.json"
SUPPORTED_OBSERVABLES = tuple(SM_PREDICTIONS)

# Kept for reference / reporting only (blocked upstream, see module docstring).
HEPDATA_INSPIRE = INSPIRE_ID
HEPDATA_JSON_URL = f"https://www.hepdata.net/record/ins{HEPDATA_INSPIRE}?format=json"


def cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def cached_series_path() -> Path:
    return DEFAULT_DATA_PATH


def metadata_path(dest: Path | str | None = None) -> Path:
    return Path(dest or cached_series_path()).parent / METADATA_NAME


def read_fetch_metadata(dest: Path | str | None = None) -> dict[str, Any]:
    path = metadata_path(dest)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def is_real_provenance(meta: dict[str, Any]) -> bool:
    src = str(meta.get("source") or "")
    return bool(src) and src not in SYNTHETIC_SOURCES and src != "cached" and meta.get("real_data") is True


_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_flavio_value(text: str) -> tuple[float, float, float]:
    """'0.521 ± 0.095 ± 0.024' or '0.255 +0.032-0.030 ± 0.007' → (value, stat, syst).

    Asymmetric errors are symmetrised as (σ₊ + σ₋)/2; extra error terms are summed in quadrature
    into the systematic.
    """
    s = str(text).replace("±", " ± ").strip()
    m = re.match(rf"^\s*({_NUM})\s*(.*)$", s)
    if not m:
        raise ValueError(f"cannot parse value: {text!r}")
    value = float(m.group(1))
    rest = m.group(2)
    errs: list[float] = []
    for tok in re.finditer(rf"±\s*({_NUM})|\+\s*({_NUM})\s*-\s*({_NUM})", rest):
        if tok.group(1) is not None:
            errs.append(abs(float(tok.group(1))))
        else:
            errs.append(0.5 * (abs(float(tok.group(2))) + abs(float(tok.group(3)))))
    if not errs:
        raise ValueError(f"no uncertainty in value: {text!r}")
    stat = errs[0]
    syst = math.sqrt(sum(e * e for e in errs[1:])) if len(errs) > 1 else 0.0
    return value, stat, syst


def _fmt_bin(x: float) -> str:
    return f"{x:g}"


def parse_lhcb2020(doc: dict[str, Any], observable: str = DEFAULT_OBSERVABLE) -> list[dict[str, Any]]:
    """Pull one observable in the 8 narrow LHCb 2020 bins out of flavio's measurements.yml."""
    name = f"<{observable}>(B0->K*mumu)"
    rows = []
    for lo, hi in LHCB_2020_BINS:
        found = None
        for basis in ("P", "S"):
            entry = doc.get(f"LHCb B->K*mumu 2020 {basis} {_fmt_bin(lo)}-{_fmt_bin(hi)}") or {}
            for item in entry.get("values") or []:
                if item.get("name") == name and abs(float(item["q2min"]) - lo) < 1e-6 and abs(
                    float(item["q2max"]) - hi
                ) < 1e-6:
                    found = (basis, item)
                    break
            if found:
                break
        if found is None:
            raise ValueError(f"{name} missing for q² bin [{lo}, {hi}] GeV² in flavio measurements")
        basis, item = found
        value, stat, syst = parse_flavio_value(item["value"])
        sm = sm_prediction(observable, lo, hi)
        if sm is None:
            raise ValueError(f"no SM prediction tabulated for {observable} in [{lo}, {hi}]")
        rows.append(
            {
                "q2_min_gev2": lo,
                "q2_max_gev2": hi,
                "q2_center_gev2": 0.5 * (lo + hi),
                "obs": value,
                "stat_err": stat,
                "syst_err": syst,
                "obs_err": math.hypot(stat, syst),
                "sm": sm[0],
                "sm_err": sm[1],
                "flavio_entry": f"LHCb B->K*mumu 2020 {basis} {_fmt_bin(lo)}-{_fmt_bin(hi)}",
                "raw_value": str(item["value"]).strip(),
            }
        )
    return rows


def rows_to_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [
            (
                r["q2_center_gev2"] * 1.0e6,
                r["q2_min_gev2"],
                r["q2_max_gev2"],
                r["q2_center_gev2"],
                r["obs"],
                r["stat_err"],
                r["syst_err"],
                r["obs_err"],
                r["sm"],
                r["sm_err"],
            )
            for r in rows
        ],
        dtype=REAL_DTYPE,
    )


def fetch_flavio_measurements(timeout: float = 60.0) -> dict[str, Any]:
    response = requests.get(
        FLAVIO_MEASUREMENTS_URL,
        timeout=timeout,
        headers={"User-Agent": "superblock-research-tool/1.0"},
    )
    response.raise_for_status()
    return yaml.safe_load(response.text)


def _cache_is_valid(dest_path: Path, observable: str) -> tuple[bool, dict[str, Any], str]:
    meta = read_fetch_metadata(dest_path)
    if not dest_path.is_file():
        return False, meta, "no cached file"
    if not is_real_provenance(meta):
        return False, meta, f"cached source {meta.get('source') or 'missing'!r} is not the real measurement"
    if meta.get("observable") != observable:
        return False, meta, f"cached observable {meta.get('observable')!r} != requested {observable!r}"
    try:
        existing = np.load(dest_path, allow_pickle=False)
    except (OSError, ValueError):
        return False, meta, "cached file unreadable"
    if not (existing.dtype.names and is_real_schema(existing)) or len(existing) != len(LHCB_2020_BINS):
        return False, meta, "cached array is not the 8-bin LHCb 2020 schema"
    return True, meta, "ok"


def download_lhcb_bmeson_dataset(
    *,
    observable: str = DEFAULT_OBSERVABLE,
    force_refresh: bool = False,
    dest: Path | str | None = None,
    n_bins: int | None = None,
    fetch: Any = None,
) -> dict[str, Any]:
    """
    Materialize the real LHCb 2020 ⟨observable⟩ series (8 narrow q² bins) under ``datasets/lhcb/``.

    ``n_bins`` is accepted for old callers but ignored: the measurement has exactly 8 bins and
    is never interpolated. A cached synthetic / interpolated file is always replaced.
    """
    observable = str(observable or DEFAULT_OBSERVABLE).strip()
    if observable not in SUPPORTED_OBSERVABLES:
        raise ValueError(f"observable must be one of {SUPPORTED_OBSERVABLES}, got {observable!r}")
    dest_path = Path(dest or cached_series_path())
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if n_bins is not None and int(n_bins) != len(LHCB_2020_BINS):
        print(
            f"[LHCb] n_bins={n_bins} ignored: LHCb 2020 publishes {len(LHCB_2020_BINS)} narrow q² bins "
            "and real data is never interpolated."
        )

    if not force_refresh:
        ok, meta, why = _cache_is_valid(dest_path, observable)
        if ok:
            print(
                f"[LHCb] Cached {observable} series ({len(LHCB_2020_BINS)} bins) is current: {dest_path}  "
                f"cached_source={meta.get('source')}"
            )
            return {
                "path": str(dest_path),
                "n_bins": len(LHCB_2020_BINS),
                "observable": observable,
                "source": "cached",
                "cached_source": meta.get("source"),
                "real_data": True,
            }
        if dest_path.is_file():
            print(f"[LHCb] Refetching: {why}.")

    print(f"[LHCb] Fetching LHCb 2020 ⟨{observable}⟩(B0→K*μμ) from {FLAVIO_MEASUREMENTS_URL}")
    doc = (fetch or fetch_flavio_measurements)()
    rows = parse_lhcb2020(doc, observable)
    data = rows_to_array(rows)
    np.save(dest_path, data)
    meta = {
        "path": str(dest_path),
        "n_bins": len(rows),
        "source": FETCH_SOURCE,
        "real_data": True,
        "url": FLAVIO_MEASUREMENTS_URL,
        "observable": observable,
        "flavio_name": f"<{observable}>(B0->K*mumu)",
        "inspire_id": INSPIRE_ID,
        "arxiv_id": ARXIV_ID,
        "citation": CITATION,
        "sm_source": SM_SOURCE,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hepdata_note": (
            f"{HEPDATA_JSON_URL} returns HTTP 403 (Cloudflare) to scripted clients and the paper has no "
            "per-bin C9 table; not used."
        ),
        "units": "q2_bin in MeV² (bin centre); q2_min/max/center in GeV²; observable dimensionless",
        "uncertainty": "obs_err = sqrt(stat² + syst²); asymmetric stat symmetrised as (σ₊+σ₋)/2",
        "bins": rows,
    }
    metadata_path(dest_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    mark_processed(["lhcb:lhcb_bmeson_data"], note=f"lhcb 2020 {observable} real 8 bins")
    print(f"[LHCb] Ready: {dest_path}  bins={len(rows)}  source={FETCH_SOURCE}")
    return meta
