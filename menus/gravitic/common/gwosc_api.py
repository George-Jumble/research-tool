"""GWOSC Event Portal API helpers (https://gwosc.org/)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

GWOSC_HOME = "https://gwosc.org/"
GWOSC_DATA = "https://gwosc.org/data/"
EVENT_API = "https://gwosc.org/eventapi/json"
ALLEVENTS_URL = f"{EVENT_API}/allevents/"

_EVENT_RE = re.compile(r"^GW\d{6}(?:_\d{6})?$", re.IGNORECASE)


@dataclass(frozen=True)
class StrainFile:
    event: str
    detector: str
    gps_start: int
    duration: int
    sampling_rate: int
    fmt: str
    url: str

    @property
    def local_name(self) -> str:
        return self.url.rsplit("/", 1)[-1]


def is_event_name(query: str) -> bool:
    clean = str(query).strip().upper()
    return bool(_EVENT_RE.match(clean))


# Bumped when the cached catalog's selection rule changes; older caches are refreshed automatically.
CATALOG_SCHEMA = 2


def _version_num(meta: dict[str, Any]) -> tuple[int, str]:
    raw = meta.get("version")
    try:
        return int(raw), str(raw)
    except (TypeError, ValueError):
        m = re.search(r"(\d+)", str(raw or ""))
        return (int(m.group(1)) if m else -1), str(raw or "")


def has_catalog_pe(meta: dict[str, Any] | None) -> bool:
    """True when a catalog entry carries final mass and redshift (preliminary entries do not)."""
    return bool(meta) and meta.get("final_mass_source") is not None and meta.get("redshift") is not None


def entry_rank(meta: dict[str, Any]) -> tuple[int, int, str]:
    """Higher is better: entries with final_mass_source + redshift first, then the highest version."""
    num, raw = _version_num(meta)
    return (1 if has_catalog_pe(meta) else 0, num, raw)


def select_best_versions(entries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One entry per commonName. Never lets a preliminary/lower version overwrite a better one."""
    out: dict[str, dict[str, Any]] = {}
    for key, meta in entries.items():
        if not isinstance(meta, dict):
            continue
        common = str(meta.get("commonName") or str(key).split("-", 1)[0]).upper()
        if common not in out or entry_rank(meta) > entry_rank(out[common]):
            out[common] = meta
    return out


def fetch_all_event_versions(*, timeout: int = 120) -> dict[str, dict[str, Any]]:
    """Raw allevents payload: every version of every event (``"GW150914-v3"`` → entry)."""
    response = requests.get(ALLEVENTS_URL, timeout=timeout)
    response.raise_for_status()
    return dict(response.json().get("events", {}))


def event_version_urls(event_name: str, versions: dict[str, dict[str, Any]]) -> list[str]:
    """Detail-JSON URLs of every version of one event, best (mass+z, highest version) first."""
    want = event_name.strip().upper()
    rows = [m for m in versions.values() if str(m.get("commonName", "")).upper() == want and m.get("jsonurl")]
    return [str(m["jsonurl"]) for m in sorted(rows, key=entry_rank, reverse=True)]


def fetch_all_events(*, timeout: int = 120) -> dict[str, dict[str, Any]]:
    response = requests.get(ALLEVENTS_URL, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return select_best_versions(payload.get("events", {}))


def _schema_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.stem + ".schema.json")


def catalog_is_stale(cache_path: Path) -> bool:
    """A cache written before schema 2 may hold clobbered (preliminary) entries."""
    side = _schema_path(cache_path)
    try:
        return int(json.loads(side.read_text(encoding="utf-8")).get("schema", 0)) < CATALOG_SCHEMA
    except (OSError, ValueError, AttributeError):
        return True


def _write_catalog(cache_path: Path, catalog: dict[str, dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8")
    _schema_path(cache_path).write_text(
        json.dumps({"schema": CATALOG_SCHEMA, "source": ALLEVENTS_URL,
                    "rule": "per commonName: entries with final_mass_source+redshift first, then highest version"},
                   indent=2) + "\n",
        encoding="utf-8",
    )


def cache_event_catalog(cache_path: Path, *, force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Cached allevents catalog (one best entry per event). A pre-schema-2 cache is re-fetched once;
    if GWOSC is unreachable the old cache is kept and returned (re-ranked within itself)."""
    if cache_path.is_file() and not force_refresh and not catalog_is_stale(cache_path):
        return json.loads(cache_path.read_text(encoding="utf-8"))
    try:
        catalog = fetch_all_events()
    except Exception:
        if cache_path.is_file() and not force_refresh:
            return select_best_versions(json.loads(cache_path.read_text(encoding="utf-8")))
        raise
    _write_catalog(cache_path, catalog)
    return catalog


def resolve_event_json_url(event_name: str, catalog: dict[str, dict[str, Any]] | None = None) -> str:
    clean = event_name.strip().upper()
    catalog = catalog or fetch_all_events()
    best = None
    for meta in catalog.values():
        if str(meta.get("commonName", "")).upper() == clean and meta.get("jsonurl"):
            if best is None or entry_rank(meta) > entry_rank(best):
                best = meta
    if best is not None:
        return str(best["jsonurl"])
    raise ValueError(f"GWOSC event {clean!r} not found in catalog")


def fetch_event_detail_url(url: str, *, timeout: int = 120) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    events = (response.json() or {}).get("events") or {}
    if not events:
        raise ValueError(f"No event payload returned from {url!r}")
    return max(events.values(), key=entry_rank)


def fetch_event_detail(event_name: str, *, catalog: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    url = resolve_event_json_url(event_name, catalog=catalog)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    payload = response.json()
    events = payload.get("events") or {}
    if not events:
        raise ValueError(f"No event payload returned for {event_name!r}")
    return next(iter(events.values()))


class StrainSelectionError(ValueError):
    """No strain entry matched the requested format/duration/rate."""


def _strain_rank(entry: dict[str, Any]) -> tuple[int, int, int]:
    """Lower is better: prefer short segments, hdf5/txt, 4 kHz."""
    fmt = str(entry.get("format", "")).lower()
    fmt_score = {"hdf5": 0, "txt": 1, "gwf": 2}.get(fmt, 3)
    try:
        duration = int(entry.get("duration", 10**9))
    except (TypeError, ValueError):
        duration = 10**9
    try:
        rate = int(entry.get("sampling_rate", 10**9))
    except (TypeError, ValueError):
        rate = 10**9
    rate_score = abs(rate - 4096)
    return (duration, fmt_score, rate_score)


def list_strain_files(
    event_name: str,
    *,
    catalog: dict[str, dict[str, Any]] | None = None,
    fmt: str | None = "hdf5",
    duration: int | None = 32,
    sampling_rate: int | None = 4096,
    detectors: set[str] | None = None,
    auto_select: bool = True,
    detail: dict[str, Any] | None = None,
) -> list[StrainFile]:
    """Strain files for an event. Pass ``detail`` (event detail JSON) to avoid a second request."""
    detail = detail if detail is not None else fetch_event_detail(event_name, catalog=catalog)
    common = str(detail.get("commonName") or event_name).upper()
    raw_entries = list(detail.get("strain") or [])
    if not raw_entries:
        raise ValueError(f"No GWOSC strain files listed for event={common!r}")

    filtered: list[dict[str, Any]] = []
    for entry in raw_entries:
        entry_fmt = str(entry.get("format", "")).lower()
        if fmt and entry_fmt != fmt.lower():
            continue
        if duration is not None and int(entry.get("duration", -1)) != duration:
            continue
        if sampling_rate is not None and int(entry.get("sampling_rate", -1)) != sampling_rate:
            continue
        detector = str(entry.get("detector", "")).upper()
        if detectors and detector not in detectors:
            continue
        if str(entry.get("url", "")).strip():
            filtered.append(entry)

    if not filtered and auto_select:
        filtered = [
            entry
            for entry in raw_entries
            if str(entry.get("url", "")).strip()
            and (not detectors or str(entry.get("detector", "")).upper() in detectors)
        ]
        chosen: dict[str, dict[str, Any]] = {}
        for entry in sorted(filtered, key=_strain_rank):
            detector = str(entry.get("detector", "")).upper()
            chosen.setdefault(detector, entry)
        filtered = list(chosen.values())

    files: list[StrainFile] = []
    for entry in filtered:
        files.append(
            StrainFile(
                event=common,
                detector=str(entry.get("detector", "")).upper(),
                gps_start=int(entry.get("GPSstart", 0)),
                duration=int(entry.get("duration", 0)),
                sampling_rate=int(entry.get("sampling_rate", 0)),
                fmt=str(entry.get("format", "")).lower(),
                url=str(entry.get("url", "")).strip(),
            )
        )
    if not files:
        raise StrainSelectionError(
            f"No GWOSC strain files matched event={common!r} "
            f"fmt={fmt!r} duration={duration!r} rate={sampling_rate!r}"
        )
    return files