"""Fetch and cache LIGO/GWOSC strain data via GWOSC API, GWDataFind, and Pelican."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Iterable

from sb_shared.batch_ledger import GWOSC_PULL_DONE, append_done_entries, read_done_list
from sb_shared.project_paths import ensure_project_dirs, move_to_finished_archive

from menus.gravitic.common.download import copy_local_file, download_uri, is_pelican_uri, is_remote_uri
from menus.gravitic.common.gwdatafind_tools import find_frame_urls, parse_segment_query
from menus.gravitic.common.gwosc_api import (
    EVENT_API,
    GWOSC_DATA,
    GWOSC_HOME,
    StrainSelectionError,
    cache_event_catalog,
    event_version_urls,
    fetch_all_event_versions,
    fetch_event_detail_url,
    has_catalog_pe,
    is_event_name,
    list_strain_files,
)

DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets" / "gwosc"
EVENTS_DIR = DATASETS_DIR / "events"
FRAMES_DIR = DATASETS_DIR / "frames"
CATALOG_CACHE = DATASETS_DIR / "event_catalog.json"
DEFAULT_DONE_FILE = GWOSC_PULL_DONE

DEFAULT_STRAIN_FMT = "hdf5"
DEFAULT_STRAIN_DURATION = 32
DEFAULT_STRAIN_RATE = 4096


@dataclass
class PullSummary:
    requested: int = 0
    downloaded: list[str] = field(default_factory=list)
    skipped_done: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def ensure_dirs() -> None:
    for path in (DATASETS_DIR, EVENTS_DIR, FRAMES_DIR):
        path.mkdir(parents=True, exist_ok=True)
    ensure_project_dirs()


def list_cached_files() -> list[Path]:
    ensure_dirs()
    files: list[Path] = []
    for root in (EVENTS_DIR, FRAMES_DIR):
        files.extend(sorted(root.rglob("*")))
    return [path for path in files if path.is_file()]


def list_catalog_events(*, force_refresh: bool = False) -> list[str]:
    catalog = cache_event_catalog(CATALOG_CACHE, force_refresh=force_refresh)
    return sorted({str(meta.get("commonName", "")).upper() for meta in catalog.values() if meta.get("commonName")})


def _strain_options(params: dict | None) -> dict:
    params = params or {}
    fmt = str(params.get("strain_format") or DEFAULT_STRAIN_FMT).strip().lower()
    try:
        duration = int(params.get("strain_duration") or DEFAULT_STRAIN_DURATION)
    except ValueError:
        duration = DEFAULT_STRAIN_DURATION
    try:
        sampling_rate = int(params.get("strain_rate") or DEFAULT_STRAIN_RATE)
    except ValueError:
        sampling_rate = DEFAULT_STRAIN_RATE
    detectors = params.get("detectors")
    if isinstance(detectors, str) and detectors.strip():
        detector_set = {part.strip().upper() for part in detectors.split(",") if part.strip()}
    else:
        detector_set = None
    return {
        "fmt": fmt,
        "duration": duration,
        "sampling_rate": sampling_rate,
        "detectors": detector_set,
    }


EVENT_DETAIL_NAME = "event_detail.json"
EVENT_VERSIONS_CACHE = DATASETS_DIR / "event_versions.json"

# Loud, heavy BBH mergers with 32 s / 4 kHz HDF5 strain and PE final mass, spin and redshift on GWOSC
# (checked against the live Event API 2026-11-06; detectors with 32 s files in brackets).
HEAVY_BBH_EVENTS = (
    "GW150914",         # GWTC-2.1 v4, 32 s files in GWTC-1 v3 [H1 L1]
    "GW170104",         # [H1 L1]
    "GW170814",         # [H1 L1 V1]
    "GW190519_153544",  # [H1 L1 V1]
    "GW190521",         # [H1 L1 V1]
    "GW190521_074359",  # [H1 L1]
    "GW190602_175927",  # [H1 L1 V1]
    "GW191109_010717",  # [H1 L1]
    "GW200129_065458",  # [H1 L1 V1]
    "GW200224_222234",  # [H1 L1 V1]
)
PRESETS = {"HEAVY_BBH": HEAVY_BBH_EVENTS}


def expand_event_query(query: str) -> list[str]:
    """``HEAVY_BBH`` → the preset; ``GW1,GW2`` → both; anything else is returned unchanged."""
    raw = str(query or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) > 1 and not all(p.upper() in PRESETS or is_event_name(p) for p in parts):
        return [raw]
    out: list[str] = []
    for part in parts:
        for item in PRESETS.get(part.upper(), (part,)):
            if item.upper() not in {o.upper() for o in out}:
                out.append(item.upper() if is_event_name(item) else item)
    return out


def _event_versions(force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    if EVENT_VERSIONS_CACHE.is_file() and not force_refresh:
        try:
            return json.loads(EVENT_VERSIONS_CACHE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    versions = fetch_all_event_versions()
    EVENT_VERSIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EVENT_VERSIONS_CACHE.write_text(json.dumps(versions, indent=1, sort_keys=True), encoding="utf-8")
    return versions


def select_event_strain(event_name: str, detail: dict[str, Any], opts: dict) -> list:
    """Exact (format, duration, rate) match from the best version's strain list; else from older versions.

    GWOSC's newest release of an event can list only the 4096 s bulk files (GW150914-v4 has 8 entries,
    all 4096 s) while an older release (GWTC-1 v3) lists the 32 s event files, ~1 MB instead of ~125 MB per
    detector. Same GPS, same data. Only when no version has the exact match does auto-selection fall back
    to the shortest available file.
    """
    clean = event_name.strip().upper()
    exact = {**opts, "auto_select": False}
    try:
        return list_strain_files(clean, detail=detail, **exact)
    except StrainSelectionError:
        pass
    try:
        urls = event_version_urls(clean, _event_versions())
    except Exception:
        urls = []
    for url in urls:
        if url.rstrip("/") == str(detail.get("jsonurl") or "").rstrip("/"):
            continue
        try:
            other = fetch_event_detail_url(url)
            return list_strain_files(clean, detail=other, **exact)
        except Exception:
            continue
    return list_strain_files(clean, detail=detail, **opts)


def event_detail_path(event_name: str, events_dir: Path | None = None) -> Path:
    return Path(events_dir or EVENTS_DIR) / event_name.strip().upper() / EVENT_DETAIL_NAME


def load_event_detail(event_name: str, events_dir: Path | None = None) -> dict[str, Any] | None:
    path = event_detail_path(event_name, events_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def save_event_detail(event_name: str, detail: dict[str, Any], events_dir: Path | None = None) -> Path:
    path = event_detail_path(event_name, events_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detail, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _is_preliminary_url(url: str) -> bool:
    import re

    return "preliminary" in url.lower() or bool(re.search(r"/v[12]/?$", url))


def ensure_event_detail(
    event_name: str,
    *,
    events_dir: Path | None = None,
    catalog: dict[str, dict[str, Any]] | None = None,
    catalog_path: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Per-event detail JSON (``meta['jsonurl']``), cached as ``events/<EVENT>/event_detail.json``.

    Holds the PE ``parameters`` (final_spin / a_final) that allevents lacks. If the catalog entry
    points at a preliminary v1/v2 record without mass/redshift, the catalog is refreshed first.
    """
    clean = event_name.strip().upper()
    if not force_refresh:
        cached = load_event_detail(clean, events_dir)
        if cached is not None:
            return cached
    cat_path = Path(catalog_path or CATALOG_CACHE)
    catalog = catalog if catalog is not None else cache_event_catalog(cat_path)
    meta = catalog.get(clean) or {}
    url = str(meta.get("jsonurl") or "")
    if not url or (_is_preliminary_url(url) and not has_catalog_pe(meta)):
        catalog = cache_event_catalog(cat_path, force_refresh=True)
        meta = catalog.get(clean) or {}
        url = str(meta.get("jsonurl") or "")
    if not url:
        raise ValueError(f"No GWOSC jsonurl for event {clean!r} in catalog {cat_path}")
    detail = fetch_event_detail_url(url)
    save_event_detail(clean, detail, events_dir)
    return detail


def pull_event_strain(
    event_name: str,
    *,
    params: dict | None = None,
    force_refresh: bool = False,
) -> PullSummary:
    ensure_dirs()
    summary = PullSummary(requested=1)
    clean = event_name.strip().upper()
    ledger_key = f"event:{clean}"
    done = read_done_list(DEFAULT_DONE_FILE)
    if ledger_key in done and not force_refresh:
        summary.skipped_done.append(ledger_key)
        # Already-pulled events still get their PE detail JSON (spin lives only there).
        if load_event_detail(clean) is None:
            try:
                ensure_event_detail(clean)
                summary.downloaded.append(str(event_detail_path(clean).relative_to(DATASETS_DIR)))
            except Exception as exc:
                summary.failed[EVENT_DETAIL_NAME] = str(exc)
        return summary

    opts = _strain_options(params)
    catalog = cache_event_catalog(CATALOG_CACHE)
    try:
        detail = ensure_event_detail(clean, catalog=catalog, force_refresh=force_refresh)
        strain_files = select_event_strain(clean, detail, opts)
    except Exception as exc:
        summary.failed[clean] = str(exc)
        return summary

    event_dir = EVENTS_DIR / clean
    event_dir.mkdir(parents=True, exist_ok=True)
    downloaded_any = False
    for strain in strain_files:
        dest = event_dir / strain.local_name
        if dest.is_file() and not force_refresh:
            summary.skipped_existing.append(str(dest.relative_to(DATASETS_DIR)))
            downloaded_any = True
            continue
        try:
            download_uri(strain.url, dest, force=force_refresh)
            summary.downloaded.append(str(dest.relative_to(DATASETS_DIR)))
            downloaded_any = True
        except Exception as exc:
            summary.failed[strain.local_name] = str(exc)

    if downloaded_any and not summary.failed:
        append_done_entries(DEFAULT_DONE_FILE, [ledger_key])
    return summary


def pull_datafind_segment(
    query: str,
    *,
    force_refresh: bool = False,
) -> PullSummary:
    ensure_dirs()
    summary = PullSummary(requested=1)
    obs, channel, gps_start, gps_end = parse_segment_query(query)
    ledger_key = f"segment:{obs}:{channel}:{gps_start}:{gps_end}"
    done = read_done_list(DEFAULT_DONE_FILE)
    if ledger_key in done and not force_refresh:
        summary.skipped_done.append(ledger_key)
        return summary

    try:
        urls = find_frame_urls(obs, channel, gps_start, gps_end)
    except Exception as exc:
        summary.failed[query] = str(exc)
        return summary

    if not urls:
        summary.failed[query] = "GWDataFind returned no frame URLs"
        return summary

    segment_dir = FRAMES_DIR / obs / channel / str(gps_start)
    segment_dir.mkdir(parents=True, exist_ok=True)
    for index, url in enumerate(urls):
        name = url.rsplit("/", 1)[-1] or f"frame_{index}.gwf"
        dest = segment_dir / name
        if dest.is_file() and not force_refresh:
            summary.skipped_existing.append(str(dest.relative_to(DATASETS_DIR)))
            continue
        try:
            download_uri(url, dest, force=force_refresh, prefer_cli=is_pelican_uri(url))
            summary.downloaded.append(str(dest.relative_to(DATASETS_DIR)))
        except Exception as exc:
            summary.failed[name] = str(exc)

    if summary.downloaded or summary.skipped_existing:
        if not summary.failed:
            append_done_entries(DEFAULT_DONE_FILE, [ledger_key])
    return summary


def pull_remote_uri(uri: str, *, force_refresh: bool = False) -> PullSummary:
    ensure_dirs()
    summary = PullSummary(requested=1)
    ledger_key = f"uri:{uri}"
    done = read_done_list(DEFAULT_DONE_FILE)
    if ledger_key in done and not force_refresh:
        summary.skipped_done.append(ledger_key)
        return summary

    name = uri.rsplit("/", 1)[-1] or "download.bin"
    dest = FRAMES_DIR / "uris" / name
    try:
        download_uri(uri, dest, force=force_refresh, prefer_cli=is_pelican_uri(uri))
        summary.downloaded.append(str(dest.relative_to(DATASETS_DIR)))
        append_done_entries(DEFAULT_DONE_FILE, [ledger_key])
    except Exception as exc:
        summary.failed[uri] = str(exc)
    return summary


def pull_selected_targets(
    targets: Iterable[str],
    *,
    params: dict | None = None,
    force_refresh: bool = False,
) -> PullSummary:
    merged = PullSummary()
    expanded: list[str] = []
    for target in targets:
        expanded.extend(expand_event_query(str(target)))
    for target in expanded:
        merged.requested += 1
        item = str(target).strip()
        if not item:
            continue
        if is_event_name(item):
            result = pull_event_strain(item, params=params, force_refresh=force_refresh)
        elif is_remote_uri(item):
            result = pull_remote_uri(item, force_refresh=force_refresh)
        elif ":" in item and not Path(item).exists():
            result = pull_datafind_segment(item, force_refresh=force_refresh)
        elif Path(item).is_file():
            name = Path(item).name
            dest = FRAMES_DIR / "local" / name
            try:
                copy_local_file(Path(item), dest, force=force_refresh)
                merged.downloaded.append(str(dest.relative_to(DATASETS_DIR)))
            except Exception as exc:
                merged.failed[item] = str(exc)
            continue
        else:
            merged.failed[item] = (
                "Unrecognized target. Use GW event ID, OBS:CHANNEL:GPS_START:GPS_END, "
                "Pelican/OSDF URI, or a local file path."
            )
            continue

        merged.downloaded.extend(result.downloaded)
        merged.skipped_done.extend(result.skipped_done)
        merged.skipped_existing.extend(result.skipped_existing)
        merged.failed.update(result.failed)
    return merged


def archive_used_datasets(paths: Iterable[Path]) -> list[Path]:
    moved: list[Path] = []
    for path in paths:
        if path.is_file():
            moved.append(move_to_finished_archive(path))
    return moved


def _print_summary(label: str, summary: PullSummary) -> None:
    print(f"[GWOSC FETCH] {label}")
    if summary.downloaded:
        print(f"  downloaded ({len(summary.downloaded)}):")
        for item in summary.downloaded[:12]:
            print(f"    - {item}")
        if len(summary.downloaded) > 12:
            print(f"    ... +{len(summary.downloaded) - 12} more")
    if summary.skipped_done:
        print(f"  skipped (ledger): {', '.join(summary.skipped_done[:6])}")
    if summary.skipped_existing:
        print(f"  skipped (cached): {len(summary.skipped_existing)} file(s)")
    if summary.failed:
        print("  failed:")
        for key, err in summary.failed.items():
            print(f"    - {key}: {err}")


def fetch_and_graph(query: str, params: dict | None = None) -> None:
    params = params or {}
    force = str(params.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
    print(f"[GWOSC FETCH] Target: {query!r}")
    print(f"[GWOSC FETCH] Repositories: {GWOSC_HOME} | {GWOSC_DATA}")
    print(f"[GWOSC FETCH] Event API: {EVENT_API}")
    summary = pull_selected_targets([query], params=params, force_refresh=force)
    _print_summary("complete", summary)
    if summary.downloaded:
        print("[GWOSC FETCH] Cached under datasets/gwosc/")