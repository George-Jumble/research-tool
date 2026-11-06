"""
ATLAS Open Data fetcher — atlasopenmagic release catalog + HTTPS/xrootd pull.

Mirrors the CERN Open Data fetcher contract: pull → ``datasets/atlas/`` cache,
ledger keys ``atlas:<target>``, optional restore from finished archives.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from menus.particle.atlas.manifest import (
    ATLAS_MANIFEST,
    DEFAULT_PROTOCOL,
    DEFAULT_RELEASE,
    resolve_target_key,
)
from sb_shared.dataset_ledger import mark_processed
from sb_shared.project_paths import (
    FINISHED_A_DIR,
    ensure_project_dirs,
    finished_a_search_dirs,
    move_to_finished_archive,
)
from sb_shared.project_paths import SUPERBLOCK_ROOT

DATASETS_DIR = SUPERBLOCK_ROOT / "datasets" / "atlas"
DONE_FILE = DATASETS_DIR / "done.txt"
TARGET_PREFIX = "atlas:"

_SIMPLECACHE_PREFIX = "simplecache::"


@dataclass
class PullSummary:
    requested: int = 0
    downloaded: list[str] = field(default_factory=list)
    skipped_done: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def ensure_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_project_dirs()


def _ledger_key(target_key: str) -> str:
    return f"{TARGET_PREFIX}{target_key}"


def read_done_list() -> set[str]:
    if not DONE_FILE.is_file():
        return set()
    return {
        line.strip()
        for line in DONE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_done_entry(key: str) -> None:
    ensure_dirs()
    done = read_done_list()
    if key in done:
        return
    with DONE_FILE.open("a", encoding="utf-8") as handle:
        handle.write(key + "\n")


def cache_dir_for_target(target_key: str) -> Path:
    spec = ATLAS_MANIFEST[target_key]
    release = str(spec.get("release") or DEFAULT_RELEASE)
    dataset_key = str(spec.get("dataset_key") or target_key)
    skim = str(spec.get("skim") or "2lep")
    return DATASETS_DIR / release / f"{dataset_key}_{skim}"


def normalize_openmagic_url(url: str) -> str:
    """Strip fsspec ``simplecache::`` wrappers for direct HTTPS download."""
    text = str(url or "").strip()
    if text.startswith(_SIMPLECACHE_PREFIX):
        return text[len(_SIMPLECACHE_PREFIX) :]
    return text


def _import_openmagic():
    try:
        import atlasopenmagic as atom
    except ImportError as exc:
        raise ImportError(
            "atlasopenmagic is required for ATLAS Open Data pulls. "
            "Install with: pip install atlasopenmagic fsspec-xrootd"
        ) from exc
    return atom


def list_available_releases() -> dict[str, str]:
    atom = _import_openmagic()
    raw = atom.available_releases()
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return {str(item): str(item) for item in raw}


def set_active_release(release: str, *, local_path: str | Path | None = None) -> str:
    atom = _import_openmagic()
    path = str(local_path) if local_path else None
    atom.set_release(str(release), local_path=path)
    current = atom.get_current_release()
    print(f"[ATLAS FETCH] Active release: {current}")
    return str(current)


def list_dataset_keys(release: str | None = None) -> list[str]:
    atom = _import_openmagic()
    if release:
        set_active_release(release)
    return [str(k) for k in atom.available_datasets()]


def list_skims(release: str | None = None) -> list[str]:
    atom = _import_openmagic()
    if release:
        set_active_release(release)
    return [str(s) for s in atom.available_skims()]


def resolve_urls(
    dataset_key: str,
    *,
    release: str = DEFAULT_RELEASE,
    skim: str = "2lep",
    protocol: str = DEFAULT_PROTOCOL,
) -> list[str]:
    """Return remote file URLs via atlasopenmagic (https / root / eos)."""
    atom = _import_openmagic()
    set_active_release(release)
    proto = (protocol or DEFAULT_PROTOCOL).strip().lower()
    if proto not in {"https", "root", "eos"}:
        proto = DEFAULT_PROTOCOL
    urls = atom.get_urls(str(dataset_key), skim=str(skim), protocol=proto)
    return [normalize_openmagic_url(u) for u in urls]


def _filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    return name or "atlas_file.root"


def _apply_file_filters(
    urls: list[str],
    *,
    filter_regexp: str | None = None,
    filter_name: str | None = None,
    max_files: int | None = None,
) -> list[str]:
    selected = list(urls)
    if filter_name:
        names = {n.strip() for n in str(filter_name).split(",") if n.strip()}
        selected = [u for u in selected if _filename_from_url(u) in names]
    if filter_regexp:
        pat = re.compile(str(filter_regexp))
        selected = [u for u in selected if pat.search(_filename_from_url(u))]
    if max_files is not None and int(max_files) > 0:
        selected = selected[: int(max_files)]
    return selected


def _download_https(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)


def _download_xrootd(url: str, dest: Path) -> None:
    """Copy root:// / xrootd URLs via fsspec-xrootd when available."""
    try:
        import fsspec
    except ImportError as exc:
        raise ImportError(
            "fsspec-xrootd is required for protocol=root. "
            "Install with: pip install fsspec-xrootd"
        ) from exc

    with fsspec.open(url, "rb") as remote, dest.open("wb") as local:
        shutil.copyfileobj(remote, local, length=1024 * 256)


def download_url(url: str, dest: Path, *, protocol: str = "https") -> None:
    clean = normalize_openmagic_url(url)
    proto = (protocol or "https").strip().lower()
    if clean.startswith("root://") or proto == "root":
        _download_xrootd(clean, dest)
        return
    if clean.startswith("http://") or clean.startswith("https://") or proto in {"https", "eos"}:
        # eos protocol often still returns https URLs after normalize
        _download_https(clean, dest)
        return
    raise ValueError(f"Unsupported ATLAS open-data URL: {clean!r}")


def _restore_from_finished_a(target_key: str, dest: Path) -> bool:
    for root in finished_a_search_dirs():
        candidate = root / f"atlas_{target_key}"
        if candidate.is_dir():
            shutil.copytree(candidate, dest, dirs_exist_ok=True)
            print(f"[ATLAS FETCH] Restored {target_key} from {candidate}")
            return True
    return False


def download_manifest_target(
    target_key: str,
    *,
    force_refresh: bool = False,
    restore_archived: bool = True,
    auto_fetch: bool = True,
    protocol: str | None = None,
) -> tuple[Path, str]:
    """
    Pull a curated ATLAS manifest target into ``datasets/atlas/``.

    Returns (cache_dir, source_tag).
    """
    if target_key not in ATLAS_MANIFEST:
        raise KeyError(f"Unknown ATLAS manifest key: {target_key}")

    ensure_dirs()
    spec = ATLAS_MANIFEST[target_key]
    dest = cache_dir_for_target(target_key)
    ledger = _ledger_key(target_key)
    proto = str(protocol or spec.get("protocol") or DEFAULT_PROTOCOL)

    if dest.is_dir() and any(dest.rglob("*.root")) and not force_refresh:
        print(f"[ATLAS FETCH] Using cached {target_key}: {dest}")
        append_done_entry(ledger)
        return dest, "cached"

    if not force_refresh and restore_archived and _restore_from_finished_a(target_key, dest):
        append_done_entry(ledger)
        mark_processed([ledger], note=f"atlas restore {target_key}")
        return dest, "restored"

    if not auto_fetch:
        raise FileNotFoundError(
            f"ATLAS dataset {target_key} not cached at {dest}. "
            "Enable auto_fetch or run ATLAS Open Data pull."
        )

    release = str(spec.get("release") or DEFAULT_RELEASE)
    dataset_key = str(spec.get("dataset_key") or "data")
    skim = str(spec.get("skim") or "2lep")
    print(
        f"[ATLAS FETCH] Pulling {target_key}: release={release} "
        f"key={dataset_key} skim={skim} protocol={proto}"
    )
    urls = resolve_urls(dataset_key, release=release, skim=skim, protocol=proto)
    filtered = _apply_file_filters(
        urls,
        filter_regexp=spec.get("filter_regexp"),
        filter_name=spec.get("filter_name"),
        max_files=spec.get("max_files"),
    )
    if not filtered:
        raise FileNotFoundError(
            f"No files matched filters for {target_key} "
            f"(release={release}, key={dataset_key}, skim={skim})"
        )

    dest.mkdir(parents=True, exist_ok=True)
    for url in filtered:
        name = _filename_from_url(url)
        out = dest / name
        if out.is_file() and out.stat().st_size > 1000 and not force_refresh:
            print(f"[ATLAS FETCH]   skip existing {name}")
            continue
        print(f"[ATLAS FETCH]   downloading {name}")
        download_url(url, out, protocol=proto)

    append_done_entry(ledger)
    mark_processed([ledger], note=f"atlas fetch {target_key}")
    meta = {
        "target_key": target_key,
        "release": release,
        "dataset_key": dataset_key,
        "skim": skim,
        "protocol": proto,
        "urls": filtered,
        "spec": {k: v for k, v in spec.items() if k != "urls"},
    }
    (dest / "atlas_fetch_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    print(f"[ATLAS FETCH] Ready: {dest} ({len(filtered)} file(s))")
    return dest, "fetched"


def ensure_atlas_target(
    target_key: str,
    *,
    force_refresh: bool = False,
    restore_archived: bool = True,
    auto_fetch: bool = True,
    protocol: str | None = None,
) -> Path:
    path, _ = download_manifest_target(
        target_key,
        force_refresh=force_refresh,
        restore_archived=restore_archived,
        auto_fetch=auto_fetch,
        protocol=protocol,
    )
    return path


def list_cached_targets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, spec in ATLAS_MANIFEST.items():
        dest = cache_dir_for_target(key)
        root_files = sorted(dest.rglob("*.root")) if dest.is_dir() else []
        rows.append(
            {
                "key": key,
                "label": spec.get("label", key),
                "release": spec.get("release"),
                "dataset_key": spec.get("dataset_key"),
                "skim": spec.get("skim"),
                "cache_dir": str(dest),
                "cached": bool(root_files),
                "n_root_files": len(root_files),
                "sample_root": str(root_files[0]) if root_files else None,
            }
        )
    return rows


def list_dataset_files() -> list[Path]:
    ensure_dirs()
    return sorted(p for p in DATASETS_DIR.rglob("*.root") if p.is_file())


def pull_selected_targets(
    names: list[str],
    *,
    params: dict | None = None,
    force_refresh: bool = False,
) -> PullSummary:
    params = params or {}
    restore = str(params.get("restore_archived", "yes")).lower() in {"yes", "y", "true", "1"}
    protocol = (params.get("protocol") or "").strip() or None
    summary = PullSummary(requested=len(names))
    done = read_done_list()

    for raw in names:
        key = resolve_target_key(raw) or raw
        if key not in ATLAS_MANIFEST:
            summary.failed[raw] = f"unknown target {raw!r}"
            continue
        ledger = _ledger_key(key)
        if ledger in done and not force_refresh:
            dest = cache_dir_for_target(key)
            if dest.is_dir() and any(dest.rglob("*.root")):
                summary.skipped_done.append(key)
                continue
        try:
            _, source = download_manifest_target(
                key,
                force_refresh=force_refresh,
                restore_archived=restore,
                auto_fetch=True,
                protocol=protocol,
            )
            if source == "cached":
                summary.skipped_existing.append(key)
            else:
                summary.downloaded.append(key)
        except Exception as exc:
            summary.failed[key] = str(exc)
    return summary


def pull_open_archives(
    *,
    force_refresh: bool = False,
    restore_archived: bool = True,
    protocol: str | None = None,
) -> PullSummary:
    return pull_selected_targets(
        list(ATLAS_MANIFEST.keys()),
        force_refresh=force_refresh,
        params={
            "restore_archived": "yes" if restore_archived else "no",
            "protocol": protocol or DEFAULT_PROTOCOL,
        },
    )


def archive_used_datasets(paths: list[Path]) -> list[Path]:
    ensure_project_dirs()
    moved: list[Path] = []
    for src in paths:
        if not src.exists():
            continue
        dest = move_to_finished_archive(
            src,
            FINISHED_A_DIR,
            also_search=finished_a_search_dirs(),
        )
        moved.append(dest)
    return moved


def resolve_root_path(query: str) -> Path | None:
    """Resolve manifest key, atlas:key, or local path to a .root file."""
    q = (query or "").strip()
    if not q:
        return None
    path = Path(q).expanduser()
    if path.is_file() and path.suffix.lower() == ".root":
        return path
    if path.is_dir():
        roots = sorted(path.rglob("*.root"))
        return roots[0] if roots else None
    key = resolve_target_key(q)
    if key:
        dest = cache_dir_for_target(key)
        roots = sorted(dest.rglob("*.root"))
        return roots[0] if roots else None
    return None


def is_atlas_mini_root(path: str | Path) -> bool:
    """True when the file looks like ATLAS Open Data outreach mini-tree ROOT."""
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".root":
        return False
    try:
        import uproot
    except ImportError:
        return "2lep" in path.name or "4lep" in path.name or "1lep" in path.name
    try:
        with uproot.open(path) as handle:
            keys = [str(k).split(";")[0] for k in handle.keys()]
            if "mini" not in keys:
                return False
            tree = handle["mini"]
            branches = set(tree.keys())
            return "lep_pt" in branches and "lep_n" in branches
    except Exception:
        return False
