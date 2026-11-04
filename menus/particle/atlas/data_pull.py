"""ATLAS Open Data — atlasopenmagic pull + mini-tree lepton analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from menus.particle.atlas.manifest import ATLAS_MANIFEST, DEFAULT_PROTOCOL, DEFAULT_RELEASE
from tav_research.data_pull_common import COMMON_BATCH_HINT
from tav_shared.llm_analysis import append_llm_entry_fields

MENU_LABEL = "ATLAS Open Data (Lepton/Photon Tracks)"

_PREFERRED_DEFAULT_KEY = "atlas_2lep_data_A"


def discover_atlas_open_data_targets(*, include_uncached: bool = False) -> list[dict[str, Any]]:
    """Discover curated ATLAS targets under ``datasets/atlas/``."""
    from menus.particle.atlas.fetcher import DATASETS_DIR, list_cached_targets

    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_paths: set[str] = set()

    for item in list_cached_targets():
        key = str(item.get("key") or "")
        sample = item.get("sample_root")
        if item.get("cached") and sample:
            path = Path(sample)
            rows.append(
                {
                    "key": key,
                    "label": f"{key} — {path.name}",
                    "path": str(path),
                    "cached": True,
                }
            )
            seen_keys.add(key)
            seen_paths.add(str(path.resolve()))
        elif include_uncached and key not in seen_keys:
            rows.append(
                {
                    "key": key,
                    "label": f"{key} — (auto-fetch via atlasopenmagic)",
                    "path": None,
                    "cached": False,
                }
            )
            seen_keys.add(key)

    if DATASETS_DIR.is_dir():
        for root_file in sorted(DATASETS_DIR.rglob("*.root")):
            resolved = str(root_file.resolve())
            if resolved in seen_paths:
                continue
            try:
                rel = root_file.relative_to(DATASETS_DIR)
            except ValueError:
                rel = Path(root_file.name)
            rows.append(
                {
                    "key": resolved,
                    "label": rel.as_posix(),
                    "path": resolved,
                    "cached": True,
                }
            )
            seen_paths.add(resolved)

    rows.sort(key=lambda row: (0 if row.get("cached") else 1, str(row.get("label") or "")))
    return rows


def _default_dataset_label(targets: list[dict[str, Any]]) -> str:
    if not targets:
        return _PREFERRED_DEFAULT_KEY
    for preferred in (_PREFERRED_DEFAULT_KEY, "atlas_2lep_data", "atlas_4lep_data"):
        for row in targets:
            if row.get("key") == preferred:
                return str(row["label"])
    return str(targets[0]["label"])


def resolve_atlas_open_data_query(query: str) -> str:
    """Map picker label, manifest key, or path to an analysis target."""
    from menus.particle.atlas.manifest import resolve_target_key

    q = (query or "").strip()
    if not q:
        targets = discover_atlas_open_data_targets()
        if targets:
            row = targets[0]
            return str(row["path"] or row["key"])
        return _PREFERRED_DEFAULT_KEY

    catalog = discover_atlas_open_data_targets(include_uncached=True)
    for row in catalog:
        if q == row.get("label") or q == row.get("key"):
            return str(row.get("path") or row["key"])

    if " — " in q:
        head = q.split(" — ", 1)[0].strip()
        for row in catalog:
            if head == row.get("key"):
                return str(row.get("path") or row["key"])
        key = resolve_target_key(head)
        if key:
            return key

    key = resolve_target_key(q)
    return key or q


def entry_instructions() -> list[str]:
    targets = discover_atlas_open_data_targets()
    n_cached = sum(1 for row in targets if row.get("cached"))
    lines = [
        f"Auto-discovered {n_cached} cached ATLAS dataset(s) under datasets/atlas/.",
        "Remote catalog: atlasopenmagic (set_release + get_urls protocol=https|root).",
        f"Default release: {DEFAULT_RELEASE}; education 2lep/4lep skims for lepton scans.",
        "Deps: atlasopenmagic, cernopendata-client, fsspec-xrootd (optional root://).",
        "Analysis: ATLAS mini-tree → shared Tav 7-fold lepton engine.",
        COMMON_BATCH_HINT,
    ]
    if targets:
        lines.append("Cached:")
        for row in targets[:6]:
            if row.get("cached"):
                lines.append(f"  • {row['label']}")
    else:
        lines.append(
            "No local cache yet — pick a manifest target to auto-fetch "
            "(default atlas_2lep_data_A)."
        )
    return lines


def entry_fields() -> list[dict]:
    targets = discover_atlas_open_data_targets(include_uncached=True)
    cached = [row for row in targets if row.get("cached")]
    picker_rows = cached or targets
    choices = [str(row["label"]) for row in picker_rows]
    if not choices:
        choices = list(ATLAS_MANIFEST.keys())

    default_label = _default_dataset_label(cached or picker_rows)
    n_cached = len(cached)

    return append_llm_entry_fields(
        [
            {
                "key": "query",
                "label": "ATLAS dataset (datasets/atlas/)",
                "default": default_label,
                "required": False,
                "choices": choices,
                "hint": (
                    f"{n_cached} cached .root file(s) — pick one or leave default "
                    "for auto-fetch via atlasopenmagic"
                ),
            },
            {
                "key": "release",
                "label": "atlasopenmagic release",
                "default": DEFAULT_RELEASE,
                "required": False,
                "hint": "e.g. 2020e-13tev (education) or 2024r-pp (research)",
                "choices": [
                    "2020e-13tev",
                    "2016e-8tev",
                    "2025e-13tev-beta",
                    "2024r-pp",
                    "2024r-hi",
                ],
            },
            {
                "key": "protocol",
                "label": "Transfer protocol",
                "default": DEFAULT_PROTOCOL,
                "required": False,
                "hint": "https (default) | root (needs fsspec-xrootd) | eos",
                "choices": ["https", "root", "eos"],
            },
            {
                "key": "lepton",
                "label": "Lepton focus",
                "default": "both",
                "required": False,
                "hint": "PDG filter on lep_type: muon=13, electron=11, both=all",
                "choices": ["both", "muon", "electron"],
            },
            {
                "key": "entry_stop",
                "label": "Max events",
                "default": "50000",
                "required": False,
                "hint": "0 = entire local ROOT file",
            },
            {
                "key": "force_refresh",
                "label": "Force refresh fetch",
                "default": "no",
                "required": False,
                "hint": "yes = re-download even if cached",
                "choices": ["no", "yes"],
            },
            {
                "key": "batch_file",
                "label": "Batch list file (optional)",
                "default": "",
                "required": False,
                "hint": "Plain-text list of manifest keys or .root paths",
            },
        ]
    )


def fetch_and_graph(query: str, params: dict[str, Any] | None = None) -> None:
    from menus.particle.atlas.analyzer import tav_analyze_atlas_leptons_from_file
    from menus.particle.atlas.fetcher import (
        ensure_atlas_target,
        is_atlas_mini_root,
        resolve_root_path,
    )
    from menus.particle.atlas.manifest import resolve_target_key
    from menus.particle.cern.tav_superblock_cms_lepton_analyzer_v2 import (
        tav_analyze_leptons_from_file,
    )

    params = params or {}
    q = resolve_atlas_open_data_query(query)
    force = str(params.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
    protocol = (params.get("protocol") or DEFAULT_PROTOCOL).strip() or DEFAULT_PROTOCOL
    lepton = (params.get("lepton") or "both").strip() or "both"
    try:
        entry_stop = int(params.get("entry_stop") or 50000)
    except ValueError:
        entry_stop = 50000
    if entry_stop <= 0:
        entry_stop = None

    key = resolve_target_key(q)
    if key and resolve_root_path(key) is None:
        print(f"[Superblock Residual Engine] Auto-fetching ATLAS target via atlasopenmagic: {key}")
        ensure_atlas_target(key, force_refresh=force, auto_fetch=True, protocol=protocol)

    root = resolve_root_path(key or q) or resolve_root_path(q)
    if root is None and key:
        root = resolve_root_path(key)
    if root is None:
        print(f"[Superblock Residual Engine] ATLAS dataset not resolved: {q!r}")
        print("  Run with a cached path or manifest key (atlas_2lep_data_A).")
        return

    print(f"[Superblock Residual Engine] ATLAS dataset resolved: {root}")
    if is_atlas_mini_root(root):
        report = tav_analyze_atlas_leptons_from_file(
            root,
            output_dir="atlas_open_data",
            entry_stop=entry_stop,
            lepton=lepton,
            save_plots=True,
            verbose=True,
        )
    else:
        # Fallback: CMS-style NanoAOD lepton path if a CMS file was selected by path
        print("[Superblock Residual Engine] File is not ATLAS mini — routing to CMS lepton analyzer.")
        report = tav_analyze_leptons_from_file(
            root,
            output_dir="atlas_open_data_cms_fallback",
            entry_stop=entry_stop,
            lepton="Muon" if lepton in {"muon", "mu"} else "Electron",
            save_plots=True,
            verbose=True,
        )

    print(f"[Superblock Residual Engine] ATLAS Open Data report: {report.get('report_path')}")
    sp = report.get("seven_periodic") or {}
    if sp.get("significance_sigma") is not None:
        print(
            f"[Superblock Residual Engine] Engine score σ: {sp['significance_sigma']:.2f} "
            f"({sp.get('interpretation_label') or 'internal score'})"
        )
