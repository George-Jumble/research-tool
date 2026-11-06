"""
CERN Open Data ↔ research_tool.py
"""

from __future__ import annotations


from menus.particle.cern.fetcher import (
    DATASETS_DIR,
    list_cached_targets,
    pull_open_archives,
    pull_selected_targets,
)
from menus.particle.cern.manifest import (
    CERN_MANIFEST,
)

MODULE_TAG = "CERN_OPENDATA"
SUBMENU_TITLE = "CERN Open Data"

MENU_ACTIONS = [
    "Pull Datasets from Open Archives",
    "Pull ATLAS Open Data (atlasopenmagic)",
    "Analyze CMS NanoAOD",
    "ATLAS Open Data Lepton Scan",
    "Analyze ALICE ROOT Sample",
    "List Cached Datasets",
]

_MANIFEST_CHOICES = list(CERN_MANIFEST.keys())


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def entry_fields(action: str) -> list[dict]:
    if action == "Pull Datasets from Open Archives":
        return [
            {
                "key": "targets",
                "label": "Targets (comma-separated)",
                "default": ",".join(_MANIFEST_CHOICES),
                "required": False,
                "hint": "Manifest keys or 'all'",
            },
            {
                "key": "force_refresh",
                "label": "Force refresh",
                "default": "no",
                "required": False,
                "hint": "yes = re-download even if cached",
                "choices": ["no", "yes"],
            },
            {
                "key": "restore_archived",
                "label": "Restore from finishedA/",
                "default": "yes",
                "required": False,
                "hint": "yes = restore archived copy before remote pull",
                "choices": ["yes", "no"],
            },
        ]
    if action == "List Cached Datasets":
        return []
    if action == "Pull ATLAS Open Data (atlasopenmagic)":
        from menus.particle.atlas.manifest import ATLAS_MANIFEST, DEFAULT_PROTOCOL, DEFAULT_RELEASE

        return [
            {
                "key": "targets",
                "label": "ATLAS targets (comma-separated)",
                "default": "atlas_2lep_data_A",
                "required": False,
                "hint": "Manifest keys or 'all' — " + ", ".join(ATLAS_MANIFEST.keys()),
            },
            {
                "key": "release",
                "label": "atlasopenmagic release",
                "default": DEFAULT_RELEASE,
                "required": False,
                "hint": "Education: 2020e-13tev | Research: 2024r-pp",
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
                "hint": "https | root (fsspec-xrootd) | eos",
                "choices": ["https", "root", "eos"],
            },
            {
                "key": "force_refresh",
                "label": "Force refresh",
                "default": "no",
                "required": False,
                "choices": ["no", "yes"],
            },
        ]
    return [
        {
            "key": "target",
            "label": "Manifest target or .root path",
            "default": "cms_nanoaod_dimu",
            "required": False,
            "hint": "Preset key, recid, or local ROOT path",
            "choices": _MANIFEST_CHOICES,
        },
        {
            "key": "entry_stop",
            "label": "Max events / tracks",
            "default": "50000",
            "required": False,
            "hint": "Uproot read limit",
        },
        {
            "key": "output_prefix",
            "label": "Report prefix",
            "default": "cern_analysis",
            "required": False,
            "hint": "JSON under artifacts/",
        },
    ]


def entry_instructions(action: str) -> list[str]:
    if action == "Pull Datasets from Open Archives":
        return [
            "Programmatic fetch via opendata.cern.ch API — no browser required.",
            f"Cache: {DATASETS_DIR}/<target>/recid_<N>/",
            "Ledger: datasets/cern/done.txt + global processed registry.",
            "Manifest: CMS NanoAOD education samples + ALICE ESD skim (1 file).",
        ]
    if action == "List Cached Datasets":
        return [
            "Shows manifest targets, recid, cache path, and sample ROOT files.",
            "Also lists ATLAS Open Data cache under datasets/atlas/.",
        ]
    if action == "Pull ATLAS Open Data (atlasopenmagic)":
        return [
            "Uses atlasopenmagic: available_releases → set_release → get_urls(protocol=https|root).",
            "Deps: atlasopenmagic, cernopendata-client, fsspec-xrootd (for root://).",
            "Cache under datasets/atlas/<release>/<key>_<skim>/.",
            "Default education release 2020e-13tev; 2lep/4lep skims for lepton work.",
        ]
    return [
        "Scored by the Superblock residual engine (template C).",
        "Run 'Pull Datasets from Open Archives' first to cache the target.",
        "Report: artifacts/superblock_residuals/CERN_OPENDATA_*.json",
    ]


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def run_action(action: str, show_plots: bool = True, options: dict | None = None) -> str | None:
    from superblock_mode import CONTINUE, begin_live_action

    routed = begin_live_action(MODULE_TAG, action, options)
    if routed is not CONTINUE:
        if isinstance(routed, dict):
            return routed.get("report_path")
        return routed
    _ = show_plots
    options = options or {}

    if action == "Pull Datasets from Open Archives":
        force = _yes(options.get("force_refresh", "no"))
        restore = _yes(options.get("restore_archived", "yes"))
        raw_targets = str(options.get("targets") or "all").strip()
        if raw_targets.lower() in {"all", "*"}:
            summary = pull_open_archives(force_refresh=force, restore_archived=restore)
        else:
            names = [t.strip() for t in raw_targets.split(",") if t.strip()]
            summary = pull_selected_targets(
                names,
                params={"restore_archived": "yes" if restore else "no"},
                force_refresh=force,
            )
        print(f"[CERN OPENDATA] Downloaded: {summary.downloaded}")
        print(f"[CERN OPENDATA] Skipped: {summary.skipped_done + summary.skipped_existing}")
        if summary.failed:
            for key, msg in summary.failed.items():
                print(f"[CERN OPENDATA] FAILED {key}: {msg}")
        return None

    if action == "List Cached Datasets":
        for row in list_cached_targets():
            status = "cached" if row["cached"] else "remote"
            print(
                f"  {row['key']:22} recid={row['recid']} [{status}] "
                f"root={row['n_root_files']} → {row['cache_dir']}"
            )
        try:
            from menus.particle.atlas.fetcher import list_cached_targets as list_atlas

            print("[CERN OPENDATA] ATLAS Open Data cache:")
            for row in list_atlas():
                status = "cached" if row["cached"] else "remote"
                print(
                    f"  {row['key']:22} [{status}] root={row['n_root_files']} "
                    f"→ {row['cache_dir']}"
                )
        except Exception as exc:
            print(f"[CERN OPENDATA] ATLAS cache list skipped: {exc}")
        return None

    if action == "Pull ATLAS Open Data (atlasopenmagic)":
        from menus.particle.atlas.fetcher import (
            pull_open_archives as atlas_pull_open_archives,
            pull_selected_targets as atlas_pull_selected_targets,
        )
        from menus.particle.atlas.manifest import ATLAS_MANIFEST

        force = _yes(options.get("force_refresh", "no"))
        protocol = str(options.get("protocol") or "https").strip() or "https"
        raw_targets = str(options.get("targets") or "atlas_2lep_data_A").strip()
        if raw_targets.lower() in {"all", "*"}:
            summary = atlas_pull_open_archives(force_refresh=force, protocol=protocol)
        else:
            names = [t.strip() for t in raw_targets.split(",") if t.strip()]
            if not names:
                names = list(ATLAS_MANIFEST.keys())[:1]
            summary = atlas_pull_selected_targets(
                names,
                force_refresh=force,
                params={"protocol": protocol, "restore_archived": "yes"},
            )
        print(f"[ATLAS OPENDATA] Downloaded: {summary.downloaded}")
        print(f"[ATLAS OPENDATA] Skipped: {summary.skipped_done + summary.skipped_existing}")
        if summary.failed:
            for key, msg in summary.failed.items():
                print(f"[ATLAS OPENDATA] FAILED {key}: {msg}")
        return None

    print(f"[CERN OPENDATA] Unknown action: {action}")
    return None
