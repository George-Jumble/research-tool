"""
Curated ATLAS Open Data targets via atlasopenmagic releases/skims.
"""

from __future__ import annotations

from typing import Any

# Education / research releases supported by atlasopenmagic.
ATLAS_RELEASES: tuple[str, ...] = (
    "2016e-8tev",
    "2020e-13tev",
    "2024r-pp",
    "2024r-hi",
    "2025e-13tev-beta",
    "2025r-evgen-13tev",
    "2025r-evgen-13p6tev",
)

DEFAULT_RELEASE = "2020e-13tev"
DEFAULT_PROTOCOL = "https"
DEFAULT_SKIM = "2lep"

# Laptop-friendly curated pulls (atlasopenmagic dataset key + skim).
ATLAS_MANIFEST: dict[str, dict[str, Any]] = {
    "atlas_2lep_data": {
        "release": DEFAULT_RELEASE,
        "dataset_key": "data",
        "skim": "2lep",
        "label": "ATLAS Open Data 13 TeV 2lep (data A–D)",
        "protocol": DEFAULT_PROTOCOL,
        "max_files": 4,
        "lepton_focus": "both",
        "analyze": "atlas_mini",
    },
    "atlas_2lep_data_A": {
        "release": DEFAULT_RELEASE,
        "dataset_key": "data",
        "skim": "2lep",
        "label": "ATLAS Open Data 13 TeV 2lep data_A only",
        "protocol": DEFAULT_PROTOCOL,
        "filter_name": "data_A.2lep.root",
        "max_files": 1,
        "lepton_focus": "both",
        "analyze": "atlas_mini",
    },
    "atlas_4lep_data": {
        "release": DEFAULT_RELEASE,
        "dataset_key": "data",
        "skim": "4lep",
        "label": "ATLAS Open Data 13 TeV 4lep (data)",
        "protocol": DEFAULT_PROTOCOL,
        "max_files": 4,
        "lepton_focus": "both",
        "analyze": "atlas_mini",
    },
    "atlas_2lep_zprime_mumu": {
        "release": DEFAULT_RELEASE,
        "dataset_key": "301220",
        "skim": "2lep",
        "label": "ATLAS MC Z'→μμ (301220) 2lep skim",
        "protocol": DEFAULT_PROTOCOL,
        "max_files": 1,
        "lepton_focus": "muon",
        "analyze": "atlas_mini",
    },
    "atlas_2lep_zprime_ee": {
        "release": DEFAULT_RELEASE,
        "dataset_key": "301215",
        "skim": "2lep",
        "label": "ATLAS MC Z'→ee (301215) 2lep skim",
        "protocol": DEFAULT_PROTOCOL,
        "max_files": 1,
        "lepton_focus": "electron",
        "analyze": "atlas_mini",
    },
}

ACTION_DEFAULT_TARGETS: dict[str, list[str]] = {
    "Pull ATLAS Open Data": list(ATLAS_MANIFEST.keys()),
    "ATLAS Open Data Lepton Scan": ["atlas_2lep_data_A"],
}


def target_id(key: str) -> str:
    return f"atlas:{key}"


def resolve_target_key(query: str) -> str | None:
    """Map menu query / atlas:key / local label to a manifest key."""
    q = (query or "").strip()
    if not q:
        return None
    if q in ATLAS_MANIFEST:
        return q
    if q.startswith("atlas:"):
        key = q.split(":", 1)[-1]
        return key if key in ATLAS_MANIFEST else None
    for key, spec in ATLAS_MANIFEST.items():
        if q == str(spec.get("label") or ""):
            return key
        if " — " in q and q.split(" — ", 1)[0].strip() == key:
            return key
    return None
