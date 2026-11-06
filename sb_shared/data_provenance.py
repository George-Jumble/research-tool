#!/usr/bin/env python3
"""
Data provenance registry for Superblock / research_tool modules.

Reports whether each menu action uses genuine repository data (cached locally),
remote-only targets, synthetic validation data, or purely theoretical inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from sb_shared.project_paths import ARTIFACTS_ROOT, SUPERBLOCK_ROOT as PROJECT_ROOT

DATASETS_DESI = PROJECT_ROOT / "datasets" / "desi"
COBAYA_DR2 = DATASETS_DESI / "bao_data" / "desi_bao_dr2"
SPARC_DATA = PROJECT_ROOT / "data" / "sparc"
SPARC_DATASETS = PROJECT_ROOT / "datasets" / "sparc"
PANTHEON = DATASETS_DESI / "pantheon_plus" / "Pantheon+SH0ES.dat"
GWOSC_DATASETS = PROJECT_ROOT / "datasets" / "gwosc"
LISA_DATASETS = PROJECT_ROOT / "datasets" / "lisa"
CERN_DATASETS = PROJECT_ROOT / "datasets" / "cern"

DataClass = Literal["real_cached", "real_remote", "synthetic", "theoretical", "mixed"]


@dataclass
class ProvenanceEntry:
    module: str
    action: str
    data_class: DataClass
    repository: str | None
    local_path: str | None
    data_available: bool
    notes: str


def _count_glob(directory: Path, pattern: str) -> int:
    if not directory.is_dir():
        return 0
    return len(list(directory.glob(pattern)))


def dataset_availability() -> dict[str, Any]:
    """Snapshot of whether key on-disk datasets are present."""
    desi_tables = _count_glob(COBAYA_DR2, "desi_gaussian_bao_*_mean.txt")
    sparc_count = _count_glob(SPARC_DATA, "*.csv") + _count_glob(SPARC_DATASETS, "*.csv")
    legacy_bao = _count_glob(DATASETS_DESI / "bao_data", "sdss_*.dat")
    return {
        "desi_dr2_cobaya_tables": desi_tables,
        "desi_dr2_path": str(COBAYA_DR2),
        "desi_dr2_available": desi_tables > 0,
        "legacy_bao_tables": legacy_bao,
        "sparc_galaxy_csv_count": sparc_count,
        "sparc_available": sparc_count > 0,
        "pantheon_plus_available": PANTHEON.is_file(),
        "pantheon_plus_path": str(PANTHEON),
        "gwosc_strain_files": _count_glob(GWOSC_DATASETS, "*") + sum(
            1 for _ in GWOSC_DATASETS.rglob("*") if _.is_file()
        ) if GWOSC_DATASETS.is_dir() else 0,
        "gwosc_available": GWOSC_DATASETS.is_dir() and any(GWOSC_DATASETS.rglob("*")),
        "lisa_files": sum(1 for _ in LISA_DATASETS.rglob("*") if _.is_file()) if LISA_DATASETS.is_dir() else 0,
        "lisa_available": LISA_DATASETS.is_dir() and any(LISA_DATASETS.rglob("*")),
        "cern_root_files": _count_glob(CERN_DATASETS, "*.root") + sum(
            1 for _ in CERN_DATASETS.rglob("*.root") if _.is_file()
        ) if CERN_DATASETS.is_dir() else 0,
        "cern_available": CERN_DATASETS.is_dir() and any(CERN_DATASETS.rglob("*.root")),
    }


def _entries_desi(avail: dict[str, Any]) -> list[ProvenanceEntry]:
    repo = "CobayaSampler/bao_data + data.desi.lbl.gov"
    path = str(COBAYA_DR2)
    ok = avail["desi_dr2_available"]
    real: DataClass = "real_cached" if ok else "real_remote"
    return [
        ProvenanceEntry("DESI_BAO", "Real DESI DR2 Scan (Cobaya)", real, repo, path, ok,
                        "Primary production scan; fails if DR2 tables missing."),
        ProvenanceEntry("DESI_BAO", "Healpy Dipole Fit", "mixed", repo, path, ok,
                        "Real BAO vector; healpy map uses synthetic sky patches unless user supplies RA/Dec."),
        ProvenanceEntry("DESI_BAO", "Change-Point Detection", real, repo, path, ok,
                        "Changepoint search on DR2 expansion residuals."),
        ProvenanceEntry("DESI_BAO", "Joint DESI+SN+Planck Fit", "mixed", repo, path, ok,
                        "DESI DR2 + Pantheon+ table if cached."),
        ProvenanceEntry("DESI_BAO", "DESI Catalog → LSS Grid", real, repo, path, ok,
                        "DESI catalog binned onto the LSS grid."),
    ]


def _entries_sparc(avail: dict[str, Any]) -> list[ProvenanceEntry]:
    repo = "https://astroweb.case.edu/SPARC/"
    path = str(SPARC_DATASETS if SPARC_DATASETS.is_dir() else SPARC_DATA)
    ok = avail["sparc_available"]
    dc: DataClass = "real_cached" if ok else "real_remote"
    return [
        ProvenanceEntry("SPARC", "Pull Batch from Repository", dc, repo, path, ok,
                        "Downloads MassModels_Lelli2016c.mrt; per-galaxy CSVs."),
        ProvenanceEntry("SPARC", "Rotation Curve Analysis", dc, repo, path, ok,
                        "Uses cached galaxy CSVs under data/sparc or datasets/sparc."),
    ]


def _entries_frb() -> list[ProvenanceEntry]:
    return [
        ProvenanceEntry(
            "FRB_WEB", "Fetch CHIME/FRB Catalog", "real_remote",
            "CHIME/FRB + CDS mirrors", str(PROJECT_ROOT / "datasets" / "frb"),
            False, "Network download; void catalog from public URL.",
        ),
        ProvenanceEntry(
            "FRB_WEB", "FRB Web Analysis", "mixed", "CHIME/FRB", None, True,
            "Real catalog when fetched; some sightline grids are deterministic synthetic.",
        ),
    ]


def _entries_gravitic(avail: dict[str, Any]) -> list[ProvenanceEntry]:
    gwosc_repo = "https://gwosc.org/ + GWDataFind + Pelican OSDF"
    lisa_repo = "IGWN OSDF (Pelican) + requests-pelican"
    gwosc_ok = avail.get("gwosc_available", False)
    lisa_ok = avail.get("lisa_available", False)
    gwosc_dc: DataClass = "real_cached" if gwosc_ok else "real_remote"
    lisa_dc: DataClass = "real_cached" if lisa_ok else "real_remote"
    return [
        ProvenanceEntry(
            "LIGO_GWOSC",
            "Pull Strain from GWOSC",
            gwosc_dc,
            gwosc_repo,
            str(GWOSC_DATASETS),
            gwosc_ok,
            "GWOSC Event API strain files; GWDataFind segments; Pelican/osdf URIs.",
        ),
        ProvenanceEntry(
            "LIGO_GWOSC",
            "Full Ringdown Report",
            gwosc_dc,
            gwosc_repo,
            str(ARTIFACTS_ROOT),
            gwosc_ok,
            "Multi-detector ringdown JSON/PNG under artifacts/.",
        ),
        ProvenanceEntry(
            "LIGO_GWOSC",
            "Scan Post-Merger Markers",
            gwosc_dc,
            gwosc_repo,
            str(ARTIFACTS_ROOT),
            gwosc_ok,
            "20–300 ms post-merger damped-sinusoid scan with time-slide FAP; JSON/PNG under artifacts/.",
        ),
        ProvenanceEntry(
            "LISA_PRE_RUNS",
            "LISA Pre-runs",
            lisa_dc,
            lisa_repo,
            str(LISA_DATASETS),
            lisa_ok,
            "IGWN/LISA mock data via requests-pelican and pelican CLI.",
        ),
    ]


def _entries_cern(avail: dict[str, Any]) -> list[ProvenanceEntry]:
    cern_ok = avail["cern_available"]
    cern_dc: DataClass = "real_cached" if cern_ok else "real_remote"
    cern_repo = "CERN Open Data Portal (opendata.cern.ch)"
    return [
        ProvenanceEntry(
            "CERN_OPENDATA",
            "Pull Datasets from Open Archives",
            cern_dc,
            cern_repo,
            str(CERN_DATASETS),
            cern_ok,
            "REST API fetch into datasets/cern/<target>/recid_<N>/; no browser.",
        ),
        ProvenanceEntry(
            "CERN_OPENDATA",
            "Analyze CMS NanoAOD",
            cern_dc,
            cern_repo,
            str(CERN_DATASETS),
            cern_ok,
            "Uproot + awkward on cached CMS NanoAOD Events tree.",
        ),
        ProvenanceEntry(
            "CERN_OPENDATA",
            "Analyze ALICE ROOT Sample",
            cern_dc,
            cern_repo,
            str(CERN_DATASETS),
            cern_ok,
            "O2 AO2D tracks when present; legacy esdTree multiplicity fallback.",
        ),
    ]


def _entries_planck() -> list[ProvenanceEntry]:
    return [
        ProvenanceEntry(
            "PLANCK_CMB", "Fetch / Analyze CMB", "real_remote",
            "Planck Legacy Archive", str(PROJECT_ROOT / "datasets" / "planck"),
            False, "Requires a Planck fetch first.",
        ),
    ]


def provenance_entries(
    *,
    scope: Literal["all", "desi", "sparc"] = "all",
) -> list[ProvenanceEntry]:
    avail = dataset_availability()
    entries: list[ProvenanceEntry] = []
    if scope in ("all", "desi"):
        entries.extend(_entries_desi(avail))
    if scope in ("all", "sparc"):
        entries.extend(_entries_sparc(avail))
    if scope == "all":
        entries.extend(_entries_frb())
        entries.extend(_entries_gravitic(avail))
        entries.extend(_entries_cern(avail))
        entries.extend(_entries_planck())
    return entries


def data_provenance_report(
    *,
    scope: Literal["all", "desi", "sparc"] = "all",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Build a JSON-serializable provenance report for research-tool menu actions.

    Parameters
    ----------
    scope:
        Limit to one module family or ``all``.
    verbose:
        Print a human-readable table to stdout.
    """
    avail = dataset_availability()
    entries = provenance_entries(scope=scope)
    by_class: dict[str, int] = {}
    for entry in entries:
        by_class[entry.data_class] = by_class.get(entry.data_class, 0) + 1

    report: dict[str, Any] = {
        "scope": scope,
        "dataset_availability": avail,
        "summary_by_data_class": by_class,
        "entries": [asdict(e) for e in entries],
    }

    if verbose:
        print("\n=== Data Provenance Report ===")
        print(f"Scope: {scope}")
        print("\nLocal dataset cache:")
        print(f"  DESI DR2 Cobaya tables : {avail['desi_dr2_cobaya_tables']} ({avail['desi_dr2_path']})")
        print(f"  Legacy BAO tables      : {avail['legacy_bao_tables']}")
        print(f"  SPARC galaxy CSVs      : {avail['sparc_galaxy_csv_count']}")
        print(f"  Pantheon+ available    : {avail['pantheon_plus_available']}")
        print("\nActions:")
        for entry in entries:
            status = "OK" if entry.data_available else "MISSING CACHE"
            print(
                f"  [{entry.module}] {entry.action}\n"
                f"    class={entry.data_class} | {status} | {entry.notes}"
            )
        print("=== End Provenance Report ===\n")

    return report


if __name__ == "__main__":
    data_provenance_report(verbose=True)