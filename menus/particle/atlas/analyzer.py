"""
ATLAS Open Data mini-tree lepton scan → shared 7-fold Tav analysis.

ATLAS outreach ROOT files use the flat ``mini`` TTree (``lep_pt``, ``lep_n``,
``lep_type``) rather than CMS NanoAOD branches. This adapter builds the same
histogram + multiplicity inputs consumed by ``tav_7fold_muon_analysis``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import uproot

from menus.particle.atlas.fetcher import is_atlas_mini_root, resolve_root_path
from menus.particle.cern.tav_superblock_cms_muon_analyzer import (
    tav_7fold_muon_analysis,
)
from tav_shared.artifact_paths import compose_dataset_slug
from tav_shared.chunked_results import should_chunk_tav_results

# PDG IDs used in ATLAS Open Data lep_type
PDG_ELECTRON = 11
PDG_MUON = 13

DEFAULT_TREE = "mini"
DEFAULT_PT_BINS = 20
DEFAULT_PT_MAX_GEV = 200.0


def _resolve_root(file_path: str | Path) -> Path:
    path = Path(file_path).expanduser()
    if path.is_file():
        return path
    resolved = resolve_root_path(str(file_path))
    if resolved is None:
        raise FileNotFoundError(f"ATLAS ROOT not found: {file_path}")
    return resolved


def _to_gev(pt_values: np.ndarray) -> np.ndarray:
    """ATLAS Open Data stores lepton pT in MeV; convert to GeV when needed."""
    arr = np.asarray(pt_values, dtype=float).ravel()
    if arr.size == 0:
        return arr
    # Heuristic: CMS NanoAOD is GeV (~ tens); ATLAS mini is MeV (~ thousands).
    median = float(np.nanmedian(np.abs(arr[np.isfinite(arr)]))) if arr.size else 0.0
    if median > 500.0:
        return arr / 1000.0
    return arr


def _filter_lepton_mask(lep_type_row: Any, lepton: str) -> np.ndarray | None:
    """Return boolean mask for PDG type, or None for all leptons."""
    focus = (lepton or "both").strip().lower()
    if focus in {"", "both", "all", "lep", "lepton"}:
        return None
    types = np.asarray(lep_type_row)
    abs_types = np.abs(types.astype(int))
    if focus in {"muon", "mu", "muons"}:
        return abs_types == PDG_MUON
    if focus in {"electron", "e", "ele", "electrons"}:
        return abs_types == PDG_ELECTRON
    return None


def scan_atlas_mini_chunked(
    file_path: str | Path,
    *,
    tree_name: str = DEFAULT_TREE,
    chunk_size: int = 100_000,
    entry_stop: int | None = 50_000,
    pt_bins: int = DEFAULT_PT_BINS,
    pt_max_gev: float = DEFAULT_PT_MAX_GEV,
    lepton: str = "both",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Iterate ATLAS ``mini`` tree and accumulate pT histogram + multiplicity.

    Parameters
    ----------
    lepton:
        ``muon``, ``electron``, or ``both`` (all ``lep_*`` objects).
    """
    path = _resolve_root(file_path)
    if not is_atlas_mini_root(path):
        raise ValueError(f"Not an ATLAS mini-tree ROOT file: {path}")

    edges = np.linspace(0.0, float(pt_max_gev), int(pt_bins) + 1)
    hist = np.zeros(int(pt_bins), dtype=np.float64)
    n_primary: list[int] = []
    n_events = 0
    n_leptons_binned = 0
    n_total = 0

    branches = ["lep_n", "lep_pt", "lep_type"]
    stop = None if entry_stop is None or int(entry_stop) <= 0 else int(entry_stop)
    step = max(1, int(chunk_size))

    if verbose:
        print(f"[ATLAS mini] Scanning {path.name} tree={tree_name} lepton={lepton}")

    with uproot.open(path) as handle:
        tree = handle[tree_name]
        n_total = int(tree.num_entries)
        if verbose:
            print(f"[ATLAS mini] Entries in file: {n_total:,}")

        for batch in tree.iterate(
            branches,
            step_size=step,
            entry_stop=stop,
            library="np",
        ):
            lep_n = np.asarray(batch["lep_n"]).ravel()
            lep_pt = batch["lep_pt"]
            lep_type = batch["lep_type"]
            n_batch = int(lep_n.size)
            for i in range(n_batch):
                pts = np.asarray(lep_pt[i], dtype=float).ravel()
                types = np.asarray(lep_type[i])
                mask = _filter_lepton_mask(types, lepton)
                if mask is not None:
                    if mask.size != pts.size:
                        # ragged length mismatch — keep all
                        selected = pts
                        count = int(pts.size)
                    else:
                        selected = pts[mask]
                        count = int(np.count_nonzero(mask))
                else:
                    selected = pts
                    count = int(lep_n[i]) if i < lep_n.size else int(pts.size)
                gev = _to_gev(selected)
                if gev.size:
                    hist += np.histogram(gev, bins=edges)[0]
                    n_leptons_binned += int(gev.size)
                n_primary.append(max(0, count))
            n_events += n_batch
            if verbose and n_events % (step * 2) == 0:
                print(f"[ATLAS mini] … {n_events:,} events")

    if verbose:
        print(
            f"[ATLAS mini] Finished {n_events:,} events "
            f"({n_leptons_binned:,} leptons binned, focus={lepton})"
        )

    return {
        "file_path": str(path),
        "tree": tree_name,
        "n_entries_in_file": n_total,
        "n_events_processed": n_events,
        "primary_pt_hist": hist,
        "bin_edges": edges,
        "n_primary_per_event": np.asarray(n_primary, dtype=int),
        "lepton_focus": lepton,
        "total_leptons_binned": n_leptons_binned,
    }


def tav_analyze_atlas_leptons_from_file(
    file_path: str | Path,
    *,
    output_dir: str = "",
    chunk_size: int = 100_000,
    entry_stop: int | None = 50_000,
    lepton: str = "both",
    pt_bins: int = DEFAULT_PT_BINS,
    pt_max_gev: float = DEFAULT_PT_MAX_GEV,
    save_plots: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Scan ATLAS mini ROOT and run shared 7-fold Tav lepton analysis."""
    path = _resolve_root(file_path)
    scan = scan_atlas_mini_chunked(
        path,
        chunk_size=chunk_size,
        entry_stop=entry_stop,
        pt_bins=pt_bins,
        pt_max_gev=pt_max_gev,
        lepton=lepton,
        verbose=verbose,
    )
    prefix = (output_dir or path.stem).strip() or path.stem
    focus = str(scan.get("lepton_focus") or lepton)
    dataset_slug = compose_dataset_slug(prefix, path.stem, f"atlas_{focus}")
    n_events = int(scan["n_events_processed"])
    auto_chunk = should_chunk_tav_results(n_events=n_events)

    tav = tav_7fold_muon_analysis(
        muon_pt_hist=scan["primary_pt_hist"],
        bin_edges=scan["bin_edges"],
        n_muon_per_event=scan["n_primary_per_event"],
        photon_pt_hist=None,
        output_dir=prefix,
        save_plots=save_plots,
        verbose=verbose,
        dataset_slug=dataset_slug,
        n_events_processed=n_events,
        chunk_results=auto_chunk,
    )
    tav = dict(tav)
    tav.update(
        {
            "action": f"ATLAS Open Data Lepton Analysis ({focus})",
            "experiment": "ATLAS",
            "format": "atlas_mini",
            "file_path": str(path),
            "lepton": focus,
            "scan_summary": {
                "n_events_processed": n_events,
                "total_leptons_binned": scan.get("total_leptons_binned"),
                "bin_edges": scan["bin_edges"].tolist(),
            },
            "n_events_processed": n_events,
            "n_events": n_events,
        }
    )
    return tav
