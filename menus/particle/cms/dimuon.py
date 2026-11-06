"""CMS NanoAOD opposite-sign dimuon mass spectrum: data vs Drell-Yan MC expectation.

Reads cached ROOT files under ``datasets/cern/<target>/recid_*/`` (same cache as the CERN fetcher).
Data: ``cms_nanoaod_dimu`` (DoubleMuParked). MC: ``cms_nanoaod_dy`` (DYJetsToLL).
The education files carry no luminosity / cross-section weights, so the MC histogram is normalised
to the data integral inside the fit range (shape comparison only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from menus.particle.cern.fetcher import DATASETS_DIR

DATA_TARGET = "cms_nanoaod_dimu"
MC_TARGET = "cms_nanoaod_dy"
MASS_RANGE_GEV = (60.0, 120.0)
N_BINS = 30
MUON_MASS_GEV = 0.1056583755


def cached_root_files(target: str, root: Path = DATASETS_DIR) -> list[Path]:
    base = Path(root) / target
    return sorted(base.rglob("*.root")) if base.is_dir() else []


def dimuon_masses_from_arrays(pt, eta, phi, charge) -> np.ndarray:
    """Leading two muons, opposite sign; per-event jagged inputs (sequence of arrays)."""
    out = []
    for p, e, f, q in zip(pt, eta, phi, charge):
        if len(p) < 2:
            continue
        idx = np.argsort(np.asarray(p))[::-1][:2]
        p2, e2, f2, q2 = (np.asarray(v, dtype=float)[idx] for v in (p, e, f, q))
        if q2[0] * q2[1] >= 0:
            continue
        px, py, pz = p2 * np.cos(f2), p2 * np.sin(f2), p2 * np.sinh(e2)
        en = np.sqrt(px**2 + py**2 + pz**2 + MUON_MASS_GEV**2)
        m2 = en.sum() ** 2 - px.sum() ** 2 - py.sum() ** 2 - pz.sum() ** 2
        out.append(np.sqrt(max(m2, 0.0)))
    return np.asarray(out, dtype=float)


def read_dimuon_masses(files: list[Path], *, max_events: int | None = 2_000_000) -> np.ndarray:
    import uproot  # optional dependency, same as the CERN analyzers

    masses = []
    left = max_events
    for path in files:
        with uproot.open(path) as handle:
            tree = handle["Events"]
            stop = None if left is None else max(0, min(left, tree.num_entries))
            arrs = tree.arrays(["Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge"], library="np", entry_stop=stop)
            masses.append(dimuon_masses_from_arrays(arrs["Muon_pt"], arrs["Muon_eta"], arrs["Muon_phi"], arrs["Muon_charge"]))
            if left is not None:
                left -= int(stop or 0)
                if left <= 0:
                    break
    return np.concatenate(masses) if masses else np.empty(0)


def histogram_comparison(m_data: np.ndarray, m_mc: np.ndarray, *, mass_range=MASS_RANGE_GEV, n_bins=N_BINS) -> dict:
    edges = np.linspace(mass_range[0], mass_range[1], n_bins + 1)
    d, _ = np.histogram(m_data, bins=edges)
    mc, _ = np.histogram(m_mc, bins=edges)
    if mc.sum() == 0 or d.sum() == 0:
        raise ValueError("empty data or MC histogram in mass range")
    scale = d.sum() / mc.sum()
    keep = (d + mc) > 0
    sigma = np.sqrt(d + scale**2 * mc)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return {"x": centers[keep], "D": d[keep].astype(float), "B": (scale * mc)[keep].astype(float),
            "sigma": sigma[keep], "mc_scale": float(scale), "n_data": int(d.sum()), "n_mc": int(mc.sum())}
