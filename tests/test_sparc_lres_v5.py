"""v5 Section 6 SPARC confrontation with the hierarchical L_res kernel, on the real SPARC table.

Data come through the repo's own SPARC fetcher (``fetch_mass_models_table``: download once, cache under
``datasets/sparc/MassModels_Lelli2016c.mrt``). Skips with a reason when the table is neither cached nor
reachable. v5 quotes 171 galaxies with >= 5 points and L_res beating pure baryons in 123/171; the exact
mean reduced chi^2 values (164.7 / 78.2) depend on an unrecorded error/grid recipe, so this test gates
the sample size and the qualitative result, not those two means.
"""

from __future__ import annotations

import numpy as np
import pytest

from kernels.l_res_rl import fit_galaxy, v_baryon


@pytest.fixture(scope="module")
def sparc_table():
    from menus.astronomical.sparc.fetcher import fetch_mass_models_table

    try:
        return fetch_mass_models_table()
    except Exception as exc:  # network down and no cache
        pytest.skip(f"SPARC MassModels table not cached and not reachable: {exc}")


def test_sparc_lres_beats_baryons_in_majority(sparc_table):
    fits = []
    for _gid, g in sparc_table.groupby("ID"):
        if len(g) < 5:
            continue
        g = g.sort_values("R")
        vb = v_baryon(g["Vgas"].fillna(0), g["Vdisk"].fillna(0), g["Vbul"].fillna(0))
        fits.append(fit_galaxy(g["R"].values, g["Vobs"].values, vb, g["e_Vobs"].values))
    assert len(fits) == 171
    wins = sum(f["chi2_SB"] < f["chi2_baryon"] for f in fits)
    assert wins >= 123
    mean_sb = float(np.mean([f["reduced_chi2_SB"] for f in fits]))
    mean_bar = float(np.mean([f["reduced_chi2_baryon"] for f in fits]))
    assert mean_sb < 0.5 * mean_bar
    assert all(f["eta"] == 0.0 and f["mass_to_light"] == 1.0 for f in fits)
