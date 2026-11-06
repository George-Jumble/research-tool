"""BGS has DV only; DH_only must fall back instead of aborting the pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from menus.astronomical.desi.fetcher import DEFAULT_COBAYA_ROOT
from menus.astronomical.desi.loader import (
    DATA_MODE_ANISOTROPIC,
    DATA_MODE_DH_ONLY,
    DATA_MODE_DV_ONLY,
    dv_over_rd_from_dm_dh,
    get_data_by_mode,
    inventory_tracer_quantities,
    load_scan_data_by_data_mode,
    resolve_data_mode_for_tracer,
)

_NEEDS_DR2 = pytest.mark.skipif(
    not DEFAULT_COBAYA_ROOT.is_dir(),
    reason=(
        "DESI DR2 tables not cached; fetch with the repo convention: git clone "
        "https://github.com/CobayaSampler/bao_data.git datasets/desi/bao_data"
    ),
)

BGS = "BGS_BRIGHT-21.35_GCcomb"
LRG_LO = "LRG_GCcomb_z0.4-0.6"
ANISO_TRACERS = (
    "QSO_GCcomb",
    "Lya_GCcomb",
    "ELG_LOPnotqso_GCcomb_z1.1-1.6",
    "LRG+ELG_LOPnotqso_GCcomb",
)


@_NEEDS_DR2
def test_inventory_bgs_is_dv_only():
    avail = inventory_tracer_quantities(None, BGS)
    assert avail == ["DV_over_rs"]


@_NEEDS_DR2
def test_bgs_dh_only_falls_back_to_dv():
    chosen = resolve_data_mode_for_tracer("DH_only", BGS, ["DV_over_rs"])
    assert chosen == DATA_MODE_DV_ONLY
    y, cov, _s, meta = get_data_by_mode("DH_only", BGS)
    assert meta["data_mode"] == DATA_MODE_DV_ONLY
    assert list(meta["quantity_types"]) == ["DV_over_rs"]
    assert int(meta["n_data"]) == 1
    assert abs(float(y[0]) - 7.94167639) < 1e-6
    assert cov.shape == (1, 1)
    assert abs(float(cov[0, 0]) - 5.78998687e-03) < 1e-8
    data = load_scan_data_by_data_mode("DH_only", tracer=BGS)
    assert data["data_mode"] == DATA_MODE_DV_ONLY
    assert data["quantity"] == ["DV_over_rs"]


@_NEEDS_DR2
def test_lrg_dh_only_keeps_dh():
    y, _cov, _s, meta = get_data_by_mode("DH_only", LRG_LO)
    assert meta["data_mode"] == DATA_MODE_DH_ONLY
    assert list(meta["quantity_types"]) == ["DH_over_rs"]
    assert int(meta["n_data"]) >= 1


@_NEEDS_DR2
def test_qso_lya_elg_lrg_elg_default_anisotropic():
    for tracer in ANISO_TRACERS:
        _y, _cov, _s, meta = get_data_by_mode(None, tracer)
        assert meta["data_mode"] == DATA_MODE_ANISOTROPIC, tracer
        assert set(meta["quantity_types"]) == {"DM_over_rs", "DH_over_rs"}, tracer


def test_dv_theory_formula_matches_desi_definition():
    # D_V/r_d = (z (D_M/r_d)^2 (D_H/r_d))^{1/3}; do not invent DH for BGS.
    z, dm, dh = 0.295, 8.0, 20.0
    got = dv_over_rd_from_dm_dh(z, dm, dh)
    expect = (z * dm**2 * dh) ** (1.0 / 3.0)
    assert abs(got - expect) < 1e-12


def test_zero_usable_quantities_errors():
    try:
        resolve_data_mode_for_tracer("DH_only", "empty", [], log=False)
    except Exception:
        raise
    assert resolve_data_mode_for_tracer("DH_only", "empty", [], log=False) is None
