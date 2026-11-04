"""Glueball ledger. 0++ is 2*m0*e (LIVE true prediction). 0-+ stays PACKED. No other J^PC packs."""

from __future__ import annotations

import math
from typing import Any

from .constants import CORPUS_DATE, DELTA_TEP_MEV

PACK_0PP = 2.0 * math.e
PACK_0MP_NOMINATED = 7.5

LATTICE_0PP_MEV = 1710.0
LATTICE_0PP_UNC_MEV = 50.0
BESIII_X2370_MEV = 2359.0
BESIII_X2370_UNC = "+13/-14"


def m_0pp_th(m0: float = DELTA_TEP_MEV) -> float:
    """True prediction: 2 * m0 * e. Coefficient is frozen. m0 is an insertion."""
    return 2.0 * float(m0) * math.e


def packed_0mp(m0: float = DELTA_TEP_MEV) -> float:
    return PACK_0MP_NOMINATED * float(m0)


def live_0pp_residual(
    lattice_mev: float = LATTICE_0PP_MEV,
    lattice_unc_mev: float = LATTICE_0PP_UNC_MEV,
    m0: float = DELTA_TEP_MEV,
) -> dict[str, Any]:
    m_th = m_0pp_th(m0)
    delta = float(lattice_mev) - m_th
    z = delta / float(lattice_unc_mev) if lattice_unc_mev else float("nan")
    return {
        "m_th": m_th,
        "delta_MeV": delta,
        "pull_z": z,
        "lattice_MeV": float(lattice_mev),
        "lattice_unc_MeV": float(lattice_unc_mev),
    }


def ledger_snapshot(m0: float = DELTA_TEP_MEV) -> dict[str, Any]:
    m0 = float(m0)
    resid = live_0pp_residual(m0=m0)
    return {
        "corpus_date": CORPUS_DATE,
        "m0_MeV": m0,
        "m0_note": "insertion (neutron dressing share); not fitted to glueball data",
        "e_status": "LIVE",
        "even_mass_formula_status": "LIVE",
        "theory_0pp_MeV": resid["m_th"],
        "theory_0pp_MeV_is_prediction": True,
        "formula_0pp": "2 * m0 * e",
        "delta_MeV": resid["delta_MeV"],
        "pull_z": resid["pull_z"],
        "packed_0mp_MeV": packed_0mp(m0),
        "packed_0mp_formula": "15 m0 / 2",
        "packed_0mp_status": "PACKED",
        "lattice_0pp_MeV": LATTICE_0PP_MEV,
        "lattice_0pp_unc_MeV": LATTICE_0PP_UNC_MEV,
        "besiii_x2370_combo_MeV": BESIII_X2370_MEV,
        "besiii_x2370_combo_unc": BESIII_X2370_UNC,
        "besiii_contact": "SCALE_CONTACT",
        "status": "TRUE_PREDICTION",
        "note": "0++ is 2*m0*e (LIVE). 0-+ 15/2 remains PACKED / SCALE_CONTACT. No other J^PC packs.",
    }
