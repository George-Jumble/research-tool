"""Glueball ledger (v5). The 0++ pack 2*Delta*e is RETIRED as a live comparison.

v5 Section 9: "The floor is not a particle pole and not a glueball; the 313 MeV glueball is retired.
The Yang-Mills mass gap is not derived." Retired from v1: "the light scalar glueball near 2*Delta as a
live comparison (not re-confirmed)". Framework model: "The 313 MeV glueball and the 1.5-1.7 GeV window
are retired." 2*Delta*e = 1702 MeV sits in that window, so the old TRUE_PREDICTION ledger is withdrawn.
The numbers are still computed so reports can show what was retired. 0-+ 15/2 stays PACKED.
"""

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
RETIRED_WINDOW_MEV = (1500.0, 1700.0)
GLUEBALL_STATUS = "RETIRED"


def m_0pp_th(m0: float = DELTA_TEP_MEV) -> float:
    """Retired pack 2 * Delta * e (record only; not a live prediction under v5)."""
    return 2.0 * float(m0) * math.e


def packed_0mp(m0: float = DELTA_TEP_MEV) -> float:
    return PACK_0MP_NOMINATED * float(m0)


def live_0pp_residual(
    lattice_mev: float = LATTICE_0PP_MEV,
    lattice_unc_mev: float = LATTICE_0PP_UNC_MEV,
    m0: float = DELTA_TEP_MEV,
) -> dict[str, Any]:
    """Residual of the retired pack against lattice. Diagnostic only; never a ledger PASS."""
    m_th = m_0pp_th(m0)
    delta = float(lattice_mev) - m_th
    z = delta / float(lattice_unc_mev) if lattice_unc_mev else float("nan")
    return {
        "m_th": m_th,
        "delta_MeV": delta,
        "pull_z": z,
        "lattice_MeV": float(lattice_mev),
        "lattice_unc_MeV": float(lattice_unc_mev),
        "diagnostic_only": True,
    }


def ledger_snapshot(m0: float = DELTA_TEP_MEV) -> dict[str, Any]:
    m0 = float(m0)
    resid = live_0pp_residual(m0=m0)
    return {
        "corpus_date": CORPUS_DATE,
        "m0_MeV": m0,
        "m0_note": "Delta = 313.1 MeV locked floor (Mirron double reduced-Compton length); not a glueball",
        "e_status": GLUEBALL_STATUS,
        "even_mass_formula_status": GLUEBALL_STATUS,
        "theory_0pp_MeV": resid["m_th"],
        "theory_0pp_MeV_is_prediction": False,
        "formula_0pp": "2 * m0 * e (retired)",
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
        "status": GLUEBALL_STATUS,
        "yang_mills_gap_derived": False,
        "note": "v5: floor is not a glueball; 1.5-1.7 GeV window retired; Yang-Mills gap not derived.",
    }
