"""Prime's sound horizon under v5: standard early-universe physics, no special factor.

v5 Section 14 / Rulings_2026-11-04: only baryons and massive persistent particles carry L_res (N11),
so hidden-domain radiation adds nothing to Prime's expansion or sound speed. The hidden domains add
only cold matter (dark/ordinary = 5.4015) once they are thermally >= ~50x out of step (N24-N30).
Prime's r_d is therefore the standard drag-epoch sound horizon with omega_c = 5.4015 * omega_b.
The CLASS v3.3 run in ``hidden_BAO_test.md`` gives 146.88 Mpc; the Aubourg et al. (2015, PRD 92,
123516, eq. 16) fit used here reproduces it to ~0.03 Mpc without running a Boltzmann code.

RETIRED (ruling N6, 2026-11-01): r_d = 10.74 * r_d,obs / gamma, which returned r_d,obs by construction.
"""

from __future__ import annotations

import math
from typing import Any

from .constants import (
    DARK_TO_ORDINARY_LOCK,
    DELTA_N_PLASMA_STATUS,
    RD_OBSERVED_ERR_MPC,
    RD_OBSERVED_MPC,
)

# Planck 2018 base-LCDM physical densities (TT,TE,EE+lowE+lensing).
OMEGA_B_H2 = 0.02237
OMEGA_C_H2_PLANCK = 0.1200
OMEGA_NU_H2 = 0.000645  # one massive neutrino, sum m_nu = 0.06 eV
CLASS_RD_V5_MPC = 146.88  # hidden_BAO_test.md, cold 5.4015 case
SOUND_HORIZON_STATUS = "RESOLVED_IN_PRINCIPLE_PENDING_AUTHOR_CONFIRMATION"


def rd_aubourg2015(omega_b_h2: float, omega_c_h2: float, omega_nu_h2: float = OMEGA_NU_H2) -> float:
    """Drag-epoch sound horizon in Mpc (Aubourg et al. 2015 eq. 16; 0.02% vs CAMB in LCDM, N_eff=3.046)."""
    wb = float(omega_b_h2)
    wcb = wb + float(omega_c_h2)
    return 55.154 * math.exp(-72.3 * (float(omega_nu_h2) + 0.0006) ** 2) / (wcb ** 0.25351 * wb ** 0.12807)


def sound_horizon(
    dark_to_ordinary: float = DARK_TO_ORDINARY_LOCK,
    omega_b_h2: float = OMEGA_B_H2,
) -> dict[str, Any]:
    """Prime's standard r_d with omega_c = (dark/ordinary) * omega_b. No gamma, no 10.74, no n_hier."""
    ratio = float(dark_to_ordinary)
    omega_c = ratio * float(omega_b_h2)
    rd = rd_aubourg2015(omega_b_h2, omega_c)
    residual = rd - RD_OBSERVED_MPC
    return {
        "method": "standard_drag_epoch_aubourg2015",
        "dark_to_ordinary": ratio,
        "omega_b_h2": float(omega_b_h2),
        "omega_c_h2": omega_c,
        "rd_predicted_mpc": rd,
        "rd_class_v5_mpc": CLASS_RD_V5_MPC,
        "rd_observed_mpc": RD_OBSERVED_MPC,
        "rd_observed_err_mpc": RD_OBSERVED_ERR_MPC,
        "residual_mpc": residual,
        "residual_sigma": residual / RD_OBSERVED_ERR_MPC,
        "special_factor": None,
        "delta_n_plasma": DELTA_N_PLASMA_STATUS,
        "status": SOUND_HORIZON_STATUS,
        "note": (
            "standard sound horizon; hidden radiation carries no L_res; hidden baryons cold "
            "(>= 50x out of thermal step). gamma = 10.74 retired (N6)."
        ),
    }

