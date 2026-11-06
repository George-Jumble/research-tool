"""Superblock / Mirron constants (flagship v5, 2026-11-04). No 142857 comb, no KK, no 5D.

Source of truth: *The Superblock Universe* v5 (Zenodo draft 23953533; concept DOI 10.5281/zenodo.22948740),
``Rulings_2026-11-01.md`` (N1-N13) and ``Rulings_2026-11-04.md`` (N13-N30).
"""

from __future__ import annotations

import math

HBAR_C_MEV_FM = 197.3269718

# Mass floor (ruling Q1, 2026-10-31; v5 Table 1): Delta = 313.1 MeV, set by the Mirron's double
# reduced-Compton length 2*hbar*c/Delta. TEP supplies only Delta > 0. The value is a locked geometric
# input, not a fit and not a theorem. The 309.3 MeV neutron-dressing print is EXCLUDED.
DELTA_FLOOR_MEV = 313.1
EXCLUDED_FLOOR_MEV = 309.3  # retired 2026-10-31; kept only so tests can assert it is not live
# Back-compat name used by the live writers. Same object as DELTA_FLOOR_MEV.
DELTA_TEP_MEV = DELTA_FLOOR_MEV
R_M_LOCKED_FM = HBAR_C_MEV_FM / DELTA_TEP_MEV          # Mirron radius r_M = hbar c / Delta = 0.6302 fm
MIRRON_DIAMETER_FM = 2.0 * R_M_LOCKED_FM               # 1.2605 fm
R_M_LOCKED_M = R_M_LOCKED_FM * 1e-15
C_M_S = 299792458.0
TICK_S = 2.0 * R_M_LOCKED_M / C_M_S                    # 4.204e-24 s
HOP_ADJACENT_MEV = 0.5 * math.pi * DELTA_FLOOR_MEV     # 491.8 MeV (theta*Delta, adjacent)
HOP_ANTIPODAL_MEV = math.pi * DELTA_FLOOR_MEV          # 983.6 MeV (antipodal)
ELL_P_M = 1.616255e-35

# Observed sound horizon (Planck 2018 TT,TE,EE+lowE+lensing, base LCDM).
RD_OBSERVED_MPC = 147.09
RD_OBSERVED_ERR_MPC = 0.26
# RETIRED 2026-11-01 (ruling N6): gamma = 10.74 returned the observed r_d by construction.
# Kept for reference only; no live engine uses it.
DELTA_N_PLASMA = 10.74
DELTA_N_PLASMA_STATUS = "RETIRED_2026-11-01_N6"
LATE_UNIVERSE_DELTA_N = 0.33
GAMMA_HEX = 1.405
Z0_OHM = 376.730313461
CORPUS_DATE = "2026-11-04"
FRAMEWORK_VERSION = "Superblock Universe v5 (Zenodo draft 23953533, 2026-11-04)"

# Scheme A quark ladder (kinematic, ruling N3; same set of factors from every domain, N22).
# gamma_q = m_q / m_u; light in domain q runs at c/gamma_q seen from any other domain's "Prime".
# It sets light speed and clock rate, NOT gravitating weight (N15).
V_ORB = (
    {"domain": 0, "flavour": "u", "role": "Prime", "beta": 0.0, "gamma": 1.0},
    {"domain": 1, "flavour": "d", "role": "near", "beta": 0.8881, "gamma": 2.176},
    {"domain": 2, "flavour": "s", "role": "mid", "beta": 0.999733, "gamma": 43.29},
    {"domain": 3, "flavour": "c", "role": "fast", "beta": 1 - 1.44e-6, "gamma": 589.4},
    {"domain": 4, "flavour": "b", "role": "fast", "beta": 1 - 1.333e-7, "gamma": 1936.6},
    {"domain": 5, "flavour": "t", "role": "ultrarel", "beta": 1 - 7.83e-11, "gamma": 7.989e4},
)
SCHEME_A_GAMMA = {row["flavour"]: float(row["gamma"]) for row in V_ORB}

# B_lin co-orbit map (F7 freeze of E8 exponents {11,13,17,19,23,29}): beta_i = (m_i - 11)/30.
# Ruling N19: the co-orbit gamma is a fixed T_priv weighting, identical from every viewpoint,
# NOT a composable velocity. Ansatz A (gamma^1, R5/T4): dark/ordinary = sum_{i=2..6} gamma_i.
E8_EXPONENTS = (11, 13, 17, 19, 23, 29)
B_LIN_BETA = tuple((m - 11) / 30.0 for m in E8_EXPONENTS)
B_LIN_GAMMA = tuple(1.0 / math.sqrt(1.0 - b * b) for b in B_LIN_BETA)
DARK_TO_ORDINARY_LOCK = 5.4015  # printed lock (v5 Section 18 item 6)

# Neutron module (ruling N21): BR_0 = 1/(gamma_d gamma_s), a local mass ratio shared by every domain.
TAU_BETA_S = 888.1
TAU_BETA_ERR_S = 2.0
