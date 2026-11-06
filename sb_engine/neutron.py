"""Neutron lifetime module under v5 (ruling N21: BR_0 = 1/(gamma_d gamma_s) is a local mass ratio).

tau_bottle = tau_beta * (1 - BR_0); beam-bottle gap = tau_beam * BR_0.
BR_0 from PDG current masses: gamma_d gamma_s = (m_d/m_u)(m_s/m_u), so BR_0 = m_u^2/(m_d m_s).
Linear error propagation reproduces the Monte Carlo of scheme_A_relative_couplings_test.md
(878.7 +- 2.1 s; gap 9.44 +- 0.63 s) to within its sampling noise.
"""

from __future__ import annotations

import math
from typing import Any

from .constants import SCHEME_A_GAMMA, TAU_BETA_ERR_S, TAU_BETA_S

# Superseded readings (domain_relativity_test.md / scheme_A_relative_couplings_test.md)
EXTRA_COUPLING_TAU_S = 888.0   # n = 2 clock x apparent coupling: +5.0 sigma, fails
DOPPLER_LADDER_TAU_S = 857.0   # reciprocal Doppler-ratio ladder: -7.3 sigma, fails


def br0_from_ladder(gamma_d: float = SCHEME_A_GAMMA["d"], gamma_s: float = SCHEME_A_GAMMA["s"]) -> float:
    return 1.0 / (float(gamma_d) * float(gamma_s))


def br0_from_masses(mu: tuple[float, float], md: tuple[float, float], ms: tuple[float, float]) -> tuple[float, float]:
    """BR_0 = m_u^2/(m_d m_s) with relative errors added in quadrature."""
    br = mu[0] ** 2 / (md[0] * ms[0])
    rel = math.sqrt((2.0 * mu[1] / mu[0]) ** 2 + (md[1] / md[0]) ** 2 + (ms[1] / ms[0]) ** 2)
    return br, br * rel


def neutron_module(
    br0: tuple[float, float],
    tau_beta: tuple[float, float] = (TAU_BETA_S, TAU_BETA_ERR_S),
    tau_beam: float = 887.7,
) -> dict[str, Any]:
    br, sbr = br0
    tau = tau_beta[0] * (1.0 - br)
    stau = math.hypot(tau_beta[1] * (1.0 - br), tau_beta[0] * sbr)
    gap = tau_beam * br
    sgap = tau_beam * sbr
    return {
        "BR_0": br,
        "BR_0_unc": sbr,
        "tau_bottle_s": tau,
        "tau_bottle_unc_s": stau,
        "gap_s": gap,
        "gap_unc_s": sgap,
        "reading": "local_mass_ratio",
    }
