"""Live Superblock / Mirron corpus gates for the test suite (flagship v5, 2026-11-04).

Updated from the 2026-09-22 gates: the floor is Delta = 313.1 MeV (309.3 excluded, Q1 2026-10-31);
the neutron module is LIVE as a local mass ratio (N21); the 0++ pack 2*m0*e is RETIRED (v5 Section 9);
Template B uses the hierarchical L_res V_RL kernel (v5 Section 6), not G_avg (retired T1 / N15).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from menus.empirical_tests.tests import (
    DELTA_TEP_MEV,
    test_light_quarks as light_quarks_row,
    test_neutron_lifetime as neutron_lifetime_row,
    test_yang_mills_glueball as glueball_row,
    theory_m0_floor,
    theory_target_br,
)
from sb_engine.constants import (
    DELTA_FLOOR_MEV,
    EXCLUDED_FLOOR_MEV,
    HOP_ADJACENT_MEV,
    HOP_ANTIPODAL_MEV,
    MIRRON_DIAMETER_FM,
    R_M_LOCKED_FM,
    TICK_S,
)
from superblock_residual import score_against_template
from sb_shared.mcmc_engine import MODEL_ORDER

ROOT = Path(__file__).resolve().parents[1]


def test_floor_locked_at_313_1():
    """v5 Table 1: Delta = 313.1 MeV; r_M = hbar c / Delta; 309.3 MeV is excluded."""
    assert theory_m0_floor() == 313.1
    assert DELTA_TEP_MEV == DELTA_FLOOR_MEV == 313.1
    assert DELTA_TEP_MEV != EXCLUDED_FLOOR_MEV
    assert abs(MIRRON_DIAMETER_FM - 1.2605) < 5e-4
    assert abs(R_M_LOCKED_FM - 0.6302) < 5e-4
    assert abs(TICK_S - 4.204e-24) < 5e-27
    assert abs(HOP_ADJACENT_MEV - 491.8) < 0.05
    assert abs(HOP_ANTIPODAL_MEV - 983.6) < 0.05


def test_comparison_models_are_conventional_halos():
    assert MODEL_ORDER == ["NFW", "Einasto", "Burkert", "Hernquist", "pISO"]


def test_neutron_module_live_local_mass_ratio():
    """N21: tau_bottle = tau_beta (1 - 1/(gamma_d gamma_s)) = 878.7 +- 2.1 s vs UCNtau 877.75 +- 0.36 s."""
    out = neutron_lifetime_row(pd.DataFrame(), plot=False)
    assert out["status"] == "LIVE"
    assert out["ledger"] == "CONSISTENCY"
    assert out["reading"] == "local_mass_ratio"
    assert out.get("pass_emitted") is False
    assert abs(out["theory_target_BR"] - theory_target_br()) < 1e-12
    assert abs(out["BR_0"] - out["theory_target_BR"]) / out["BR_0"] < 2e-3  # masses vs locked ladder
    assert abs(out["tau_bottle_s"] - 878.7) < 0.1
    assert abs(out["tau_bottle_unc_s"] - 2.1) < 0.1
    assert abs(out["z_bottle"] - 0.4) < 0.1
    assert abs(out["gap_s"] - 9.44) < 0.05
    assert abs(out["gap_unc_s"] - 0.63) < 0.02
    assert abs(out["z_gap"] - (-0.2)) < 0.1
    assert "epsilon_applies_to_jparc" not in out
    assert out["status"] != "HOLD"


def test_glueball_0pp_pack_is_retired():
    """v5 Section 9: the floor is not a glueball; 1.5-1.7 GeV window retired; Yang-Mills gap not derived."""
    frame = pd.DataFrame(
        [{"observable": "Lightest 0++ glueball mass (lattice)", "value": 1710, "unc": 50}]
    )
    out = glueball_row(frame, plot=False)
    assert out["status"] == "RETIRED"
    assert out["e_status"] == "RETIRED"
    assert out["theory_0pp_MeV_is_prediction"] is False
    assert out["yang_mills_gap_derived"] is False
    assert out["pass_emitted"] is False
    assert abs(out["theory_0pp_MeV"] - 2.0 * 313.1 * np.e) < 1e-9
    lo, hi = out["retired_window_MeV"]
    assert lo <= out["theory_0pp_MeV"] <= hi + 5.0  # 1702 MeV: inside the retired window
    assert out["packed_0mp_status"] == "PACKED"
    assert out["status"] != "TRUE_PREDICTION"


def test_light_quarks_consistency_not_pass():
    frame = pd.DataFrame(
        [
            {"observable": "u current mass (MSbar 2 GeV)", "value": 2.16, "unc": 0.07, "source": "PDG 2024"},
            {"observable": "d current mass (MSbar 2 GeV)", "value": 4.70, "unc": 0.07, "source": "PDG 2024"},
        ]
    )
    out = light_quarks_row(frame, plot=False)
    assert out["ledger"] == "CONSISTENCY"
    assert out["status"] == "CONSISTENCY"
    assert out["pass_emitted"] is False
    assert out["m0_MeV_frozen"] == 313.1
    assert out["current_masses"]["ms"]["value"] is None
    # (mp+mn)/6 = 312.97 +- 8 MeV band: the floor sits inside it
    assert out["constituent_floor"]["tension_sigma"] < 0.1


def test_sparc_template_b_uses_hierarchical_lres_kernel():
    """v5 Section 6: V_RL = V_flat sqrt(r/(r+R_core)) in quadrature with V_bar; one R_core per galaxy."""
    from kernels.l_res_rl import v_rl

    r = np.linspace(0.5, 30.0, 25)
    v_bar = 90.0 * np.sqrt(r / (r + 2.0)) * np.exp(-r / 40.0)
    v_flat_true = 150.0
    v_obs = np.sqrt(v_bar**2 + v_rl(r, v_flat_true, 4.0) ** 2)
    sigma = np.full_like(r, 5.0)
    scored = score_against_template(
        None, template="B", v_obs=v_obs, v_baryon=v_bar, sigma=sigma, radius=r
    )
    assert scored["kernel"] == "L_res_V_RL_hierarchical"
    assert scored["retired_kernel_not_evaluated"] == "G_avg_equal_tick_gamma_inv2"
    assert scored["verdict"] == "fit_with_free_core"
    assert scored["inherits_m0"] is False
    assert scored["n_galaxies"] == 1
    assert scored["chi2_SB"] < scored["chi2_Lambda"]
    assert scored["chi2_SB"] / len(r) < 1.0


def test_template_b_without_radii_is_inconclusive_not_g_avg():
    v_bar = np.array([80.0, 100.0, 120.0])
    scored = score_against_template(None, template="B", v_obs=v_bar * 1.2, v_baryon=v_bar)
    assert scored["verdict"] == "inconclusive"
    assert scored["chi2_SB"] is None
    assert scored["kernel"] != "G_avg_equal_tick_gamma_inv2"


def test_live_writers_have_no_struck_strings():
    blobs = [
        (ROOT / "menus" / "empirical_tests" / "tests.py").read_text(encoding="utf-8"),
        (ROOT / "research_tool.py").read_text(encoding="utf-8"),
        (ROOT / "kernels" / "l_res_rl.py").read_text(encoding="utf-8"),
        (ROOT / "superblock_residual.py").read_text(encoding="utf-8"),
    ]
    joined = "\n".join(blobs)
    assert "0.0094" not in joined
    assert "joint fit" not in joined
    assert "numeric kernel is TODO" not in joined
    assert "BBN Abundance Confrontation" not in joined
    assert "309.3" not in joined
    assert "TRUE_PREDICTION" not in joined
    assert "Prime-invisible leak" not in joined
