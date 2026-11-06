"""Superblock flagship v5 (Zenodo draft 23953533, 2026-11-04) confrontation gates.

Each test reads its reference data through the repo's empirical layer (``menus.empirical_tests.tests``
REPOS keys -> curated DataFrame -> CSV cache) and scores it with a ledger writer, exactly like the
light-quark / neutron / glueball rows. Framework-side numbers come from the v5 test notes in
``/workspace/forks/open_questions/`` (quoted in the ``superblock_v5_predictions`` table); observational
anchors carry their published references. Open questions are ``xfail(strict=True)`` with the reason.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from menus.empirical_tests import tests as empirical
from sb_engine import constants as C
from sb_engine.dark_matter import (
    dark_to_ordinary,
    viewpoint_ratios_composed,
    viewpoint_ratios_tpriv,
)
from sb_engine.early_universe import (
    D_LINK_HALF_BOUNCE,
    X_PASS,
    hidden_drag_redshift,
    n_eff_prime,
    relay_structures,
)

V5_KEYS = (
    "planck2018_cosmology",
    "superblock_v5_predictions",
    "hidden_bao_class_grid",
    "relay_class_configs",
)


# ----------------------------------------------------------------------------- data layer
@pytest.mark.parametrize("key", V5_KEYS + ("bbn_abundances", "neutron_lifetime", "pdg_light_quarks"))
def test_v5_sources_use_repo_curated_cache(key, tmp_path):
    """New sources follow the REPOS contract and materialize to CSV like the existing curated tables."""
    repo = empirical.REPOS[key]
    assert {"description", "url", "local_csv", "type", "parser"} <= set(repo)
    assert repo["type"] in ("curated", "pdg_or_curated")
    frame = empirical.load_or_download(key)
    assert frame is not None and len(frame) > 0
    assert {"observable", "value", "unc"} <= set(frame.columns)
    parsed = getattr(empirical, repo["parser"])(frame)
    assert len(parsed) == len(frame)
    dest = tmp_path / f"{key}.csv"
    assert empirical.materialize_curated_cache(key, dest)
    back = pd.read_csv(dest)
    assert list(back["observable"]) == list(frame["observable"])


def test_v5_menu_action_auto_fetches_v5_sources():
    from sb_shared.empirical_action_targets import empirical_target_ids

    ids = empirical_target_ids("EMPIRICAL_TESTS", "Superblock v5 Confrontations")
    for key in V5_KEYS:
        assert f"empirical:{key}" in ids


def test_bbn_and_neutron_anchors_are_the_v5_references():
    bbn = empirical.curated_bbn_abundances().set_index("observable")
    assert bbn.loc["Y_p_mass_He4", "value"] == 0.2453 and bbn.loc["Y_p_mass_He4", "unc"] == 0.0034
    assert bbn.loc["D_H", "value"] == 2.527e-5 and bbn.loc["D_H", "unc"] == 0.030e-5
    assert "Aver" in bbn.loc["Y_p_mass_He4", "reference"]
    assert "Cooke" in bbn.loc["D_H", "reference"]
    n = empirical.empirical_neutron_dataframe().set_index("observable")
    assert n.loc["magnetic_bottle", "value"] == 877.75 and n.loc["magnetic_bottle", "unc"] == 0.36
    assert n.loc["beam_yue2013", "value"] == 887.7


# ----------------------------------------------------------------------------- 5.4015
def test_dark_to_ordinary_tpriv_weighting_vs_planck():
    """R5/T4, N17, N19: sum of B_lin gammas = 5.4015 vs Planck 0.1200/0.02237 = 5.364 +- 0.065 (+0.58 sigma)."""
    out = empirical.test_dark_to_ordinary(empirical.load_or_download("planck2018_cosmology"))
    assert abs(out["theory_ratio"] - 5.401514) < 5e-6
    assert abs(out["theory_ratio"] - C.DARK_TO_ORDINARY_LOCK) < 5e-5
    assert abs(out["planck_ratio"] - 5.364) < 1e-3
    assert abs(out["planck_ratio_unc"] - 0.065) < 1e-3
    assert abs(out["pull_sigma"] - 0.58) < 0.01
    assert out["ledger"] == "CONSISTENCY" and out["pass_emitted"] is False
    assert out["gamma1_derived"] is False


def test_strict_local_rest_mass_reading_fails():
    """local_gravity_coorbit_test: equal weights give 5.00 (-5.6 sigma)."""
    out = empirical.test_dark_to_ordinary()
    assert out["local_rest_mass_ratio"] == 5.0
    assert out["local_rest_mass_pull_sigma"] < -5.5


def test_every_domain_is_prime_same_ratio_from_every_viewpoint():
    """N18/N19: the co-orbit gamma is a fixed T_priv weighting -> identical 5.4015 for all six observers."""
    ratios = viewpoint_ratios_tpriv()
    assert len(ratios) == 6
    assert max(ratios.values()) - min(ratios.values()) == 0.0


def test_composable_velocity_reading_is_retired():
    """domain_relativity_test: B_lin as real velocities -> fastest envelope sees >= 5.70 (+5 sigma), spread >= 0.54."""
    out = empirical.test_dark_to_ordinary()
    assert out["viewpoint_max_composed"] >= 5.70
    assert (out["viewpoint_max_composed"] - out["planck_ratio"]) / out["planck_ratio_unc"] > 5.0
    assert out["viewpoint_spread_composed"] >= 0.53
    assert out["viewpoint_spread_tpriv"] == 0.0
    assert min(viewpoint_ratios_composed().values()) < 5.2


def test_quark_ladder_does_not_gravitate():
    """N15: gamma_q drops out of gravity (ladder sum ~ 8e4 would be absurd as a weight)."""
    assert dark_to_ordinary("ladder_gamma1") > 1e4
    assert abs(dark_to_ordinary("kernel_gamma3") - 6.44) < 0.01  # literal kernel, not used


# ----------------------------------------------------------------------------- Scheme A kinematics
def test_scheme_a_kinematic_means_live():
    """N3: <gamma>_harm = 4.041, <gamma>_arith = 1.3744e4, Lambda_tw = 2.00e-10 GeV^-2."""
    g = list(C.SCHEME_A_GAMMA.values())
    harm = len(g) / sum(1.0 / x for x in g)
    arith = sum(g) / len(g)
    lam_tw = (1.1663787e-5 / math.sqrt(2.0)) / (0.5 * sum(g))
    assert abs(harm - 4.041) < 1e-3
    assert abs(arith - 1.3744e4) < 1.0
    assert abs(lam_tw - 2.00e-10) < 0.005e-10
    for q, gam in C.SCHEME_A_GAMMA.items():
        beta = math.sqrt(1.0 - 1.0 / gam**2)
        assert abs((1.0 / gam) ** 2 + beta**2 - 1.0) < 1e-12  # (c/gamma)^2 + (beta c)^2 = c^2


def test_coupling_constants_are_local():
    """N14: pure numbers invariant under 1,000 random frame maps; literal gamma_q*alpha superseded."""
    out = empirical.test_coupling_invariance(n_maps=1000)
    assert out["max_abs_dimensionless_exponent"] < 1e-12
    assert out["literal_alpha_t"] > 500.0          # alpha ~ 583 in the t domain
    # literal alpha_s(MZ)*gamma: non-perturbative (> 1) from s on, beyond 4 pi from c on
    assert out["literal_alpha_s_s"] > 1.0
    from sb_engine.couplings import literal_scaled_couplings

    assert literal_scaled_couplings()["c"]["alpha_s"] > 4.0 * math.pi
    assert out["ledger"] == "NULL_PREDICTION"


# ----------------------------------------------------------------------------- early universe
def test_neff_standard_hidden_radiation_carries_no_lres():
    """N11: N_eff = 3.044 vs Planck 2.99 +- 0.17; the retired gravitating-radiation reading fails (Delta N_eff ~ 37)."""
    out = empirical.test_neff_v5(empirical.load_or_download("planck2018_cosmology"))
    assert out["theory"] == 3.044
    assert abs(out["z"]) < 1.0
    assert abs(out["retired_hidden_radiation_gravitates"] - 3.044 - 37.2) < 0.1
    assert out["retired_z"] > 100.0


def test_bbn_standard_after_n11():
    """BBN_physics_superblock: Y_p 0.2469 (+0.5 sigma vs Aver 2021), D/H 2.51e-5 (-0.1 sigma vs Cooke 2018)."""
    out = empirical.test_bbn_v5(
        empirical.load_or_download("bbn_abundances"),
        empirical.load_or_download("superblock_v5_predictions"),
    )
    assert abs(out["Y_p"]["z"] - 0.5) < 0.1
    assert abs(out["D_H"]["z"] - (-0.1)) < 0.1
    assert out["ledger"] == "CONSISTENCY" and out["pass_emitted"] is False
    assert out["Li7_H"]["status"] == "OPEN"
    assert 2.5 < out["Li7_H"]["ratio_to_spite"] < 3.5


@pytest.mark.xfail(strict=True, reason="open question 6: lithium-7 stays ~3x the Spite plateau; no Superblock mechanism")
def test_lithium7_matches_spite_plateau():
    out = empirical.test_bbn_v5()
    assert out["Li7_H"]["ratio_to_spite"] < 1.5


def test_prime_sound_horizon_is_standard_no_gamma_factor():
    """Hidden-BAO test: r_d = 146.88 Mpc (CLASS) vs Planck 147.09 +- 0.26 (-0.8 sigma); 10.74 retired (N6)."""
    out = empirical.test_sound_horizon_v5(
        empirical.load_or_download("planck2018_cosmology"),
        empirical.load_or_download("superblock_v5_predictions"),
    )
    assert out["rd_class_mpc"] == 146.88
    assert abs(out["rd_fit_mpc"] - 146.88) < 0.1
    assert abs(out["z_class"] - (-0.8)) < 0.05
    assert abs(out["z_fit"]) < 1.5
    assert out["special_factor"] is None
    assert out["status"] == "RESOLVED_IN_PRINCIPLE_PENDING_AUTHOR_CONFIRMATION"


def test_gamma_10_74_sound_horizon_factor_retired():
    import sb_engine
    import sb_engine.sound_horizon as shmod
    from sb_engine.sound_horizon import rd_aubourg2015, sound_horizon

    # gamma = 10.74 retired (N6): calibrate_gamma() removed from the engine entirely.
    assert not hasattr(shmod, "calibrate_gamma") and not hasattr(sb_engine, "calibrate_gamma")
    sh = sound_horizon()
    assert sh["delta_n_plasma"].startswith("RETIRED")
    assert sh["residual_mpc"] != 0.0  # no longer zero by construction
    assert abs(rd_aubourg2015(0.02237, 0.1200) - 147.09) < 0.1  # same fit reproduces Planck LCDM


def test_hidden_bao_in_step_fails_offset_passes():
    """N23-N26: in-step acoustic hidden baryons fail (sigma8 0.157); x <= 0.02 (>= 50x out of step) acts as CDM."""
    out = empirical.test_hidden_bao_v5(empirical.load_or_download("hidden_bao_class_grid"))
    assert out["in_step_fails"] is True
    assert abs(out["in_step_sigma8"] - 0.157) < 1e-6
    assert out["x_pass"] == X_PASS == 0.02
    assert out["rule_consistent_with_class"] is True
    by_x = {r["x"]: r for r in out["grid"]}
    assert by_x[0.02]["class_pass"] and by_x[0.01]["class_pass"]
    assert not by_x[0.05]["class_pass"] and not by_x[0.1]["class_pass"]
    assert hidden_drag_redshift(0.02) > 5e4  # hidden drag before Prime equality


def test_relay_hierarchy_d_must_not_be_adjacent():
    """relay_hierarchy_BAO_test: 9,576 hierarchies; F1/F3 pass exactly the 4,800 with d relayed; F2 never."""
    out = empirical.test_relay_hierarchy_v5(empirical.load_or_download("relay_class_configs"))
    assert out["n_structures"] == 133
    assert out["n_hierarchies"] == 9576
    assert out["unmatched_configs"] == []
    for form in ("F1", "F3"):
        assert out["counts"][form]["pass"] == 4800
        assert out["counts"][form]["d_direct_pass"] == 0
        assert out["counts"][form]["d_relayed_fail"] == 0
    assert out["counts"]["F2"]["pass"] == 0
    assert out["ledger"] == "CONSTRAINT"


def test_d_link_is_three_half_bounce_steps():
    """N30 / Q9: d's link to Prime is (-1, +1, -1), summing to -1; full photonic bounce of six ticks sums to 0."""
    assert D_LINK_HALF_BOUNCE == (-1, +1, -1)
    assert sum(D_LINK_HALF_BOUNCE) == -1
    assert len(D_LINK_HALF_BOUNCE) == 3
    assert sum(D_LINK_HALF_BOUNCE + (+1, -1, +1)) == 0
    assert len(relay_structures()) == 133


def test_mirror_chains_fail_only_with_d_at_minus_t():
    """Prime->T->B and Prime->-T->-B with AP off -T: 48/72 pass; the 24 failures have d at -T."""
    from sb_engine.early_universe import class_config_key, flavour_assignments, link_offsets

    par = {"T": "Prime", "B": "T", "-T": "Prime", "-B": "-T", "AP": "-T"}
    table = empirical.relay_class_configs_dataframe()
    ok_keys = {("cold",)} | {
        (round(float(r.weight_fraction), 4), float(r.x_max))
        for r in table.itertuples()
        if r.config != "cold" and r.value <= 3.0
    }
    passes, fails_d_at_mt = 0, 0
    for fl in flavour_assignments():
        key = class_config_key({q: v["F1"] for q, v in link_offsets(par, fl).items()})
        if key in ok_keys:
            passes += 1
        elif fl["-T"] == "d":
            fails_d_at_mt += 1
    assert passes == 48
    assert fails_d_at_mt == 24


@pytest.mark.xfail(strict=True, reason="open question 4/5: gamma^1 weighting is motivated (gamma M c^2 on T_priv), not derived")
def test_gamma1_weighting_derived():
    assert empirical.test_dark_to_ordinary()["gamma1_derived"] is True


@pytest.mark.xfail(strict=True, reason="open question 3 (v5 item 21): Lyman-alpha check owed; s linked directly cuts P(k=1) by 10-20%")
def test_lyman_alpha_small_scale_power_cleared():
    table = empirical.relay_class_configs_dataframe().set_index("config")
    assert table.loc["s_direct", "pk1_over_cold"] >= 0.9
