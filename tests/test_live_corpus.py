"""Live Superblock / Mirron corpus gates for the test suite (2026-09-22)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kernels.g_avg import g_avg_over_gn
from menus.empirical_tests.tests import (
    DELTA_TEP_MEV,
    test_light_quarks as light_quarks_row,
    test_neutron_lifetime as neutron_lifetime_row,
    test_yang_mills_glueball as glueball_row,
    theory_m0_floor,
    theory_target_br,
)
from superblock_mode import tau_templates_enabled
from superblock_residual import score_against_template
from tav_shared.mcmc_engine import MODEL_ORDER

ROOT = Path(__file__).resolve().parents[1]


def test_m0_frozen():
    assert theory_m0_floor() == 309.3
    assert DELTA_TEP_MEV == 309.3


def test_tau_templates_off():
    assert tau_templates_enabled() is False


def test_tav_not_a_live_comparison_model():
    assert "Tav" not in MODEL_ORDER


def test_neutron_hold_appearance_not_epsilon():
    out = neutron_lifetime_row(pd.DataFrame(), plot=False)
    assert out["status"] == "HOLD"
    assert out["ledger"] == "HOLD"
    assert out["jparc_class"] == "appearance"
    assert out["epsilon_applies_to_jparc"] is False
    assert out["channel"] == "Prime-invisible leak"
    assert abs(out["theory_target_BR"] - theory_target_br()) < 1e-12
    assert out.get("pass_emitted") is not True


def test_glueball_0pp_is_live_prediction():
    frame = pd.DataFrame(
        [{"observable": "Lightest 0++ glueball mass (lattice)", "value": 1710, "unc": 50}]
    )
    out = glueball_row(frame, plot=False)
    assert out["e_status"] == "LIVE"
    assert out["theory_0pp_MeV_is_prediction"] is True
    assert out["status"] == "TRUE_PREDICTION"
    assert abs(out["theory_0pp_MeV"] - 1681.5291390847653) < 1e-9
    assert abs(out["delta_MeV"] - 28.4708609152347) < 1e-9
    assert abs(out["pull_z"] - 0.569417218304694) < 1e-9
    assert out["packed_0mp_status"] == "PACKED"
    assert out["besiii_contact"] == "SCALE_CONTACT"


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
    assert out["m0_MeV_frozen"] == 309.3
    assert out["current_masses"]["ms"]["value"] is None


def test_sparc_template_b_uses_frozen_g_avg():
    k = g_avg_over_gn()
    assert abs(k - 0.20195517812232902) < 1e-12
    v_bar = np.array([80.0, 100.0, 120.0])
    from kernels.g_avg import v_sb

    scored = score_against_template(
        None,
        template="B",
        v_obs=v_sb(v_bar),
        v_baryon=v_bar,
    )
    assert scored["kernel"] == "G_avg_equal_tick_gamma_inv2"
    assert abs(float(scored["S_k"]) - k) < 1e-12
    assert scored["verdict"] == "scale_only"
    assert scored["inherits_m0"] is False
    assert isinstance(scored["chi2_SB"], float)
    assert isinstance(scored["chi2_Lambda"], float)


def test_live_writers_have_no_tav_engine_or_struck_strings():
    blobs = [
        (ROOT / "menus" / "empirical_tests" / "tests.py").read_text(encoding="utf-8"),
        (ROOT / "research_tool.py").read_text(encoding="utf-8"),
        (ROOT / "kernels" / "g_avg.py").read_text(encoding="utf-8"),
        (ROOT / "superblock_residual.py").read_text(encoding="utf-8"),
    ]
    joined = "\n".join(blobs)
    assert "TAV ENGINE" not in joined
    assert "0.0094" not in joined
    assert "joint fit" not in joined
    assert "numeric kernel is TODO" not in joined
    assert "BBN Abundance Confrontation" not in joined
