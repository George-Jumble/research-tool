"""Template C Mirron kernel: shape, deterministic α_s running, LHCb χ²_SB."""

import math

import numpy as np

from kernels import mirron_c as mc


def test_shape_is_unit_at_zero_and_half_at_m2():
    m2 = mc.M_MIR_GEV**2
    s = mc.s_c([0.0, m2, 100.0], 1.0)
    assert math.isclose(s[0], 1.0)
    assert math.isclose(s[1], 0.5)
    assert np.all(np.diff(s) < 0) and s[2] > 0
    assert math.isclose(mc.M_MIR_GEV, 0.3131)


def test_alpha_s_running_deterministic_and_sane():
    a = mc.alpha_s(1.777, loops=4)["alpha_s"]
    assert a == mc.alpha_s(1.777, loops=4)["alpha_s"]
    assert 0.30 < a < 0.34  # α_s(m_τ) ≈ 0.32
    assert 0.22 < mc.alpha_s(mc.M_B_GEV, loops=4)["alpha_s"] < 0.23


def test_alpha_s_at_mirron_scale_reports_instability():
    four = mc.alpha_s(loops=4)
    one = mc.alpha_s(loops=1)
    assert not four["stable"] and four["status"] == "landau_pole"
    assert one["stable"] and 0.8 < one["alpha_s"] < 1.0


def test_lhcb_template_c_produces_chi2_sb(tmp_path):
    from menus.particle.lhcb.flavor import DEFAULT_DATA_PATH, load_lhcb_data, residual_arrays

    data, _ = load_lhcb_data(str(DEFAULT_DATA_PATH))
    arr = residual_arrays(data)
    D, B, sig, q2 = arr["D"], arr["B"], arr["sigma"], arr["q2_gev2"]
    from superblock_residual import score_against_template

    out = score_against_template(D - B, template="C", sigma=sig, q2=q2)
    assert out["chi2_SB"] is not None and math.isfinite(out["chi2_SB"])
    assert out["N"] == len(D)
    assert out["verdict"] in {"SB_better", "baseline_better"}
    assert out["A"] == mc.alpha_s(loops=out["A_loops_used"])["alpha_s"]
