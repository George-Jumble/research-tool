import numpy as np
import pytest

import sb_shared.mcmc_engine as me


def _synthetic_burkert(x, rho_0=12.0, r_0=2.5):
    xr = x / r_0
    return rho_0 / ((1 + xr) * (1 + xr ** 2))


def test_all_models_return_finite_aic_on_clean_data():
    x = np.linspace(0.5, 20.0, 80)
    y = _synthetic_burkert(x) + np.random.default_rng(0).normal(0, 0.05, size=x.size)

    for model_type in me.MODEL_ORDER:
        rss, aic = me.run_mcmc_sweep(x, y, model_type=model_type)
        assert np.isfinite(rss), f"{model_type} RSS not finite"
        assert np.isfinite(aic), f"{model_type} AIC not finite"


def test_model_comparison_ranks_burkert_near_top_on_burkert_data():
    x = np.linspace(0.5, 20.0, 100)
    y = _synthetic_burkert(x) + np.random.default_rng(1).normal(0, 0.02, size=x.size)

    results = me.run_model_comparison(x, y)
    top_models = {row["model_type"] for row in results[:2]}
    assert "Burkert" in top_models


def test_catalog_contains_requested_profiles():
    expected = {"NFW", "Einasto", "Burkert", "Hernquist", "pISO", "Mirron"}
    assert expected == set(me.MODEL_CATALOG.keys())
    assert "Mirron" not in me.MODEL_ORDER


def test_format_comparison_report_includes_models():
    x = np.linspace(1.0, 10.0, 40)
    y = me.nfw_model(x, 8.0, 3.0)
    report = me.format_comparison_report(me.run_model_comparison(x, y))
    for label in ("NFW", "Einasto", "Burkert", "Hernquist", "pISO"):
        assert label in report
    assert "Superblock" not in report

