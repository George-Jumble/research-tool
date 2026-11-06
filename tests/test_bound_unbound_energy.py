"""Bound vs unbound energy partition — continuum must not stay in E_b."""

from __future__ import annotations

import numpy as np

from sb_shared.bound_unbound_energy import (
    partition_energy,
    partition_periodogram_totals,
)


def test_aggregator_without_tests_is_not_bound():
    e_tot = np.array([1.0, 2.0, 3.0])
    split = partition_energy(e_tot, from_total_aggregator=True, ledger="content")
    assert split["sum_E_b"] == 0.0
    assert split["n_bound"] == 0
    assert split["n_ambiguous"] == 3
    assert split["sum_E_u"] == float(np.sum(e_tot))


def test_discrete_peak_above_continuum_is_bound():
    freqs = np.linspace(0.05, 0.5, 200)
    power = np.full_like(freqs, 1.0)
    i = int(np.argmin(np.abs(freqs - 0.22)))
    power[i] = 20.0
    tot = partition_periodogram_totals(freqs, power, f_bound=0.22, s_min=3.0)
    assert tot["E_bound_clean"] > 0.0
    assert tot["E_unbound_clean"] > tot["E_bound_clean"]
    assert tot["bound_flag_clean"] is True


def test_negative_delta_is_not_more_bound():
    e_obs = np.array([5.0, 5.0])
    e_free = np.array([4.0, 3.0])  # ΔE_bind < 0
    split = partition_energy(
        e_obs,
        e_free=e_free,
        e_th=10.0,
        is_discrete_harmonic=True,
        bind_ok=True,
        significance=10.0,
        ledger="deficit",
    )
    assert split["sum_E_b"] == 0.0
    assert np.all(split["delta_e_bind"] < 0)
