"""
DEPRECATED (2026-11-04): toy BBN network, kept for history only. Not a baseline.

Author ruling 2026-11-04: the Prime-past standard-BBN baseline is AlterBBN v2.2
(``bbn_isolated.alterbbn``; gates G1–G5 in ``bbn_isolated/runner.py``; residual loader
``superblock_residual._load_prime_past``). This network hard-wires deuterium production to zero,
has no weak rates, ignores eta and caps helium, so it fails G1–G5 by construction. Calling
``run_bbn`` / ``run_bbn_comparison`` emits a DeprecationWarning. Not deleted.

Minimal standard-BBN network on the radiation-era expansion rate.

Integrates abundances vs u = ln(T_START / T) from T_START → T_FINAL (radiation-era cooling).
Used by Superblock Empirical Tests → BBN integrator status. Only massive particles carry L_res,
so the expansion rate is the standard one (no Superblock lever is added here).
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sb_shared.project_paths import SUPERBLOCK_ROOT

ARTIFACTS_DIR = SUPERBLOCK_ROOT / "artifacts" / "prime_past"

DEPRECATED = True
DEPRECATION_NOTE = (
    "menus/prime_past/bbn_network.py is a deprecated toy network; the Prime-past BBN baseline is "
    "AlterBBN v2.2 (bbn_isolated.alterbbn, gates in bbn_isolated/runner.py)."
)


def _warn_deprecated() -> None:
    warnings.warn(DEPRECATION_NOTE, DeprecationWarning, stacklevel=3)

# Standard BBN / cosmology anchors
ETA_BBN: float = 6.1e-10
T_START_MEV: float = 0.8
T_FINAL_MEV: float = 0.01
M_PL_MEV: float = 1.22e22
G_STAR: float = 10.75
Q_NP_MEV: float = 1.293
B_D_MEV: float = 2.22


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def hubble_rate_MeV(T: float) -> float:
    """Radiation-dominated H(T) in MeV (natural units)."""
    T = max(float(T), 1e-4)
    return float(1.66 * np.sqrt(G_STAR) * (T**2) / M_PL_MEV)


def _y_n_equilibrium(T: float) -> float:
    return 1.0 / (1.0 + np.exp(Q_NP_MEV / max(T, 1e-4)))


def _temp_gate(T: float, T_on: float, *, width: float = 0.02) -> float:
    """Smooth turn-on as temperature drops below T_on (MeV)."""
    return float(0.5 * (1.0 + np.tanh((T_on - T) / width)))


def _normalize_abundances(y: np.ndarray) -> np.ndarray:
    """Keep mass fractions in [0, 1] with baryon budget ≤ 1."""
    Y_n, Y_D, Y_He4, Y_Li7 = [max(float(v), 0.0) for v in y]
    total = Y_n + 2.0 * Y_D + 4.0 * Y_He4 + 7.0 * Y_Li7
    if total > 0.999:
        scale = 0.999 / total
        Y_n *= scale
        Y_D *= scale
        Y_He4 *= scale
        Y_Li7 *= scale
    return np.array([Y_n, Y_D, Y_He4, Y_Li7], dtype=float)


def bbn_rhs(u: float, y: np.ndarray, *, T0: float) -> np.ndarray:
    """
    RHS for state [Y_n, Y_D, Y_He4, Y_Li7] with u ≥ 0 and T = T0 * exp(-u).

    As u increases, temperature cools and nuclear yields build up.
    """
    T = float(T0 * np.exp(-u))
    Y_n, Y_D, Y_He4, Y_Li7 = _normalize_abundances(y)
    Y_p = max(1.0 - Y_n - 2.0 * Y_D - 4.0 * Y_He4 - 7.0 * Y_Li7, 1e-12)

    dY_n = 0.0
    dY_D = 0.0
    dY_He4 = 0.0
    dY_Li7 = 0.0

    gate_he = _temp_gate(T, 0.18, width=0.025)
    if gate_he > 1e-4:
        # Effective 2n + 2p → He4 channel (post-D bottleneck, illustrative)
        rate = gate_he * 0.28 * Y_n * Y_p
        cap = max(0.26 - Y_He4, 0.0)
        take = min(rate, cap * 0.18, Y_n * 0.48, Y_p * 0.48)
        dY_He4 += take
        dY_n -= 2.0 * take

    gate_li = _temp_gate(T, 0.08, width=0.01)
    if gate_li > 1e-4 and Y_He4 > 1e-6:
        dY_Li7 += gate_li * 1.5e-6 * Y_He4

    deriv = np.array([dY_n, dY_D, dY_He4, dY_Li7], dtype=float)
    return np.clip(deriv, -0.08, 0.08)


def run_bbn(*, label: str = "BBN run", verbose: bool = True) -> dict[str, Any]:
    _warn_deprecated()
    """Evolve abundances from T_START to T_FINAL; return final yields."""
    T0 = T_START_MEV
    # Post-weak-freeze-out n/p (~0.16 n at T ≈ 0.8 MeV)
    y = np.array([0.16, 0.0, 0.0, 0.0], dtype=float)

    u_end = float(np.log(T0 / T_FINAL_MEV))
    u_grid = np.linspace(0.0, u_end, 400)
    for idx in range(len(u_grid) - 1):
        du = u_grid[idx + 1] - u_grid[idx]
        y = _normalize_abundances(y + bbn_rhs(u_grid[idx], y, T0=T0) * du)

    Y_n, Y_D, Y_He4, Y_Li7 = _normalize_abundances(y)
    T_final = float(T0 * np.exp(-u_grid[-1]))
    Y_p = max(1.0 - Y_n - 2.0 * Y_D - 4.0 * Y_He4 - 7.0 * Y_Li7, 0.0)
    Y_p_mass = 4.0 * Y_He4

    result = {
        "label": label,
        "T_start_MeV": T0,
        "T_final_MeV": T_final,
        "Y_n": Y_n,
        "Y_p": Y_p,
        "Y_D": Y_D,
        "Y_He4": Y_He4,
        "Y_Li7": Y_Li7,
        "Y_p_mass": Y_p_mass,
        "Li7_H": Y_Li7,
        "Li_H": Y_Li7,
        "D_H": Y_D,
        "integration_success": True,
        "n_steps": int(len(u_grid)),
        "integrator": "explicit_euler_lnT",
    }

    if verbose:
        print(f"\n=== {label} ===")
        print(f"  T: {T0:.3f} → {T_final:.4f} MeV  ({result['n_steps']} steps)")
        print(f"  Y_n = {Y_n:.5f}   Y_p = {Y_p:.5f}")
        print(f"  Y_D = {Y_D:.2e}   Y_He4 = {Y_He4:.5f}")
        print(f"  ^4He mass fraction Y_p ≈ {Y_p_mass:.4f}")
        print(f"  ^7Li/H ≈ {Y_Li7:.2e}")

    return result


def run_bbn_comparison(
    params: dict[str, Any] | None = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the (deprecated) toy network and report integrator status. No yield score."""
    _warn_deprecated()
    _ = params
    std_run = run_bbn(label="standard network", verbose=verbose)

    from menus.empirical_tests.tests import bbn_integrator_failure

    reason = bbn_integrator_failure(std_run)
    comparison: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "standard": std_run,
        "deprecated": DEPRECATION_NOTE,
    }
    if reason:
        comparison["status"] = "FAIL_INTEGRATOR"
        comparison["ledger"] = "FAIL_INTEGRATOR"
        comparison["reason"] = reason
        if verbose:
            print("\n[Superblock Residual Engine] BBN status=FAIL_INTEGRATOR")
            print(f"[Superblock Residual Engine] {reason}")
        return comparison

    comparison["status"] = "UNSCORED"
    comparison["ledger"] = "FAIL_INTEGRATOR"
    comparison["reason"] = "public BBN code not wired; no Superblock lever added"
    return comparison


def save_bbn_report(comparison: dict[str, Any], *, prefix: str = "bbn_network") -> str:
    """Write JSON report under artifacts/prime_past/."""
    ensure_artifacts_dir()
    from sb_shared.artifact_paths import TestSlug, artifact_path, compose_dataset_slug

    path = artifact_path(TestSlug.PRIME_PAST, compose_dataset_slug(prefix), "report", "json")
    path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"[Superblock Residual Engine] BBN report saved: {path}")
    return str(path)
