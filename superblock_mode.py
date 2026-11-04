"""
Superblock-only boot gate (2026-09-21).

Live science is 3+1D Superblock / Mirron. Tau, Tav, KK, 142857, compact S¹,
and 5D templates are not configurable and are not on the default path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "superblock_only.json"

RETIRED_ON = "2026-09-21"
ENGINE_PREFIX = "Superblock Residual Engine"
TOOL_TITLE = "Superblock Research Tool"

RETIRED_NOTE = (
    f"This Tau/Tav/5D item was retired {RETIRED_ON}. "
    "The live tool is Superblock / Mirron only (3+1D). "
    "Tau, Tav, KK, 142857, and compact-S¹ templates are not evaluated. "
    "Tau/5D templates were not applied."
)


class RetiredModuleError(RuntimeError):
    """Raised when a retired Tau/Tav/5D entry point is called."""


# Entire modules disconnected from the GUI import graph.
RETIRED_MODULE_TAGS = frozenset(
    {
        "TAV_RESONANCE",
        "RGC_MOCK_CATALOG",
        "TAV_BBN",
        "BERARD_FRAMEWORK",
        "TAV_DATA_INTEGRATOR",
        "TSB_RESEARCH",
    }
)

# Actions that stay on the original loader (no Tau scoring).
LOADER_ACTIONS: dict[str, frozenset[str]] = {
    "SPARC": frozenset(
        {
            "Pull Batch from Repository",
            "Pull Single Galaxy CSV",
        }
    ),
    "LHCB_TAV_ECHO": frozenset({"Download LHCb B-meson Dataset"}),
    "CERN_OPENDATA": frozenset(
        {
            "Pull Datasets from Open Archives",
            "Pull ATLAS Open Data (atlasopenmagic)",
            "List Cached Datasets",
            "Preregistered Recoil q_T Study (HEP Controls)",
        }
    ),
    "LIGO_GWOSC": frozenset({"Pull Strain from GWOSC"}),
    "LISA_PRE_RUNS": frozenset(
        {
            "Pull OSDF Target",
            "List Cached Mock Data",
            "OSDF Connectivity Check",
        }
    ),
    "FRB_COSMIC_WEB_TAV": frozenset({"Pull Datasets from Open Archives"}),
    "TSB_CASIMIR": frozenset(
        {
            "Download Zenodo Casimir Drums",
            "Download GitHub Magnetic-Fluid Data",
        }
    ),
    "EMPIRICAL_TESTS": frozenset(
        {
            "Run All Tests (Curated)",
            "Light Quarks Confrontation",
            "Neutron Lifetime Test",
            "Yang-Mills Glueball Test",
            "BBN Abundance Confrontation",
            "Full Pipeline (Plot + LaTeX)",
        }
    ),
    "REMOTE_AI": frozenset(),  # filled at runtime: all remote-ai actions are keepers
}

# Explicit Tau/5D/KK menu items. Not shown. Calling them raises.
RETIRED_ACTIONS: dict[str, frozenset[str]] = {
    "TAU_SB_DESI": frozenset(
        {
            "Periodicity Scan (1/7 Tracks)",
            "Harmonic Binding Detection Kit",
            "Model Compare (ΛCDM vs aDE vs Tau-SB)",
            "MCMC Posteriors (emcee)",
            "Injection/Recovery Tests",
            "Full Production Pipeline",
            "Mock Catalog Recovery (1/7 + Δn)",
            "LSS Comb Falsification (P(k))",
            "S(n) Node Cross-Correlation",
            "Gridded Mock Comb Recovery",
            "Method 10 Joint Likelihood",
            "Test 2 Redshift Law (λ(z))",
            "LSS Publication Figures",
            "Cosmology Falsification (Methods 3, 5)",
            "EKK BAO Re-analysis (DR2 Covariance)",
            "Batch Tracer Scan (All DR2)",
        }
    ),
    "CERN_OPENDATA": frozenset(
        {
            "CMS 7-Fold Muon Tav Analysis",
            "CMS Full-Dataset 7-Fold Scan",
            "Run MC Validation Suite",
        }
    ),
    "LHCB_TAV_ECHO": frozenset(
        {
            "Run Tav-Echo Correlation",
            "Plot Residuals vs Kernel",
            "Full Analysis (Stats + Plot)",
        }
    ),
    "LIGO_GWOSC": frozenset({"Ringdown Harmonic Scan (1/7)"}),
    "FRB_COSMIC_WEB_TAV": frozenset(
        {
            "Classify FRB Paths",
            "Scan 1/7 Tav Harmonics",
            "FRB Dispersion Tav Test (KS)",
            "Full FRB Tav Analysis",
        }
    ),
    "PLANCK_CMB_TAV": frozenset(
        {
            "Scan 1/7 Tav Harmonics",
            "Full CMB Tav Analysis",
        }
    ),
    "TSB_CASIMIR": frozenset(
        {
            "Example Casimir Tau-Search",
            "TSB Falsification Test Suite (5 predictions)",
            "TSB Falsification Diagnostics (controls + audit)",
            "Full Casimir Tau-Search Pipeline",
            "Null-Hypothesis False-Positive Suite",
            "Blind Periodicity Scan",
            "Lock Blind Results",
            "Compare Blind vs Informed",
        }
    ),
    "PRIME_PAST_HARMONIC": frozenset(
        {
            "BBN Interference Scan",
            "BBN Confrontation Scan",
            "Enhanced BBN Confrontation",
            "Final BBN Confrontation Tool",
            "Full Module Demo",
        }
    ),
}

# Everything else on a live module is the 3+1D residual path (Step 0–4).
RESIDUAL_BY_DEFAULT = frozenset(
    {
        "TAU_SB_DESI",
        "SPARC",
        "LHCB_TAV_ECHO",
        "CERN_OPENDATA",
        "LIGO_GWOSC",
        "FRB_COSMIC_WEB_TAV",
        "PLANCK_CMB_TAV",
        "TSB_CASIMIR",
        "PRIME_PAST_HARMONIC",
    }
)

CONTINUE = object()


def load_boot_config() -> dict[str, Any]:
    """Default boot. tau_5d_templates is forced false even if the file is edited."""
    payload: dict[str, Any] = {
        "mode": "superblock_mirron_3p1d",
        "superblock_only": True,
        "tau_5d_templates": False,
        "comparison_models": ["LCDM", "GR", "SM", "Newtonian_SPARC"],
        "alternative_under_test": "Superblock/Mirron",
        "retired": RETIRED_ON,
    }
    if CONFIG_PATH.is_file():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    payload["superblock_only"] = True
    payload["tau_5d_templates"] = False
    return payload


def superblock_only() -> bool:
    return True


def tau_templates_enabled() -> bool:
    """Retired. Always false. Not a hidden switch."""
    return False


def retired_message(module_tag: str, action: str | None = None) -> str:
    target = f"{module_tag}: {action}" if action else module_tag
    return (
        f"module retired {RETIRED_ON}; Superblock/Mirron only. "
        f"{target}. {RETIRED_NOTE}"
    )


def classify_action(module_tag: str, action: str) -> str:
    """Return 'retired', 'loader', 'keep', or 'residual'."""
    tag = str(module_tag or "")
    name = str(action or "")
    if tag in RETIRED_MODULE_TAGS:
        return "retired"
    if name in RETIRED_ACTIONS.get(tag, frozenset()):
        return "retired"
    if tag == "REMOTE_AI" or tag == "EMPIRICAL_TESTS":
        return "keep"
    if name in LOADER_ACTIONS.get(tag, frozenset()):
        return "loader"
    if tag in RESIDUAL_BY_DEFAULT:
        return "residual"
    return "keep"


def live_menu_actions(module_tag: str, actions: Iterable[str]) -> list[str]:
    """Drop retired Tau/5D items from a GUI menu. Order preserved."""
    if module_tag in RETIRED_MODULE_TAGS:
        return []
    kept: list[str] = []
    for action in actions:
        if classify_action(module_tag, action) == "retired":
            continue
        kept.append(action)
    return kept


def print_retired(module_tag: str, action: str | None = None) -> None:
    print(f"[{ENGINE_PREFIX}] {retired_message(module_tag, action)}")


def begin_live_action(
    module_tag: str,
    action: str,
    options: dict | None = None,
) -> Any:
    """
    Call at the start of a live module ``run_action``.

    Returns ``CONTINUE`` when the original loader/keeper body may run.
    Returns a residual report dict when the action is retargeted.
    Raises ``RetiredModuleError`` for Tau/Tav/5D actions.
    """
    route = classify_action(module_tag, action)
    if route == "retired":
        print_retired(module_tag, action)
        raise RetiredModuleError(retired_message(module_tag, action))
    if route == "residual":
        from superblock_residual import run_superblock_residual

        return run_superblock_residual(module_tag, action, options or {})
    return CONTINUE
