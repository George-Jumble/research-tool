"""
Superblock boot gate.

Live science is 3+1D Superblock / Mirron. Loader actions fetch and cache data; every other
action on a live module is scored by the Superblock residual engine (Steps 0–4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "superblock_only.json"

ENGINE_PREFIX = "Superblock Residual Engine"
TOOL_TITLE = "Superblock Research Tool"


# Actions that stay on the original loader (fetch / cache only).
LOADER_ACTIONS: dict[str, frozenset[str]] = {
    "SPARC": frozenset(
        {
            "Pull Batch from Repository",
            "Pull Single Galaxy CSV",
        }
    ),
    "LHCB_FLAVOR": frozenset({"Download LHCb B-meson Dataset"}),
    "PRIME_PAST_HARMONIC": frozenset({"Fetch & Build AlterBBN v2.2", "BBN Gate Check (AlterBBN)"}),
    "CERN_OPENDATA": frozenset(
        {
            "Pull Datasets from Open Archives",
            "Pull ATLAS Open Data (atlasopenmagic)",
            "List Cached Datasets",
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
    "FRB_DM": frozenset({"Pull Datasets from Open Archives"}),
    "CASIMIR": frozenset(
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

# Everything else on a live module is the 3+1D residual path (Step 0–4).
RESIDUAL_BY_DEFAULT = frozenset(
    {
        "DESI_BAO",
        "SPARC",
        "LHCB_FLAVOR",
        "CERN_OPENDATA",
        "LIGO_GWOSC",
        "FRB_DM",
        "PLANCK_CMB",
        "CASIMIR",
        "PRIME_PAST_HARMONIC",
    }
)

CONTINUE = object()


def load_boot_config() -> dict[str, Any]:
    """Default boot config, merged with config/superblock_only.json when present."""
    payload: dict[str, Any] = {
        "mode": "superblock_mirron_3p1d",
        "superblock_only": True,
        "comparison_models": ["LCDM", "GR", "SM", "Newtonian_SPARC"],
        "alternative_under_test": "Superblock/Mirron",
    }
    if CONFIG_PATH.is_file():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    payload["superblock_only"] = True
    return payload


def superblock_only() -> bool:
    return True


def classify_action(module_tag: str, action: str) -> str:
    """Return 'loader', 'keep', or 'residual'."""
    from sb_shared.legacy_names import canonical_module_tag
    module_tag = canonical_module_tag(module_tag)
    tag = str(module_tag or "")
    name = str(action or "")
    if tag == "REMOTE_AI" or tag == "EMPIRICAL_TESTS":
        return "keep"
    if name in LOADER_ACTIONS.get(tag, frozenset()):
        return "loader"
    if tag in RESIDUAL_BY_DEFAULT:
        return "residual"
    return "keep"


def begin_live_action(
    module_tag: str,
    action: str,
    options: dict | None = None,
) -> Any:
    """
    Call at the start of a live module ``run_action``.

    Returns ``CONTINUE`` when the module's own loader/keeper body may run.
    Returns a residual report dict when the action is scored by the residual engine.
    """
    route = classify_action(module_tag, action)
    if route == "residual":
        from superblock_residual import run_superblock_residual

        return run_superblock_residual(module_tag, action, options or {})
    return CONTINUE
