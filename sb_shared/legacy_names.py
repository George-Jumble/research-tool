"""
Legacy-name aliases kept so pre-rename data on disk keeps loading.

Earlier versions used framework-specific names for module tags, the
processed-archive staging directory, artifact sub-directories, ledger
prefixes and environment variables. Those names were removed from the
code base; this module is the single place where the old spellings are
still recognised. Nothing here moves, renames or deletes user files: old
locations are only *read* (or reused when they already hold data).
"""

from __future__ import annotations

import os
from pathlib import Path

# Old module tag -> current module tag (registry keys, ledgers, report names).
LEGACY_MODULE_TAGS: dict[str, str] = {
    "TAU_SB_DESI": "DESI_BAO",
    "LHCB_TAV_ECHO": "LHCB_FLAVOR",
    "FRB_COSMIC_WEB_TAV": "FRB_DM",
    "PLANCK_CMB_TAV": "PLANCK_CMB",
    "TSB_CASIMIR": "CASIMIR",
}

# Old processed-ledger id prefixes still honoured when clearing/reading ledgers.
LEGACY_LEDGER_PREFIXES: dict[str, tuple[str, ...]] = {
    "DESI_BAO": ("tau_sb_desi:",),
    "CASIMIR": ("tsb_casimir:",),
}

# Filename regexes of old flat artifacts -> artifacts/ sub-directory used by
# ``organize_artifacts`` (old directory names kept verbatim so old files are
# filed exactly where earlier versions put them).
LEGACY_ARTIFACT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^tau_sb_desi", "desi_bao_scan"),
    (r"^tau_sb_mock_catalog", "mock_catalog_recovery"),
    (r"^tau_sb_mcmc", "mcmc_posteriors"),
    (r"^tau_sb_nested|^tau_sb_jax", "nested_sampling"),
    (r"^tsb_casimir", "casimir"),
    (r"^tsb_research|^tsb_falsification", "tsb_research"),
    (r"^tav_resonance|^tav_bbn", "prime_past_bbn"),
)
# Old artifacts/ sub-directories merged by ``organize_artifacts``.
LEGACY_ARTIFACT_SUBDIR_MERGE: dict[str, str] = {
    "rgc": "rgc_mock_catalog",
}

# Processed-archive staging directory (repo-relative) and old ProtonDrive folder.
LEGACY_STAGING_DIRNAME = ".tav_project_staging"
LEGACY_PROTON_DIRNAME = "tav_project"

# Current env var -> old env var accepted as a fallback.
LEGACY_ENV_VARS: dict[str, str] = {
    "SB_USE_MOCK_DATA": "TAV_USE_MOCK_DATA",
    "SB_CACHE_DIR": "TAV_CACHE_DIR",
    "SB_INGESTION_CONFIG": "TAV_INGESTION_CONFIG",
    "SB_COMPARISON_CONFIG": "TAV_COMPARISON_CONFIG",
    "SB_REMOTE_TRANSFER": "TAV_REMOTE_TRANSFER",
}


def canonical_module_tag(tag: str | None) -> str:
    """Map a legacy module tag onto its current name (identity otherwise)."""
    if tag is None:
        return ""
    key = str(tag).strip()
    return LEGACY_MODULE_TAGS.get(key.upper(), key)


def getenv_compat(name: str, default: str | None = None) -> str | None:
    """``os.getenv`` that also honours the pre-rename variable name."""
    value = os.getenv(name)
    if value is None and name in LEGACY_ENV_VARS:
        value = os.getenv(LEGACY_ENV_VARS[name])
    return default if value is None else value


def resolve_staging_dir(project_root: Path, new_name: str = ".sb_project_staging") -> Path:
    """
    Return the processed-archive staging dir.

    If only the legacy directory exists (existing user archives), keep using it
    in place rather than moving or copying anything.
    """
    new = (project_root / new_name).resolve()
    legacy = (project_root / LEGACY_STAGING_DIRNAME).resolve()
    if not new.exists() and legacy.is_dir():
        return legacy
    return new
