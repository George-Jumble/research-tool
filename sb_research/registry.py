"""
Module registry — loads every ``*_extension`` submenu and dispatches UI/run hooks.

Each submenu module is imported here once.  Optional imports are guarded so a
missing dependency (healpy, awkward, …) does not break the whole menu.
"""

from __future__ import annotations

from typing import Any

import menus.empirical_tests.extension as empirical_tests_extension
import menus.particle.lhcb.extension as lhcb_flavor_extension
import menus.prime_past.extension as prime_past_extension
import menus.remote_ai.extension as remote_ai_extension
import menus.astronomical.sparc.extension as sparc_extension

from sb_research.base_module import (
    call_pre_form_hook,
    resolve_entry_fields,
    resolve_entry_instructions,
)

# =============================================================================
# BLOCK: Optional extension imports (graceful degradation)
# =============================================================================

try:
    import menus.planck_cmb.extension as planck_cmb_extension
except ImportError as _planck_import_error:
    planck_cmb_extension = None
    PLANCK_IMPORT_ERROR = _planck_import_error
else:
    PLANCK_IMPORT_ERROR = None

try:
    import menus.astronomical.frb.extension as frb_web_extension
except ImportError as _frb_import_error:
    frb_web_extension = None
    FRB_IMPORT_ERROR = _frb_import_error
else:
    FRB_IMPORT_ERROR = None


try:
    import menus.astronomical.desi.extension as desi_bao_extension
except ImportError as _desi_import_error:
    desi_bao_extension = None
    DESI_IMPORT_ERROR = _desi_import_error
else:
    DESI_IMPORT_ERROR = None


try:
    import menus.particle.casimir.extension as casimir_extension
except ImportError as _casimir_import_error:
    casimir_extension = None
    CASIMIR_IMPORT_ERROR = _casimir_import_error
else:
    CASIMIR_IMPORT_ERROR = None


try:
    import menus.gravitic.ligo.extension as ligo_gwosc_extension
except ImportError as _ligo_import_error:
    ligo_gwosc_extension = None
    LIGO_IMPORT_ERROR = _ligo_import_error
else:
    LIGO_IMPORT_ERROR = None

try:
    import menus.gravitic.lisa.extension as lisa_pre_runs_extension
except ImportError as _lisa_import_error:
    lisa_pre_runs_extension = None
    LISA_IMPORT_ERROR = _lisa_import_error
else:
    LISA_IMPORT_ERROR = None

try:
    import menus.particle.cern.extension as cern_opendata_extension
except ImportError as _cern_import_error:
    cern_opendata_extension = None
    CERN_IMPORT_ERROR = _cern_import_error
else:
    CERN_IMPORT_ERROR = None

# =============================================================================
# BLOCK: MODULE_EXTENSIONS map (tag → extension module)
# =============================================================================

MODULE_EXTENSIONS: dict[str, Any] = {
    prime_past_extension.MODULE_TAG: prime_past_extension,
    empirical_tests_extension.MODULE_TAG: empirical_tests_extension,
    sparc_extension.MODULE_TAG: sparc_extension,
    lhcb_flavor_extension.MODULE_TAG: lhcb_flavor_extension,
    remote_ai_extension.MODULE_TAG: remote_ai_extension,
}

if planck_cmb_extension is not None:
    MODULE_EXTENSIONS[planck_cmb_extension.MODULE_TAG] = planck_cmb_extension
if frb_web_extension is not None:
    MODULE_EXTENSIONS[frb_web_extension.MODULE_TAG] = frb_web_extension
if desi_bao_extension is not None:
    MODULE_EXTENSIONS[desi_bao_extension.MODULE_TAG] = desi_bao_extension
if casimir_extension is not None:
    MODULE_EXTENSIONS[casimir_extension.MODULE_TAG] = casimir_extension
if ligo_gwosc_extension is not None:
    MODULE_EXTENSIONS[ligo_gwosc_extension.MODULE_TAG] = ligo_gwosc_extension
if lisa_pre_runs_extension is not None:
    MODULE_EXTENSIONS[lisa_pre_runs_extension.MODULE_TAG] = lisa_pre_runs_extension
if cern_opendata_extension is not None:
    MODULE_EXTENSIONS[cern_opendata_extension.MODULE_TAG] = cern_opendata_extension


# =============================================================================
# BLOCK: Dispatch helpers
# =============================================================================


def get_extension(module_tag: str) -> Any | None:
    """Return the extension module for ``module_tag``, or None."""
    from sb_shared.legacy_names import canonical_module_tag
    module_tag = canonical_module_tag(module_tag)
    return MODULE_EXTENSIONS.get(module_tag)


def module_entry_fields(module_tag: str, action: str) -> list[dict]:
    """Entry-form fields for a module action (delegates to the extension)."""
    ext = get_extension(module_tag)
    if ext is None:
        return []
    return resolve_entry_fields(ext, action)


def module_entry_instructions(module_tag: str, action: str) -> list[str]:
    """Help bullets for a module action (delegates to the extension)."""
    ext = get_extension(module_tag)
    if ext is None:
        return ["Follow field hints; leave optional entries blank to use engine defaults."]
    return resolve_entry_instructions(ext, action)


def handle_module_pre_form(
    module_tag: str,
    stdscr: Any,
    selection: str,
    params: dict[str, str],
) -> str | None:
    """Run a submenu's optional pre-form hook (n selector, warnings, …)."""
    ext = get_extension(module_tag)
    if ext is None:
        return None
    return call_pre_form_hook(ext, stdscr, selection, params)