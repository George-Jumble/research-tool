"""
Auto dataset pipeline wired into every analysis run.

Stages (unchanged from Dataset Manager design):
  1. **pull**   — fetch from remote / git / wget
  2. **cache**  — store under ``datasets/<module>/``
  3. **store**  — ``mark_processed`` after successful fetch/run
  4. **archive** — ``delete_previous_archived_runs`` before moving to finishedA/

Mock/synthetic data is not used: every module requires live data.
"""

from __future__ import annotations

from typing import Any, Callable

from sb_shared.dataset_registry import fetch_selected, get_provider
from sb_shared.empirical_action_targets import (
    empirical_keys_for_action,
    ensure_empirical_targets,
)

# (module_tag, action) pairs that intentionally use synthetic data
MOCK_ONLY_ACTIONS: set[tuple[str, str]] = set()

_ModuleEnsure = Callable[[str, dict[str, Any]], None]

_MODULE_ENSURES: dict[str, _ModuleEnsure] = {}


def _register_ensures() -> None:
    if _MODULE_ENSURES:
        return

    def _desi(action: str, options: dict[str, Any]) -> None:
        from menus.astronomical.desi.fetcher import ensure_desi_bao_data

        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        dr2 = ensure_desi_bao_data(force_refresh=force, auto_fetch=True)
        if not (options.get("cobaya_path") or "").strip():
            options["cobaya_path"] = str(dr2.parent)

    def _frb(action: str, options: dict[str, Any]) -> None:
        if action == "Pull Datasets from Open Archives":
            return
        from menus.astronomical.frb.fetcher import ensure_frb_datasets

        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        restore = str(options.get("restore_archived", "yes")).lower() in {"yes", "y", "true", "1"}
        print("[Superblock Residual Engine] Auto-fetch: FRB + void catalogs")
        ensure_frb_datasets(force_refresh=force, restore_archived=restore, auto_fetch=True)

    def _sparc(action: str, options: dict[str, Any]) -> None:
        if action in {"Pull Batch from Repository", "Pull Single Galaxy CSV"}:
            return
        galaxy = (options.get("galaxy_name") or options.get("query") or "").strip()
        if not galaxy:
            return
        from menus.astronomical.sparc.fetcher import ensure_galaxy_csv

        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        print(f"[Superblock Residual Engine] Auto-fetch: SPARC galaxy {galaxy}")
        ensure_galaxy_csv(galaxy, force_refresh=force)

    def _registry(module_tag: str) -> _ModuleEnsure:
        def _ensure(action: str, options: dict[str, Any]) -> None:
            provider = get_provider(module_tag)
            if provider is None:
                return
            targets = _default_registry_targets(module_tag, action, options)
            if not targets:
                return
            print(f"[Superblock Residual Engine] Auto-fetch via registry: {module_tag} ({len(targets)} target(s))")
            result = fetch_selected(module_tag, targets, options)
            if result.fetched:
                print(f"[Superblock Residual Engine] Fetched: {', '.join(result.fetched[:6])}")
            if result.failed:
                for key, msg in result.failed.items():
                    raise RuntimeError(f"Dataset fetch failed for {key}: {msg}")

        return _ensure

    def _empirical_module(module_tag: str) -> _ModuleEnsure:
        def _ensure(action: str, options: dict[str, Any]) -> None:
            keys = empirical_keys_for_action(module_tag, action)
            if not keys:
                return
            ensure_empirical_targets(module_tag, action, options, verbose=True)

        return _ensure

    def _cern(action: str, options: dict[str, Any]) -> None:
        # These actions perform the pull themselves (form protocol/release).
        if action in {
            "Pull Datasets from Open Archives",
            "Pull ATLAS Open Data (atlasopenmagic)",
            "List Cached Datasets",
        }:
            return
        from menus.particle.atlas.manifest import (
            ATLAS_MANIFEST,
            resolve_target_key as atlas_resolve,
        )
        from menus.particle.cern.manifest import ACTION_DEFAULT_TARGETS

        targets = ACTION_DEFAULT_TARGETS.get(action, [])
        form = str(options.get("target") or options.get("targets") or "").strip()
        if form and form.lower() not in {"all", "*"}:
            targets = [t.strip() for t in form.split(",") if t.strip()]
        if not targets:
            return

        atlas_names = [
            raw
            for raw in targets
            if atlas_resolve(raw) or raw in ATLAS_MANIFEST
        ]
        cern_names = [raw for raw in targets if raw not in atlas_names]
        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        restore = str(options.get("restore_archived", "yes")).lower() in {"yes", "y", "true", "1"}

        if atlas_names:
            from menus.particle.atlas.fetcher import pull_selected_targets as atlas_pull

            protocol = str(options.get("protocol") or "https").strip() or "https"
            print(f"[Superblock Residual Engine] Auto-fetch: ATLAS Open Data ({', '.join(atlas_names)})")
            result = atlas_pull(
                atlas_names,
                params={
                    "restore_archived": "yes" if restore else "no",
                    "protocol": protocol,
                },
                force_refresh=force,
            )
            if result.failed:
                for key, msg in result.failed.items():
                    raise RuntimeError(f"ATLAS fetch failed for {key}: {msg}")

        if cern_names:
            from menus.particle.cern.fetcher import pull_selected_targets

            print(f"[Superblock Residual Engine] Auto-fetch: CERN Open Data ({', '.join(cern_names)})")
            result = pull_selected_targets(
                cern_names,
                params={"restore_archived": "yes" if restore else "no"},
                force_refresh=force,
            )
            if result.failed:
                for key, msg in result.failed.items():
                    raise RuntimeError(f"CERN fetch failed for {key}: {msg}")

    _MODULE_ENSURES["DESI_BAO"] = _desi
    _MODULE_ENSURES["FRB_DM"] = _frb
    _MODULE_ENSURES["SPARC"] = _sparc
    _MODULE_ENSURES["CERN_OPENDATA"] = _cern
    def _lhcb(action: str, options: dict[str, Any]) -> None:
        if action == "Download LHCb B-meson Dataset":
            return
        _registry("LHCB_FLAVOR")(action, options)

    def _prime_past(action: str, options: dict[str, Any]) -> None:
        # Loader actions do their own fetch/build. Scoring actions lazily bootstrap AlterBBN; a failure
        # is printed and the residual loader reports a clean load_error (never aborts the menu).
        from menus.prime_past.extension import LEGACY_ACTIONS, LOADER_ACTIONS

        action = LEGACY_ACTIONS.get(action, action)
        if action in LOADER_ACTIONS:
            return
        from bbn_isolated import alterbbn

        if alterbbn.find_binary(alterbbn.BINARY_NAME) and alterbbn.find_binary(alterbbn.FULL_BINARY_NAME):
            return
        if not alterbbn.auto_build_enabled():
            print("[Superblock Residual Engine] AlterBBN missing and SB_ALTERBBN_NO_AUTO_BUILD is set")
            return
        print("[Superblock Residual Engine] Auto-fetch: AlterBBN v2.2 (pinned tarball) → datasets/prime_past/")
        try:
            alterbbn.ensure_alterbbn(verbose=True)
        except alterbbn.AlterBBNUnavailable as exc:
            print(f"[Superblock Residual Engine] AlterBBN bootstrap failed: {exc}")

    _MODULE_ENSURES["PRIME_PAST_HARMONIC"] = _prime_past
    _MODULE_ENSURES["LHCB_FLAVOR"] = _lhcb
    for tag in ("EMPIRICAL_TESTS", "PLANCK_CMB"):
        _MODULE_ENSURES[tag] = _registry(tag)


def _default_registry_targets(module_tag: str, action: str, options: dict[str, Any]) -> list[str]:
    provider = get_provider(module_tag)
    if provider is None:
        return []
    remote = [t.id for t in provider.list_fetchable() if t.state.name in {"REMOTE", "ARCHIVED"}]
    if module_tag in {"EMPIRICAL_TESTS", "PRIME_PAST_HARMONIC"}:
        keys = empirical_keys_for_action(module_tag, action)
        if keys:
            return [f"empirical:{key}" for key in keys]
        if module_tag == "EMPIRICAL_TESTS":
            return remote[:3] if remote else []
        return []
    if module_tag == "LHCB_FLAVOR":
        return [t.id for t in provider.list_fetchable() if "lhcb:" in t.id][:1]
    if module_tag == "CERN_OPENDATA":
        from menus.particle.atlas.manifest import ATLAS_MANIFEST, resolve_target_key as atlas_resolve
        from menus.particle.cern.manifest import ACTION_DEFAULT_TARGETS

        ids: list[str] = []
        for key in ACTION_DEFAULT_TARGETS.get(action, []):
            if atlas_resolve(key) or key in ATLAS_MANIFEST:
                ids.append(f"atlas:{key}")
            else:
                ids.append(f"cern:{key}")
        return ids
    return remote


def ensure_datasets_before_run(
    module_tag: str,
    action: str,
    options: dict[str, Any] | None = None,
) -> None:
    """
    Pull required datasets before ``run_action``. Raises on missing live data.
    """
    if options is None:
        options = {}
    if (module_tag, action) in MOCK_ONLY_ACTIONS:
        print(f"[Superblock Residual Engine] Skipping live-data fetch for mock/toy action: {action}")
        return

    _register_ensures()
    ensure_fn = _MODULE_ENSURES.get(module_tag)
    if ensure_fn is None:
        ext_hook = options.pop("_extension_ensure", None)
        if callable(ext_hook):
            ext_hook(action, options)
        return

    print(f"[Superblock Residual Engine] Dataset pipeline — pull→cache for {module_tag} / {action}")
    ensure_fn(action, options)