#!/usr/bin/env python3
"""
Unified registry for fetch-capable research-tool modules.

Each provider exposes listable dataset targets, fetch, archive, and optional run hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

from sb_shared.dataset_ledger import is_processed, mark_processed, sync_archived_as_processed

from sb_shared.project_paths import (
    FINISHED_A_DIR,
    LEGACY_FINISHED_A_DIR,
    ensure_project_dirs,
    move_to_finished_archive,
)

from sb_shared.project_paths import SUPERBLOCK_ROOT as PROJECT_ROOT


class TargetState(str, Enum):
    REMOTE = "remote"
    CACHED = "cached"
    ARCHIVED = "archived"
    PROCESSED = "processed"


@dataclass
class DatasetTarget:
    """One fetchable or runnable dataset entry."""

    id: str
    module_tag: str
    label: str
    state: TargetState
    path: Optional[Path] = None
    group: str = ""

    @property
    def local_name(self) -> str:
        return self.id.split(":", 1)[-1]


@dataclass
class FetchResult:
    module_tag: str
    fetched: List[str]
    skipped: List[str]
    failed: Dict[str, str]


ProviderListFn = Callable[[], List[DatasetTarget]]
ProviderFetchFn = Callable[[List[str], dict], FetchResult]
ProviderArchiveFn = Callable[[List[str]], List[Path]]
ProviderRunFn = Callable[[str, List[str], dict], None]


@dataclass
class DatasetProvider:
    module_tag: str
    title: str
    intake_dir: Path
    list_fetchable: ProviderListFn
    list_runnable: ProviderListFn
    fetch_targets: ProviderFetchFn
    archive_targets: ProviderArchiveFn
    run_action: Optional[ProviderRunFn] = None
    run_actions: Optional[List[str]] = None


def _state_from_path(path: Optional[Path], intake: Path) -> TargetState:
    if path is None or not path.exists():
        return TargetState.REMOTE
    try:
        path.resolve().relative_to(intake.resolve())
        return TargetState.CACHED
    except ValueError:
        return TargetState.CACHED


def _apply_processed_state(target: DatasetTarget) -> DatasetTarget:
    """Upgrade target state when it appears in the global processed ledger."""
    if is_processed(target.id):
        return DatasetTarget(
            id=target.id,
            module_tag=target.module_tag,
            label=target.label,
            state=TargetState.PROCESSED,
            path=target.path,
            group=target.group,
        )
    return target


def _stamp_processed(targets: List[DatasetTarget]) -> List[DatasetTarget]:
    return [_apply_processed_state(target) for target in targets]


def _should_skip_fetch(target_id: str, force: bool) -> bool:
    return not force and is_processed(target_id)


def _sparc_provider() -> DatasetProvider:
    from menus.astronomical.sparc.fetcher import (
        DATASETS_DIR,
        archive_used_datasets,
        fetch_mass_models_table,
        list_catalog_galaxies,
        list_local_galaxies,
        pull_selected_galaxies,
        read_done_list,
    )

    def list_fetchable() -> List[DatasetTarget]:
        targets: List[DatasetTarget] = []
        catalog_name = "MassModels_Lelli2016c.mrt"
        catalog_path = DATASETS_DIR / catalog_name
        targets.append(
            DatasetTarget(
                id=f"sparc:{catalog_name}",
                module_tag="SPARC",
                label="SPARC mass-model catalog (.mrt)",
                state=_state_from_path(catalog_path, DATASETS_DIR),
                path=catalog_path if catalog_path.is_file() else None,
                group="catalog",
            )
        )
        done = read_done_list()
        cached = set(list_local_galaxies())
        try:
            mass_models = fetch_mass_models_table(cache_path=catalog_path)
            galaxies = list_catalog_galaxies(mass_models)
        except Exception:
            galaxies = sorted(cached)
        for galaxy in galaxies:
            csv_path = DATASETS_DIR / f"{galaxy}.csv"
            if galaxy in cached or csv_path.is_file():
                state = TargetState.CACHED
            elif galaxy in done:
                state = TargetState.ARCHIVED
            else:
                state = TargetState.REMOTE
            targets.append(
                DatasetTarget(
                    id=f"sparc:{galaxy}",
                    module_tag="SPARC",
                    label=f"{galaxy} rotation curve",
                    state=state,
                    path=csv_path if csv_path.is_file() else None,
                    group="galaxy",
                )
            )
        return _stamp_processed(targets)

    def list_runnable() -> List[DatasetTarget]:
        from menus.astronomical.sparc.fetcher import iter_local_csv_files

        return _stamp_processed(
            [
                DatasetTarget(
                    id=f"sparc:{path.stem}",
                    module_tag="SPARC",
                    label=f"{path.stem} rotation curve",
                    state=TargetState.CACHED,
                    path=path,
                    group="galaxy",
                )
                for path in iter_local_csv_files()
            ]
        )

    def fetch_targets(target_ids: List[str], options: dict) -> FetchResult:
        result = FetchResult(module_tag="SPARC", fetched=[], skipped=[], failed={})
        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        names = [
            tid.split(":", 1)[-1]
            for tid in target_ids
            if tid.startswith("sparc:") and not _should_skip_fetch(tid, force)
        ]
        result.skipped.extend(
            tid for tid in target_ids if _should_skip_fetch(tid, force)
        )
        if "MassModels_Lelli2016c.mrt" in names:
            fetch_mass_models_table(
                cache_path=DATASETS_DIR / "MassModels_Lelli2016c.mrt",
                force_refresh=force,
            )
            result.fetched.append("MassModels_Lelli2016c.mrt")
            names = [name for name in names if name != "MassModels_Lelli2016c.mrt"]
        if names:
            summary = pull_selected_galaxies(
                names,
                force_refresh_catalog=force,
            )
            result.fetched.extend(summary.downloaded)
            result.skipped.extend(summary.skipped_done)
            result.skipped.extend(summary.skipped_existing)
            result.failed.update(summary.failed)
        return result

    def archive_targets(target_ids: List[str]) -> List[Path]:
        from menus.astronomical.sparc.fetcher import resolve_galaxy_csv_path

        paths = [
            resolve_galaxy_csv_path(tid.split(":", 1)[-1])
            for tid in target_ids
            if tid.startswith("sparc:")
        ]
        moved = archive_used_datasets(paths)
        sync_archived_as_processed(
            [tid for tid in target_ids if tid.startswith("sparc:")],
            note="sparc archive",
        )
        return moved

    def run_action(action: str, target_ids: List[str], options: dict) -> None:
        import menus.astronomical.sparc.extension as sparc_extension

        galaxies = [tid.split(":", 1)[-1] for tid in target_ids if tid.startswith("sparc:")]
        merged = {**options, "selected_datasets": ",".join(galaxies)}
        sparc_extension.run_action(action, options=merged)
        mark_processed(
            [tid for tid in target_ids if tid.startswith("sparc:")],
            note=f"sparc run: {action}",
        )

    return DatasetProvider(
        module_tag="SPARC",
        title="SPARC",
        intake_dir=DATASETS_DIR,
        list_fetchable=list_fetchable,
        list_runnable=list_runnable,
        fetch_targets=fetch_targets,
        archive_targets=archive_targets,
        run_action=run_action,
        run_actions=[
            action
            for action in __import__("sparc_extension").MENU_ACTIONS
            if "Pull" not in action
        ],
    )


def _frb_provider() -> DatasetProvider:
    from menus.astronomical.frb.fetcher import (
        DATASETS_DIR,
        DEFAULT_FRB_CSV,
        DEFAULT_VOID_CSV,
        FRB_TARGET_CHIME,
        FRB_TARGET_VOID,
        archive_used_datasets,
        ensure_chime_catalog,
        ensure_void_catalog,
        list_dataset_files,
    )

    def _frb_targets() -> List[DatasetTarget]:
        targets = [
            DatasetTarget(
                id="frb:chime_frb_catalog",
                module_tag="FRB_DM",
                label="CHIME FRB catalog (CSV)",
                state=_state_from_path(DEFAULT_FRB_CSV, DATASETS_DIR),
                path=DEFAULT_FRB_CSV if DEFAULT_FRB_CSV.is_file() else None,
                group="frb",
            ),
            DatasetTarget(
                id="frb:sdss_void_catalog",
                module_tag="FRB_DM",
                label="SDSS DR7 void catalog (Douglass+ 2023)",
                state=_state_from_path(DEFAULT_VOID_CSV, DATASETS_DIR),
                path=DEFAULT_VOID_CSV if DEFAULT_VOID_CSV.is_file() else None,
                group="void",
            ),
        ]
        for path in list_dataset_files():
            stem = path.stem
            if stem in {"chime_frb_catalog", "sdss_void_catalog"}:
                continue
            tag = "frb" if "frb" in stem or "chime" in stem else "void"
            targets.append(
                DatasetTarget(
                    id=f"frb:{stem}",
                    module_tag="FRB_DM",
                    label=path.name,
                    state=TargetState.CACHED,
                    path=path,
                    group=tag,
                )
            )
        return _stamp_processed(targets)

    def fetch_targets(target_ids: List[str], options: dict) -> FetchResult:
        result = FetchResult(module_tag="FRB_DM", fetched=[], skipped=[], failed={})
        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        ids = {
            tid.split(":", 1)[-1]
            for tid in target_ids
            if tid.startswith("frb:") and not _should_skip_fetch(tid, force)
        }
        result.skipped.extend(
            tid for tid in target_ids if _should_skip_fetch(tid, force)
        )
        restore = str(options.get("restore_archived", "yes")).lower() in {
            "yes",
            "y",
            "true",
            "1",
        }
        try:
            if "chime_frb_catalog" in ids:
                if _should_skip_fetch(FRB_TARGET_CHIME, force) and DEFAULT_FRB_CSV.is_file():
                    result.skipped.append(FRB_TARGET_CHIME)
                else:
                    _, source = ensure_chime_catalog(
                        force_refresh=force,
                        restore_archived=restore,
                        auto_fetch=True,
                    )
                    if source in {"cached", "restored"} and not force:
                        result.skipped.append(FRB_TARGET_CHIME)
                    else:
                        result.fetched.append("chime_frb_catalog")
            if "sdss_void_catalog" in ids:
                if _should_skip_fetch(FRB_TARGET_VOID, force) and DEFAULT_VOID_CSV.is_file():
                    result.skipped.append(FRB_TARGET_VOID)
                else:
                    _, source = ensure_void_catalog(
                        force_refresh=force,
                        restore_archived=restore,
                        auto_fetch=True,
                    )
                    if source in {"cached", "restored"} and not force:
                        result.skipped.append(FRB_TARGET_VOID)
                    else:
                        result.fetched.append("sdss_void_catalog")
        except Exception as exc:
            result.failed["fetch"] = str(exc)
        return result

    def archive_targets(target_ids: List[str]) -> List[Path]:
        paths = []
        for tid in target_ids:
            if not tid.startswith("frb:"):
                continue
            name = tid.split(":", 1)[-1]
            candidate = DATASETS_DIR / f"{name}.csv"
            if candidate.is_file():
                paths.append(candidate)
        moved = archive_used_datasets(paths)
        sync_archived_as_processed(
            [tid for tid in target_ids if tid.startswith("frb:")],
            note="frb archive",
        )
        return moved

    def run_action(action: str, target_ids: List[str], options: dict) -> None:
        import menus.astronomical.frb.extension as frb_web_extension

        merged = dict(options)
        for tid in target_ids:
            if not tid.startswith("frb:"):
                continue
            name = tid.split(":", 1)[-1]
            if "void" in name:
                merged["void_catalog"] = str(DATASETS_DIR / f"{name}.csv")
            else:
                merged["frb_catalog"] = str(DATASETS_DIR / f"{name}.csv")
        frb_web_extension.run_action(action, options=merged)
        mark_processed(
            [tid for tid in target_ids if tid.startswith("frb:")],
            note=f"frb run: {action}",
        )

    return DatasetProvider(
        module_tag="FRB_DM",
        title="FRB Dispersion Residuals",
        intake_dir=DATASETS_DIR,
        list_fetchable=_frb_targets,
        list_runnable=_frb_targets,
        fetch_targets=fetch_targets,
        archive_targets=archive_targets,
        run_action=run_action,
        run_actions=[
            action
            for action in __import__("frb_web_extension").MENU_ACTIONS
            if "Pull" not in action
        ],
    )


def _planck_provider() -> Optional[DatasetProvider]:
    try:
        from menus.planck_cmb.fetcher import (
            CMB_MAP_URL,
            CMB_MASK_URL,
            EXAMPLE_CMB_MAP,
            EXAMPLE_CMB_MASK,
            FITS_DIR,
            archive_used_fits,
            download_planck_mask_for_map,
            list_fits_files,
        )
    except ImportError:
        return None

    def _planck_targets() -> List[DatasetTarget]:
        targets = [
            DatasetTarget(
                id=f"planck:{EXAMPLE_CMB_MAP}",
                module_tag="PLANCK_CMB",
                label=f"Example CMB map ({EXAMPLE_CMB_MAP})",
                state=TargetState.REMOTE,
                group="remote",
            ),
            DatasetTarget(
                id=f"planck:{EXAMPLE_CMB_MASK}",
                module_tag="PLANCK_CMB",
                label=f"Example CMB mask ({EXAMPLE_CMB_MASK})",
                state=TargetState.REMOTE,
                group="remote",
            ),
        ]
        for path in list_fits_files():
            group = "mask" if "mask" in path.name.lower() else "map"
            targets.append(
                DatasetTarget(
                    id=f"planck:{path.name}",
                    module_tag="PLANCK_CMB",
                    label=path.name,
                    state=TargetState.CACHED,
                    path=path,
                    group=group,
                )
            )
        return _stamp_processed(targets)

    def fetch_targets(target_ids: List[str], options: dict) -> FetchResult:
        import subprocess

        result = FetchResult(module_tag="PLANCK_CMB", fetched=[], skipped=[], failed={})
        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        FITS_DIR.mkdir(parents=True, exist_ok=True)
        names = {
            tid.split(":", 1)[-1]
            for tid in target_ids
            if tid.startswith("planck:") and not _should_skip_fetch(tid, force)
        }
        result.skipped.extend(
            tid for tid in target_ids if _should_skip_fetch(tid, force)
        )
        url_map = {
            EXAMPLE_CMB_MAP: CMB_MAP_URL,
            EXAMPLE_CMB_MASK: CMB_MASK_URL,
        }
        for name, url in url_map.items():
            if name not in names:
                continue
            dest = FITS_DIR / name
            if dest.is_file() and dest.stat().st_size > 1000:
                result.skipped.append(name)
                continue
            proc = subprocess.run(
                ["wget", "-q", "--timeout=180", "-O", str(dest), url],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0 and dest.is_file():
                result.fetched.append(name)
            else:
                result.failed[name] = proc.stderr.strip() or "wget failed"

        maps = [n for n in names if "mask" not in n.lower() and n.endswith(".fits")]
        for map_name in maps:
            map_path = FITS_DIR / map_name
            if not map_path.is_file():
                continue
            try:
                mask_path = download_planck_mask_for_map(map_path)
                result.fetched.append(mask_path.name)
            except Exception as exc:
                result.failed[f"mask_for_{map_name}"] = str(exc)
        return result

    def archive_targets(target_ids: List[str]) -> List[Path]:
        paths = []
        for tid in target_ids:
            if not tid.startswith("planck:"):
                continue
            path = FITS_DIR / tid.split(":", 1)[-1]
            if path.is_file():
                paths.append(path)
        if len(paths) >= 2:
            archive_used_fits(paths[0], paths[1])
        elif paths:
            archive_used_fits(paths[0], paths[0])
        sync_archived_as_processed(
            [tid for tid in target_ids if tid.startswith("planck:")],
            note="planck archive",
        )
        return paths

    def run_action(action: str, target_ids: List[str], options: dict) -> None:
        import menus.planck_cmb.extension as planck_cmb_extension

        merged = dict(options)
        for tid in target_ids:
            if not tid.startswith("planck:"):
                continue
            name = tid.split(":", 1)[-1]
            if "mask" in name.lower():
                merged["cmb_mask"] = str(FITS_DIR / name)
            elif name.endswith(".fits"):
                merged["cmb_map"] = str(FITS_DIR / name)
        planck_cmb_extension.run_action(action, options=merged)
        mark_processed(
            [tid for tid in target_ids if tid.startswith("planck:")],
            note=f"planck run: {action}",
        )

    return DatasetProvider(
        module_tag="PLANCK_CMB",
        title="Planck CMB Residuals",
        intake_dir=FITS_DIR,
        list_fetchable=_planck_targets,
        list_runnable=_planck_targets,
        fetch_targets=fetch_targets,
        archive_targets=archive_targets,
        run_action=run_action,
        run_actions=[
            action
            for action in __import__("planck_cmb_extension").MENU_ACTIONS
        ],
    )


def _empirical_provider() -> DatasetProvider:
    from menus.empirical_tests import tests as empirical

    DATASETS_DIR = PROJECT_ROOT / "datasets" / "empirical"
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    def _targets() -> List[DatasetTarget]:
        targets: List[DatasetTarget] = []
        for key, repo in empirical.REPOS.items():
            cache = empirical.cache_path(key)
            dataset_cache = DATASETS_DIR / f"{key}.csv"
            path = dataset_cache if dataset_cache.is_file() else cache
            state = TargetState.CACHED if path.is_file() else TargetState.REMOTE
            if repo.get("url") is None and repo["type"] in {"curated", "pdg_or_curated"}:
                state = TargetState.CACHED
            targets.append(
                DatasetTarget(
                    id=f"empirical:{key}",
                    module_tag="EMPIRICAL_TESTS",
                    label=repo["description"],
                    state=state,
                    path=path if path.is_file() else None,
                    group=repo["type"],
                )
            )
        return _stamp_processed(targets)

    def fetch_targets(target_ids: List[str], options: dict) -> FetchResult:
        result = FetchResult(module_tag="EMPIRICAL_TESTS", fetched=[], skipped=[], failed={})
        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        keys = [
            tid.split(":", 1)[-1]
            for tid in target_ids
            if tid.startswith("empirical:") and not _should_skip_fetch(tid, force)
        ]
        result.skipped.extend(
            tid for tid in target_ids if _should_skip_fetch(tid, force)
        )
        for key in keys:
            if key not in empirical.REPOS:
                result.failed[key] = "unknown source"
                continue
            repo = empirical.REPOS[key]
            dest = DATASETS_DIR / f"{key}.csv"
            if dest.is_file() and not force:
                result.skipped.append(key)
                continue
            if repo.get("url"):
                if empirical.download_file(repo["url"], dest):
                    result.fetched.append(key)
                elif empirical.materialize_curated_cache(key, dest):
                    print(
                        f"[Superblock Residual Engine] Download failed for {key}; "
                        "using PDG/literature fallback."
                    )
                    result.fetched.append(key)
                else:
                    result.failed[key] = "download failed"
            else:
                if empirical.materialize_curated_cache(key, dest):
                    result.fetched.append(key)
                else:
                    frame = empirical.load_or_download(key, force=force)
                    if frame is not None and len(frame):
                        result.skipped.append(key)
                    else:
                        result.failed[key] = "curated data unavailable"
        return result

    def archive_targets(target_ids: List[str]) -> List[Path]:
        moved: List[Path] = []
        ensure_project_dirs()
        for tid in target_ids:
            if not tid.startswith("empirical:"):
                continue
            key = tid.split(":", 1)[-1]
            for candidate in (
                DATASETS_DIR / f"{key}.csv",
                empirical.cache_path(key),
            ):
                if candidate.is_file():
                    dest = move_to_finished_archive(
                        candidate,
                        FINISHED_A_DIR,
                        also_search=(LEGACY_FINISHED_A_DIR,),
                    )
                    moved.append(dest)
                    break
        sync_archived_as_processed(
            [tid for tid in target_ids if tid.startswith("empirical:")],
            note="empirical archive",
        )
        return moved

    def run_action(action: str, target_ids: List[str], options: dict) -> None:
        import menus.empirical_tests.extension as empirical_tests_extension

        keys = [tid.split(":", 1)[-1] for tid in target_ids if tid.startswith("empirical:")]
        merged = {**options, "selected_sources": ",".join(keys)}
        empirical_tests_extension.run_action(action, options=merged)
        mark_processed(
            [tid for tid in target_ids if tid.startswith("empirical:")],
            note=f"empirical run: {action}",
        )

    return DatasetProvider(
        module_tag="EMPIRICAL_TESTS",
        title="Superblock Empirical Tests",
        intake_dir=DATASETS_DIR,
        list_fetchable=_targets,
        list_runnable=_targets,
        fetch_targets=fetch_targets,
        archive_targets=archive_targets,
        run_action=run_action,
        run_actions=list(__import__("empirical_tests_extension").MENU_ACTIONS),
    )


def _lhcb_provider() -> DatasetProvider:
    from menus.particle.lhcb.flavor import DEFAULT_DATA_PATH, DEFAULT_N_BINS, clip_n_bins
    from menus.particle.lhcb.fetcher import download_lhcb_bmeson_dataset

    DATASETS_DIR = PROJECT_ROOT / "datasets" / "lhcb"
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    default_path = DATASETS_DIR / "lhcb_bmeson_data.npy"

    def _targets() -> List[DatasetTarget]:
        path = default_path if default_path.is_file() else DEFAULT_DATA_PATH
        return _stamp_processed(
            [
                DatasetTarget(
                    id="lhcb:lhcb_bmeson_data",
                    module_tag="LHCB_FLAVOR",
                    label="LHCb B-meson C9 structured array",
                    state=_state_from_path(path, DATASETS_DIR),
                    path=path if path.is_file() else None,
                    group="particle",
                )
            ]
        )

    def fetch_targets(target_ids: List[str], options: dict) -> FetchResult:
        result = FetchResult(module_tag="LHCB_FLAVOR", fetched=[], skipped=[], failed={})
        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        n_bins = clip_n_bins(options.get("n_q2_bins") or DEFAULT_N_BINS)
        if _should_skip_fetch("lhcb:lhcb_bmeson_data", force):
            result.skipped.append("lhcb:lhcb_bmeson_data")
            return result
        if default_path.is_file() and not force:
            import numpy as np

            existing = np.load(default_path, allow_pickle=False)
            if len(existing) >= n_bins:
                result.skipped.append("lhcb:lhcb_bmeson_data")
                return result
        download_lhcb_bmeson_dataset(
            n_bins=n_bins,
            force_refresh=True,
            dest=default_path,
        )
        result.fetched.append("lhcb_bmeson_data")
        return result

    def archive_targets(target_ids: List[str]) -> List[Path]:
        ensure_project_dirs()
        if default_path.is_file():
            dest = move_to_finished_archive(
                default_path,
                FINISHED_A_DIR,
                also_search=(LEGACY_FINISHED_A_DIR,),
            )
            sync_archived_as_processed(["lhcb:lhcb_bmeson_data"], note="lhcb archive")
            return [dest]
        return []

    def run_action(action: str, target_ids: List[str], options: dict) -> None:
        import menus.particle.lhcb.extension as lhcb_flavor_extension

        merged = {**options, "data_file": str(default_path if default_path.is_file() else DEFAULT_DATA_PATH)}
        lhcb_flavor_extension.run_action(action, options=merged)
        mark_processed(["lhcb:lhcb_bmeson_data"], note=f"lhcb run: {action}")

    return DatasetProvider(
        module_tag="LHCB_FLAVOR",
        title="LHCb Flavor Residuals",
        intake_dir=DATASETS_DIR,
        list_fetchable=_targets,
        list_runnable=_targets,
        fetch_targets=fetch_targets,
        archive_targets=archive_targets,
        run_action=run_action,
        run_actions=[
            action
            for action in __import__("lhcb_flavor_extension").MENU_ACTIONS
        ],
    )


def _prime_past_provider() -> DatasetProvider:
    """AlterBBN v2.2 (pinned upstream tarball, built locally) + its cached Planck-input baseline."""
    from bbn_isolated import alterbbn

    DATASETS_DIR = alterbbn.PRIME_PAST_DATA
    BIN_ID = "prime_past:alterbbn_v2.2"
    BASE_ID = "prime_past:alterbbn_planck_baseline"

    def _baseline_csv() -> Optional[Path]:
        hits = sorted(alterbbn.CACHE_DIR.glob("alterbbn__sb_bbn__eta6.1140e-10_neff3.0440_tau877.75_fs1.csv"))
        return hits[0] if hits else None

    def _targets() -> List[DatasetTarget]:
        exe = alterbbn.find_binary(alterbbn.BINARY_NAME)
        full = alterbbn.find_binary(alterbbn.FULL_BINARY_NAME)
        base = _baseline_csv()
        return _stamp_processed([
            DatasetTarget(
                id=BIN_ID,
                module_tag="PRIME_PAST_HARMONIC",
                label="AlterBBN v2.2 binaries (sb_bbn.x + sb_bbn_full.x)",
                state=TargetState.CACHED if exe and full else TargetState.REMOTE,
                path=exe if exe and full else None,
                group="tool",
            ),
            DatasetTarget(
                id=BASE_ID,
                module_tag="PRIME_PAST_HARMONIC",
                label="AlterBBN standard BBN at Planck 2018 inputs (CSV)",
                state=_state_from_path(base, DATASETS_DIR),
                path=base,
                group="baseline",
            ),
        ])

    def fetch_targets(target_ids: List[str], options: dict) -> FetchResult:
        from menus.prime_past.extension import fetch_and_build

        result = FetchResult(module_tag="PRIME_PAST_HARMONIC", fetched=[], skipped=[], failed={})
        wanted = [t for t in target_ids if t in {BIN_ID, BASE_ID}]
        if not wanted:
            return result
        try:
            out = fetch_and_build(options)
        except alterbbn.AlterBBNUnavailable as exc:
            for tid in wanted:
                result.failed[tid.split(":", 1)[-1]] = str(exc)
            return result
        info = out["alterbbn"]
        if BIN_ID in wanted:
            (result.fetched if info.get("built") else result.skipped).append("alterbbn_v2.2")
        if BASE_ID in wanted:
            (result.skipped if out["baseline"].get("from_cache") else result.fetched).append(
                "alterbbn_planck_baseline"
            )
        return result

    def archive_targets(target_ids: List[str]) -> List[Path]:
        # Binaries and the baseline are tools/caches, not run inputs; nothing is moved.
        return []

    def run_action(action: str, target_ids: List[str], options: dict) -> None:
        import menus.prime_past.extension as prime_past_extension

        prime_past_extension.run_action(action, options=options)
        mark_processed([t for t in target_ids if t.startswith("prime_past:")], note=f"prime past run: {action}")

    import menus.prime_past.extension as _pp

    return DatasetProvider(
        module_tag="PRIME_PAST_HARMONIC",
        title="Prime Past Residuals",
        intake_dir=DATASETS_DIR,
        list_fetchable=_targets,
        list_runnable=_targets,
        fetch_targets=fetch_targets,
        archive_targets=archive_targets,
        run_action=run_action,
        run_actions=list(_pp.MENU_ACTIONS),
    )


def _desi_provider() -> DatasetProvider:
    from menus.astronomical.desi.fetcher import (
        BAO_DATA_DIR,
        DEFAULT_COBAYA_ROOT,
        DESI_DATASET_ID,
        ensure_desi_bao_data,
    )
    from menus.astronomical.desi.loader import list_desi_tracers

    intake = BAO_DATA_DIR.parent

    def _targets() -> List[DatasetTarget]:
        targets: List[DatasetTarget] = [
            DatasetTarget(
                id=DESI_DATASET_ID,
                module_tag="DESI_BAO",
                label="Cobaya DESI DR2 BAO tables (desi_bao_dr2)",
                state=_state_from_path(DEFAULT_COBAYA_ROOT, intake),
                path=DEFAULT_COBAYA_ROOT if DEFAULT_COBAYA_ROOT.is_dir() else None,
                group="bao_repo",
            )
        ]
        if DEFAULT_COBAYA_ROOT.is_dir():
            for tracer in list_desi_tracers(DEFAULT_COBAYA_ROOT.parent):
                targets.append(
                    DatasetTarget(
                        id=f"desi:{tracer}",
                        module_tag="DESI_BAO",
                        label=f"DESI DR2 tracer {tracer}",
                        state=TargetState.CACHED,
                        path=DEFAULT_COBAYA_ROOT,
                        group="tracer",
                    )
                )
        return _stamp_processed(targets)

    def fetch_targets(target_ids: List[str], options: dict) -> FetchResult:
        result = FetchResult(module_tag="DESI_BAO", fetched=[], skipped=[], failed={})
        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        wants_repo = any(
            tid in {DESI_DATASET_ID, "desi:bao_data_dr2"} or tid.startswith("desi:")
            for tid in target_ids
        )
        if not wants_repo:
            return result
        if _should_skip_fetch(DESI_DATASET_ID, force) and DEFAULT_COBAYA_ROOT.is_dir():
            result.skipped.append(DESI_DATASET_ID)
            return result
        try:
            ensure_desi_bao_data(force_refresh=force, auto_fetch=True)
            result.fetched.append("bao_data_dr2")
        except Exception as exc:
            result.failed[DESI_DATASET_ID] = str(exc)
        return result

    def archive_targets(target_ids: List[str]) -> List[Path]:
        return []

    def run_action(action: str, target_ids: List[str], options: dict) -> None:
        import menus.astronomical.desi.extension as desi_extension

        desi_extension.run_action(action, options=options)
        mark_processed([DESI_DATASET_ID], note=f"desi run: {action}")

    return DatasetProvider(
        module_tag="DESI_BAO",
        title="DESI DR2 BAO Residuals",
        intake_dir=intake,
        list_fetchable=_targets,
        list_runnable=_targets,
        fetch_targets=fetch_targets,
        archive_targets=archive_targets,
        run_action=run_action,
        run_actions=list(__import__("desi_bao_extension").MENU_ACTIONS),
    )


def _cern_opendata_provider() -> DatasetProvider:
    from menus.particle.cern import extension as cern_opendata_extension
    from menus.particle.cern.fetcher import (
        DATASETS_DIR,
        archive_used_datasets,
        cache_dir_for_target,
        list_dataset_files,
        pull_selected_targets,
    )
    from menus.particle.cern.manifest import CERN_MANIFEST

    def _targets() -> List[DatasetTarget]:
        targets: List[DatasetTarget] = []
        for key, spec in CERN_MANIFEST.items():
            dest = cache_dir_for_target(key)
            cached = dest.is_dir() and any(dest.rglob("*"))
            targets.append(
                DatasetTarget(
                    id=f"cern:{key}",
                    module_tag="CERN_OPENDATA",
                    label=spec["label"],
                    state=_state_from_path(dest if cached else None, DATASETS_DIR),
                    path=dest if cached else None,
                    group=str(spec.get("group", "cern")),
                )
            )
        for path in list_dataset_files():
            if path.name in {"done.txt", "record_metadata.json"}:
                continue
            rel = path.relative_to(DATASETS_DIR)
            stem = str(rel).replace("/", "_").replace(".", "_")
            targets.append(
                DatasetTarget(
                    id=f"cern:{stem}",
                    module_tag="CERN_OPENDATA",
                    label=path.name,
                    state=TargetState.CACHED,
                    path=path,
                    group="cached",
                )
            )
        return _stamp_processed(targets)

    def fetch_targets(target_ids: List[str], options: dict) -> FetchResult:
        result = FetchResult(module_tag="CERN_OPENDATA", fetched=[], skipped=[], failed={})
        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        names = [
            tid.split(":", 1)[-1]
            for tid in target_ids
            if tid.startswith("cern:") and not _should_skip_fetch(tid, force)
        ]
        result.skipped.extend(tid for tid in target_ids if _should_skip_fetch(tid, force))
        if not names:
            return result
        summary = pull_selected_targets(names, params=options, force_refresh=force)
        result.fetched.extend(summary.downloaded)
        result.skipped.extend(summary.skipped_done)
        result.skipped.extend(summary.skipped_existing)
        result.failed.update(summary.failed)
        return result

    def archive_targets(target_ids: List[str]) -> List[Path]:
        paths: List[Path] = []
        for tid in target_ids:
            if not tid.startswith("cern:"):
                continue
            key = tid.split(":", 1)[-1]
            if key in CERN_MANIFEST:
                candidate = cache_dir_for_target(key)
                if candidate.is_dir():
                    paths.append(candidate)
        moved = archive_used_datasets(paths)
        sync_archived_as_processed(
            [tid for tid in target_ids if tid.startswith("cern:")],
            note="cern archive",
        )
        return moved

    def run_action(action: str, target_ids: List[str], options: dict) -> None:
        merged = dict(options)
        if target_ids:
            key = target_ids[0].split(":", 1)[-1]
            if key in CERN_MANIFEST:
                merged.setdefault("target", key)
        cern_opendata_extension.run_action(action, options=merged)
        mark_processed(
            [tid for tid in target_ids if tid.startswith("cern:")],
            note=f"cern run: {action}",
        )

    return DatasetProvider(
        module_tag="CERN_OPENDATA",
        title="CERN Open Data",
        intake_dir=DATASETS_DIR,
        list_fetchable=_targets,
        list_runnable=_targets,
        fetch_targets=fetch_targets,
        archive_targets=archive_targets,
        run_action=run_action,
        run_actions=list(cern_opendata_extension.MENU_ACTIONS),
    )


def _ligo_target_options(target_ids: List[str], options: dict) -> dict:
    """event / strain_path implied by selected GWOSC targets.

    ``gwosc:GW150914`` → event; ``gwosc:events/GW150914/<file>`` → event + strain_path (one file
    selected); ``gwosc:event_catalog.json`` and other non-event files imply nothing. An explicit
    event (other than ALL/*/blank) or strain_path in ``options`` wins.
    """
    from menus.gravitic.ligo.fetcher import DATASETS_DIR, PRESETS

    out: dict = {}
    events: list[str] = []
    files: list[Path] = []
    for tid in target_ids or []:
        if not tid.startswith("gwosc:"):
            continue
        rest = tid.split(":", 1)[-1]
        parts = Path(rest).parts
        if parts and parts[0] == "events" and len(parts) >= 2:
            events.append(parts[1].upper())
            if len(parts) >= 3:
                files.append(DATASETS_DIR / rest)
        elif rest.upper() in PRESETS:
            continue  # a batch preset implies every cached event (event stays ALL)
        elif len(parts) == 1 and "." not in rest and rest:
            events.append(rest.upper())
    explicit_event = str(options.get("event") or "").strip().upper() not in {"", "ALL", "*"}
    uniq = sorted(set(events))
    if not explicit_event and len(uniq) == 1:
        out["event"] = uniq[0]
    if not str(options.get("strain_path") or "").strip() and len(files) == 1 and len(uniq) == 1:
        out["strain_path"] = str(files[0])
    return out


def _ligo_gwosc_provider() -> DatasetProvider:
    from menus.gravitic.ligo import extension as ligo_gwosc_extension
    from menus.gravitic.ligo.fetcher import (
        DATASETS_DIR,
        archive_used_datasets,
        list_cached_files,
        list_catalog_events,
        pull_selected_targets,
    )

    def _targets() -> List[DatasetTarget]:
        targets: List[DatasetTarget] = []
        try:
            events = list_catalog_events()
        except Exception:
            events = []
        cached_events = {path.parent.name for path in list_cached_files() if path.parent.parent.name == "events"}
        from menus.gravitic.ligo.fetcher import HEAVY_BBH_EVENTS

        n_cached = sum(1 for ev in HEAVY_BBH_EVENTS if ev in cached_events)
        targets.append(
            DatasetTarget(
                id="gwosc:HEAVY_BBH",
                module_tag="LIGO_GWOSC",
                label=f"HEAVY_BBH preset: {len(HEAVY_BBH_EVENTS)} loud heavy BBH events, 32 s 4 kHz ({n_cached} cached)",
                state=TargetState.CACHED if n_cached == len(HEAVY_BBH_EVENTS) else TargetState.REMOTE,
                path=None,
                group="preset",
            )
        )
        for event in events:
            state = TargetState.CACHED if event in cached_events else TargetState.REMOTE
            targets.append(
                DatasetTarget(
                    id=f"gwosc:{event}",
                    module_tag="LIGO_GWOSC",
                    label=f"{event} strain (GWOSC)",
                    state=state,
                    path=DATASETS_DIR / "events" / event if event in cached_events else None,
                    group="event",
                )
            )
        for path in list_cached_files():
            rel = path.relative_to(DATASETS_DIR)
            targets.append(
                DatasetTarget(
                    id=f"gwosc:{rel}",
                    module_tag="LIGO_GWOSC",
                    label=str(rel),
                    state=TargetState.CACHED,
                    path=path,
                    group="cached",
                )
            )
        return _stamp_processed(targets)

    def fetch_targets(target_ids: List[str], options: dict) -> FetchResult:
        result = FetchResult(module_tag="LIGO_GWOSC", fetched=[], skipped=[], failed={})
        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        names = [
            tid.split(":", 1)[-1]
            for tid in target_ids
            if tid.startswith("gwosc:") and not _should_skip_fetch(tid, force)
        ]
        result.skipped.extend(tid for tid in target_ids if _should_skip_fetch(tid, force))
        if not names:
            return result
        summary = pull_selected_targets(names, params=options, force_refresh=force)
        result.fetched.extend(summary.downloaded)
        result.skipped.extend(summary.skipped_done)
        result.skipped.extend(summary.skipped_existing)
        result.failed.update(summary.failed)
        return result

    def archive_targets(target_ids: List[str]) -> List[Path]:
        paths = []
        for tid in target_ids:
            if not tid.startswith("gwosc:"):
                continue
            name = tid.split(":", 1)[-1]
            candidate = DATASETS_DIR / name
            if candidate.is_file():
                paths.append(candidate)
        moved = archive_used_datasets(paths)
        sync_archived_as_processed(
            [tid for tid in target_ids if tid.startswith("gwosc:")],
            note="gwosc archive",
        )
        return moved

    def run_action(action: str, target_ids: List[str], options: dict) -> None:
        merged = dict(options)
        merged.update(_ligo_target_options(target_ids, merged))
        ligo_gwosc_extension.run_action(action, options=merged)
        mark_processed(
            [tid for tid in target_ids if tid.startswith("gwosc:")],
            note=f"ligo run: {action}",
        )

    return DatasetProvider(
        module_tag="LIGO_GWOSC",
        title="LIGO GWOSC (Strain Data)",
        intake_dir=DATASETS_DIR,
        list_fetchable=_targets,
        list_runnable=_targets,
        fetch_targets=fetch_targets,
        archive_targets=archive_targets,
        run_action=run_action,
        run_actions=list(ligo_gwosc_extension.MENU_ACTIONS),
    )


def _lisa_provider() -> DatasetProvider:
    from menus.gravitic.lisa import extension as lisa_pre_runs_extension
    from menus.gravitic.lisa.fetcher import (
        DATASETS_DIR,
        LISA_OSDF_TARGETS,
        archive_used_datasets,
        list_cached_files,
        pull_selected_targets,
    )

    def _targets() -> List[DatasetTarget]:
        targets: List[DatasetTarget] = []
        for key, uri in LISA_OSDF_TARGETS.items():
            name = uri.rsplit("/", 1)[-1]
            path = DATASETS_DIR / "osdf" / name
            targets.append(
                DatasetTarget(
                    id=f"lisa:{key}",
                    module_tag="LISA_PRE_RUNS",
                    label=f"{key} ({uri})",
                    state=_state_from_path(path, DATASETS_DIR),
                    path=path if path.is_file() else None,
                    group="osdf",
                )
            )
        for path in list_cached_files():
            rel = path.relative_to(DATASETS_DIR)
            targets.append(
                DatasetTarget(
                    id=f"lisa:{rel}",
                    module_tag="LISA_PRE_RUNS",
                    label=str(rel),
                    state=TargetState.CACHED,
                    path=path,
                    group="cached",
                )
            )
        return _stamp_processed(targets)

    def fetch_targets(target_ids: List[str], options: dict) -> FetchResult:
        result = FetchResult(module_tag="LISA_PRE_RUNS", fetched=[], skipped=[], failed={})
        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        names = [
            tid.split(":", 1)[-1]
            for tid in target_ids
            if tid.startswith("lisa:") and not _should_skip_fetch(tid, force)
        ]
        result.skipped.extend(tid for tid in target_ids if _should_skip_fetch(tid, force))
        if not names:
            return result
        summary = pull_selected_targets(names, force_refresh=force)
        result.fetched.extend(summary.downloaded)
        result.skipped.extend(summary.skipped_done)
        result.skipped.extend(summary.skipped_existing)
        result.failed.update(summary.failed)
        return result

    def archive_targets(target_ids: List[str]) -> List[Path]:
        paths = []
        for tid in target_ids:
            if not tid.startswith("lisa:"):
                continue
            name = tid.split(":", 1)[-1]
            candidate = DATASETS_DIR / name
            if candidate.is_file():
                paths.append(candidate)
        moved = archive_used_datasets(paths)
        sync_archived_as_processed(
            [tid for tid in target_ids if tid.startswith("lisa:")],
            note="lisa archive",
        )
        return moved

    def run_action(action: str, target_ids: List[str], options: dict) -> None:
        merged = dict(options)
        if target_ids and action == "Pull OSDF Target":
            merged.setdefault("query", target_ids[0].split(":", 1)[-1])
        lisa_pre_runs_extension.run_action(action, options=merged)
        mark_processed(
            [tid for tid in target_ids if tid.startswith("lisa:")],
            note=f"lisa run: {action}",
        )

    return DatasetProvider(
        module_tag="LISA_PRE_RUNS",
        title="LISA Pre-runs",
        intake_dir=DATASETS_DIR,
        list_fetchable=_targets,
        list_runnable=_targets,
        fetch_targets=fetch_targets,
        archive_targets=archive_targets,
        run_action=run_action,
        run_actions=list(lisa_pre_runs_extension.MENU_ACTIONS),
    )


_PROVIDERS: Optional[Dict[str, DatasetProvider]] = None


def get_providers() -> Dict[str, DatasetProvider]:
    global _PROVIDERS
    if _PROVIDERS is not None:
        return _PROVIDERS

    providers: Dict[str, DatasetProvider] = {
        "SPARC": _sparc_provider(),
        "FRB_DM": _frb_provider(),
        "EMPIRICAL_TESTS": _empirical_provider(),
        "LHCB_FLAVOR": _lhcb_provider(),
        "LIGO_GWOSC": _ligo_gwosc_provider(),
        "LISA_PRE_RUNS": _lisa_provider(),
        "CERN_OPENDATA": _cern_opendata_provider(),
        "PRIME_PAST_HARMONIC": _prime_past_provider(),
    }
    try:
        providers["DESI_BAO"] = _desi_provider()
    except ImportError:
        pass
    planck = _planck_provider()
    if planck is not None:
        providers["PLANCK_CMB"] = planck
    _PROVIDERS = providers
    return providers


def list_modules() -> List[DatasetProvider]:
    return list(get_providers().values())


def get_provider(module_tag: str) -> Optional[DatasetProvider]:
    from sb_shared.legacy_names import canonical_module_tag
    module_tag = canonical_module_tag(module_tag)
    return get_providers().get(module_tag)


def fetch_selected(module_tag: str, target_ids: List[str], options: dict | None = None) -> FetchResult:
    from sb_shared.dataset_ledger import filter_for_fetch

    provider = get_provider(module_tag)
    if provider is None:
        raise KeyError(f"Unknown module: {module_tag}")

    options = options or {}
    force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
    fetchable, skipped_processed = filter_for_fetch(target_ids, force_refresh=force)

    if skipped_processed:
        print(
            f"[DATASET LEDGER] Skipping {len(skipped_processed)} already-processed "
            f"target(s) (use force_refresh=yes to override)"
        )

    if not fetchable:
        return FetchResult(
            module_tag=module_tag,
            fetched=[],
            skipped=skipped_processed,
            failed={},
        )

    result = provider.fetch_targets(fetchable, options)
    result.skipped.extend(skipped_processed)
    return result


def archive_selected(module_tag: str, target_ids: List[str]) -> List[Path]:
    provider = get_provider(module_tag)
    if provider is None:
        raise KeyError(f"Unknown module: {module_tag}")
    return provider.archive_targets(target_ids)


def run_selected(
    module_tag: str,
    action: str,
    target_ids: List[str],
    options: dict | None = None,
) -> None:
    provider = get_provider(module_tag)
    if provider is None or provider.run_action is None:
        raise KeyError(f"Module {module_tag} does not support dataset runs")
    provider.run_action(action, target_ids, options or {})