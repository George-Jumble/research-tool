"""
Processed-dataset archive paths for the Superblock research tool.

Local machine keeps a staging mirror under the research_tool tree.
Finished archives can be transferred via SFTP/SCP to a configured remote host
when enabled.

Remote transfer is currently disabled (local staging only).

Existing archives in the pre-rename staging directory are reused in place;
see ``sb_shared.legacy_names``.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

from sb_shared.legacy_names import LEGACY_PROTON_DIRNAME, resolve_staging_dir

# Repository root (parent of sb_shared/)
SUPERBLOCK_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SUPERBLOCK_ROOT
DATASETS_ROOT = SUPERBLOCK_ROOT / "datasets"
ARTIFACTS_ROOT = SUPERBLOCK_ROOT / "artifacts"

# Local staging (no longer assumes ~/Public/ProtonDrive on this machine)
STAGING_DIR = resolve_staging_dir(PROJECT_ROOT)
FINISHED_A_DIR = STAGING_DIR / "finishedA"
FINISHED2_DIR = STAGING_DIR / "finished2"
FINISHED_DIR = STAGING_DIR / "finished"

LEGACY_FINISHED_A_DIR = PROJECT_ROOT / "finishedA"
LEGACY_FINISHED2_DIR = PROJECT_ROOT / "finished2"
LEGACY_FINISHED_DIR = PROJECT_ROOT / "finished"

# Also search old local ProtonDrive path if it still exists on this host
_LEGACY_PROTON = (Path.home() / "Public" / "ProtonDrive" / LEGACY_PROTON_DIRNAME).resolve()
_LEGACY_PROTON_A = _LEGACY_PROTON / "finishedA"
_LEGACY_PROTON_2 = _LEGACY_PROTON / "finished2"
_LEGACY_PROTON_F = _LEGACY_PROTON / "finished"


def _scripts_on_path() -> None:
    """Allow importing Research/scripts/remote_transfer.py from research_tool."""
    scripts = Path(__file__).resolve().parents[2]  # .../Research/scripts
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))


def remote_transfer_enabled() -> bool:
    # Remote SSH host updates disabled — local staging only.
    return False
    # return os.environ.get("SB_REMOTE_TRANSFER", "1").strip() not in (
    #     "0",
    #     "false",
    #     "no",
    #     "off",
    # )


def _remote():
    _scripts_on_path()
    import remote_transfer as rt

    return rt


def ensure_project_dirs() -> None:
    """Create local staging dirs and ensure remote Public tree exists."""
    for path in (STAGING_DIR, FINISHED_A_DIR, FINISHED2_DIR, FINISHED_DIR):
        path.mkdir(parents=True, exist_ok=True)

    # Remote SSH mkdir disabled — local staging dirs only.
    # if not remote_transfer_enabled():
    #     return
    # try:
    #     rt = _remote()
    #     for sub in ("finishedA", "finished2", "finished"):
    #         remote_dir = rt.remote_tav_subdir(sub)
    #         result = rt.sftp_mkdir_p(remote_dir)
    #         if not result.ok:
    #             print(
    #                 f"[Superblock Residual Engine] Warning: could not create remote {remote_dir}: "
    #                 f"{result.stderr or result.stdout}"
    #             )
    # except Exception as e:
    #     print(f"[Superblock Residual Engine] Warning: remote mkdir skipped ({e})")


def _dir_exists(path: Path) -> bool:
    """True when ``path`` is a directory. A stale mount (EIO) counts as absent."""
    try:
        return path.is_dir()
    except OSError:
        return False


def finished_a_search_dirs() -> tuple[Path, ...]:
    """Archive locations for SPARC / FRB CSV restores (local staging + legacy)."""
    dirs = [FINISHED_A_DIR, LEGACY_FINISHED_A_DIR]
    if _dir_exists(_LEGACY_PROTON_A):
        dirs.append(_LEGACY_PROTON_A)
    return tuple(dirs)


def finished_fits_search_dirs() -> tuple[Path, ...]:
    """Archive locations for Planck / integrator FITS restores."""
    dirs = [
        FINISHED2_DIR,
        FINISHED_DIR,
        LEGACY_FINISHED2_DIR,
        LEGACY_FINISHED_DIR,
    ]
    if _dir_exists(_LEGACY_PROTON_2):
        dirs.append(_LEGACY_PROTON_2)
    if _dir_exists(_LEGACY_PROTON_F):
        dirs.append(_LEGACY_PROTON_F)
    return tuple(dirs)


_ARCHIVE_STAMP_RE = re.compile(r"^(.+)_(\d{8}_\d{6})$")


def logical_archive_key(filename: str | Path) -> str:
    """Normalize a filename so timestamp-suffixed archives group with the base dataset."""
    path = Path(filename)
    stem = path.stem
    match = _ARCHIVE_STAMP_RE.match(stem)
    if match:
        stem = match.group(1)
    return f"{stem}{path.suffix.lower()}"


def delete_previous_archived_runs(
    filename: str | Path,
    archive_dir: Path | str,
    *,
    also_search: tuple[Path, ...] = (),
) -> list[Path]:
    """
    Delete prior archived copies of the same logical dataset before saving the current run.

    Removes exact-name matches and ``{stem}_YYYYMMDD_HHMMSS{suffix}`` variants in
    ``archive_dir`` and any optional ``also_search`` directories (e.g. legacy finished/).
    """
    key = logical_archive_key(filename)
    removed: list[Path] = []
    seen: set[Path] = set()

    for directory in (Path(archive_dir), *also_search):
        if not _dir_exists(directory):
            continue
        for candidate in directory.iterdir():
            if not candidate.is_file() or candidate in seen:
                continue
            if logical_archive_key(candidate) != key:
                continue
            candidate.unlink()
            seen.add(candidate)
            removed.append(candidate)

    if removed:
        names = ", ".join(p.name for p in removed[:4])
        if len(removed) > 4:
            names += f", ... (+{len(removed) - 4})"
        print(
            f"[Superblock Residual Engine] Removed {len(removed)} prior archive(s) for "
            f"{Path(filename).name}: {names}"
        )
    return removed


def archive_destination(
    src: Path | str,
    archive_dir: Path | str,
    *,
    also_search: tuple[Path, ...] = (),
) -> Path:
    """Return the local staging destination after deleting previous archives."""
    src = Path(src)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    delete_previous_archived_runs(src.name, archive_dir, also_search=also_search)
    return archive_dir / src.name


def _subdir_name(archive_dir: Path) -> str:
    name = archive_dir.name
    if name in ("finishedA", "finished2", "finished"):
        return name
    # fallback
    return "finished"


def push_to_remote_public(
    local_file: Path | str,
    archive_dir: Path | str,
) -> str | None:
    """
    SFTP/SCP one staged archive file to a configured remote archive path.

    Returns remote path on success, None if remote transfer disabled.
    Raises RuntimeError on hard failure when transfer is enabled.
    """
    # Remote SSH/SFTP upload disabled.
    return None
    # if not remote_transfer_enabled():
    #     return None
    #
    # local_file = Path(local_file)
    # archive_dir = Path(archive_dir)
    # rt = _remote()
    # sub = _subdir_name(archive_dir)
    # remote_path = rt.remote_tav_subdir(sub, local_file.name)
    # print(f"[Superblock Residual Engine] SFTP → {rt.remote_spec()}:{remote_path}")
    # result = rt.sftp_put_file(local_file, remote_path)
    # if not result.ok:
    #     raise RuntimeError(
    #         f"SFTP upload failed for {local_file.name}: "
    #         f"{result.stderr or result.stdout or result.command}"
    #     )
    # print(f"[Superblock Residual Engine] Uploaded {local_file.name} to remote Public archive")
    # return remote_path


def move_to_finished_archive(
    src: Path | str,
    archive_dir: Path | str,
    *,
    also_search: tuple[Path, ...] = (),
) -> Path:
    """
    Move ``src`` into local staging finished folder, then upload to a configured
    remote archive path when transfer is enabled.
    """
    src = Path(src)
    dest = archive_destination(src, archive_dir, also_search=also_search)
    shutil.move(str(src), str(dest))
    # Remote SSH push disabled — archive stays in local staging only.
    # try:
    #     push_to_remote_public(dest, archive_dir)
    # except Exception as e:
    #     print(f"[Superblock Residual Engine] Warning: remote Public transfer failed: {e}")
    #     print(f"[Superblock Residual Engine] File remains in local staging: {dest}")
    return dest


def sync_staging_tree_to_remote() -> dict:
    """
    Push the local staging tree to a configured remote archive target (rsync over ssh).
    """
    # Remote SSH rsync disabled.
    return {"ok": True, "skipped": True, "reason": "remote SSH updates disabled"}
    # if not remote_transfer_enabled():
    #     return {"ok": False, "error": "SB_REMOTE_TRANSFER disabled"}
    # ensure_project_dirs()
    # rt = _remote()
    # result = rt.rsync_to_remote(STAGING_DIR, rt.REMOTE_TAV_PROJECT)
    # return {
    #     "ok": result.ok,
    #     "remote": result.remote_path,
    #     "stderr": result.stderr,
    #     "stdout": result.stdout,
    #     "returncode": result.returncode,
    # }
