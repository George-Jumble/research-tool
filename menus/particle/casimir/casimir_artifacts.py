"""
Casimir artifact layout: one day folder, series token on every filename.

    artifacts/casimir/27-07-2026/
        casimir__dielptag005__report__20-56_tsb_casimir.json
        casimir__dielptal005__report__20-56_tsb_casimir.json
        ...

A series run shares ``HH-MM_tsb_casimir`` so all materials from one scan
are globbable without a folder per file.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tav_shared.artifact_paths import TestSlug, slugify, test_artifacts_dir

_FOLDER_RE = re.compile(r"^(\d{2}-\d{2}-\d{4})_(\d{2}-\d{2})_(.+)$")
_DAY_ONLY_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")


@dataclass
class CasimirArtifactSeries:
    when: datetime
    day_dir: Path
    token: str


_ACTIVE: CasimirArtifactSeries | None = None


def casimir_series_label(output_prefix: str | None) -> str:
    slug = slugify(output_prefix or "tsb_casimir")
    if slug.startswith("blind"):
        return slug[:48] or "blind"
    if slug.startswith("tsb_casimir"):
        return "tsb_casimir"
    if slug.startswith("integrity"):
        return "integrity"
    head = slug.split("_", 1)[0]
    return head or "tsb_casimir"


def start_casimir_artifact_series(
    output_prefix: str = "tsb_casimir",
    *,
    when: datetime | None = None,
) -> CasimirArtifactSeries:
    """Begin a series: date folder ``DD-MM-YYYY``, token ``HH-MM_tsb_casimir``."""
    global _ACTIVE
    stamp = when or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    stamp = stamp.astimezone(timezone.utc)
    token = f"{stamp.strftime('%H-%M')}_{casimir_series_label(output_prefix)}"
    day_dir = test_artifacts_dir(TestSlug.CASIMIR) / stamp.strftime("%d-%m-%Y")
    day_dir.mkdir(parents=True, exist_ok=True)
    _ACTIVE = CasimirArtifactSeries(when=stamp, day_dir=day_dir, token=token)
    print(f"[CASIMIR ARTIFACTS] Series {token} → {day_dir}/")
    return _ACTIVE


def current_casimir_series() -> CasimirArtifactSeries | None:
    return _ACTIVE


def end_casimir_artifact_series() -> None:
    global _ACTIVE
    _ACTIVE = None


def ensure_casimir_series(output_prefix: str = "tsb_casimir") -> CasimirArtifactSeries:
    return _ACTIVE or start_casimir_artifact_series(output_prefix)


def casimir_output_path(
    dataset_slug: str,
    artifact_kind: str,
    ext: str,
    *,
    output_prefix: str = "tsb_casimir",
    series: CasimirArtifactSeries | None = None,
) -> Path:
    """Write into the day folder; append the series token to the filename."""
    active = series or ensure_casimir_series(output_prefix)
    dataset = slugify(dataset_slug) or "default"
    kind = slugify(artifact_kind) or "output"
    extension = str(ext).lstrip(".").lower() or "bin"
    name = f"casimir__{dataset}__{kind}__{active.token}.{extension}"
    dest = active.day_dir / name
    if dest.exists():
        stamp = active.when.strftime("%H%M%S")
        name = f"casimir__{dataset}__{kind}__{active.token}_{stamp}.{extension}"
        dest = active.day_dir / name
    return dest


def migrate_casimir_run_folders(*, dry_run: bool = False) -> dict[str, int]:
    """
    Collapse per-material run folders into ``artifacts/casimir/DD-MM-YYYY/``
    and append ``HH-MM_tsb_casimir`` (or equivalent) to each filename.
    """
    root = test_artifacts_dir(TestSlug.CASIMIR, create=True)
    stats = {"moved": 0, "skipped": 0, "removed_dirs": 0, "errors": 0}
    for item in sorted(root.iterdir()):
        if not item.is_dir() or _DAY_ONLY_RE.match(item.name):
            continue
        match = _FOLDER_RE.match(item.name)
        if not match:
            stats["skipped"] += 1
            continue
        day, hhmm, rest = match.groups()
        label = casimir_series_label(rest)
        token = f"{hhmm}_{label}"
        dest_dir = root / day
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(item.iterdir()):
            if not src.is_file():
                continue
            stem, suffix = src.stem, src.suffix
            if token in stem:
                new_name = src.name
            else:
                new_name = f"{stem}__{token}{suffix}"
            dest = dest_dir / new_name
            if dest.exists():
                stats["skipped"] += 1
                continue
            if dry_run:
                print(f"[dry-run] {src.relative_to(root)} → {day}/{new_name}")
                stats["moved"] += 1
                continue
            try:
                shutil.move(str(src), str(dest))
                stats["moved"] += 1
            except OSError:
                stats["errors"] += 1
        if not dry_run:
            try:
                leftover = list(item.iterdir())
                if not leftover:
                    item.rmdir()
                    stats["removed_dirs"] += 1
            except OSError:
                stats["errors"] += 1
    return stats
