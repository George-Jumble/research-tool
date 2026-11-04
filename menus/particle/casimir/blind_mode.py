"""
Blind-analysis mode flags, report prefixing, lock, and delayed comparison.

Blind results must be locked before any informed-vs-blind table is written.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tav_shared.artifact_paths import TestSlug, test_artifacts_dir

LOCK_NAME = "blind_lock.json"


def is_blind(options: dict[str, Any] | None = None) -> bool:
    if options:
        raw = str(options.get("blind") or options.get("analysis_mode") or "").strip().lower()
        if raw in {"1", "true", "yes", "on", "blind"}:
            return True
        if raw in {"0", "false", "no", "off", "informed"}:
            return False
    return str(os.getenv("TAV_BLIND", "")).strip().lower() in {"1", "true", "yes", "on"}


def set_blind(enabled: bool) -> None:
    os.environ["TAV_BLIND"] = "1" if enabled else "0"


def apply_blind_prefix(output_prefix: str | None) -> str:
    prefix = (output_prefix or "tsb_casimir").strip() or "tsb_casimir"
    if not prefix.startswith("blind_"):
        return f"blind_{prefix}"
    return prefix


def lock_path() -> Path:
    return test_artifacts_dir(TestSlug.CASIMIR) / LOCK_NAME


def find_blind_reports() -> list[Path]:
    root = test_artifacts_dir(TestSlug.CASIMIR, create=False)
    if not root.is_dir():
        return []
    hits: list[Path] = []
    for path in root.rglob("*.json"):
        name = path.name.lower()
        if "blind_" in name or name.startswith("blind_"):
            if path.name == LOCK_NAME:
                continue
            if "comparison" in name:
                continue
            hits.append(path)
    return sorted(hits)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_blind_results() -> dict[str, Any]:
    """Freeze current blind reports so later informed runs cannot rewrite them."""
    reports = find_blind_reports()
    if not reports:
        raise FileNotFoundError(
            "No blind reports found under artifacts/casimir/. Run with --blind first."
        )
    payload = {
        "locked": True,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "n_reports": len(reports),
        "reports": [
            {"path": str(p.resolve()), "sha256": _file_digest(p), "name": p.name}
            for p in reports
        ],
    }
    dest = lock_path()
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[BLIND] Locked {len(reports)} report(s) → {dest}")
    return payload


def load_lock() -> dict[str, Any] | None:
    path = lock_path()
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def verify_lock_intact(lock: dict[str, Any]) -> None:
    """Refuse comparison if locked files were altered after lock."""
    for row in lock.get("reports") or []:
        path = Path(row["path"])
        if not path.is_file():
            raise RuntimeError(f"Locked blind report missing: {path}")
        digest = _file_digest(path)
        if digest != row.get("sha256"):
            raise RuntimeError(
                f"Locked blind report was modified after lock: {path}. "
                "Comparison aborted."
            )


def _label_from_report(payload: dict[str, Any], path: Path) -> str:
    return str(payload.get("data_label") or path.stem.split("__")[1] if "__" in path.stem else path.stem)


def _feature_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {feat.get("type"): feat for feat in payload.get("detected_features") or [] if feat.get("type")}


def write_blind_comparison_table() -> Path:
    """Side-by-side informed vs blind table. Requires an intact lock."""
    lock = load_lock()
    if not lock or not lock.get("locked"):
        raise RuntimeError(
            "Blind results are not locked. After the blind run, execute "
            "`python research_tool.py --lock-blind` (or the Casimir lock action) "
            "before requesting a comparison."
        )
    verify_lock_intact(lock)

    from menus.particle.casimir.casimir_artifacts import casimir_output_path, ensure_casimir_series

    blind_by_label: dict[str, dict[str, Any]] = {}
    for row in lock["reports"]:
        path = Path(row["path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("analysis_mode") not in {None, "blind"} and "blind_" not in path.name:
            continue
        label = _label_from_report(payload, path)
        blind_by_label[label] = payload

    informed: dict[str, dict[str, Any]] = {}
    root = test_artifacts_dir(TestSlug.CASIMIR, create=False)
    for path in sorted(root.rglob("*__report__*.json")):
        if "blind_" in path.name.lower() or "comparison" in path.name.lower():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("analysis_mode") == "blind":
            continue
        label = _label_from_report(payload, path)
        informed[label] = payload

    rows = []
    for label in sorted(set(blind_by_label) | set(informed)):
        b = _feature_map(blind_by_label.get(label) or {})
        i = _feature_map(informed.get(label) or {})
        b_per = b.get("periodicity_search") or {}
        i_clk = i.get("seven_phase_clockwork") or {}
        b_tav = (blind_by_label.get(label) or {}).get("tav_harmonics")
        i_tav = (informed.get(label) or {}).get("tav_harmonics") or {}
        rows.append(
            {
                "data_label": label,
                "blind_strongest_period": b_per.get("strongest_period"),
                "blind_strongest_frequency": b_per.get("strongest_frequency"),
                "blind_periodicity_p": b_per.get("p_value_holm", b_per.get("p_value")),
                "informed_expected_1_over_7": i_clk.get("expected_freq_1_over_7"),
                "informed_observed_frequency": i_clk.get("observed_frequency"),
                "informed_1_over_7_p_holm": i_clk.get("p_value_holm", i_clk.get("p_value")),
                "informed_tav_resonance_detected": (i_tav or {}).get("tav_resonance_detected"),
                "blind_has_tav_block": b_tav is not None,
                "informed_present": label in informed,
                "blind_present": label in blind_by_label,
            }
        )

    ensure_casimir_series("blind_compare")
    dest = casimir_output_path("blind_vs_informed", "comparison", "json", output_prefix="blind_compare")
    table = {
        "locked_at": lock.get("locked_at"),
        "n_blind": len(blind_by_label),
        "n_informed": len(informed),
        "rows": rows,
        "note": (
            "Comparison released only after lock_blind_results(). "
            "Blind periodicity has no preferred 1/7 frequency."
        ),
    }
    dest.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    _print_comparison(rows)
    print(f"[BLIND] Comparison table: {dest}")
    return dest


def _print_comparison(rows: list[dict[str, Any]]) -> None:
    print("\n=== Blind vs informed (post-lock) ===")
    print(
        f"{'label':<16} {'blind_P':>10} {'blind_p':>10} "
        f"{'inf_f1/7':>10} {'inf_pHolm':>10} {'tav?':>6}"
    )
    for row in rows:
        print(
            f"{str(row['data_label'])[:16]:<16} "
            f"{_fmt(row['blind_strongest_period']):>10} "
            f"{_fmt(row['blind_periodicity_p']):>10} "
            f"{_fmt(row['informed_expected_1_over_7']):>10} "
            f"{_fmt(row['informed_1_over_7_p_holm']):>10} "
            f"{str(row['informed_tav_resonance_detected']):>6}"
        )
    print()


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
