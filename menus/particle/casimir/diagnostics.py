"""
TSB Casimir falsification diagnostics: observable classification, hash ledger,
family preregistration, and console/JSON extras for controls + audits.

This module does not score p-values. Tests live in test_framework.py;
bound-energy residuals live in bound_energy.py.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from tav_shared.artifact_paths import ARTIFACTS_DIR

TSB_TEST_RESULTS_DIR = ARTIFACTS_DIR / "tsb_test_results"
HASH_LEDGER_PATH = TSB_TEST_RESULTS_DIR / "casimir_hash_ledger.jsonl"
PREREGISTER_PATH = TSB_TEST_RESULTS_DIR / "preregister_casimir_families.json"

TSB_EVIDENCE_ALPHA = 0.05

FAMILY_A_LABELS: tuple[str, ...] = (
    "dielPTAg005",
    "dielPTAl005",
    "dielPTGold005",
    "dielPTLi005",
)
FAMILY_B_LABELS: tuple[str, ...] = (
    "dielPTBen_Au",
    "dielPTCyc_Au",
    "dielPTOct_Au",
)

PREREGISTER_TESTS: tuple[str, ...] = (
    "discrete_steps",
    "hysteresis",
    "periodicity",
    "log_modulations",
    "powerlaw_index",
    "anisotropy_channels",
)

_FREQ_TOKENS = (
    "matsubara",
    "frequency",
    "freq",
    "xi_n",
    "xi",
    "omega",
    "wavenumber",
)
_SEP_TOKENS = (
    "separation",
    "distance",
    "gap",
    "displacement",
    "sep",
)
_FORCE_TOKENS = ("force", "pressure", "stress", "pn")
_ENERGY_TOKENS = ("energy", "free_energy", "delta_e", "e_obs", "e_lifshitz")
_DIELECTRIC_TOKENS = ("dielectric", "permittivity", "eps_pri", "eps_")
_RESIDUAL_TOKENS = ("delta_e", "residual", "bound_energy")

_TWO_PASS_FWD = ("forward", "approach")
_TWO_PASS_REV = ("reverse", "retract")
_TWO_PASS_UP = ("up",)
_TWO_PASS_DOWN = ("down",)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_name(name: str) -> str:
    return str(name or "").strip().lower()


def _has_token(name: str, tokens: Iterable[str]) -> bool:
    lowered = _norm_name(name)
    return any(tok in lowered for tok in tokens)


def dielectric_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None:
        return []
    cols: list[str] = []
    for col in df.columns:
        text = str(col)
        low = text.lower()
        if low.startswith("dielectric_response"):
            cols.append(text)
        elif low in {"eps_pri", "eps"} or low.startswith("eps_"):
            cols.append(text)
    return cols


def aux_dielectric_columns(
    df: pd.DataFrame | None,
    *,
    primary: str | None = None,
) -> list[str]:
    cols = dielectric_columns(df)
    if primary is not None:
        cols = [c for c in cols if c != primary]
    else:
        cols = [c for c in cols if str(c).lower() != "dielectric_response_primary"]
    return cols


def n_aux_dielectric_channels(
    df: pd.DataFrame | None,
    *,
    primary: str | None = None,
) -> int:
    return len(aux_dielectric_columns(df, primary=primary))


def classify_x_kind(param_col: str) -> str:
    name = _norm_name(param_col)
    if _has_token(name, _FREQ_TOKENS) or name in {"matsubara_frequency"}:
        return "frequency"
    if _has_token(name, _SEP_TOKENS) or name in {"d", "separation_nm"}:
        return "separation"
    if "magnetic" in name or name in {"b", "field"}:
        return "field"
    if name in {"time", "time_s", "t"}:
        return "time"
    return "unknown"


def classify_y_kind(value_col: str) -> str:
    name = _norm_name(value_col)
    if _has_token(name, _RESIDUAL_TOKENS) or name in {"delta_e"}:
        return "residual"
    if _has_token(name, _DIELECTRIC_TOKENS) or name.startswith("dielectric_response"):
        return "dielectric"
    if _has_token(name, _FORCE_TOKENS) or name in {"force_pn", "f"}:
        return "force"
    if _has_token(name, _ENERGY_TOKENS):
        return "energy"
    return "unknown"


def observable_kind(param_col: str, value_col: str) -> str:
    y_kind = classify_y_kind(value_col)
    if y_kind in {"dielectric", "force", "energy", "residual"}:
        return y_kind
    x_kind = classify_x_kind(param_col)
    if x_kind == "frequency" and y_kind == "unknown":
        return "dielectric"
    return y_kind


def powerlaw_index_is_legal(param_col: str, value_col: str) -> bool:
    return classify_x_kind(param_col) == "separation" and classify_y_kind(value_col) in {
        "force",
        "energy",
    }


def expected_lifshitz_index(value_col: str) -> float | None:
    y_kind = classify_y_kind(value_col)
    if y_kind == "force":
        return -4.0
    if y_kind == "energy":
        return -3.0
    return None


def applicable_tests(
    *,
    param_col: str,
    value_col: str,
    df: pd.DataFrame | None = None,
) -> list[str]:
    tests = ["discrete_steps", "hysteresis", "periodicity", "log_modulations"]
    if powerlaw_index_is_legal(param_col, value_col):
        tests.append("powerlaw_index")
    n_diel = len(dielectric_columns(df))
    if n_diel >= 2:
        tests.append("anisotropy_channels")
    return tests


def classify_casimir_table(
    df: pd.DataFrame,
    param_col: str,
    value_col: str,
) -> dict[str, Any]:
    kind = observable_kind(param_col, value_col)
    n_aux = n_aux_dielectric_channels(df, primary=value_col)
    n_diel = len(dielectric_columns(df))
    tests = applicable_tests(param_col=param_col, value_col=value_col, df=df)
    return {
        "observable_kind": kind,
        "x_kind": classify_x_kind(param_col),
        "y_kind": classify_y_kind(value_col),
        "n_aux_dielectric_channels": int(n_aux),
        "n_dielectric_columns": int(n_diel),
        "applicable_tests": tests,
        "powerlaw_index_legal": powerlaw_index_is_legal(param_col, value_col),
        "expected_lifshitz_index": expected_lifshitz_index(value_col),
    }


def print_casimir_load_context(
    *,
    observable_kind_value: str,
    n_aux: int,
    applicable: Sequence[str],
) -> None:
    tests = ",".join(applicable) if applicable else "(none)"
    print(
        f"[CASIMIR LOAD] observable_kind={observable_kind_value} "
        f"n_aux_dielectric_channels={int(n_aux)} "
        f"applicable_tests={tests}"
    )


def family_label_from_name(name: str) -> str | None:
    stem = Path(str(name)).stem
    for label in FAMILY_A_LABELS + FAMILY_B_LABELS:
        if label.lower() in stem.lower() or stem.lower() in label.lower():
            return label
    return None


def family_of_label(label: str, lock: dict[str, Any] | None = None) -> str | None:
    lock = lock or load_family_lock()
    fam_a = {str(x) for x in (lock or {}).get("family_A") or FAMILY_A_LABELS}
    fam_b = {str(x) for x in (lock or {}).get("family_B") or FAMILY_B_LABELS}
    resolved = family_label_from_name(label) or label
    if resolved in fam_a or any(x.lower() in str(label).lower() for x in fam_a):
        return "A"
    if resolved in fam_b or any(x.lower() in str(label).lower() for x in fam_b):
        return "B"
    return None


def preregister_split(
    *,
    relock: bool = False,
    path: Path | None = None,
    family_a: Sequence[str] | None = None,
    family_b: Sequence[str] | None = None,
    alpha: float = TSB_EVIDENCE_ALPHA,
    tests: Sequence[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """
    Write the family lockfile only if it does not exist, or if ``relock`` is set.

    Scoring a new folder must load this lock first and attach it to the report.
    Do not infer families after looking at p-values.
    """
    dest = path or PREREGISTER_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not relock:
        return json.loads(dest.read_text(encoding="utf-8"))
    payload = {
        "family_A": list(family_a if family_a is not None else FAMILY_A_LABELS),
        "family_B": list(family_b if family_b is not None else FAMILY_B_LABELS),
        "alpha": float(alpha),
        "tests": list(tests if tests is not None else PREREGISTER_TESTS),
        "created_utc": _utc_now(),
        "note": note
        or "scored after lock; do not edit labels after first score of a new folder",
    }
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[CASIMIR LOAD] preregistered families → {dest}")
    return payload


def load_family_lock(path: Path | None = None) -> dict[str, Any] | None:
    dest = path or PREREGISTER_PATH
    if not dest.is_file():
        return None
    try:
        return json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_ledger_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return rows


def record_hash_ledger(
    *,
    data_label: str,
    path: str | Path | None,
    H: str,
    n_rows: int,
    n_cols: int,
    stamp: str | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """
    Append-only ledger. Same data_label with a different H refuses pooling
    and prints HASH MISMATCH; the old line is never overwritten.
    """
    dest = ledger_path or HASH_LEDGER_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    prior = _read_ledger_lines(dest)
    mismatch = False
    old_h: str | None = None
    for row in prior:
        if str(row.get("data_label") or "") != str(data_label):
            continue
        prev = str(row.get("H") or "")
        if prev and prev != str(H):
            mismatch = True
            old_h = prev
    if mismatch:
        print(
            f"[CASIMIR LOAD] HASH MISMATCH for label={data_label} "
            f"old={old_h} new={H}"
        )
    entry = {
        "data_label": str(data_label),
        "path": str(Path(path).resolve()) if path else None,
        "H": str(H),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "stamp": stamp or _utc_now(),
    }
    with dest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return {
        **entry,
        "mismatch": mismatch,
        "old_H": old_h,
        "pool_allowed": not mismatch,
        "ledger_path": str(dest),
    }


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return value
    return obj


def _token_bounded(name: str, token: str) -> bool:
    return re.search(rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)", name.lower()) is not None


def find_two_pass_columns(df: pd.DataFrame | None) -> tuple[str | None, str | None]:
    """Detect a real forward/reverse Casimir loop pair from column names."""
    if df is None:
        return None, None
    cols = [str(c) for c in df.columns]
    pairs = (
        (_TWO_PASS_FWD, _TWO_PASS_REV),
        (_TWO_PASS_UP, _TWO_PASS_DOWN),
    )
    for fwd_tokens, rev_tokens in pairs:
        fwd = next((c for c in cols if any(_token_bounded(c, t) for t in fwd_tokens)), None)
        rev = next((c for c in cols if any(_token_bounded(c, t) for t in rev_tokens)), None)
        if fwd and rev and fwd != rev:
            return fwd, rev
    return None, None


def extra_fields_for_console(name: str, result: dict[str, Any]) -> list[str]:
    extras: list[str] = []
    status = result.get("status")
    if status and status != "ok":
        extras.append(f"status={status}")
        if result.get("not_applicable_reason"):
            extras.append(f"n/a={result['not_applicable_reason']}")
    if "identically_null" in result:
        extras.append(f"identically_null={result['identically_null']}")
    if "grid_locked" in result:
        extras.append(f"dominant_period={result.get('dominant_period')}")
        extras.append(f"Delta_xi={result.get('Delta_xi')}")
        extras.append(f"grid_locked={result['grid_locked']}")
        if result.get("grid_locked_reason"):
            extras.append(str(result["grid_locked_reason"]))
    if "control_failed" in result:
        extras.append(f"control_failed={result['control_failed']}")
    if result.get("effect_size_ci_lo") is not None:
        extras.append(
            f"CI=[{result.get('effect_size_ci_lo')}, {result.get('effect_size_ci_hi')}]"
        )
    if result.get("p_ci_lo") is not None:
        extras.append(f"p_CI=[{result.get('p_ci_lo')}, {result.get('p_ci_hi')}]")
    if name == "log_modulations":
        if result.get("a") is not None:
            extras.append(f"a={result.get('a')}")
        if result.get("chi2_raw") is not None:
            extras.append(f"chi2_raw={result.get('chi2_raw')}")
        if result.get("effect_size") is not None:
            extras.append(f"|a|/rms(y)={result.get('effect_size')}")
    _ = name
    return extras


def format_test_console_line(name: str, result: dict[str, Any]) -> str:
    p_value = result.get("p_value")
    if p_value is None:
        p_txt = "null"
    else:
        try:
            p_txt = f"{float(p_value):.4g}"
        except (TypeError, ValueError):
            p_txt = str(p_value)
    line = (
        f"{name}: p={p_txt} | Evidence for TSB: {bool(result.get('evidence_for_TSB'))}"
    )
    extras = extra_fields_for_console(name, result)
    if extras:
        line += " | " + " | ".join(extras)
    return line


def print_falsification_report(
    results: dict[str, dict[str, Any]],
    *,
    s_tsb: float | None = None,
    bound_energy: dict[str, Any] | None = None,
) -> None:
    print("\n=== TSB Casimir Falsification Test Report ===")
    for name, res in results.items():
        print(format_test_console_line(name, res if isinstance(res, dict) else {}))
    if s_tsb is not None:
        print(f"S_TSB: {s_tsb:.4g}")
    if bound_energy:
        print("\n=== TSB Bound-Energy Residual Report ===")
        print(
            f"identity_control_ok={bound_energy.get('identity_control_ok')} | "
            f"E_bind_claimed={bound_energy.get('E_bind_claimed')} | "
            f"floor_p={bound_energy.get('floor_p')}"
        )
        nested = bound_energy.get("delta_e_tests") or {}
        for name, res in nested.items():
            if isinstance(res, dict) and "evidence_for_TSB" in res:
                print("  " + format_test_console_line(name, res))


def score_s_tsb(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored: list[tuple[str, dict[str, Any]]] = []
    for name, res in results.items():
        if not isinstance(res, dict):
            continue
        if res.get("status") == "not_applicable":
            continue
        if res.get("counted_in_S_TSB") is False:
            continue
        scored.append((name, res))
    n = len(scored)
    n_hit = sum(1 for _, res in scored if res.get("evidence_for_TSB"))
    s_tsb = (n_hit / n) if n else None
    return {
        "S_TSB": s_tsb,
        "n_applicable": n,
        "n_evidence": n_hit,
        "applicable_tests": [name for name, _ in scored],
    }


def run_diagnostics_battery(
    *,
    param_col: str | None = None,
    value_col: str | None = None,
    output_prefix: str = "tsb_diagnostics",
    plot: bool = False,
    show_popup: bool = False,
    relock: bool = False,
) -> dict[str, Any]:
    """Audit/control battery on auto-discovered datasets/casimir tables."""
    from menus.particle.casimir.test_framework import run_falsification_on_local_downloads

    preregister_split(relock=relock)
    bundle = run_falsification_on_local_downloads(
        param_col=param_col,
        value_col=value_col,
        output_prefix=output_prefix,
        plot=plot,
        show_popup=show_popup,
    )
    bundle["diagnostics"] = True
    return bundle
