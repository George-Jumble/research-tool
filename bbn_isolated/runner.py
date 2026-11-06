"""
Standard-BBN gates G1–G5, computed from AlterBBN v2.2 (author ruling 2026-11-04).

The baseline is the external AlterBBN run at Planck 2018 inputs (eta = 6.114e-10,
tau_n = 877.75 s, N_eff = 3.044). The old in-repo toy network (menus/prime_past/bbn_network.py) is
deprecated and is not used here.
No observational table. No dark-sector switch.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from bbn_isolated import alterbbn
from bbn_isolated.reference import PRIMAT_REF, PRIMAT_TAU_N, primat_at_eta

HERE = Path(__file__).resolve().parent

# Names this package must not load.
_BANNED = (
    "menus.empirical_tests",
    "empirical_tests",
    "neutron",
    "sparc",
    "desi",
    "cms",
    "casimir",
    "bounce",
)

GATE_IDS = ("G1", "G2", "G3", "G4", "G5")
G1_TOL = 1e-8
G3_DH_RANGE = (1e-6, 1e-3)
G4_YP_TOL = 0.005
G4_DH_REL_TOL = 0.20
G5_REL_TOL = 0.05
# failsafe 1 is the nominal run; 0 ≈ double step (coarse), 2 ≈ half step, 3 = tighter tolerance.
NOMINAL_FAILSAFE = 1
G5_VARIANTS = {"double_step": 0, "half_step": 2, "tighter_tolerance": 3}


def assert_isolated() -> None:
    loaded = [name for name in _BANNED if name in sys.modules]
    if loaded:
        raise RuntimeError("bbn_isolated refuses loaded modules: " + ", ".join(loaded))


def first_failed(gate_map: dict[str, bool]) -> str:
    for name in GATE_IDS:
        if not gate_map[name]:
            return name
    return ""


def _rel_change(a: float, ref: float) -> float:
    if ref == 0.0:
        return 0.0 if a == 0.0 else math.inf
    return abs(a / ref - 1.0)


def _point_gates(short: dict[str, Any], full: dict[str, Any] | None) -> dict[str, Any]:
    """G1–G4 for one integrator setting."""
    vals = short["values"]
    detail: dict[str, Any] = {}
    # G1: baryon number from the full vector. Charge neutrality is imposed inside AlterBBN (the
    # electron chemical potential is solved from Σ Z_i Y_i), so it is not an independent output.
    if full is None:
        detail["G1"] = {"pass": False, "reason": f"{alterbbn.FULL_BINARY_NAME} not available; baryon closure not evaluable"}
    else:
        closure = alterbbn.baryon_closure(full)
        detail["G1"] = {
            "pass": bool(math.isfinite(closure) and abs(closure) < G1_TOL),
            "baryon_closure": closure,
            "tol": G1_TOL,
            "charge": "imposed by AlterBBN (phi_e from sum Z_i Y_i); not independently testable",
        }
    # G2: every printed abundance finite and non-negative, integrator error flag 0.
    pool = list(vals.values()) + (list(full["ratioH"][1:]) if full else [])
    ok = all(math.isfinite(v) and v >= 0.0 for v in pool)
    err = int(short.get("err", 0)) or (int(full.get("err", 0)) if full else 0)
    detail["G2"] = {
        "pass": bool(ok and err == 0),
        "min_value": min(pool),
        "err_flag": err,
        "note": "AlterBBN floors unpopulated species at ~1e-30 and applies max(0, .) only after integration",
    }
    # G3: nonzero deuterium in range (AlterBBN final state, after D freeze-out).
    dh = vals["D_H"]
    y_d = dh * full["ratioH"][2] if full else None
    detail["G3"] = {
        "pass": bool(G3_DH_RANGE[0] < dh < G3_DH_RANGE[1] and (y_d is None or y_d > 0)),
        "D_H": dh,
        "Y_D": y_d,
        "range": list(G3_DH_RANGE),
    }
    # G4: published SBBN table at the same eta.
    try:
        ref = primat_at_eta(short["eta"])
        d_yp = vals["Y_p"] - ref["Y_p"]
        d_dh = vals["D_H"] / ref["D_H"] - 1.0
        detail["G4"] = {
            "pass": bool(abs(d_yp) < G4_YP_TOL and abs(d_dh) < G4_DH_REL_TOL),
            "reference": PRIMAT_REF,
            "reference_tau_n": PRIMAT_TAU_N,
            "reference_Y_p": ref["Y_p"],
            "reference_D_H": ref["D_H"],
            "delta_Y_p": d_yp,
            "rel_delta_D_H": d_dh,
            "tol": {"Y_p_abs": G4_YP_TOL, "D_H_rel": G4_DH_REL_TOL},
        }
    except ValueError as exc:
        detail["G4"] = {"pass": False, "reason": str(exc)}
    return detail


def evaluate(
    *,
    eta: float = alterbbn.PLANCK_INPUTS["eta"],
    n_eff: float = alterbbn.PLANCK_INPUTS["n_eff"],
    tau_n: float = alterbbn.PLANCK_INPUTS["tau_n"],
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Run AlterBBN (nominal + G5 variants) and compute G1–G5. Raises AlterBBNUnavailable."""
    kw = dict(eta=eta, n_eff=n_eff, tau_n=tau_n, cache_dir=cache_dir, use_cache=use_cache)

    def _pair(fs: int) -> tuple[dict, dict | None]:
        short = alterbbn.run(failsafe=fs, **kw)
        try:
            full = alterbbn.run(failsafe=fs, full=True, **kw)
        except alterbbn.AlterBBNUnavailable:
            full = None
        return short, full

    nominal, nominal_full = _pair(NOMINAL_FAILSAFE)
    points = {"nominal": _point_gates(nominal, nominal_full)}
    if nominal_full is not None:
        drift = max(_rel_change(nominal_full["values"][k], nominal["values"][k]) for k in ("Y_p", "D_H"))
        points["nominal"]["driver_agreement_rel"] = drift
    g5: dict[str, Any] = {}
    g5_ok = True
    for label, fs in G5_VARIANTS.items():
        short, full = _pair(fs)
        pg = _point_gates(short, full)
        rel = {k: _rel_change(short["values"][k], nominal["values"][k]) for k in ("Y_p", "D_H")}
        inner = all(pg[g]["pass"] for g in ("G1", "G2", "G3", "G4"))
        ok = inner and all(v < G5_REL_TOL for v in rel.values())
        g5[label] = {"failsafe": fs, "G1_G4": inner, "rel_change": rel, "Y_p": short["values"]["Y_p"],
                     "D_H": short["values"]["D_H"], "pass": ok}
        g5_ok = g5_ok and ok
    gate_map = {g: bool(points["nominal"][g]["pass"]) for g in ("G1", "G2", "G3", "G4")}
    gate_map["G5"] = bool(g5_ok)
    return {
        "inputs": {"eta": eta, "n_eff": n_eff, "tau_n": tau_n, "failsafe": NOMINAL_FAILSAFE},
        "nominal": nominal,
        "nominal_full": nominal_full,
        "gate_detail": {**points["nominal"], "G5": {"tol_rel": G5_REL_TOL, "variants": g5}},
        "gates": gate_map,
    }


def gates(**kw: Any) -> dict[str, bool]:
    """G1–G5 from AlterBBN; all False when AlterBBN is unavailable."""
    try:
        return evaluate(**kw)["gates"]
    except (alterbbn.AlterBBNUnavailable, RuntimeError, OSError):
        return {g: False for g in GATE_IDS}


def _rel(path: str | None) -> str | None:
    if not path:
        return path
    try:
        return str(Path(path).resolve().relative_to(alterbbn.REPO_ROOT))
    except ValueError:
        return path


def report(**kw: Any) -> dict[str, Any]:
    try:
        ev = evaluate(**kw)
    except (alterbbn.AlterBBNUnavailable, RuntimeError, OSError) as exc:
        gate_map = {g: False for g in GATE_IDS}
        return {
            "status": "ALTERBBN_UNAVAILABLE",
            "failed_gate": first_failed(gate_map),
            "reason": str(exc),
            "gates": gate_map,
            "network_name": alterbbn.ALTERBBN_VERSION,
            "observational_table": None,
        }
    gate_map = ev["gates"]
    failed = first_failed(gate_map)
    vals = ev["nominal"]["values"]
    full = ev["nominal_full"]
    y_h = full["ratioH"][2] if full else None
    return {
        "status": "PASS_GATES" if not failed else "FAIL_GATES",
        "failed_gate": failed,
        "reason": "" if not failed else f"{failed} failed; see gate_detail",
        "gates": gate_map,
        "gate_detail": ev["gate_detail"],
        "Y_H": y_h,
        "Y_D": vals["D_H"] * y_h if y_h else None,
        "Yp_mass": vals["Y_p"],
        "Yp_definition": "4 * n_He4 / n_b (AlterBBN ratioH[6])",
        "D_over_H": vals["D_H"],
        "He3_over_H": vals["He3_H"],
        "Li7_over_H": vals["Li7_H"],
        "eta": ev["inputs"]["eta"],
        "n_eff": ev["inputs"]["n_eff"],
        "tau_n_used": ev["inputs"]["tau_n"],
        "network_name": f"{alterbbn.ALTERBBN_VERSION} ({alterbbn.ALTERBBN_REF})",
        "binary": ev["nominal"].get("binary"),
        "binary_sha256": ev["nominal"].get("binary_sha256"),
        "cache_path": _rel(ev["nominal"].get("cache_path")),
        "conservation_max_rel": ev["gate_detail"]["G1"].get("baryon_closure"),
        "observational_table": None,
    }


def run(*, check_isolated: bool = True, **kw: Any) -> dict:
    """Write gates.json / report.json. The research tool calls it with check_isolated=False because the
    menu process has other modules loaded; the gates themselves never import them."""
    if check_isolated:
        assert_isolated()
    payload = report(**kw)
    (HERE / "gates.json").write_text(
        json.dumps(payload["gates"], indent=2) + "\n",
        encoding="utf-8",
    )
    (HERE / "report.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["gates"], indent=2))
    print(f"status={payload['status']} failed_gate={payload['failed_gate']}")
    if payload.get("reason"):
        print(payload["reason"])
    return payload


def main() -> int:
    run()
    return 0
