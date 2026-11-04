"""
Standard-BBN recovery skeleton.

The reaction network is not finished. Deuterium is not claimed.
Status stays FAIL_INTEGRATOR until G3 is actually true.
No observational table. No dark-sector switch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


def assert_isolated() -> None:
    loaded = [name for name in _BANNED if name in sys.modules]
    if loaded:
        raise RuntimeError("bbn_isolated refuses loaded modules: " + ", ".join(loaded))


def gates() -> dict[str, bool]:
    """G3 is false because the network has not been integrated."""
    return {
        "G1": False,
        "G2": False,
        "G3": False,
        "G4": False,
        "G5": False,
    }


def first_failed(gate_map: dict[str, bool]) -> str:
    for name in ("G1", "G2", "G3", "G4", "G5"):
        if not gate_map[name]:
            return name
    return ""


def report() -> dict:
    gate_map = gates()
    failed = first_failed(gate_map)
    # Y_D is omitted, not stored as zero. A zero yield is not a result.
    return {
        "status": "FAIL_INTEGRATOR",
        "failed_gate": failed,
        "reason": "network not finished; deuterium yield not claimed",
        "gates": gate_map,
        "Y_n": None,
        "Y_p": None,
        "Y_D": None,
        "Y_T": None,
        "Y_He3": None,
        "Y_He4": None,
        "D_over_H": None,
        "Yp_mass": None,
        "eta": 6.12e-10,
        "tau_n_used": 879.4,
        "network_name": "skeleton; Wagoner-class rates not integrated",
        "conservation_max_rel": None,
        "observational_table": None,
    }


def run() -> dict:
    assert_isolated()
    payload = report()
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
    print(payload["reason"])
    return payload


def main() -> int:
    run()
    return 0
