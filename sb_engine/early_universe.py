"""Early universe under v5: N_eff, hidden acoustics (BAO) and the t_priv relay hierarchy.

Rulings: N11 (only baryons and massive persistent particles carry L_res), N23 (each domain has its
own BAO, coupled to the others only by gravity), N24-N26 (inter-domain time variability; only gravity
propagates along T_priv; |offset| matters, not its sign), N29-N30 (t_priv hierarchy; d's link to Prime
is (-1, +1, -1), never adjacent). Test write-ups: delta_Neff_Lres_test.md, hidden_BAO_test.md,
relay_hierarchy_BAO_test.md. The CLASS outputs those notes produced are curated reference tables in
``menus.empirical_tests.tests`` (REPOS keys ``hidden_bao_class_grid`` and ``relay_class_configs``);
this module only does the bookkeeping that decides which CLASS configuration applies.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable

from .constants import B_LIN_GAMMA, SCHEME_A_GAMMA

# ---------------------------------------------------------------- N_eff
N_EFF_STANDARD = 3.044
PHOTON_NEFF_UNITS = (8.0 / 7.0) * (11.0 / 4.0) ** (4.0 / 3.0)  # one photon bath in N_eff units
N_HIDDEN_DOMAINS = 5


def n_eff_prime(hidden_radiation_carries_lres: bool = False, x: float = 1.0) -> float:
    """N_eff seen by Prime. N11: hidden photons/neutrinos/e+- carry no L_res -> standard 3.044.

    The superseded reading (hidden radiation gravitates on Prime) adds five copies of photons plus
    neutrinos at temperature ratio x: Delta N_eff = 5 (4.40 + 3.044) x^4 ~ 37 for x = 1.
    """
    if not hidden_radiation_carries_lres:
        return N_EFF_STANDARD
    per_domain = (PHOTON_NEFF_UNITS + N_EFF_STANDARD) * float(x) ** 4
    return N_EFF_STANDARD + N_HIDDEN_DOMAINS * per_domain


# ---------------------------------------------------------------- hidden acoustics
X_PASS = 0.02   # every hidden domain >= ~50x out of thermal step (v5 Section 14)
X_COLD = 0.005  # bracket used by the relay test: x <= 0.005 treated as cold


def hidden_drag_redshift(x: float, z_drag_prime: float = 1060.0) -> float:
    """Hidden drag at Prime redshift ~ z_drag / x (same local physics, colder by x)."""
    return float(z_drag_prime) / float(x)


def hidden_domain_acts_as_cdm(x: float) -> bool:
    return float(x) <= X_PASS


# ---------------------------------------------------------------- relay hierarchy
POS = ("AP", "T", "B", "-T", "-B")
RELAY = ("T", "B", "-T", "-B")
FWD = frozenset({"Prime", "T", "B"})
REVERSED = frozenset({"AP", "-T", "-B"})
B_LIN_SLOT_WEIGHT = dict(zip("dscbt", B_LIN_GAMMA[1:]))  # slot-order convention (relay test)
D_LINK_HALF_BOUNCE = (-1, +1, -1)                         # ruling N30 / Q9


def relay_structures() -> list[dict[str, str]]:
    """Parent maps: each relay links to Prime or to one other depth-1 relay; AP is a leaf; depth <= 2."""
    out = []
    opts = {p: ["Prime"] + [r for r in RELAY if r != p] for p in POS}
    for combo in itertools.product(*[opts[p] for p in POS]):
        par = dict(zip(POS, combo))
        if all(par[p] == "Prime" or par[par[p]] == "Prime" for p in POS):
            out.append(par)
    return out


def flavour_assignments() -> list[dict[str, str]]:
    """d sits in a reversed-time position (AP, -T, -B); s, c, b, t permute over the rest."""
    out = []
    for dpos in ("AP", "-T", "-B"):
        rest = [p for p in POS if p != dpos]
        for perm in itertools.permutations("scbt"):
            fl = {dpos: "d"}
            fl.update(dict(zip(rest, perm)))
            out.append(fl)
    return out


def path_from_prime(par: dict[str, str], pos: str) -> list[str]:
    chain = [pos]
    while par[chain[-1]] != "Prime":
        chain.append(par[chain[-1]])
    return chain[::-1]


def link_offsets(par: dict[str, str], fl: dict[str, str]) -> dict[str, dict[str, float]]:
    """x per flavour under F1 (clock product), F2 (signed ratio, telescopes) and F3 (|ratio| per link)."""
    g = SCHEME_A_GAMMA
    out: dict[str, dict[str, float]] = {}
    for p in POS:
        nodes = ["u"] + [fl[n] for n in path_from_prime(par, p)]
        f1 = math.prod(1.0 / g[n] for n in nodes[1:])
        sg = math.prod(g[a] / g[b] for a, b in zip(nodes[:-1], nodes[1:]))
        f2 = min(sg, 1.0 / sg)
        f3 = math.prod(min(g[a], g[b]) / max(g[a], g[b]) for a, b in zip(nodes[:-1], nodes[1:]))
        out[fl[p]] = {"F1": f1, "F2": f2, "F3": f3, "depth": len(nodes) - 1}
    return out


def class_config_key(xs: dict[str, float]) -> tuple:
    """Conservative CLASS bracket of relay_hierarchy_BAO_test.py: acoustic part = all x > X_COLD at max x."""
    acoustic = [q for q, x in xs.items() if x > X_COLD]
    if not acoustic:
        return ("cold",)
    w = sum(B_LIN_SLOT_WEIGHT.values())
    frac = round(sum(B_LIN_SLOT_WEIGHT[q] for q in acoustic) / w, 4)
    return (frac, float("%.3g" % max(xs[q] for q in acoustic)))


def d_is_adjacent_to_prime(par: dict[str, str], fl: dict[str, str]) -> bool:
    dpos = next(p for p, q in fl.items() if q == "d")
    return par[dpos] == "Prime"


def enumerate_hierarchies() -> Iterable[dict[str, Any]]:
    for par in relay_structures():
        for fl in flavour_assignments():
            yield {"parent": par, "flavour": fl, "x": link_offsets(par, fl)}


def half_bounce_sum(steps: tuple[int, ...] = D_LINK_HALF_BOUNCE) -> int:
    return int(sum(steps))
