"""Published SBBN table for gate G4 (not an observation).

PRIMAT (Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 754 (2018) 1, arXiv:1801.08023; update
arXiv:2011.11320), table ``PRIMAT_Yp_DH_ErrorMC_2024.dat`` as distributed with CAMB (last update
2024-09-20, LUNA d(p,γ)³He rate, τ_n = 878.4 s, N_eff = 3.044 at ΔN = 0). Rows: ΔN = 0, ω_b h² 0.0220–0.0226.
Columns: (ω_b h², η10, Y_p^BBN = 4 Y_He4, D/H). Interpolated linearly in η10.
"""

from __future__ import annotations

PRIMAT_REF = (
    "PRIMAT 2024 table (Pitrou et al. 2018, Phys. Rep. 754, 1; arXiv:1801.08023, 2011.11320), "
    "PRIMAT_Yp_DH_ErrorMC_2024.dat, DeltaN = 0, tau_n = 878.4 s"
)
PRIMAT_TAU_N = 878.4

PRIMAT_ROWS: tuple[tuple[float, float, float, float], ...] = (
    (0.022000, 6.022456, 0.246826, 2.514174e-5),
    (0.022200, 6.077209, 0.246913, 2.477195e-5),
    (0.022400, 6.131963, 0.247000, 2.441024e-5),
    (0.022600, 6.186716, 0.247086, 2.405636e-5),
)


def primat_at_eta(eta: float) -> dict[str, float]:
    """Y_p^BBN and D/H at the same η (η10 = 1e10 η); linear in η10 inside the vendored rows."""
    e10 = float(eta) * 1e10
    rows = PRIMAT_ROWS
    if not rows[0][1] <= e10 <= rows[-1][1]:
        raise ValueError(f"eta10={e10:.4f} outside the vendored PRIMAT rows {rows[0][1]}–{rows[-1][1]}")
    for lo, hi in zip(rows, rows[1:]):
        if lo[1] <= e10 <= hi[1]:
            t = (e10 - lo[1]) / (hi[1] - lo[1])
            return {
                "eta10": e10,
                "omega_b_h2": lo[0] + t * (hi[0] - lo[0]),
                "Y_p": lo[2] + t * (hi[2] - lo[2]),
                "D_H": lo[3] + t * (hi[3] - lo[3]),
            }
    raise AssertionError("unreachable")
