# Isolated standard-BBN gates (AlterBBN baseline)

This tree is not an evidence line. It does not import the empirical suite.

Author ruling 2026-11-04: the Prime-past standard-BBN baseline is **AlterBBN v2.2**, run as an
external binary at Planck 2018 inputs (eta = 6.114e-10, tau_n = 877.75 s, N_eff = 3.044). The old
toy network `menus/prime_past/bbn_network.py` is deprecated (kept, not deleted; it emits a
DeprecationWarning) and is not used here.

`tsb_on` is not a switch in this package. No dark-sector switch is added.

## AlterBBN binaries

Out of the box: if no binary is found, `ensure_alterbbn()` (also `python -m bbn_isolated.alterbbn ensure`,
the Prime Past menu action "Fetch & Build AlterBBN v2.2", the Dataset Manager target
`prime_past:alterbbn_v2.2`, the lazy first use, and `scripts/run_v5_tests.sh`) fetches the pinned
upstream tarball `https://alterbbn.hepforge.org/downloads?f=alterbbn_v2.2.tgz` (GPLv3, sha256
`46b94253cde6afce15ddb9367fa454bac91b25c9a1d3412cc19e512ec3515c17`) to `datasets/prime_past/alterbbn_v2.2.tgz`,
unpacks it to `datasets/prime_past/alterbbn_v2.2/`, and compiles both drivers there with the first of `$CC`,
cc, gcc, clang (`-O3 -ffast-math -fno-finite-math-only -std=gnu99`, no OpenMP). `SB_ALTERBBN_NO_AUTO_BUILD=1`
disables fetch/build; `SB_ALTERBBN_TARBALL` uses a local copy of the tarball.

| Binary | Role |
|---|---|
| `sb_bbn.x` (`alterbbn/sb_bbn.c`) | baseline B: prints Y_p, D/H, 3He/H, 7Li/H, 7Be/H |
| `sb_bbn_full.x` (`alterbbn/sb_bbn_full.c`) | same argv and physics; prints the whole abundance vector for G1 |

Lookup: `$SB_ALTERBBN_DIR`, then `datasets/prime_past/alterbbn_v2.2`, `../bbncodes/alterbbn_v2.2`,
`~/bbncodes/alterbbn_v2.2`, then `$PATH`. AlterBBN's default Nnu is 3.046, so N_eff = 3.044 is passed
as dNnu = -0.002. Runs are cached as CSV under `datasets/prime_past/alterbbn/` (gitignored), one file
per driver and (eta, N_eff, tau_n, failsafe), keyed to the binary's sha256.

If AlterBBN is missing and nothing is cached, `report()` returns `status = ALTERBBN_UNAVAILABLE` with
all gates false, and the tests skip.

## Gates

| ID | Gate | Pass |
|---|---|---|
| G1 | Conservation | absolute change in baryon number, \|Σ A_i Y_i − 1\| < 1e-8, from the full-vector driver. Charge neutrality is imposed inside AlterBBN (phi_e is solved from Σ Z_i Y_i), so it is reported, not tested. |
| G2 | Positivity | every printed abundance finite and non-negative, integrator error flag 0. (AlterBBN floors unpopulated species at ~1e-30 and applies max(0, ·) only after integration.) |
| G3 | Nonzero deuterium | Y_D > 0 and D/H between 1e-6 and 1e-3 in the final state. |
| G4 | Published SBBN table | at the same eta, helium within 0.005 and D/H within 20 percent of PRIMAT 2024 (`reference.py`). Not an observation. |
| G5 | Convergence | G1–G4 still hold at failsafe 0 (≈ double step), 2 (≈ half step) and 3 (tighter tolerance); relative change in Y_p and D/H vs failsafe 1 below 0.05. |

## Status

`python -m bbn_isolated` writes `gates.json` and `report.json` in this directory.
`status` is `PASS_GATES` when G1–G5 are green, `FAIL_GATES` otherwise, `ALTERBBN_UNAVAILABLE` without
AlterBBN. No observational sigma is written here; the Prime Past residual loader
(`superblock_residual._load_prime_past`) scores the baseline against Cooke 2018 D/H and Aver 2021 Y_p,
and only when G1–G5 are green.

## Refuse

Do not import the empirical suite, neutron, glueball, sparc, desi, cms, casimir, or bounce.
Do not read the frozen mass floor into the Friedmann equation.
Do not score a zero deuterium yield against observations.
