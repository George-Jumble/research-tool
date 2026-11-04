# Retired Tau / Tav / 5D modules (2026-09-21)

The live tool is **Superblock / Mirron only** (3+1D).

If a menu item, CLI, or direct `run_action` hits a retired entry point, the tool shows:

> This Tau/Tav/5D item was retired 2026-09-21. The live tool is Superblock / Mirron only (3+1D). Tau, Tav, KK, 142857, and compact-S¹ templates are not evaluated. Tau/5D templates were not applied.

Calling a retired module raises `RetiredModuleError`: `module retired 2026-09-21; Superblock/Mirron only`.

## Disconnected from the GUI (files left on disk)

| Module | Why |
|---|---|
| Tav-Resonance package | Tav ladder / 1/7 cosmological search |
| RGC Resonance Mock Catalog | 1/7 + Δn mock as a cosmology backend |
| Tav BBN | Tav cylinder BBN interference |
| Berard Resonance Cosmology | tau-cosmology backend |
| Tav Framework Integrator | Tav correlation engine |
| Tau-Superblock Research Engine | cylinder state, fractal Tau circle |

## Removed from default menus (loaders kept)

1/7 tracks, 7-fold CMS/LHCb/LIGO scores, EKK BAO, KK/comb recovery, Casimir Tau falsification suite, Planck/FRB Tav harmonic scans.

Comparison models on the default path: ΛCDM, GR, SM, Newtonian SPARC. Superblock / Mirron is the alternative under test. `tau_5d_templates` is forced false.
