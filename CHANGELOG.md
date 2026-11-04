# Changelog

## 2026.10.13 — Superblock / Mirron residual engine

Corpus date: 2026-10-13. Default path is 3+1D Superblock / Mirron. Tau / Tav / compact \(S^1_\tau\) / 142857 / 7 Mpc 5D are off the live menu.

### Live ledger

- Topological floor \(m_0 = 309.3\) MeV (insertion; old print 313.1 was \(M_n/3\)). \(r_M = \hbar c / m_0\).
- `sb_engine/constants.py` is the single source for \(m_0\), \(V_{\mathrm{orb}}\), \(\Delta N_{\mathrm{plasma}}\), \(r_{d,\mathrm{obs}}\).
- SPARC Template B: frozen \(G_{\mathrm{avg}}/G_N\) on the full local SPARC table (not CamB-only). Kernel `G_avg_equal_tick_gamma_inv2`.
- DESI: \(r_d = \Delta N_{\mathrm{plasma}}\, r_{d,\mathrm{obs}} / \gamma\). Residual identically zero when \(\gamma = \Delta N_{\mathrm{plasma}}\). No \(n_{\mathrm{hier}}\).
- Glueball \(0^{++}\): \(m^{\mathrm{th}} = 2 m_0 e\), \(e\) LIVE, TRUE_PREDICTION vs lattice 1710 ± 50 MeV (\(\Delta \approx 28.47\) MeV, \(z \approx 0.57\)). \(0^{-+}\): \(15 m_0 / 2\) stays PACKED / SCALE_CONTACT.
- Neutron: HOLD. J-PARC is appearance (arXiv:2412.19519). \(\varepsilon\) does not apply to J-PARC. Channel: Prime-invisible leak.
- BBN empirical row: FAIL_INTEGRATOR. Recovery skeleton in `bbn_isolated/` (not wired to the empirical pipeline).
- FRB Template E: loads CHIME DM; Macquart residual only with independent host \(z\). No 1/7 Tav score.

### Tooling

- Boot: `config/superblock_only.json`. `tau_5d_templates` forced false.
- Residual scorer: `superblock_residual.py`. Synthetic illustrations in `sb_engine/{cms,casimir,sparc}.py` are labeled SYNTHETIC and are not Open Data / SPARC / Casimir detections.
- Casimir diagnostics: self-normalized \(\chi^2\) removed; hash ledger; bound-energy module present but \(\Delta E\) is not the live claim.
- Empirical artifacts use `artifacts/empirical_tests/{DD-MM-YYYY}_{HH-MM}_{dataset}/` naming.
- Live pytest gate: `tests/test_live_corpus.py`. Retired 1/7 / Tav / Berard tests are skipped, not deleted.

### Not in this release as live science

- Extra \(J^{PC}\) glueball packs (2++, 1+-, 0++*).
- Invented unquenched / string-tension lattice tables.
- Treating planted 7-fold CMS or log7 Casimir illustrations as detections.
