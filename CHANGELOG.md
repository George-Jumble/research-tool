# Changelog

## 2026.11.06 — Superblock flagship v5

Release of the local `v5-tests-update` line onto `main`. Corpus date stays **2026-11-04** (Superblock Universe v5; rulings through N30). Tool version is **2026.11.06**.

### Live ledger

- Floor \(\Delta = 313.1\) MeV. \(r_M = 0.6302\) fm. The 309.3 MeV print is excluded.
- Template B is the hierarchical \(L_{\mathrm{res}}\) kernel \(V_{\mathrm{RL}}\), one core radius per galaxy. \(G_{\mathrm{avg}}\) is retired and is not evaluated.
- Prime sound horizon is the standard \(r_d\). `calibrate_gamma()` is gone (\(\gamma = 10.74\) retired).
- Template C (Mirron) is \(S_C(q^2) = \alpha_s(313.1\,\mathrm{MeV})\, M^2/(M^2+q^2)\) with \(M = 313.1\) MeV and no fitted amplitude. One-loop \(\alpha_s = 0.887\) is used because higher-loop running hits the Landau pole. The kernel scores **LHCb only**. CERN dimuon and Casimir keep \(\chi^2_\Lambda\) and refuse Template C pending a mapping ruling.
- LHCb data are the published 2020 \(B^0 \to K^{*0}\mu^+\mu^-\) angular bins (8 narrow \(q^2\) bins; default \(\langle P_5'\rangle\)), not an interpolated \(C_9\) series. Legacy \(C_9\) files still load and are marked synthetic.
- Glueball \(0^{++}\) pack is RETIRED. Neutron row is a live local mass-ratio consistency check (bottle lifetime and beam gap), not an appearance claim.
- Prime Past baseline is AlterBBN v2.2 at Planck 2018 inputs (\(\eta = 6.114\times 10^{-10}\), \(\tau_n = 877.75\) s, \(N_{\mathrm{eff}} = 3.044\)). The repo fetches the sha256-pinned tarball into `datasets/prime_past/` and builds `sb_bbn.x` and `sb_bbn_full.x` with no OpenMP. Gates G1–G5 are computed from that output. The v5 D/H \(2.51\times 10^{-5}\) figure is the Yeh 2021 (LUNA) reference plus an AlterBBN shift, not the raw AlterBBN yield.
- LIGO ringdown: one catalog entry per event (mass and redshift preferred, then highest version), final spin from the per-event PE JSON, whitened strain, and a blind \((f,\tau)\) fit. Default stack is coherent H1+L1+V1 with a null-stream glitch veto, plus a 20–300 ms post-merger scan. `HEAVY_BBH` is the 10-event preset. Residual rows are \(f\) and \(\gamma\), both in Hz.
- Tau/Tav menus, scripts, and guard names are removed. Old module tags, staging directory, ledger prefixes, and `TAV_*` environment variables still resolve through `sb_shared/legacy_names.py`. Nothing already on disk is moved.

### Tooling

- Curses line editor: digits and the keypad no longer submit the field; only Enter does. Backspace works. Numpad Enter is Enter, not cancel.
- A missing or stale archive path (I/O error on `~/Public` → `Storage7TB`) is treated as absent, so startup does not abort.
- `scripts/run_v5_tests.sh` bootstraps AlterBBN, then pytest. On 2026-11-06, with DESI DR2, SPARC, AlterBBN, and GW150914 strain cached, Planck TT absent, and CMS ROOT files present: **150 passed, 2 skipped, 3 xfailed**. The xfails are lithium-7, the \(\gamma^1\) weighting, and the Lyman-\(\alpha\) check. Further skips appear when DESI, AlterBBN, or the Planck spectrum are absent.

### Not in this release as live science

- Lithium-7 versus the Spite plateau, a derivation of the \(\gamma^1\) weight, and the Lyman-\(\alpha\) small-scale power check.
- A Template C mapping for CERN dimuon or Casimir.
- A Superblock formula for the Prime Past (Template D) prediction. \(\chi^2_{\mathrm{SB}}\) stays unset there.

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
