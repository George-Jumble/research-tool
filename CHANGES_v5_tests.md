# Test-suite update for Superblock flagship v5 (2026-11-04)

Branch `v5-tests-update`, based on `main` @ `8a2e32b` (tag `v2026.10.13`). Released as **2026.11.06**.

**Framework source of truth:** *The Superblock Universe* v5 (Zenodo draft 23953533; concept DOI
10.5281/zenodo.22948740), `Rulings_2026-11-01.md` (N1–N13), `Rulings_2026-11-04.md` (N13–N30),
`Current_Canon_2026-11-04.md`, `Superblock_Framework_Changes_Log_2026-11-04.md`, and the test notes
`scheme_A_light_speed_test`, `delta_Neff_Lres_test`, `BBN_physics_superblock`,
`scheme_A_relative_couplings_test`, `local_gravity_coorbit_test`, `domain_relativity_test`,
`hidden_BAO_test` and `relay_hierarchy_BAO_test`.

**Conventions kept:** every new data source is a `REPOS` entry in `menus/empirical_tests/tests.py`
(`type: curated`, a `parse_*` pass-through, a `get_curated_dataframe` branch with `DataPoint`-style
columns `observable/value/unc/source/year/units/reference`). That means `materialize_curated_cache`, the
`EMPIRICAL_TESTS` dataset provider (`empirical:{key}` ids, CSV cache under `datasets/empirical/`) and
`ensure_empirical_targets` pick them up with no new plumbing. New scoring is done by ledger writers named
`test_*` with `__test__ = False`, returning `status / ledger / pass_emitted / _dependency_block` dicts, like
the light-quark, neutron and glueball rows. Run reports go to the same `artifacts/empirical_tests/` run
folders via `run_pipeline`. Real data (DESI DR2, SPARC) is fetched only through the repo's own fetchers
(`datasets/desi/bao_data`, `fetch_mass_models_table`).

## Results

| | before (main @ 8a2e32b) | after (v5-tests-update) |
|---|---|---|
| `pytest` at repo root | collection aborted (2 errors: `test_tk.py` needs tkinter, `menus/astronomical/frb/dispersion_test.py` needs healpy) | 82 passed, 3 xfailed |
| maintained suites only (63 items) | 51 passed, 8 skipped, 4 failed (DESI tables missing) | 85 items: 82 passed, 3 xfailed (DESI tables cached); 78 passed, 4 skipped, 3 xfailed without them |

## Removed (11 dead test functions, all already skipped; plus `test_m0_frozen`, replaced)

| Test | Reason |
|---|---|
| `tests/test_berard_units.py` (3 tests, module-level skip) | Berard halo backend retired 2026-09-22; not in v5 |
| `tests/test_flux_choking_outliers.py` (module-level skip) | `TauSuperblockSolver` is the retired Tau metric package (Tau retired 10-31) |
| `test_harmonic_binding_kit.py::test_synthetic_1_7_periodicity_and_kit_agree_true`, `::test_synthetic_1_7_whim_not_auto_true` (skipped) | 1/7 Tau periodicity in DESI is retired (Tau and 7 Mpc phenomenology, 10-31) |
| `test_periodicity_detection.py::test_synthetic_sinusoid_at_1_7_detects`, `::test_lomb_scargle_cyclic_peaks_at_one_seventh` (skipped) | same |
| `test_mcmc_engine.py::test_softened_tav_returns_sigma_seed_params`, `::test_softened_tav_competitive_on_piso_like_core` (skipped) | Tav is not a comparison model |
| `test_live_corpus.py::test_m0_frozen` | asserted the 309.3 MeV floor, which is excluded (Q1 2026-10-31; floor Δ = 313.1 MeV). Replaced by `test_floor_locked_at_313_1` |

`test_tk.py` (a manual Tk backend check) and the retired Tav FRB runner `menus/astronomical/frb/dispersion_test.py` are left in the tree but no longer collected. The new `pytest.ini` sets `testpaths`.

## Fixed / flipped (6 rewritten tests + 4 DESI tests data-gated, plus the code they gate)

| Test | Change | v5 basis |
|---|---|---|
| `test_live_corpus::test_floor_locked_at_313_1` (was `test_m0_frozen`) | Δ = 313.1 MeV; r_M = 0.6302 fm, diameter 1.2605 fm, tick 4.204×10⁻²⁴ s, hops 491.8 / 983.6 MeV; 309.3 asserted not live | Q1, v5 Table 1 |
| `test_live_corpus::test_neutron_module_live_local_mass_ratio` (was `test_neutron_hold_appearance_not_epsilon`) | HOLD → LIVE consistency: τ_bottle = 878.7 ± 2.1 s (z = +0.4 vs UCNτ 877.75 ± 0.36), gap 9.44 ± 0.63 s (z = −0.2 vs 9.95 ± 2.28); ε / J-PARC appearance fields dropped | N21, v5 §14 |
| `test_live_corpus::test_glueball_0pp_pack_is_retired` (was `test_glueball_0pp_is_live_prediction`) | TRUE_PREDICTION → RETIRED; 2Δe = 1702 MeV sits in the retired 1.5–1.7 GeV window; Yang–Mills gap not derived | v5 §9 |
| `test_live_corpus::test_light_quarks_consistency_not_pass` | m0 313.1; floor inside the (m_p+m_n)/6 band | Q1 |
| `test_live_corpus::test_sparc_template_b_uses_hierarchical_lres_kernel` (was `..._uses_frozen_g_avg`) | Template B = V_RL = V_flat·√(r/(r+R_core)) in quadrature, one R_core per galaxy, M/L = 1, η = 0. G_avg = ⟨1/γ²⟩ never evaluated | T1 (averaging kernel with Scheme A weights retired), N15, v5 §6 |
| `test_live_corpus::test_template_b_without_radii_is_inconclusive_not_g_avg` (new; counted under Added) | no radii means inconclusive, never G_avg | same |
| `test_live_corpus::test_live_writers_have_no_tav_engine_or_struck_strings` | also bans `309.3`, `TRUE_PREDICTION` and `Prime-invisible leak` in the live writers | Q1, v5 §9, N21 |
| `test_desi_bgs_mode` (4 tests) | `skipif` with the clone command when DESI DR2 tables are not cached, instead of failing | loader tests are still live; data is gitignored |

Engine changes these gates required:
- `sb_engine/constants.py`: floor and Mirron quantities; B_lin β/γ; 5.4015; `DELTA_N_PLASMA` kept but marked RETIRED for the legacy DESI and TSB importers.
- `sb_engine/sound_horizon.py`: standard r_d (Aubourg et al. 2015 fit at ω_c = 5.4015 ω_b). `calibrate_gamma()` removed entirely (γ = 10.74 retired; see 2026-11-04 follow-up below).
- `sb_engine/glueball.py`: status RETIRED.
- `kernels/g_avg.py`: marked RETIRED. New `kernels/l_res_rl.py`.
- `superblock_residual.py`: Template B now uses V_RL, and the SPARC loader carries radii and galaxy ids. DESI reports the standard r_d.
- `menus/empirical_tests/tests.py` (neutron, glueball, LaTeX and pipeline text):
  - BBN anchors are Aver 2021 Y_p 0.2453 ± 0.0034 and Cooke 2018 D/H (2.527 ± 0.030)×10⁻⁵.
  - The UCNτ bottle error is the combined 0.36 s, and the Yue 2013 beam comparator was added.
  - The s-quark current mass (93.5 ± 0.8 MeV, PDG 2024) was added to the curated PDG table.
  - `get_curated_dataframe("pdg_light_quarks")` now materializes (before, the registry cache stage returned nothing for it).

## Added (30 new test cases: 27 in `test_superblock_v5.py`, 1 SPARC, 1 boot, 1 live-corpus)

New data sources (curated `REPOS` keys): `planck2018_cosmology`, `superblock_v5_predictions`, `hidden_bao_class_grid`, `relay_class_configs`. They are also on the new menu action **Superblock v5 Confrontations**.

New engine modules: `sb_engine/dark_matter.py`, `neutron.py`, `couplings.py`, `early_universe.py`.

New writers: `test_dark_to_ordinary`, `test_coupling_invariance`, `test_bbn_v5`, `test_neff_v5`, `test_sound_horizon_v5`, `test_hidden_bao_v5`, `test_relay_hierarchy_v5`.

`tests/test_superblock_v5.py` (27):

| Test | Reference data / tolerance |
|---|---|
| `test_v5_sources_use_repo_curated_cache[×7]` | REPOS schema; each table materializes to CSV via `materialize_curated_cache` |
| `test_v5_menu_action_auto_fetches_v5_sources` | `empirical:{key}` ids come from the menu action |
| `test_bbn_and_neutron_anchors_are_the_v5_references` | Aver 2021, Cooke 2018, UCNτ 0.36 s, Yue 2013 |
| `test_dark_to_ordinary_tpriv_weighting_vs_planck` | Σγ(B_lin) = 5.401514; Planck 0.1200/0.02237 = 5.364 ± 0.065; pull +0.58 ± 0.01 |
| `test_strict_local_rest_mass_reading_fails` | 5.00, pull < −5.5 |
| `test_every_domain_is_prime_same_ratio_from_every_viewpoint` | spread 0 (N18/N19) |
| `test_composable_velocity_reading_is_retired` | fastest observer ≥ 5.70 (> +5σ), spread ≥ 0.53 |
| `test_quark_ladder_does_not_gravitate` | N15; the literal γ³ kernel gives 6.44 (not used) |
| `test_scheme_a_kinematic_means_live` | ⟨γ⟩_harm 4.041, ⟨γ⟩_arith 1.3744×10⁴, Λ_tw 2.00×10⁻¹⁰ GeV⁻², (c/γ)² + (βc)² = c² |
| `test_coupling_constants_are_local` | 1,000 random frame maps, \|exponent\| < 10⁻¹²; literal α_t > 500 (superseded) |
| `test_neff_standard_hidden_radiation_carries_no_lres` | 3.044 vs 2.99 ± 0.17; retired reading ΔN_eff = 37.2 |
| `test_bbn_standard_after_n11` | Y_p z = +0.5 ± 0.1, D/H z = −0.1 ± 0.1; Li-7 2.5–3.5× Spite (OPEN) |
| `test_lithium7_matches_spite_plateau` | **xfail (strict)**: open question 6 |
| `test_prime_sound_horizon_is_standard_no_gamma_factor` | CLASS r_d 146.88 (z = −0.8 ± 0.05 vs 147.09 ± 0.26); fit within 0.1 Mpc of 146.88 |
| `test_gamma_10_74_sound_horizon_factor_retired` | `calibrate_gamma` no longer exists in the engine; residual no longer zero by construction; fit gives Planck ΛCDM 147.09 ± 0.1 |
| `test_hidden_bao_in_step_fails_offset_passes` | x = 1: σ8 0.157 (fail); x ≤ 0.02 passes (Δχ² ≤ 3, \|Δσ8\| < 0.02); x ≥ 0.05 fails |
| `test_relay_hierarchy_d_must_not_be_adjacent` | 133 structures × 72 assignments = 9,576 hierarchies; F1 and F3 each pass 4,800, all with d relayed; F2 passes 0 |
| `test_d_link_is_three_half_bounce_steps` | (−1, +1, −1) sums to −1; the six-tick bounce sums to 0 |
| `test_mirror_chains_fail_only_with_d_at_minus_t` | 48/72 pass; the 24 failures all have d at −T |
| `test_gamma1_weighting_derived` | **xfail (strict)**: γ¹ is motivated, not derived |
| `test_lyman_alpha_small_scale_power_cleared` | **xfail (strict)**: Lyman-α check still owed (s linked directly cuts P(k = 1) by 10–20%) |

`tests/test_sparc_lres_v5.py` (1): uses the real SPARC MassModels table through the repo fetcher, with 171 galaxies of at least 5 points. L_res beats pure baryons in ≥ 123/171, and the mean reduced χ²(L_res) is under half the baryonic mean. It skips with a reason if the table is not cached and can't be reached. **Caveat:** the manuscript's mean χ² values (164.7 / 78.2) are not reproduced exactly, because the error and grid recipe behind them is not recorded. This run gives 240.8 / 43.4 with 137 wins, so the test gates the sample size and the qualitative result only.

`tests/test_superblock_only_boot.py::test_v5_retired_geometry_absent_from_default_menu` (1): no Kaluza–Klein / Fork III / Conformal KK / 5D / S³ labels in the default menu.

## Kept unchanged (still relevant)

- `test_superblock_only_boot` (4): Tau/5D refused at boot.
- `test_bound_unbound_energy` (4).
- Casimir `test_diagnostics` (7): Lifshitz is a floor object; no E_bind source (author cut, N13).
- `test_harmonic_binding_kit` (12) and `test_periodicity_detection` (6): negative controls showing the DESI loaders never emit a false 1/7 detection.
- `test_mcmc_engine` (4): conventional halo baselines only.

## Not changed / for the author

- `VERSION` is **2026.11.06**. `CORPUS_DATE` stays `2026-11-04`.
- The retired DESI/TSB research modules (`menus/tsb_research`, `menus/astronomical/desi/bao_ekk_geometry.py`, …) still carry their own 10.74 constants. They are off the default menu and are not edited here.

## Follow-up (2026-11-04): `calibrate_gamma()` removed

- `sb_engine/sound_horizon.py`: deleted `calibrate_gamma()` (γ = 10.74 sound-horizon factor retired, ruling N6).
- `sb_engine/__init__.py`: dropped its import and `__all__` entry.
- `tests/test_superblock_v5.py::test_gamma_10_74_sound_horizon_factor_retired`: now asserts the function is absent instead of expecting `RuntimeError`.
- Not touched: the unrelated DESI scanner option `auto_calibrate_gamma` / `calibrate_gamma_from_hierarchy` (fits γ to the z-hierarchy; not the 10.74 sound-horizon calibration).

## Follow-up (2026-11-04): LHCb action routing fixed (no plot / pop-up)

Symptom: "Full Flavor Residual Report" wrote an `N=0`, `verdict=inconclusive` JSON and drew no plot.
Cause (already in `main` 8a2e32b): the LHCb menu was renamed to "Score / Plot Flavor Residuals" and
"Full Flavor Residual Report", but `_ACTION_CONFIG` only knew the old Tav-Echo names (which
`superblock_mode.RETIRED_ACTIONS` retires), and the residual engine had no LHCb loader, so the generic
"loader not asked to score" note was returned.

- `superblock_residual.py`: new `_load_lhcb(options)` Step 0 loader using the existing
  `menus.particle.lhcb.flavor.load_lhcb_data` (same `datasets/lhcb/` cache, bundled/curated fallback).
  D = C9_exp, B = C9_SM (SM baseline), σ = C9_exp_err, x = q². `run_superblock_residual` now passes
  `options` (data_file, n_q2_bins) through. The Tav-Echo 1/(m₀²+q²) correlation is not applied.
- `menus/particle/lhcb/extension.py`: `_ACTION_CONFIG` gets the three new names (Score: plot False;
  Plot and Full Report: plot True). After the residual report, plot actions draw ΔC9 = C9_exp − C9_SM
  vs q² to `artifacts/lhcb/lhcb_flavor_residuals.png`; the pop-up opens when Graphics = `popup`.
  The "Next:" hint after download names the new actions.
- `superblock_mode.py`: unchanged. The old Tav-Echo names stay retired (they are Tau/Tav items).
- New `tests/test_lhcb_routing.py` (5 tests): every non-download LHCb action is live and configured,
  plot flags, old names stay retired, the residual engine loads LHCb data, and the plot is written.
- Suite: **87 passed, 3 xfailed**.
- Same gap, not fixed here: CERN_OPENDATA, LIGO_GWOSC, PLANCK_CMB_TAV, TSB_CASIMIR and
  PRIME_PAST_HARMONIC are residual-by-default but have no residual-engine loader, so their live
  non-loader actions also return the N=0 "inconclusive" note.

## Follow-up (2026-11-04): Tau / Tav removed, including the names

Removed code (still in git history):
- Retired menus: `menus/astronomical/tav_resonance`, `menus/astronomical/berard`, `menus/particle/rgc`,
  `menus/integrator`, `menus/tsb_research`, plus their root `*_extension.py` shims and `install_tav_resonance.sh`.
- Tav BBN scripts (`tav_bbn_*`, `enhanced_/final_tav_bbn_*`, `minimal_bbn_tav.py`, `tavModel.py`,
  `theoretical_model.py`, `mock_generator.py`, `desi_dashboard.py`, `tav_superblock_cms_*`), Prime Past
  BBN-Tav files (`bbn_menu*.py`, `bbn_confrontation.py`, `bbn_enhanced.py`, `harmonic.py`), `threadChokeSim.py`.
- Dead post-gate scoring code in residual-routed modules: DESI scanner / EKK / LSS / likelihood / nested
  sampling files, FRB cosmic-web-Tav + dispersion test, SPARC analyzer, LIGO ringdown, ATLAS analyzer, CERN
  7-fold / mod-7 analyzers, CMS HEP-statistics / preregistered-recoil q_T study (7-fold study; was a loader
  action), Planck `tav.py`, Casimir falsification suite + `bound_energy.py` + its tests and changelog.
- Guard plumbing: `RETIRED_*`, `RetiredModuleError`, `refuse_tau_template`, `tau_templates_enabled`,
  `tau_5d_templates` boot key, the `negative_control` / `T_τ` report fields, `--lock-blind/--compare-blind/--blind`.
- Fractal-tau ingestion schemas/normalizers/bounds, Tau-SB aDE three-way LLM comparison block, the `Tav`
  model branch in `mcmc_engine` (its profile is the cataloged `Mirron` model), `tav-resonance` requirement.
- Docs: `docs/retired/RETIRED_TAU_MODULES.md`, `docs/rejected/*`, `BLIND_ANALYSIS_PROTOCOL.md`.
- Tests: `test_harmonic_binding_kit.py`, `test_periodicity_detection.py`, Casimir periodogram test, Tau corpus test.

Kept loaders were extracted into small modules: `desi/loader.py`, `desi/covariance.py`, `frb/catalog.py`,
`casimir/fetcher.py`, `planck_cmb/fetcher.py`; `lhcb/echo.py` → `lhcb/flavor.py`;
`prime_past/bbn_interference.py` → `bbn_network.py` (standard network only).

Renames (`git mv` where files moved):
- Packages `tav_shared` → `sb_shared`, `tav_research` → `sb_research`; `tav_project_paths` → `project_paths`;
  `TAU_SUPERBLOCK_ROOT` → `SUPERBLOCK_ROOT`, `TAV_PROJECT_DIR` → `STAGING_DIR`, `ensure_tav_project_dirs` → `ensure_project_dirs`.
- Module tags `TAU_SB_DESI` → `DESI_BAO`, `LHCB_TAV_ECHO` → `LHCB_FLAVOR`, `FRB_COSMIC_WEB_TAV` → `FRB_DM`,
  `PLANCK_CMB_TAV` → `PLANCK_CMB`, `TSB_CASIMIR` → `CASIMIR`; menu titles "… Residuals".
- Root shims `tau_sb_desi_extension.py` → `desi_bao_extension.py`, `lhcb_echo_extension.py` → `lhcb_flavor_extension.py`,
  `tsb_casimir_extension.py` → `casimir_extension.py`; `e8_tav_superblock_*` → `e8_superblock_*`,
  `tav_superblock_interactive_model.html` → `superblock_interactive_model.html`.
- Env vars `TAV_USE_MOCK_DATA/TAV_CACHE_DIR/TAV_INGESTION_CONFIG/TAV_COMPARISON_CONFIG/TAV_REMOTE_TRANSFER` → `SB_*`.
- Artifact slugs `tau_sb_desi_scan` → `desi_bao_scan`, `lhcb_echo` → `lhcb_flavor`; log slugs follow the new tags.

Legacy aliases (`sb_shared/legacy_names.py`, nothing on disk is moved or deleted):
- `canonical_module_tag()` maps old tags in `classify_action`, `run_superblock_residual`, `get_provider`,
  `get_extension`, the runner, test-set reset and cache clear.
- Staging dir: new `.sb_project_staging/`; an existing `.tav_project_staging/` is reused in place.
  Old `~/Public/ProtonDrive/tav_project` archives still searched. Both stay in `.gitignore`.
- Old ledger prefixes (`tau_sb_desi:`, `tsb_casimir:`), old artifact filename patterns and the `rgc` subdir
  are still recognised; `getenv_compat()` falls back to the old `TAV_*` env names.

- New `tests/test_legacy_names.py` (5 tests) covers the alias lookup.
- Suite: **61 passed, 3 xfailed** (was 87: the deleted Tau-only test files held the difference). Root-level `test_mcmc_engine.py` (not collected by pytest.ini): 4 passed.

## Follow-up (2026-11-04): Template C (Mirron) numeric kernel

Author ruling: S_C(q²) = A·M²/(M²+q²), M = M_mir = Δ = 313.1 MeV (existing floor constant),
A = α_s(μ = ħc/r_M = 313.1 MeV). Zero free parameters; A is not fitted.

- New `kernels/mirron_c.py`: MS-bar α_s running from α_s(M_Z) = 0.1180 (RK4 in ln μ²; 1–4-loop β;
  thresholds at m_b(m_b) = 4.18 GeV, m_c(m_c) = 1.27 GeV; 3-loop decoupling for loops = 4, continuous
  matching below). Checks: α_s(m_b) = 0.2246, α_s(m_τ) = 0.320 (4-loop).
- At μ = 313.1 MeV the running is **unstable**: 2-, 3- and 4-loop hit the Landau pole (4-loop α_s
  passes 1 at μ ≈ 650 MeV, 2-loop at ≈ 527 MeV). 1-loop gives α_s = 0.887. The kernel uses the highest
  loop order that stays finite and below 1, so **A = 0.887 (1-loop)**; the report carries
  `alpha_s_by_loops`, `alpha_s_status_by_loops` and `A_loops_used`.
- `superblock_residual.score_against_template` scores Template C (new `q2` argument): r = C9_exp − C9_SM,
  chi2_SB = Σ((r − S_C)/σ)², chi2_Lambda = Σ(r/σ)², verdict `SB_better` / `baseline_better` from the
  χ² comparison. A diagnostic least-squares A ± σ for the fixed shape is reported and not used in the score.
- `_load_lhcb` now passes q² in GeV² (flavor.py stores MeV²).
- LHCb, 24 bins (bundled `datasets/lhcb/lhcb_bmeson_data.npy`, interpolated from 6 anchors, so the bins
  are correlated): **chi2_SB = 12.47 vs chi2_Lambda = 11.05, verdict baseline_better**. Diagnostic best-fit
  A = −4.84 ± 2.56 (wrong sign; the ruled +0.887 is about 2.2σ from it).
- New `tests/test_mirron_c_kernel.py` (4 tests). Suite: **65 passed, 3 xfailed**.

## Follow-up (2026-11-04): residual-engine loaders for CERN, LIGO, Planck, Casimir, Prime Past

Each loader returns D, B, σ, x and notes like `_load_lhcb` and reuses the module's own cache paths.
A missing cache fails cleanly: `load_error` is set, N = 0 and chi2_Lambda = None. No data is bundled
or invented. chi2_SB stays None for templates A/D/E, and the Template C Mirron kernel is now passed q²
only for LHCB_FLAVOR, so CERN/Casimir (also template C) do not get it.

- PLANCK_CMB (`menus/planck_cmb/spectrum.py`): PLA `COM_PowerSpect_CMB-TT-binned_R3.01.txt` in
  `menus/planck_cmb/fits/` (restored from the finished-FITS archives; wget from IRSA when `auto_fetch`).
  D = D_ℓ, B = BestFit ΛCDM column, σ = mean(±dD_ℓ), x = ℓ; bins treated as independent.
- LIGO_GWOSC (`menus/gravitic/ligo/qnm.py`): per cached event in `datasets/gwosc/events/` with
  `event_catalog.json`: D = periodogram peak of the band-passed strain in the merger+0…0.05 s window,
  B = GR Kerr f_220 (Berti et al. 2006 fit) from (1+z)·M_f,source and final spin, σ = 1/T ⊕ catalog
  uncertainty. Events without final spin in the catalog are skipped with a note.
- CERN_OPENDATA (`menus/particle/cms/dimuon.py`): opposite-sign dimuon mass, 60–120 GeV, 30 bins,
  `cms_nanoaod_dimu` data vs `cms_nanoaod_dy` MC normalised to the data integral (education files have
  no lumi/xsec weights), σ = sqrt(N_d + s²N_MC). Needs uproot.
- CASIMIR (`menus/particle/casimir/force_table.py`): user-selected `datasets/casimir/**/*force*.csv`
  (`separation_m, force_N[, sigma_N]`) and geometry (`sphere_radius_m` or `plate_area_m2`), against the
  ideal-conductor T = 0 PFA force; Lifshitz corrections not applied. None of the existing downloads is
  such a table.
- PRIME_PAST_HARMONIC: no loader. The only baseline in the repo is the unfinished standard BBN network
  (fails G1–G5; scoring it is refused by `bbn_isolated/README.md`). A note explains this.
- New `tests/test_residual_loaders.py` (13 tests; real-data checks skip when uncached).
  Suite: **77 passed, 1 skipped, 3 xfailed**.

## Follow-up (2026-11-04): Prime Past on AlterBBN; Mirron kernel kept on LHCb only

Author rulings: (1) the Prime-past standard-BBN baseline is **AlterBBN v2.2**; (2) the Template C
Mirron kernel stays on **LHCb only**.

AlterBBN baseline:
- New `bbn_isolated/alterbbn/` (wrapper package). It runs the local `sb_bbn.x` driver (found via
  `$SB_ALTERBBN_DIR`, `../bbncodes/alterbbn_v2.2`, `/workspace/bbncodes/alterbbn_v2.2`, `~/bbncodes/…`, `$PATH`)
  at Planck 2018 inputs η = 6.114×10⁻¹⁰, τ_n = 877.75 s, N_eff = 3.044 (AlterBBN's Nnu default is 3.046,
  so dNnu = −0.002). Result: **Y_p = 0.24678, D/H = 2.4250×10⁻⁵** (³He/H 1.028×10⁻⁵, ⁷Li/H 5.49×10⁻¹⁰).
- `sb_bbn_full.c` + `build_full_driver.sh`: the same driver, printing the whole abundance vector
  (Y_H, A_i, Z_i) so G1 can check baryon number. Built with gcc and the AlterBBN Makefile flags
  next to `sb_bbn.x`; it reproduces `sb_bbn.x` to 1.6×10⁻⁷. `sb_bbn.x` and AlterBBN's `src/` are not modified.
- Cache: one CSV per (driver, η, N_eff, τ_n, failsafe) under `datasets/prime_past/alterbbn/` (gitignored),
  `DataPoint` columns `observable/value/unc/source/year/units/notes/reference` plus run parameters and the
  binary's sha256, rows in a fixed order. A cached run is reused when the parameters (and, if the binary is
  present, its hash) match. New curated `REPOS` key `alterbbn_planck_baseline` (`parse_alterbbn_baseline`,
  `get_curated_dataframe` branch) exposes it to `materialize_curated_cache` / `empirical:` ids; it is empty
  when AlterBBN is absent and nothing is cached.

`bbn_isolated/runner.py`: gates are now computed from the AlterBBN output (no longer hard-coded False).

| Gate | Check | Result |
|---|---|---|
| G1 | \|Σ A_i Y_i − 1\| < 1e-8 (full driver). Charge neutrality is imposed inside AlterBBN, so it is reported, not tested | pass, 2.0×10⁻¹² |
| G2 | all abundances finite, ≥ 0; AlterBBN error flag 0 | pass |
| G3 | Y_D > 0, 1e-6 < D/H < 1e-3 | pass, Y_D = 1.83×10⁻⁵ |
| G4 | vs PRIMAT 2024 table at the same η (`bbn_isolated/reference.py`; Y_p^BBN 0.24697, D/H 2.4529×10⁻⁵): \|ΔY_p\| < 0.005, \|ΔD/H\| < 20 % | pass, ΔY_p = −0.00019, ΔD/H = −1.1 % |
| G5 | G1–G4 hold and \|Δ\|/x < 5 % at failsafe 0 (≈ double step), 2 (≈ half step), 3 (tighter) | pass; worst ΔD/H 3.6 % (failsafe 0) |

Status `PASS_GATES` (`bbn_isolated/gates.json`, `report.json` regenerated). Without AlterBBN, `report()`
returns `ALTERBBN_UNAVAILABLE` with all gates false.

`superblock_residual._load_prime_past`: D = Cooke 2018 D/H (2.527 ± 0.030)×10⁻⁵ and Aver 2021 Y_p 0.2453 ± 0.0034
(curated `bbn_abundances`), B = AlterBBN, σ = observational only. B is used only when G1–G5 are green.
Template D has no formula: `sb_prediction = "pending"`, chi2_SB = None, S_k = None (no formula invented).

| | D | B (AlterBBN) | r = D − B | pull |
|---|---|---|---|---|
| D/H | 2.527×10⁻⁵ ± 0.030 | 2.4250×10⁻⁵ | +1.020×10⁻⁶ | **+3.40σ** |
| Y_p | 0.2453 ± 0.0034 | 0.24678 | −0.00148 | −0.44σ |

**chi2_Lambda = 11.75 (N = 2), chi2_SB = None, verdict inconclusive.** Caveats: no AlterBBN rate (theory)
uncertainty is in σ; with a ~0.04×10⁻⁵ theory term the D/H pull would drop to about 2σ. AlterBBN's D/H is
1.1 % below PRIMAT (older d(p,γ)³He rate). The AlterBBN Y_p is 4·n_He4/n_b (Y_p^BBN), and Aver quotes a mass
fraction (about 0.5 % lower in the atomic-mass convention). The v5 table `superblock_v5_predictions` still
quotes D/H 2.51×10⁻⁵ for "AlterBBN v2.2, N11", which this Planck-input run does not reproduce (2.425×10⁻⁵).

`menus/prime_past/bbn_network.py`: marked **DEPRECATED** (docstring, `DEPRECATED = True`, DeprecationWarning
from `run_bbn` / `run_bbn_comparison`). Not deleted. The empirical "BBN integrator status" action still runs
it and now also attaches the AlterBBN gate report.

Mirron kernel scope:
- `superblock_residual`: `MIRRON_C_SCOPE = {LHCB_FLAVOR}`; `score_against_template(…, dataset=)` refuses
  Template C for `CERN_OPENDATA` and `CASIMIR` (even if q² is supplied) with
  `template_C_refused = "awaiting author ruling on mapping"`, `kernel = "refused"`, chi2_SB = None; their
  reports also carry the note. chi2_Lambda (data vs SM / ideal-PFA baseline) is still computed. LHCb is unchanged
  (chi2_SB 12.47 vs chi2_Lambda 11.05).

Tests:
- New `tests/test_prime_past_alterbbn.py` (10): Planck baseline values and cache layout; missing binary raises;
  gates all green from AlterBBN; gates follow the output (fake runs flip G1/G2/G3/G4/G5); unavailable report;
  loader χ² vs Cooke/Aver with SB pending; clean failure without AlterBBN; refusal when a gate fails; curated
  REPOS entry; toy network deprecated but present. AlterBBN-dependent tests skip when `sb_bbn.x` (or
  `sb_bbn_full.x`) is not found.
- New `tests/test_mirron_scope.py` (8): scope constants; refusal for CERN/Casimir in `score_against_template`,
  `residual_report` and full runs; LHCb still scored.
- `test_residual_loaders::test_prime_past_reports_missing_baseline` → `test_prime_past_has_an_alterbbn_loader`.
- Suite: **95 passed, 1 skipped, 3 xfailed** (skip: Planck TT not cached). With AlterBBN hidden: 90 passed,
  6 skipped, 3 xfailed.

## Follow-up (2026-11-06): AlterBBN works out of the box (bootstrap, Prime Past menu, D/H 2.51 explained)

On a machine without AlterBBN, the previous patch skipped the Prime Past tests. Now the repo fetches and
builds AlterBBN itself, into its own gitignored `datasets/prime_past/`. Nothing outside the repo and no
Python environment is touched.

Source and bootstrap (`bbn_isolated/alterbbn/`):
- Upstream: `https://alterbbn.hepforge.org/downloads?f=alterbbn_v2.2.tgz` (live over HTTPS, 4,499,370 bytes,
  sha256 `46b94253cde6afce15ddb9367fa454bac91b25c9a1d3412cc19e512ec3515c17`, identical to the box copy), GPLv3.
  Unpacked it is 22 MB / 40 files (18.7 MB is `src/contrib/newreac/results02200820.dat`), so it is fetched and
  cached on demand rather than vendored. Only the two small Superblock drivers are in git:
  `sb_bbn.c` (the box's original driver, now vendored) and `sb_bbn_full.c`.
- `ensure_alterbbn()`: if `sb_bbn.x` or `sb_bbn_full.x` is not found, download the tarball to
  `datasets/prime_past/alterbbn_v2.2.tgz` (sha256 checked; a mismatch is rejected and removed), unpack it with
  member checks into `datasets/prime_past/alterbbn_v2.2/`, and compile both drivers there.
- Compiler: the first of `$CC`, cc, gcc, clang. Flags `-O3 -fomit-frame-pointer -ffast-math -fno-finite-math-only
  -std=gnu99 -Wno-unknown-pragmas`, **no `-fopenmp`** (Apple Clang lacks it). A serial build from the pristine
  tarball gives output **identical to all printed digits** to the box's OpenMP `sb_bbn.x` at failsafe 0–3
  (the OpenMP pragmas are only in the error-propagation loop, which these drivers don't use).
- Lookup order: `$SB_ALTERBBN_DIR`, `datasets/prime_past/alterbbn_v2.2/`, `../bbncodes/alterbbn_v2.2/`,
  `~/bbncodes/alterbbn_v2.2/`, `$PATH` (the hard-coded `/workspace/…` path was removed).
- Switches: `SB_ALTERBBN_NO_AUTO_BUILD=1` (never fetch/build), `SB_ALTERBBN_TARBALL=/path/alterbbn_v2.2.tgz`
  (offline copy), `SB_ALTERBBN_DIR` (existing tree). `SB_KAPPA` is stripped from the AlterBBN environment so the
  baseline can never pick up the lever-(e) rate scaling.
- The box's `src/bbnrate.c` carries the local `SB_KAPPA` patch (a no-op unless `SB_KAPPA` is set). The bootstrap
  builds the pristine upstream file, and the numbers are the same.
- CLI: `python -m bbn_isolated.alterbbn status|ensure|build [--force] [--dir TREE]`.
  `build_full_driver.sh` now wraps `build`.
- Lazy use: `alterbbn.run()` bootstraps once per process when a binary is missing (unless disabled).

Research tool (Prime Past):
- Before: the menu had "Verification Suite / Generate Visualizations / Export LaTeX Appendix / Monte-Carlo Domain
  Scan". All of them were residual-routed and returned only a report: no plot, no fetch, no gate check. There was
  no Dataset Manager provider for Prime Past, and the auto-dataset hook pulled nothing.
- Now `menus/prime_past/extension.py` has:
  - **Fetch & Build AlterBBN v2.2** (loader): bootstrap, cache the Planck-input baseline CSV, write
    `bbn_isolated/gates.json` + `report.json`.
  - **BBN Gate Check (AlterBBN)** (loader): G1–G5, writes the two JSONs.
  - **Score BBN Residuals**, **Plot BBN Residuals**, **Full BBN Residual Report** (residual engine). The plot
    actions save the pulls chart `artifacts/prime_past/prime_past_bbn_residuals.png` and open a pop-up when
    Graphics = `popup`; with no residuals they print why and skip the plot.
  - The old action names still resolve (`LEGACY_ACTIONS`), but they are off the menu.
- `superblock_mode.LOADER_ACTIONS` lists the two loader actions.
- `sb_shared/dataset_registry.py`: new `PRIME_PAST_HARMONIC` provider with targets `prime_past:alterbbn_v2.2`
  (binaries) and `prime_past:alterbbn_planck_baseline` (CSV). Dataset Manager fetch = Fetch & Build.
- `sb_shared/auto_dataset.py`: before a scoring action, bootstrap AlterBBN if it is missing. A failure is printed
  and the loader returns a clean `load_error`, so the menu never aborts.
- `bbn_isolated.runner.run(check_isolated=False)` is what the menu calls (the menu process has other modules
  loaded; the gate code still imports none of them).
- The residual report now also carries `table` (D, B, σ, pull per observable).

`scripts/run_v5_tests.sh`: runs `python -m bbn_isolated.alterbbn ensure` before pytest and prints where AlterBBN
resolved, or that its tests will skip. `--no-alterbbn` sets `SB_ALTERBBN_NO_AUTO_BUILD=1`.

Why v5 quotes D/H = 2.51×10⁻⁵ for "AlterBBN v2.2, N11": that number is **not an AlterBBN output**.
`forks/open_questions/BBN_physics_superblock.md` (§ Calibration, l. 6, and l. 254/258) runs AlterBBN at
η = 6.104×10⁻¹⁰, τ_n = 878.4 s, dNnu = 0 (Nnu 3.046). That gives Y_p 0.2469 and **D/H 2.43×10⁻⁵**; re-run here,
the result is 2.4315×10⁻⁵. The note then reports every scenario as AlterBBN's *shift* added onto literature
reference values: Y_p 0.24691 (Fields et al. 2020) and **D/H (2.51 ± 0.11)×10⁻⁵ from Yeh, Olive & Fields 2021,
which uses the LUNA 2020 d(p,γ)³He rate**. AlterBBN v2.2 (2019) predates LUNA, and its older, faster d(p,γ) rate
burns more D, so its absolute D/H is about 3 % low. The 2.425×10⁻⁵ here is the raw AlterBBN value at the ruled
Planck inputs (η 6.114, τ_n 877.75, N_eff 3.044); the change of inputs moves it only −0.27 %. So the gap is
**reference anchoring plus the d(p,γ) rate, not inputs or an AlterBBN version**. The v5 row label
"D/H (AlterBBN v2.2, N11)" is a misnomer for "Yeh 2021 + AlterBBN shift". Canon files are not edited.

Tests:
- New `tests/test_alterbbn_bootstrap.py` (12): pinned URL/sha/paths, no OpenMP flag; the no-auto-build switch;
  sha256-mismatch rejection; unsafe tar members refused; portable build in a scratch tree identical to the
  resolved binary; CLI; the `run_v5_tests.sh` hook; Prime Past menu routing / legacy aliases / menu tree +
  registry provider; plot handler (PNG and empty case); Gate Check action; Fetch action without AlterBBN.
- `test_prime_past_alterbbn.py` / `test_residual_loaders.py`: the skip markers now try one bootstrap first. The
  missing-binary tests use a `no_alterbbn` fixture (nothing found + `SB_ALTERBBN_NO_AUTO_BUILD=1`), so they still
  exercise the missing path and never download.
- Suite: **107 passed, 1 skipped, 3 xfailed** (skip: Planck TT not cached). A clean copy with no AlterBBN anywhere
  bootstraps (download, verify, compile) and gives the same 107/1/3 via `scripts/run_v5_tests.sh`. With
  `--no-alterbbn`: **101 passed, 7 skipped, 3 xfailed**.

## Follow-up (2026-11-06): LIGO GWOSC ringdown fixed end to end (catalog versions, PE spin, whitening)

**Why the user's run failed:** (1) `fetch_all_events` keyed allevents by `commonName` with last-wins, so a
preliminary v1/v2 entry (no `final_mass_source` / `redshift`) could overwrite a confident v3/v4; in today's
allevents (671 entries, 433 events) that rule loses mass/z for 33 events. (2) `final_spin` is **never** in
allevents; it lives only in the per-event detail JSON (`jsonurl`) under `parameters` (GWTC-2.1/3:
`final_spin`, GWTC-1: `a_final`), which was never read. So every event was skipped and the loader raised
without its notes.

- `gwosc_api`: `select_best_versions` / `entry_rank` (entries with mass+z first, then highest numeric
  version); `fetch_all_events` uses it; `cache_event_catalog` writes a `event_catalog.schema.json` sidecar
  (schema 2) and auto-refreshes a cache without it (clobbered) — offline it keeps and re-ranks the old cache.
  `fetch_event_detail_url`; `list_strain_files(detail=...)`.
- `fetcher`: `ensure_event_detail` / `load_event_detail` / `save_event_detail` cache
  `events/<EVENT>/event_detail.json`; a preliminary v1/v2 `jsonurl` without mass/z forces a catalog refresh
  first. `pull_event_strain` saves it on every pull, including when `done.txt` already marks the event.
- `qnm`: `extract_pe_parameters` (prefers `is_preferred` PE sets), `resolve_event_metadata`;
  `gr_prediction` resolves a `parameters` dict, reads spin bounds from `final_spin_*`/`a_final_*`/`chi_f_*`,
  and propagates redshift bounds into σ(M_det). `catalog_entry` uses numeric versions.
- `qnm.measure_ringdown` (new) / `measured_peak_hz` (same signature): never uses f_220; ≥4 s segments are
  whitened with a Welch median PSD (±0.5 s around the event excluded, ±32 s used), 35–512 Hz band-pass,
  Hilbert-envelope peak within ±0.10 s of the GPS (t_peak, SNR), 0.05 s post-peak window with a one-sided
  1→0 taper; σ = 1/T. Short synthetic segments skip whitening.
- `_load_ligo`: one point per detector file (`"GW150914 (H1)"`), options `event` (ALL/*/blank = all cached;
  unknown ID → FileNotFoundError listing cached events), `strain_path`, `auto_fetch_detail` (backfills the
  detail JSON only when a field is missing), `min_snr` (default 4; low-SNR segments skipped with a note).
  Report JSON gains `points` (f_obs, f220, σ parts, pull, SNR, t_peak).
- `run_superblock_residual` keeps loader notes when the loader raises (`LoaderError.loader_notes`).
- LIGO menu: `event` defaults to `ALL`, not required; `show_graphics` artifacts | popup | no; Full Ringdown
  Report saves `artifacts/ligo_gwosc/ligo_ringdown_residuals.png` (D vs B, pulls). Dataset Manager:
  `_ligo_target_options` — `gwosc:events/<EV>/<file>` sets event + strain_path, `gwosc:event_catalog.json`
  sets nothing, explicit event wins.

**Real GW150914 (GWTC-2.1 v4: M_src 61.5, z 0.10, PE χ_f 0.68 ± 0.05), 4096 s 4 kHz files:**

| detector | f_obs [Hz] | f_220^GR [Hz] | σ [Hz] (1/T 20.0 ⊕ GR 17.5) | pull | SNR | t_peak − GPS |
|---|---|---|---|---|---|---|
| H1 | 245.0 | 251.6 | 26.6 | −0.25 | 7.4 | +23.6 ms |
| L1 | 241.0 | 251.6 | 26.6 | −0.40 | 6.7 | +18.2 ms |

χ²_Λ = 0.22 (N = 2). Tests: `tests/test_ligo_ringdown.py` (17). Full suite 124 passed, 1 skipped, 3 xfailed
with GWOSC data cached; 123 passed, 2 skipped, 3 xfailed without it.

## Follow-up (2026-11-06): ringdown upgrades — blind (f, τ) fit, coherent stack + null veto, post-merger scan, 32 s strain, HEAVY_BBH

1. **Blind damped-sinusoid fit + GR (f, τ, Q).** `gr_prediction` adds Berti–Cardoso–Will (2,2,0)
   `q220` = 0.7000 + 1.4187(1−a)^−0.4990, `tau220_s` = Q/(πf), `gamma220_hz` = f/(2Q) and propagated
   `sigma_q`, `sigma_tau_s`, `sigma_gamma_hz` (M_src, z, a_f); old keys unchanged. `qnm.fit_damped_ringdown`
   fits e^{−t/τ}[c1 cos 2πft + c2 sin 2πft] on [t_peak + 2 ms, + 30 ms] by variable projection (exact LS for
   c1, c2; grid f 35–512 Hz in 2 Hz + top spectral peaks × τ ∈ {1.5, 3, 6, 12, 25} ms; bounded
   `least_squares`, f ∈ [35, 512] Hz, τ ∈ [0.8, 80] ms). σ from the sandwich (JᵀJ)⁻¹JᵀΣ_nJ(JᵀJ)⁻¹ with Σ_n
   the Toeplitz off-source autocovariance of the whitened, band-passed data. No GR quantity is an input
   (checked by a test that changes GR and gets identical D). Monte Carlo (red noise + damped sinusoid,
   32 s, envelope SNR ≈ 20, 60 draws): pull std ≈ 1.1 (f) / 1.0 (τ). At SNR ≲ 12 the f errors get
   non-Gaussian tails; at very high SNR (≳ 100) PSD-estimation noise in the whitening dominates. Welch
   segments are now 0.5 s (≈ 120 median averages on 32 s) because a noisy PSD distorts the whitened signal.
   `measured_peak_hz` keeps its periodogram behaviour. Residual rows are now **f and γ = 1/(2πτ), both in
   Hz** (γ only when τ is inside its bounds and σ_τ < τ). τ, Q, τ_220, Q_220 are in `report["points"]`.
   Two existing assertions change from `D.size == 1` to `D.size == 2` (f + γ row).
2. **Coherent stack + null stream** (`menus/gravitic/ligo/coherent.py`, default `detector_mode=coherent`).
   Each detector is whitened by its own PSD and scaled to unit off-source σ. The highest-SNR detector is the
   reference; the others are aligned by complex analytic cross-correlation in [t_peak − 60 ms, + 30 ms]
   (±12 ms H1–L1, ±30 ms with V1) with an FFT sub-sample shift + phase. Stack weights w_d ∝ SNR_d with
   Σw² = 1. Null h = (a h_ref − h̃_2)/√(1+a²), a clipped to [0.5, 2]. The veto fires when χ²_null,red > 4
   **and** the null holds > 15 % of the stack's excess energy. 15 %, not 50 %: with the clip a lone glitch
   leaves ≈ 25 % in the null, a real signal < 15 %. Detectors under SNR 4 are left out of the stack, with a note.
   For narrow-band ringdowns Δt is only fixed to ~ms, because Δφ absorbs the rest.
3. **Post-merger scan** (`menus/gravitic/ligo/postmerger.py`, new menu action *Scan Post-Merger Markers*,
   also part of *Full Ringdown Report*). Bank: f 50–400 Hz (10 Hz) × τ {4, 10, 25} ms orthonormal quadrature
   atoms. Delays 20–300 ms. Statistic: √(|ρ_stack|² − |ρ_null|²). Background: the same max-statistic on
   off-source windows (25 ms steps, ±1 s and 0.5 s edges excluded) at zero lag and under 4 time slides
   (±0.1, ±0.2 s per extra detector). That gives N_bg ≈ 5700 for 2–3 detectors and ≈ 1100 for one.
   Reported: FAP, FAR, σ_equiv; detected = σ_equiv ≥ 3 and the null checks pass.
4. **32 s strain + HEAVY_BBH + joint constraints.**
   - Why 4096 s files were downloaded: GWOSC's newest release of GW150914 (GWTC-2.1 v4) lists only
     8 strain entries, all 4096 s. Format and rate strings were fine. The 32 s files are listed in GWTC-1 v3.
     `fetcher.select_event_strain` now asks for an exact (hdf5, 32 s, 4096 Hz) match, then tries the older
     versions (`event_versions.json` cache of the raw allevents), and only then falls back to the shortest
     file. That is about 1 MB instead of about 125 MB per detector.
   - The loader uses one file per detector: the shortest that contains the GPS time.
   - `HEAVY_BBH_EVENTS` (10 events, all checked live for 32 s 4 kHz HDF5 + PE mass/spin/z): GW150914,
     GW170104, GW170814, GW190519_153544, GW190521, GW190521_074359, GW190602_175927, GW191109_010717,
     GW200129_065458, GW200224_222234. Pull accepts `HEAVY_BBH` and comma lists. The Dataset Manager shows a
     `gwosc:HEAVY_BBH` target.
   - `report["joint_constraints"]`: inverse-variance ⟨δf/f⟩, ⟨δγ/γ⟩, ⟨δτ/τ⟩, network SNRs, post-merger
     summary, χ²_Λ/N. The 2×2 figure shows f, γ (with τ), pulls with SNR/null badges, and the scan. The scan
     action also writes `ligo_postmerger_scan.png`.

**Real data, all 10 HEAVY_BBH events (32 s files, coherent mode):**

| event (detectors) | f_obs [Hz] | f_220^GR [Hz] | τ_obs [ms] | τ_220^GR [ms] | γ pull | f pull | net SNR | null χ²_red | post-merger loudest (delay, f, σ_equiv) |
|---|---|---|---|---|---|---|---|---|---|
| GW150914 (H1+L1) | 245.2 ± 7.8 | 251.6 ± 17.5 | 4.65 ± 1.21 | 4.06 ± 0.25 | -0.55 | -0.33 | 9.3 | 0.67 | 68 ms, 60 Hz, 0.70σ |
| GW170104 (H1+L1) | 261.7 ± 8.9 | 291.4 ± 34.5 | 5.42 ± 1.83 | 3.46 ± 0.38 | -1.49 | -0.84 | 7.4 | 0.88 | 198 ms, 220 Hz, -0.65σ |
| GW170814 (H1+L1) | 282.1 ± 7.7 | 292.3 ± 24.9 | 7.47 ± 2.74 | 3.68 ± 0.28 | -2.59 | -0.39 | 6.6 | 1.57 | 98 ms, 50 Hz, 1.17σ |
| GW190519_153544 (H1+L1) | 112.9 ± 2.1 | 129.1 ± 27.1 | 28.67 ± 10.04 | 9.34 ± 1.93 | -2.86 | -0.60 | 5.9 | 1.46 | 21 ms, 360 Hz, -0.18σ |
| GW190521 (H1+L1+V1) | 61.8 ± 3.1 | 70.9 ± 22.4 | 21.71 ± 9.02 | 13.47 ± 3.85 | -0.99 | -0.40 | 7.0 | 1.39 | 115 ms, 350 Hz, -0.00σ |
| GW190521_074359 (H1+L1) | 174.6 ± 3.7 | 198.4 ± 25.0 | 7.93 ± 1.34 | 5.35 ± 0.65 | -1.96 | -0.94 | 10.2 | 1.26 | 227 ms, 320 Hz, -0.70σ |
| GW190602_175927 (L1) | 94.9 ± 5.5 | 106.7 ± 25.7 | 8.82 ± 3.26 | 10.08 ± 2.28 | +0.30 | -0.45 | 5.7 | — | 131 ms, 370 Hz, 1.05σ |
| GW191109_010717 (H1+L1) | 103.0 ± 1.4 | 121.0 ± 28.1 | 36.07 ± 12.72 | 7.81 ± 1.59 | -3.61 | -0.64 | 6.4 | 0.87 | 20 ms, 80 Hz, 2.86σ |
| GW200129_065458 (H1+L1+V1) | 230.5 ± 7.1 | 249.4 ± 22.9 | 3.79 ± 0.76 | 4.37 ± 0.38 | +0.62 | -0.79 | 10.2 | 1.41 | 42 ms, 320 Hz, -0.08σ |
| GW200224_222234 (H1+L1) | 182.9 ± 4.6 | 195.4 ± 24.1 | 8.61 ± 2.01 | 5.58 ± 0.66 | -1.83 | -0.51 | 7.4 | 1.03 | 273 ms, 390 Hz, 0.72σ |

Joint: ⟨δf/f⟩ = −0.066 ± 0.039 (N = 10), ⟨δγ/γ⟩ = −0.372 ± 0.072, ⟨δτ/τ⟩ = +0.21 ± 0.12.
χ²_Λ = 42.89 over N = 20 rows (χ²_Λ/N = 2.14). No null-stream veto fired. No post-merger marker was
detected; the loudest is GW191109 at 2.86σ, 20 ms / 80 Hz, at the window edge, next to that event's long
fitted primary (τ 36 ms).

Reading: f agrees with Kerr within σ for every event, mostly because σ_GR from the catalog dominates.
The fitted damping is systematically slower than Kerr (γ low, τ long; the largest are GW191109 and
GW190519). The likely cause is the 2 ms start, which still includes merger/overtone content, and the
whitening filter's effect on the waveform. It is not evidence against GR. Later starts (`fit_dt_start_ms`)
are the next check.

Tests: `tests/test_ligo_ringdown_upgrades.py` (13). Full suite: 137 passed, 1 skipped, 3 xfailed with
GWOSC data cached; 136 passed, 2 skipped, 3 xfailed without it. Real 10-event Full Ringdown Report: ~2.5 min.
