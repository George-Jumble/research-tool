# Superblock-only inventory (2026-09-21)

Live call graph of the research tool after the prune. Verdicts: KEEP, PRUNE, DISABLE, ARCHIVE-ONLY, RETARGET.

Historical Tau kernels remain on disk. They are not imported by the GUI and their `run_action` / scoring CLI entry points do not run.

| path | current role | verdict |
|---|---|---|
| research_tool.py | GUI entry. No Tav-resonance bootstrap | KEEP |
| config/superblock_only.json | 3+1D Superblock boot. tau_5d_templates forced false | KEEP |
| superblock_mode.py | Menu filter, retired-module errors, route to residuals | KEEP |
| superblock_residual.py | Steps 0–4. S_k only. T_τ refused | KEEP |
| tav_research/menu_tree.py | Primary Matrix. Retired actions dropped | RETARGET |
| tav_research/registry.py | Imports live loaders only | RETARGET |
| tav_research/runner.py | Runs menu actions. Catches RetiredModuleError | RETARGET |
| tav_research/gateway_launcher.py | Gateway label is Superblock Research Tool | RETARGET |
| menus/astronomical/desi/extension.py | DESI DR2 loader + residual route. 1/7 menu removed | RETARGET |
| menus/astronomical/desi/scanner.py | Loader kept. Scoring CLI returns retired note | RETARGET |
| menus/astronomical/desi/fetcher.py | DESI DR2 file loader | KEEP |
| menus/astronomical/sparc/extension.py | SPARC CSV pull kept. Fits route to template B | RETARGET |
| menus/astronomical/sparc/fetcher.py | SPARC loader | KEEP |
| menus/astronomical/sparc/analyzer.py | TauSuperblockSolver left on disk, not the default path | DISABLE |
| menus/astronomical/frb/extension.py | FRB pull kept. 1/7 scans removed. DM action is residual E | RETARGET |
| menus/astronomical/frb/fetcher.py | FRB catalog loader | KEEP |
| menus/astronomical/tav_resonance/extension.py | Tav-resonance package. run_action raises | DISABLE |
| menus/astronomical/berard/extension.py | tau-cosmology Berard. run_action raises | DISABLE |
| menus/planck_cmb/extension.py | Power-spectrum action is residual E. 1/7 scans removed | RETARGET |
| menus/planck_cmb/tav.py | Planck FITS loader plus old 1/7 search. Not the default score | DISABLE |
| menus/gravitic/ligo/extension.py | GWOSC pull kept. 1/7 ringdown removed. Report is residual E | RETARGET |
| menus/gravitic/ligo/fetcher.py | Strain loader | KEEP |
| menus/gravitic/ligo/ringdown.py | Old 1/7 ringdown. Not called from the default menu | DISABLE |
| menus/gravitic/lisa/extension.py | OSDF mock-data loader only | KEEP |
| menus/particle/cern/extension.py | CMS/ATLAS/ALICE pull kept. 7-fold actions removed | RETARGET |
| menus/particle/cern/fetcher.py | CERN Open Data loader | KEEP |
| menus/particle/lhcb/extension.py | LHCb download kept. Echo actions are residual C | RETARGET |
| menus/particle/lhcb/echo.py | Old Tav-echo kernel. Not the default score | DISABLE |
| menus/particle/casimir/extension.py | Downloads kept. Tau falsification suite not on the menu | RETARGET |
| menus/particle/casimir/scanner.py | Scoring CLI retired. Loader functions remain | RETARGET |
| menus/particle/rgc/extension.py | 1/7 mock catalog as cosmology. run_action raises | PRUNE |
| menus/integrator/extension.py | Tav Framework Integrator. run_action raises | DISABLE |
| menus/tsb_research/extension.py | Cylinder / fractal Tau engine. run_action raises | PRUNE |
| menus/tsb_research/fractal_tau_mcmc.py | Fractal Tau likelihood. Not imported by the GUI | PRUNE |
| menus/prime_past/extension.py | Down-quark / isospin menu. BBN-Tav actions removed | RETARGET |
| menus/prime_past/bbn_menu_extension.py | Tav BBN. Not on the menu. run_action raises | PRUNE |
| menus/prime_past/harmonic.py | Historical harmonic engine. Not the default score | DISABLE |
| menus/empirical_tests/extension.py | Quarks, neutron lifetime, Yang-Mills, BBN abundances | KEEP |
| menus/empirical_tests/tests.py | Superblock empirical confrontations | KEEP |
| menus/remote_ai/extension.py | Remote AI review. Not a cosmology backend | KEEP |
| tav_shared/mcmc_engine.py | Default halo comparison is NFW/Einasto/Burkert/Hernquist/pISO. Mirron name replaces soliton | RETARGET |
| tav_shared/bound_unbound_energy.py | Energy partition helper. Not a 5D backend | KEEP |
| tav_shared/tav_resonance_bootstrap.py | Not called at boot | DISABLE |
| tav_bbn_confrontation.py | Historical Tav BBN CLI script. Not on the menu | ARCHIVE-ONLY |
| tav_bbn_confrontation.md | Moved to docs/rejected/ | ARCHIVE-ONLY |
| preregistration.md | v4 Tau-cylinder preregistration. Moved to docs/rejected/ | ARCHIVE-ONLY |
| docs/RESIDUAL_ANALYSIS.md | Live question and rejected questions | KEEP |
| docs/retired/RETIRED_TAU_MODULES.md | Note the GUI prints for a retired item | KEEP |
| theoretical_model.py | Shared model helpers. Not a menu cosmology | KEEP |
| superblock.py | Superblock helpers | KEEP |

## Diff summary

**Pruned from the default path (not deleted):** RGC mock catalog, Tau-Superblock Research Engine (cylinder / fractal Tau), Tav BBN scans, 1/7 and 7-fold and EKK and comb menu items.

**Disabled (file on disk, GUI does not import, `run_action` raises):** Tav-Resonance, Berard tau-cosmology, Tav Framework Integrator, Tav-resonance bootstrap at startup.

**Retargeted:** DESI, SPARC, FRB, Planck, LIGO, CERN, LHCb, Casimir, Prime Past. Loaders stay. Scoring actions call `superblock_residual` (templates A–E). Gateway title and engine prefix renamed.

**Archive-only:** `docs/rejected/preregistration.md`, `docs/rejected/tav_bbn_confrontation.md`.

**Renames:** `TAU-SB Gateway` → `Superblock Research Tool`. `[Superblock Residual Engine]` on the live shell → `[Superblock Residual Engine]`. User-facing soliton profile → Mirron. Default MCMC model is NFW, not Tav.

## Default config

See `config/superblock_only.json`. `superblock_mode.load_boot_config()` forces `superblock_only: true` and `tau_5d_templates: false`.
