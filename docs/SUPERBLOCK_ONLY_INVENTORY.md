# Superblock-only inventory (2026-11-04)

Live call graph of the research tool. The retired extra-dimension / 7-fold
template code, its menus and its guard plumbing were deleted (they remain in
git history only).

| path | current role |
|---|---|
| research_tool.py | GUI entry |
| config/superblock_only.json | 3+1D Superblock boot config |
| superblock_mode.py | Routes actions: loader, keep, or residual engine |
| superblock_residual.py | Steps 0–4. Scores residuals against S_k only |
| sb_research/ | Curses UI, menu tree, registry, runner |
| sb_shared/ | Paths, ledgers, dataset registry, LLM review, remote AI |
| sb_shared/legacy_names.py | Old tag / dir / env-var names → current names (keeps existing data loading) |
| menus/astronomical/desi/ | DESI DR2 loader (`loader.py`, `covariance.py`, `fetcher.py`) + residual route |
| menus/astronomical/sparc/ | SPARC CSV pull; fits route to template B |
| menus/astronomical/frb/ | FRB pull + catalog loader (`catalog.py`); DM action is residual E |
| menus/planck_cmb/ | Planck FITS fetcher; power-spectrum action is residual E |
| menus/gravitic/ligo/, lisa/ | GWOSC / OSDF loaders |
| menus/particle/cern/, cms/, atlas/ | CERN Open Data loaders; analysis is residual (CERN_OPENDATA) |
| menus/particle/lhcb/ | LHCb download + `flavor.py` loader; residual C |
| menus/particle/casimir/ | Downloads kept; scoring is residual |
| menus/prime_past/ | Residual-scored menu (AlterBBN v2.2 baseline via `bbn_isolated/`); toy `bbn_network.py` deprecated |
| menus/empirical_tests/ | Quarks, neutron lifetime, Yang-Mills, BBN abundances |
| menus/remote_ai/ | Remote AI review. Not a cosmology backend |
| sb_shared/mcmc_engine.py | Default halo comparison NFW/Einasto/Burkert/Hernquist/pISO |
| docs/RESIDUAL_ANALYSIS.md | Live question and rejected questions |

## Renamed

`tav_shared` → `sb_shared`, `tav_research` → `sb_research`,
`tav_project_paths` → `project_paths`, `TAU_SUPERBLOCK_ROOT` → `SUPERBLOCK_ROOT`.
Module tags: `TAU_SB_DESI` → `DESI_BAO`, `LHCB_TAV_ECHO` → `LHCB_FLAVOR`,
`FRB_COSMIC_WEB_TAV` → `FRB_DM`, `PLANCK_CMB_TAV` → `PLANCK_CMB`,
`TSB_CASIMIR` → `CASIMIR`. Env vars `TAV_*` → `SB_*` (old names still read).
Staging dir `.sb_project_staging/` (an existing legacy staging dir is reused in place).

## Default config

See `config/superblock_only.json`. `superblock_mode.load_boot_config()` forces `superblock_only: true`.
