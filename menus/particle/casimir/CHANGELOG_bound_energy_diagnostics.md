# Casimir bound-energy diagnostics (2026-09-15)

Fix the five-test falsification battery, add controls/audits, and restore
ΔE = E_obs − E_Lifshitz. Do not loosen `TSB_EVIDENCE_ALPHA = 0.05`.

Local hashing and 7-file auto-discovery under `datasets/casimir` are preserved.

## New modules

| File | Role |
|---|---|
| `diagnostics.py` | Observable classification, hash ledger, family preregistration, console extras, diagnostics menu entry |
| `lifshitz_approx.py` | Approximate planar Matsubara / TE-TM Fresnel integrator (not a scattering code) |
| `bound_energy.py` | ΔE pipeline, identity subtraction, floor test, bound-energy menu entry |
| `tests/test_diagnostics.py` | Seven acceptance cases |

## Functions changed

### `test_framework.py`

- `test_log_modulations` — permutation p-value + mandatory data/shuffle/surrogate audit; raw `a, f, phi, c, chi2_raw, N`; never divides residuals by `std(residuals)` for a p-value; `identically_null` when p or `chi2_raw/N` agree to 3 digits on data vs shuffle
- `test_periodicity` — adds `dominant_period`, `Delta_xi`, `ratio`, `grid_locked` (ratio within 2% of an integer)
- `test_hysteresis` — residual-block bootstrap (B=1000) with effect-size and p CIs; two-pass loop area when up/down, forward/reverse, retract/approach columns exist
- `test_anisotropy_or_backreaction` — deprecated dispatcher only
- **New** `test_powerlaw_index` — legal only for force/energy vs separation; n/a on frequency/dielectric; not counted in `S_TSB` when n/a
- **New** `test_anisotropy_channels` — aux vs `eps_pri` spread; mandatory broadcast-`eps_pri` negative control (`control_failed`)
- `run_full_test_suite` — persists `raw_series_sha256`, ledger, preregister lock, `observable_kind`, extra console fields; V_PPR JSON key is `v_ppr_calibration_residual_pN` (not ΔE)
- `run_falsification_from_source` / `run_falsification_on_local_downloads` — refuse to pool hash-mismatched labels; `--relock`, `--bound-energy`

### `scanner.py`

- `detect_tau_resonances` — classifies `observable_kind`; does not treat ε(iξ) vs ξ as a d⁻⁴ force law; runs `anisotropy_channels` + control when aux columns exist; optional ΔE second pass labeled `observable="bound_energy_residual"`
- `_try_tav_harmonic_check` — JSON name `detrended_harmonic_residual` (not ΔE)
- `_write_report` / `CasimirScanResult` — `raw_series_sha256`, `param_col`, `value_col`, `observable_kind`
- `_run_casimir_scan_body` — hash ledger + family lock attached to the scan report
- `plot_and_save` — dielectric tables are not titled as a force power-law

### `extension.py`

- Menu actions added: `TSB Falsification Diagnostics (controls + audit)`, `TSB Bound-Energy Residual Search`
- `handle_pre_form`, `entry_fields`, `entry_instructions`, `run_action` — skip-form + prefixes `tsb_diagnostics_` / `tsb_bound_energy_`
- Existing `TSB Falsification Test Suite (5 predictions)` kept

### `diagnostics.py` (new)

- `classify_casimir_table`, `print_casimir_load_context`
- `preregister_split` / `load_family_lock`
- `record_hash_ledger` (append-only; HASH MISMATCH refuses pooling)
- `run_diagnostics_battery`
- `format_test_console_line`, `print_falsification_report`, `score_s_tsb`

### `lifshitz_approx.py` (new)

- `free_energy_density`, `free_energy_density_quadrature_check`, `infer_xi_rad_s`
- Approximations listed in `LIFSHITZ_APPROXIMATIONS` and copied into JSON

### `bound_energy.py` (new)

- `estimate_bound_energy`, `identity_subtraction`, `floor_test_E_bind`
- `run_bound_energy_search` — pairs dielectric and force/energy tables by Ag/Al/Au/Li/Ben_Au/Cyc_Au/Oct_Au

### Unchanged on purpose

- `series_integrity.py` hashing + distinctive-column auto-discovery
- `discover_local_casimir_files` 7-file discovery
- DESI / Berard / CMS modules
- Curses UI beyond the two new actions and their entry copy

## Acceptance tests (`tests/test_diagnostics.py`)

1. `test_log_modulations_permutation_distinguishes_sine_from_noise` — planted sine vs white noise permutation p-values differ; old self-std χ² path is gone
2. `test_anisotropy_identical_aux_is_null_and_control_ok` — identical aux copies → evidence False, `control_failed` False
3. `test_powerlaw_index_dielectric_vs_xi_not_applicable`
4. `test_powerlaw_index_lifshitz_force_law_is_not_tsb` — y ∝ d⁻⁴ → evidence False
5. `test_periodicity_sampling_comb_is_grid_locked`
6. `test_identity_lifshitz_subtraction_has_no_e_bind`
7. `test_hash_ledger_stable_then_mismatch` — same bytes twice does not warn; changing one value does

Run:

```
python -m pytest menus/particle/casimir/tests/test_diagnostics.py -q
python menus/particle/casimir/tests/test_diagnostics.py
```
