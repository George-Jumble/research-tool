# Blind-analysis protocol (Casimir / TSB signatures)

This document is the contract an independent reviewer can use to confirm that
the **blind** path does not receive a 1/7 (or seven-phase / TSB) prior.

## How to run

```bash
# Blind run (generic strongest-period search; reports use a blind_ prefix)
python research_tool.py --blind
# or, Casimir CLI:
python -m menus.particle.casimir.scanner --data local --local-csv PATH --blind --no-plot

# After the analyst is satisfied the blind reports are final:
python research_tool.py --lock-blind

# Informed run (default; 1/7 residual test)
python -m menus.particle.casimir.scanner --data local --local-csv PATH --no-plot

# Side-by-side table — refused unless the lock file is present and hashes match
python research_tool.py --compare-blind
```

`--compare-blind` **will not** write a table until `--lock-blind` has frozen
the current `*blind_*` JSON reports. If a locked file is edited, comparison
aborts.

Lock file: `artifacts/casimir/blind_lock.json`

## Code paths (exact)

| Concern | Informed | Blind |
|---------|----------|-------|
| Detector module | `menus/particle/casimir/scanner.py` → `detect_tau_resonances` | `menus/particle/casimir/blind_detectors.py` → `detect_generic_signatures` |
| Periodicity test | `significance.residual_periodogram_significance` scores **pre-specified** `expected_one_seventh_frequency` = `7 / span(x)` | `blind_detectors.generic_strongest_period` scores **argmax** residual power; **no target frequency** |
| Seven-phase grid | `_seven_phase_positions` (k/7 of the parameter span) | **not called** |
| Step alignment | `random_placement_pvalue` vs seven-phase markers | residual exceedances only; **no alignment grid** |
| Claim flag | `tav_resonance_detected` iff Holm-adjusted 1/7 p < α (default 0.01) | **flag is omitted** (`tav_harmonics` is `null`) |
| Feature names | `seven_phase_clockwork`, `discrete_steps_thresholds`, `hysteresis_asymmetry` | `periodicity_search`, `residual_steps`, `residual_asymmetry` |
| Report prefix | `tsb_casimir` (or caller prefix) | always `blind_<prefix>` |
| Null-hypothesis gate | applied | **not** applied (would leak informed envelope) |
| Plot | expected 1/7 line; 1/7 autocorrelation lags | strongest-peak line only; no preferred lag |

Shared, **prior-free** helpers (used by both paths):

- `significance.fit_smooth_trend` (power-law / Drude+interband)
- `significance.residual_step_pvalue`
- `significance.hysteresis_pvalue`
- `significance.pelt_changepoints_with_bootstrap`
- `significance.holm_adjust`

These helpers do not encode a 1/7 frequency.

## Grep audit (blind module)

The following strings **must not** appear in
`menus/particle/casimir/blind_detectors.py`:

```
1/7
1 / 7
0.142857
142857
seven_phase
clockwork
tav_resonance
TAU_RESONANCE
ONE_SEVENTH
expected_one_seventh
```

Dispatch is a single branch at the top of `detect_tau_resonances`:

```
if blind:
    return detect_generic_signatures(...)
```

No 1/7 constant is passed into the blind function.

## Report schema differences

Informed JSON includes:

- `analysis_mode: "informed"`
- `detected_features[].type == "seven_phase_clockwork"` with `expected_freq_1_over_7`
- `tav_harmonics.tav_resonance_detected`

Blind JSON includes:

- `analysis_mode: "blind"`
- `detected_features[].type == "periodicity_search"` with `strongest_period` / `strongest_frequency`
- `tav_harmonics: null`

Filenames contain `blind_` in the series token, e.g.
`casimir__dielptag005__report__HH-MM_blind_tsb_casimir.json`.

## Comparison table

Written only after a valid lock. Columns:

- blind strongest period and p-value
- informed expected 1/7 frequency and Holm p-value
- informed `tav_resonance_detected`
- `blind_has_tav_block` (must be false)

The table is a **post-hoc** document. It is not an input to the blind detectors.
