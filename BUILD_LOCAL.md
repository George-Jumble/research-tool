# Build and run locally (v5 tests update)

This tree is `George-Jumble/research-tool` `main` @ `8a2e32b` (tag `v2026.10.13`) plus the local branch
`v5-tests-update`. See `CHANGES_v5_tests.md` for what changed. It uses only the repo's existing tooling
(Python 3.11+, `venv`, `requirements.txt`), with nothing new to install system-wide.

## Option A: your existing research_tool venv (no changes)
If you already run `research_tool.py` from `./venv`, it has everything the tests need. Then:
```bash
./scripts/run_v5_tests.sh            # uses ./venv/bin/python automatically
```

## Option B: fresh checkout, the README's own setup
```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt        # full tool, as in README
#   or, tests only:  ./venv/bin/pip install -r requirements-tests.txt
./scripts/run_v5_tests.sh
```

## Option C: no venv at all
If the `python3` on your PATH already has numpy, scipy, pandas, matplotlib, pydantic, requests, PyYAML and
pytest, the script uses it as is (`PYTHON=/path/to/python ./scripts/run_v5_tests.sh` to choose one). It
only checks imports and tells you what is missing. It never installs anything.

## Real data (optional, same fetchers the tool uses)
```bash
./scripts/run_v5_tests.sh --fetch-data
```
This caches DESI DR2 (`datasets/desi/bao_data`, Cobaya clone) and the SPARC MassModels table
(`datasets/sparc/`, via `fetch_mass_models_table`). Without them, 4 DESI loader tests skip with the
clone command in the reason, and the SPARC test downloads the table once (or skips offline).

## Expected result
`77 passed, 1 skipped, 3 xfailed` with DESI tables cached (`73 passed, 5 skipped, 3 xfailed` without them).
The 3 xfails are open questions: lithium-7, the γ¹ derivation, and the Lyman-α check.

## Git
If you got the `.bundle`: `git clone research-tool_v5_tests.bundle research-tool && cd research-tool &&
git checkout v5-tests-update`. Or apply `research-tool_v5_tests.diff` to a checkout of `8a2e32b` with
`git apply research-tool_v5_tests.diff`.

The v5 confrontations can also be run from the curses UI: Empirical Tests → **Superblock v5 Confrontations**,
or `python -m menus.empirical_tests.tests --tests v5 --sources planck2018_cosmology,superblock_v5_predictions,bbn_abundances,hidden_bao_class_grid,relay_class_configs`.
