# Superblock Research Tool

Interactive curses console for **3+1D Superblock / Mirron** residuals. Loaders for DESI DR2, SPARC, CMS, LHCb, LIGO/LISA, and Casimir stay. Default scoring does not apply KK, 142857, or 5D templates; the old extra-dimension template code has been removed.

**Entry point:** `research_tool.py`

Boot file: `config/superblock_only.json`.

## Residual analysis (live question)

After a standard 3+1D ΛCDM, SM, GR, or Newtonian SPARC reduction, do the residuals match Superblock / Mirron predictions and not a compact extra dimension?

Rejected on the default path: 7-fold / KK signatures, a 7 h⁻¹ Mpc compact S¹, the 142857 comb, RGC as a cosmological backend, and a 5D roll or Coleman bounce.

Full key and pipeline: [docs/RESIDUAL_ANALYSIS.md](docs/RESIDUAL_ANALYSIS.md).

Inventory: [docs/SUPERBLOCK_ONLY_INVENTORY.md](docs/SUPERBLOCK_ONLY_INVENTORY.md).

## Version

See [VERSION](VERSION) and [CHANGELOG.md](CHANGELOG.md). Current corpus: **2026.10.13**.

## Command list

- `python research_tool.py geometry` — prints \(m_0\), \(r_M\) from `sb_engine/constants.py`
- Live residual path (flagship v5, 2026-11-04): SPARC Template B = hierarchical \(L_{\rm res}\) kernel (`kernels/l_res_rl.py`), Prime's standard sound horizon (`sb_engine/sound_horizon.py`, no \(\gamma=10.74\)), FRB DM load
- live consistency: neutron module as a local mass ratio (`sb_engine/neutron.py`, 878.7 ± 2.1 s)
- v5 confrontations (menu **Superblock v5 Confrontations**; `tests/test_superblock_v5.py`): 5.4015 vs Planck, \(N_{\rm eff}\), BBN, \(r_d\), hidden BAO, relay hierarchy, local couplings
- retired: \(0^{++}=2m_0e\) glueball pack, G_avg kernel, 309.3 MeV floor (floor is \(\Delta=313.1\) MeV)
- not live: in-repo BBN yield network beyond the standard comparison, extra \(J^{PC}\) glueball packs
- Test changes for v5: [CHANGES_v5_tests.md](CHANGES_v5_tests.md)
- `sb_engine/{cms,casimir,sparc}.py` — SYNTHETIC illustrations only

The empirical menu does not feed PASS into SPARC, DESI, CMS, Casimir, or bounce.

## Historical notes (not the live path)

## Latest upgrade highlights

- **Merged fork code** — this V2 build includes the merged code from Clay Gatlin's fork, plus bug fixes.
- **Global fetch enforcement** — shared `DataManager` wiring is now available across runtime dispatch and ingestion validators, with `data_key`-based caching.
- **Persistent logging policy** — data operations now write to `research_tool.log` for fetch, cache-hit, and failure traceability.
- **Mock data fallback** — `use_mock` allows pipeline validation when live CERN/DESI sources are unavailable.
- **Constraint hardening** — resonance-style checks enforce minimum sample-size constraints (e.g., underpowered `n < 8` guards before periodogram paths).

## Gateway integration

This tree is merged into the Research **LLM Gateway** (`../gateway.py`):

```bash
cd /home/wyle-e/Research
./scripts/llm                 # gateway REPL
# then: /research   or  /rt   → full curses UI
#       /rt-status  /rt-modules  /rt-providers  /rt-run …

./scripts/rt                  # launch research_tool UI directly
./scripts/rt status           # bridge health check
```

API keys: `config/api_keys.env` (see example) or the same env vars the gateway uses
(`XAI_API_KEY`, `NVIDIA_API_KEY` / `NV_API_KEY`, `OLLAMA_BASE_URL`, …).
Gateway persistent memory (name, notes) is passed into the child process env.

## Legacy names

Module tags, directories and env vars were renamed (e.g. `tav_shared` → `sb_shared`, `tav_research` → `sb_research`, `LHCB_TAV_ECHO` → `LHCB_FLAVOR`). `sb_shared/legacy_names.py` maps the old names so existing datasets, staging archives, ledgers and env settings keep working; nothing on disk is moved or deleted.

## Remote archive transfer

Finished datasets are staged locally under `.sb_project_staging/` (an existing pre-rename staging folder is reused in place) and can be copied to a configured remote host when enabled.

Remote SSH host updates are **disabled in code** (local staging only).

**To re-enable SSH/SFTP/rsync sync:** see [enable_ssh_sync.txt](enable_ssh_sync.txt) for file-by-file uncomment instructions, required environment variables, and verification steps.

## Features

- **Modular menus** — each domain lives in `menus/*/` with thin `*_extension.py` shims at the project root for backward compatibility
- **Dataset manager** — fetch, cache, and resume batch downloads (DESI, Planck, SPARC, FRB, LHCb, …)
- **Prime Past / BBN** — standard BBN network comparison scored by the residual engine
- **Remote AI** — Grok, Gemini, OpenAI, Claude, OpenRouter, Ollama, NVIDIA NIM (`integrate.api.nvidia.com`)
- **Post-run LLM review** — set **AI verbose review = yes** on any module entry form

## Quick start

```bash
cd research_tool
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp config/api_keys.env.example config/api_keys.env
# Edit config/api_keys.env with your API keys (optional; needed for Remote AI)

./venv/bin/python research_tool.py
```

Or use the helper script:

```bash
./research.sh
```

## Configuration

| File | Purpose |
|------|---------|
| `config/api_keys.env` | LLM provider keys (gitignored; copy from `api_keys.env.example`) |
| `.env` | Optional fallback for the same keys |
| `requirements-desi-production.txt` | Extra DESI MCMC dependencies (`emcee`, `ruptures`, …) |

**Never commit** `config/api_keys.env` or `.env`.

## Project layout

```
research_tool.py          # Thin entry → sb_research/runner.py
sb_research/             # Curses UI, menu tree, registry, data pull
sb_shared/               # LLM analysis, remote AI, run output, paths
menus/                    # Domain modules (prime_past, astronomical, particle, …)
config/                   # api_keys.env.example
scripts/                  # Maintenance utilities
```

## Empirical auto-fetch (`empirical:` targets)

Before each run, `ensure_datasets_before_run` materializes curated empirical tables:

| Module | Actions | Targets |
|--------|---------|---------|
| `EMPIRICAL_TESTS` | per-action keys from the extension | `empirical:*` |

Config: `sb_shared/empirical_action_targets.py`. Provenance is stored on `options['_empirical_provenance']` and written into JSON reports.

## Automated ingestion validation

Pydantic-backed validation runs before data reaches Superblock solvers and `log_likelihood`:

| Package | Role |
|---------|------|
| `sb_shared/ingestion/schemas.py` | CMS dimuon, BBN empirical models |
| `sb_shared/ingestion/normalize.py` | CSV / JSON / ROOT-derived dict → canonical internal format |
| `sb_shared/ingestion/pipeline.py` | `IngestionValidationReport`, batch validators, anomaly filtering |
| `sb_shared/ingestion/config.py` | Physical bounds; override via `SB_INGESTION_CONFIG` JSON path (old `TAV_` name still accepted) |

**Integrated entry points:** `extract_dimuon_kinematics_from_nanoaod`, `fetch_empirical_data` (BBN).

## Standardized dataset comparison

Apples-to-apples MC vs data and model vs observed checks during preprocessing:

| Package | Role |
|---------|------|
| `sb_shared/dataset_comparison/scaling.py` | `StandardScaler`, `MinMaxScaler` (numpy, sklearn-compatible API) |
| `sb_shared/dataset_comparison/keys.py` | Primary-key rounding, sort, inner-join alignment |
| `sb_shared/dataset_comparison/tolerance.py` | `tolerant_diff`, `compare_mass_gap_mev` with configurable thresholds |
| `sb_shared/dataset_comparison/pipeline.py` | `prepare_datasets_for_comparison`, `compare_histogram_pair`, reports |

Override tolerances and scalers via `SB_COMPARISON_CONFIG` JSON path (old `TAV_` name still accepted).

**Integrated entry points:** `normalize_dimuon_events` (key-sorted), `log_likelihood_breakdown` (model vs observed tolerance block).

## Dependencies

Core scientific stack: NumPy, SciPy, Matplotlib, Astropy, healpy, uproot/xrootd for ROOT/FITS I/O.

Large datasets and run artifacts are **not** in the repo — they are downloaded or written under `datasets/` and `artifacts/` at runtime (both gitignored).

## Publishing to GitHub

This directory is a self-contained git export (no venv, artifacts, or secrets). After creating a repo on GitHub:

```bash
cd /path/to/research_tool
./scripts/push_to_github.sh git@github.com:YOUR_USER/YOUR_REPO.git
```

Or set the remote yourself:

```bash
git remote add origin git@github.com:YOUR_USER/YOUR_REPO.git
git push -u origin main
```

## License
