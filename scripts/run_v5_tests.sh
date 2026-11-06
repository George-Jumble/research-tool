#!/usr/bin/env bash
# Run the research-tool test suite (Superblock v5 gates included) with the repo's existing tooling.
#
#   ./scripts/run_v5_tests.sh              # use ./venv if it exists, else the python3 on PATH
#   ./scripts/run_v5_tests.sh --fetch-data # first cache DESI DR2 + SPARC with the repo's own fetchers
#   ./scripts/run_v5_tests.sh -k v5        # extra args go straight to pytest
#   ./scripts/run_v5_tests.sh --no-alterbbn # do not fetch/build AlterBBN (its tests then skip)
#
# AlterBBN v2.2 (Prime Past baseline): if sb_bbn.x / sb_bbn_full.x are not found, the pinned upstream
# tarball (sha256-checked) is fetched into datasets/prime_past/ and compiled there with cc/gcc/clang.
# Nothing outside the repo's ignored datasets/ directory is touched. Offline or no compiler: tests skip.
#
# It never creates or modifies an environment. If a package is missing it tells you what to install.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/venv/bin/python" ]]; then
  PY="$ROOT/venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

FETCH=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--fetch-data" ]]; then FETCH=1
  elif [[ "$a" == "--no-alterbbn" ]]; then export SB_ALTERBBN_NO_AUTO_BUILD=1
  else ARGS+=("$a"); fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"$PY" - <<'PYCHECK'
import importlib.util, sys
need = ["numpy", "scipy", "pandas", "matplotlib", "pydantic", "requests", "yaml", "pytest"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    print("Missing Python packages: " + ", ".join(missing))
    print("Install into the interpreter you already use for research_tool, e.g.:")
    print("  ./venv/bin/pip install -r requirements-tests.txt   (or -r requirements.txt)")
    sys.exit(2)
PYCHECK

if [[ "$FETCH" == "1" ]]; then
  if [[ ! -d datasets/desi/bao_data/desi_bao_dr2 ]]; then
    mkdir -p datasets/desi
    git clone --depth 1 https://github.com/CobayaSampler/bao_data.git datasets/desi/bao_data
  fi
  "$PY" -c "from menus.astronomical.sparc.fetcher import fetch_mass_models_table as f; f()"
fi

if [[ -z "${SB_ALTERBBN_NO_AUTO_BUILD:-}" ]]; then
  "$PY" -m bbn_isolated.alterbbn ensure --quiet >/dev/null \
    && echo "[run_v5_tests] AlterBBN ready: $("$PY" -c 'from bbn_isolated import alterbbn as a; print(a.find_binary(a.BINARY_NAME))')" \
    || echo "[run_v5_tests] AlterBBN not available (offline or no C compiler); its tests will skip"
fi

exec "$PY" -m pytest -rsxX "${ARGS[@]}"
