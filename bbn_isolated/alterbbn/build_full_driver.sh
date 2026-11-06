#!/usr/bin/env bash
# Build sb_bbn.x and sb_bbn_full.x in an AlterBBN v2.2 tree (portable: $CC, cc, gcc or clang; no OpenMP).
# Same code path as the automatic bootstrap in bbn_isolated.alterbbn.build().
#   usage: bbn_isolated/alterbbn/build_full_driver.sh [ALTERBBN_DIR]   (default: datasets/prime_past/alterbbn_v2.2)
#   To fetch the pinned upstream source as well:  python -m bbn_isolated.alterbbn ensure
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
[[ -x "$ROOT/venv/bin/python" ]] && PY="$ROOT/venv/bin/python"
if [[ $# -ge 1 ]]; then
  exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m bbn_isolated.alterbbn build --dir "$1"
fi
exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m bbn_isolated.alterbbn build
