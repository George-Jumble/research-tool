#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p session_out
output_file="session_out/output_terminal_$(date -u +%Y%m%d_%H%M%S).txt"

PYTHON_BIN="${ROOT_DIR}/../.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ $# -eq 0 ]]; then
  test_cmd=("$PYTHON_BIN" -m unittest -v)
else
  test_cmd=("$@")
fi

{
  echo "[SESSION OUT] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[PWD] $PWD"
  echo "[CMD] ${test_cmd[*]}"
  echo
  "${test_cmd[@]}"
} 2>&1 | tee "$output_file"

echo "$output_file"

"$PYTHON_BIN" - <<'PY' "$output_file"
import sys
from tav_research.external_editor import open_in_external_editor

path = sys.argv[1]
if not open_in_external_editor(path):
    print(f"[WARN] Could not launch configured editor for: {path}")
PY
