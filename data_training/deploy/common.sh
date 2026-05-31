#!/usr/bin/env bash
set -euo pipefail

export EWALD_REPO_ROOT="${EWALD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export EWALD_RUN_BASE="${EWALD_RUN_BASE:-$EWALD_REPO_ROOT/data_training/runs}"
export EWALD_RUN_ID="${EWALD_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export EWALD_CONDA_ENV="${EWALD_CONDA_ENV:-}"

mkdir -p "$EWALD_RUN_BASE"

run_python() {
  if [[ -n "${EWALD_CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
    conda run --no-capture-output -n "$EWALD_CONDA_ENV" python "$@"
  else
    python "$@"
  fi
}

write_run_info() {
  local run_root="$1"
  mkdir -p "$run_root"
  {
    echo "run_id=$EWALD_RUN_ID"
    echo "repo=$EWALD_REPO_ROOT"
    echo "host=$(hostname)"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$run_root/RUN_INFO.txt"
}
