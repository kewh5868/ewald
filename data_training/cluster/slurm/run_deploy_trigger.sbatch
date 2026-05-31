#!/usr/bin/env bash
#SBATCH --job-name=ewald-trigger
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

: "${EWALD_RUNTIME_DIR:?Set EWALD_RUNTIME_DIR before submitting}"
: "${EWALD_SCRATCH_DIR:?Set EWALD_SCRATCH_DIR before submitting}"
: "${EWALD_TRIGGER_DIR:?Set EWALD_TRIGGER_DIR, for example data_training/deploy/01_generate_simulations}"

EWALD_CONDA_ENV="${EWALD_CONDA_ENV:-ewald-py312}"
EWALD_CONDA_SH="${EWALD_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

if [[ -n "${EWALD_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  for module_name in $EWALD_MODULES; do
    module load "$module_name"
  done
fi

if [[ -f "$EWALD_CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$EWALD_CONDA_SH"
  conda activate "$EWALD_CONDA_ENV"
fi

export EWALD_REPO_ROOT="$EWALD_RUNTIME_DIR/repo"
export EWALD_RUN_BASE="${EWALD_RUN_BASE:-$EWALD_SCRATCH_DIR/runs}"

cd "$EWALD_RUNTIME_DIR/repo"
bash "$EWALD_TRIGGER_DIR/run.sh"

