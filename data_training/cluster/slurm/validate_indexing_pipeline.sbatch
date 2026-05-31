#!/usr/bin/env bash
#SBATCH --job-name=ewald-validate-indexing
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

EWALD_TRAINING_MANIFEST="${EWALD_TRAINING_MANIFEST:-$EWALD_SCRATCH_DIR/manifests/training-manifest.jsonl}"
EWALD_VALIDATE_CMD="${EWALD_VALIDATE_CMD:-python -m ewald.data_training.validate_indexing_pipeline}"
EWALD_CONDA_ENV="${EWALD_CONDA_ENV:-ewald-training}"
RUN_ID="${SLURM_JOB_ID:-local}"
RUN_DIR="$EWALD_SCRATCH_DIR/validation-runs/$RUN_ID"

mkdir -p "$RUN_DIR" "$EWALD_RUNTIME_DIR/logs"

if [[ -n "${EWALD_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  for module_name in $EWALD_MODULES; do
    module load "$module_name"
  done
fi

if [[ -f "${EWALD_CONDA_SH:-}" ]]; then
  # shellcheck source=/dev/null
  source "$EWALD_CONDA_SH"
  conda activate "$EWALD_CONDA_ENV"
fi

cd "$EWALD_RUNTIME_DIR/repo"

# shellcheck disable=SC2086
$EWALD_VALIDATE_CMD \
  --manifest "$EWALD_TRAINING_MANIFEST" \
  --output-dir "$RUN_DIR" \
  --metrics "$RUN_DIR/indexing-metrics.json"

echo "Validation outputs: $RUN_DIR"
