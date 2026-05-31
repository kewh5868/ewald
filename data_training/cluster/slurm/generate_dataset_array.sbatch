#!/usr/bin/env bash
#SBATCH --job-name=ewald-generate
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-100

set -euo pipefail

: "${EWALD_RUNTIME_DIR:?Set EWALD_RUNTIME_DIR before submitting}"
: "${EWALD_SCRATCH_DIR:?Set EWALD_SCRATCH_DIR before submitting}"

EWALD_PLAN_FILE="${EWALD_PLAN_FILE:-$EWALD_RUNTIME_DIR/plans/generation_plan.jsonl}"
EWALD_GENERATE_CMD="${EWALD_GENERATE_CMD:-python data_training/scripts/generate_dataset.py}"
EWALD_CONDA_ENV="${EWALD_CONDA_ENV:-ewald-training}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-1}"
TASK_PADDED="$(printf "%06d" "$TASK_INDEX")"
RUN_DIR="$EWALD_SCRATCH_DIR/generated/task-$TASK_PADDED"

mkdir -p "$RUN_DIR" "$EWALD_SCRATCH_DIR/manifests" "$EWALD_RUNTIME_DIR/logs"

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

PLAN_LINE="$(sed -n "${TASK_INDEX}p" "$EWALD_PLAN_FILE")"
if [[ -z "$PLAN_LINE" ]]; then
  echo "No generation plan row $TASK_INDEX in $EWALD_PLAN_FILE" >&2
  exit 2
fi

printf "%s\n" "$PLAN_LINE" > "$RUN_DIR/task_plan.json"

cd "$EWALD_RUNTIME_DIR/repo"

# shellcheck disable=SC2086
$EWALD_GENERATE_CMD \
  --plan "$RUN_DIR/task_plan.json" \
  --output-root "$RUN_DIR" \
  --manifest "$RUN_DIR/manifest.jsonl"

cp "$RUN_DIR/manifest.jsonl" "$EWALD_SCRATCH_DIR/manifests/generate-${SLURM_JOB_ID:-local}-$TASK_PADDED.jsonl"
echo "Wrote generation manifest for task $TASK_PADDED"
