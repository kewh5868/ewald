#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-data_training/cluster/alpine.paths.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing environment file: $ENV_FILE" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

REMOTE="${EWALD_CLUSTER_USER}@${EWALD_CLUSTER_HOST}"

ssh "$REMOTE" "bash -lc '
set -euo pipefail
export EWALD_RUNTIME_DIR=\"$EWALD_RUNTIME_DIR\"
export EWALD_SCRATCH_DIR=\"$EWALD_SCRATCH_DIR\"
export EWALD_PLAN_FILE=\"$EWALD_PLAN_FILE\"
export EWALD_GENERATE_CMD=\"$EWALD_GENERATE_CMD\"
export EWALD_CONDA_ENV=\"$EWALD_CONDA_ENV\"
export EWALD_CONDA_SH=\"$EWALD_CONDA_SH\"
export EWALD_MODULES=\"${EWALD_MODULES:-}\"
cd \"$EWALD_RUNTIME_DIR/repo\"
sbatch data_training/cluster/slurm/generate_dataset_array.sbatch
'"
