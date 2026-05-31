#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-data_training/cluster/alpine.paths.env}"
TRIGGER_DIR="${2:-}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing environment file: $ENV_FILE" >&2
  exit 2
fi

if [[ -z "$TRIGGER_DIR" ]]; then
  echo "Usage: $0 path/to/alpine.paths.env data_training/deploy/NN_stage" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

REMOTE="${EWALD_CLUSTER_USER}@${EWALD_CLUSTER_HOST}"

ssh "$REMOTE" "bash -lc '
set -euo pipefail
export EWALD_RUNTIME_DIR=\"$EWALD_RUNTIME_DIR\"
export EWALD_SCRATCH_DIR=\"$EWALD_SCRATCH_DIR\"
export EWALD_CONDA_ENV=\"$EWALD_CONDA_ENV\"
export EWALD_CONDA_SH=\"$EWALD_CONDA_SH\"
export EWALD_MODULES=\"${EWALD_MODULES:-}\"
export EWALD_TRIGGER_DIR=\"$TRIGGER_DIR\"
cd \"$EWALD_RUNTIME_DIR/repo\"
mkdir -p \"$EWALD_RUNTIME_DIR/logs\"
sbatch_opts=(
  --output=\"$EWALD_RUNTIME_DIR/logs/%x-%j.out\"
  --error=\"$EWALD_RUNTIME_DIR/logs/%x-%j.err\"
)
if [[ -n \"${EWALD_ALPINE_ACCOUNT:-}\" ]]; then
  sbatch_opts+=(--account=\"$EWALD_ALPINE_ACCOUNT\")
fi
if [[ -n \"${EWALD_ALPINE_PARTITION:-}\" ]]; then
  sbatch_opts+=(--partition=\"$EWALD_ALPINE_PARTITION\")
fi
if [[ -n \"${EWALD_ALPINE_QOS:-}\" ]]; then
  sbatch_opts+=(--qos=\"$EWALD_ALPINE_QOS\")
fi
if [[ -n \"${EWALD_TRIGGER_TIME:-}\" ]]; then
  sbatch_opts+=(--time=\"$EWALD_TRIGGER_TIME\")
fi
if [[ -n \"${EWALD_TRIGGER_CPUS:-}\" ]]; then
  sbatch_opts+=(--cpus-per-task=\"$EWALD_TRIGGER_CPUS\")
fi
if [[ -n \"${EWALD_TRIGGER_MEM:-}\" ]]; then
  sbatch_opts+=(--mem=\"$EWALD_TRIGGER_MEM\")
fi
sbatch \"\${sbatch_opts[@]}\" --export=ALL,EWALD_TRIGGER_DIR=\"$TRIGGER_DIR\" data_training/cluster/slurm/run_deploy_trigger.sbatch
'"
