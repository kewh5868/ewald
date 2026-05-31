#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-data_training/cluster/alpine.paths.env}"
shift || true

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing environment file: $ENV_FILE" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 path/to/alpine.paths.env COMMAND [ARGS...]" >&2
  echo "Example: $0 data_training/cluster/alpine.paths.env python --version" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

REMOTE="${EWALD_CLUSTER_USER}@${EWALD_CLUSTER_HOST}"
printf -v REMOTE_COMMAND "%q " "$@"

ssh "$REMOTE" "bash -lc '
set -euo pipefail
export EWALD_RUNTIME_DIR=\"$EWALD_RUNTIME_DIR\"
export EWALD_SCRATCH_DIR=\"$EWALD_SCRATCH_DIR\"
export EWALD_PLAN_FILE=\"$EWALD_PLAN_FILE\"
export EWALD_TRAINING_MANIFEST=\"$EWALD_TRAINING_MANIFEST\"

if [[ -n \"${EWALD_MODULES:-}\" ]] && command -v module >/dev/null 2>&1; then
  for module_name in $EWALD_MODULES; do
    module load \"\$module_name\" || true
  done
fi
if [[ -f \"$EWALD_CONDA_SH\" ]]; then
  source \"$EWALD_CONDA_SH\"
  conda activate \"$EWALD_CONDA_ENV\" || true
fi

cd \"$EWALD_RUNTIME_DIR/repo\"
$REMOTE_COMMAND
'"

