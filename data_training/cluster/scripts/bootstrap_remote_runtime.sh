#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-data_training/cluster/alpine.paths.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing environment file: $ENV_FILE" >&2
  echo "Copy data_training/cluster/alpine.paths.example.env and edit it first." >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

REMOTE="${EWALD_CLUSTER_USER}@${EWALD_CLUSTER_HOST}"

ssh "$REMOTE" "bash -lc '
set -euo pipefail
mkdir -p \"$EWALD_RUNTIME_DIR/repo\" \"$EWALD_RUNTIME_DIR/cluster\" \"$EWALD_RUNTIME_DIR/plans\" \"$EWALD_RUNTIME_DIR/logs\"
mkdir -p \"$EWALD_SCRATCH_DIR/generated\" \"$EWALD_SCRATCH_DIR/manifests\" \"$EWALD_SCRATCH_DIR/training-runs\" \"$EWALD_SCRATCH_DIR/validation-runs\"

echo \"runtime=$EWALD_RUNTIME_DIR\"
echo \"scratch=$EWALD_SCRATCH_DIR\"

if [[ -n \"${EWALD_MODULES:-}\" ]] && command -v module >/dev/null 2>&1; then
  for module_name in $EWALD_MODULES; do
    module load \"\$module_name\" || true
  done
fi

if [[ -f \"$EWALD_CONDA_SH\" ]]; then
  # shellcheck source=/dev/null
  source \"$EWALD_CONDA_SH\"
  if conda env list | awk \"{print \\\$1}\" | grep -qx \"$EWALD_CONDA_ENV\"; then
    conda activate \"$EWALD_CONDA_ENV\"
    python --version
  elif [[ \"${EWALD_BOOTSTRAP_CREATE_ENV:-0}\" == \"1\" ]]; then
    cd \"$EWALD_RUNTIME_DIR/repo\"
    conda env create -f requirements/ewald-py312.yml
    conda activate \"$EWALD_CONDA_ENV\"
    python -m pip install -e .
    python --version
  else
    echo \"Conda env $EWALD_CONDA_ENV is not present. Set EWALD_BOOTSTRAP_CREATE_ENV=1 to create it after staging.\" >&2
  fi
else
  echo \"EWALD_CONDA_SH does not exist: $EWALD_CONDA_SH\" >&2
fi

if [[ -d \"$EWALD_RUNTIME_DIR/repo/.git\" ]]; then
  cd \"$EWALD_RUNTIME_DIR/repo\"
  git status --short || true
fi
'"
