#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-data_training/cluster/alpine.paths.env}"
SESSION_NAME="${2:-ewald-training}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing environment file: $ENV_FILE" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

REMOTE="${EWALD_CLUSTER_USER}@${EWALD_CLUSTER_HOST}"

ssh -t "$REMOTE" "bash -lc '
set -euo pipefail
mkdir -p \"$EWALD_RUNTIME_DIR/repo\" \"$EWALD_RUNTIME_DIR/logs\" \"$EWALD_SCRATCH_DIR\"
cd \"$EWALD_RUNTIME_DIR/repo\"
if command -v tmux >/dev/null 2>&1; then
  tmux new-session -A -s \"$SESSION_NAME\"
else
  echo \"tmux not found; opening an interactive shell in $EWALD_RUNTIME_DIR/repo\"
  exec \"\${SHELL:-/bin/bash}\" -l
fi
'"
