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
echo \"== squeue ==\"
squeue -u \"$EWALD_CLUSTER_USER\" || true
echo
echo \"== manifests ==\"
find \"$EWALD_SCRATCH_DIR/manifests\" -maxdepth 1 -type f -print 2>/dev/null | sort | tail -20 || true
echo
echo \"== generated shards ==\"
find \"$EWALD_SCRATCH_DIR/generated\" -maxdepth 1 -type d -name \"task-*\" -print 2>/dev/null | wc -l || true
echo
echo \"== pipeline outputs ==\"
find \"$EWALD_SCRATCH_DIR\" -maxdepth 3 \( -name \"hybrid3_structure_catalog.yaml\" -o -name \"vector_ranker.json\" -o -name \"feedback_metrics.json\" -o -name \"ranked_guesses.json\" \) -print 2>/dev/null | sort | tail -40 || true
'"
