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
EWALD_LOCAL_SYNC_DIR="${EWALD_LOCAL_SYNC_DIR:-data_training/cluster/_synced_outputs}"

mkdir -p "$EWALD_LOCAL_SYNC_DIR/manifests" "$EWALD_LOCAL_SYNC_DIR/training-runs" "$EWALD_LOCAL_SYNC_DIR/validation-runs" "$EWALD_LOCAL_SYNC_DIR/logs" "$EWALD_LOCAL_SYNC_DIR/models" "$EWALD_LOCAL_SYNC_DIR/metrics" "$EWALD_LOCAL_SYNC_DIR/structure-guesses" "$EWALD_LOCAL_SYNC_DIR/libraries"

rsync -az "$REMOTE:$EWALD_SCRATCH_DIR/manifests/" "$EWALD_LOCAL_SYNC_DIR/manifests/"
rsync -az "$REMOTE:$EWALD_SCRATCH_DIR/training-runs/" "$EWALD_LOCAL_SYNC_DIR/training-runs/" || true
rsync -az "$REMOTE:$EWALD_SCRATCH_DIR/validation-runs/" "$EWALD_LOCAL_SYNC_DIR/validation-runs/" || true
rsync -az "$REMOTE:$EWALD_SCRATCH_DIR/models/" "$EWALD_LOCAL_SYNC_DIR/models/" || true
rsync -az "$REMOTE:$EWALD_SCRATCH_DIR/metrics/" "$EWALD_LOCAL_SYNC_DIR/metrics/" || true
rsync -az "$REMOTE:$EWALD_SCRATCH_DIR/structure-guesses/" "$EWALD_LOCAL_SYNC_DIR/structure-guesses/" || true
rsync -az "$REMOTE:$EWALD_SCRATCH_DIR/libraries/" "$EWALD_LOCAL_SYNC_DIR/libraries/" || true
rsync -az "$REMOTE:$EWALD_RUNTIME_DIR/logs/" "$EWALD_LOCAL_SYNC_DIR/logs/" || true

if [[ "${EWALD_SYNC_LARGE_OUTPUTS:-0}" == "1" ]]; then
  mkdir -p "$EWALD_LOCAL_SYNC_DIR/generated"
  rsync -az "$REMOTE:$EWALD_SCRATCH_DIR/generated/" "$EWALD_LOCAL_SYNC_DIR/generated/"
fi

echo "Synced compact cluster outputs to $EWALD_LOCAL_SYNC_DIR"
