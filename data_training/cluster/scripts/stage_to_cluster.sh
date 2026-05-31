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

EWALD_LOCAL_REPO="${EWALD_LOCAL_REPO:-$PWD}"
REMOTE="${EWALD_CLUSTER_USER}@${EWALD_CLUSTER_HOST}"

if [[ ! -d "$EWALD_LOCAL_REPO" ]]; then
  echo "EWALD_LOCAL_REPO does not exist: $EWALD_LOCAL_REPO" >&2
  exit 2
fi

ssh "$REMOTE" "mkdir -p '$EWALD_RUNTIME_DIR/repo' '$EWALD_RUNTIME_DIR/cluster' '$EWALD_RUNTIME_DIR/plans' '$EWALD_RUNTIME_DIR/logs' '$EWALD_SCRATCH_DIR/manifests'"

RSYNC_DELETE_ARGS=()
if [[ "${EWALD_RSYNC_DELETE:-0}" == "1" ]]; then
  RSYNC_DELETE_ARGS=(--delete)
fi

rsync -az "${RSYNC_DELETE_ARGS[@]}" \
  --exclude ".git/" \
  --exclude ".mypy_cache/" \
  --exclude ".pytest_cache/" \
  --exclude ".ruff_cache/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude "site/" \
  --exclude "data_training/cluster/_synced_outputs/" \
  "$EWALD_LOCAL_REPO/" \
  "$REMOTE:$EWALD_RUNTIME_DIR/repo/"

rsync -az "$EWALD_LOCAL_REPO/data_training/cluster/" "$REMOTE:$EWALD_RUNTIME_DIR/cluster/"

if [[ -n "${EWALD_LOCAL_CATALOG_DIR:-}" ]]; then
  rsync -az "$EWALD_LOCAL_CATALOG_DIR/" "$REMOTE:$EWALD_RUNTIME_DIR/catalogs/"
fi

if [[ -n "${EWALD_LOCAL_PLAN_FILE:-}" ]]; then
  rsync -az "$EWALD_LOCAL_PLAN_FILE" "$REMOTE:$EWALD_RUNTIME_DIR/plans/generation_plan.jsonl"
fi

echo "Staged EWALD runtime to $REMOTE:$EWALD_RUNTIME_DIR"
echo "Scratch root is $REMOTE:$EWALD_SCRATCH_DIR"
