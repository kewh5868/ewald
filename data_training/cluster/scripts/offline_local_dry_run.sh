#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-data_training/cluster/alpine.paths.env}"
TASK_INDEX="${2:-1}"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

EWALD_LOCAL_REPO="${EWALD_LOCAL_REPO:-$PWD}"
EWALD_PLAN_FILE="${EWALD_LOCAL_PLAN_FILE:-data_training/cluster/templates/generation_plan.example.jsonl}"
EWALD_SCRATCH_DIR="${EWALD_SCRATCH_DIR:-/tmp/ewald-training-scratch}"
EWALD_GENERATE_CMD="${EWALD_GENERATE_CMD:-python data_training/scripts/generate_dataset.py}"

PLAN_LINE="$(sed -n "${TASK_INDEX}p" "$EWALD_PLAN_FILE")"
if [[ -z "$PLAN_LINE" ]]; then
  echo "No plan row $TASK_INDEX in $EWALD_PLAN_FILE" >&2
  exit 2
fi

RUN_DIR="$EWALD_SCRATCH_DIR/offline-task-$(printf "%06d" "$TASK_INDEX")"
mkdir -p "$RUN_DIR"
printf "%s\n" "$PLAN_LINE" > "$RUN_DIR/task_plan.json"

cd "$EWALD_LOCAL_REPO"
echo "Running offline dry run in $RUN_DIR"

# shellcheck disable=SC2086
$EWALD_GENERATE_CMD \
  --plan "$RUN_DIR/task_plan.json" \
  --output-root "$RUN_DIR" \
  --manifest "$RUN_DIR/manifest.jsonl" \
  --dry-run

echo "Dry-run manifest: $RUN_DIR/manifest.jsonl"
