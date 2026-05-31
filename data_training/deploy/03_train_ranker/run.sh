#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"

RUN_ROOT="${EWALD_TRAIN_RUN_ROOT:-$EWALD_RUN_BASE/03_train_ranker_$EWALD_RUN_ID}"
SOURCE_ROOT="${EWALD_TRAIN_SOURCE_ROOT:-}"
SOURCE_MANIFEST="${EWALD_TRAIN_MANIFEST:-}"
MODEL_PATH="${EWALD_MODEL_PATH:-$RUN_ROOT/model/vector_ranker.json}"
DRY_RUN="${EWALD_TRAIN_DRY_RUN:-0}"

if [[ -z "$SOURCE_MANIFEST" || -z "$SOURCE_ROOT" ]]; then
  echo "Set EWALD_TRAIN_MANIFEST and EWALD_TRAIN_SOURCE_ROOT." >&2
  exit 2
fi

write_run_info "$RUN_ROOT"

args=(
  "$EWALD_REPO_ROOT/data_training/scripts/train_ranker.py"
  --manifest "$SOURCE_MANIFEST"
  --root "$SOURCE_ROOT"
  --output "$MODEL_PATH"
)

if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

run_python "${args[@]}" | tee "$RUN_ROOT/train.log"
echo "$MODEL_PATH" > "$RUN_ROOT/model_path.txt"
echo "model=$MODEL_PATH"

