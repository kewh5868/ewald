#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"

RUN_ROOT="${EWALD_GUESS_RUN_ROOT:-$EWALD_RUN_BASE/05_export_structure_guesses_$EWALD_RUN_ID}"
MODEL="${EWALD_MODEL_PATH:-}"
EVAL_ROOT="${EWALD_GUESS_ROOT:-${EWALD_EVAL_ROOT:-}}"
EVAL_MANIFEST="${EWALD_GUESS_MANIFEST:-${EWALD_EVAL_MANIFEST:-}}"
OUTPUT_ROOT="${EWALD_GUESS_OUTPUT_ROOT:-$RUN_ROOT/structure_guesses}"
TOP_K="${EWALD_GUESS_TOP_K:-5}"
MAX_SAMPLES="${EWALD_GUESS_MAX_SAMPLES:-}"

if [[ -z "$MODEL" || -z "$EVAL_ROOT" || -z "$EVAL_MANIFEST" ]]; then
  echo "Set EWALD_MODEL_PATH, EWALD_GUESS_ROOT, and EWALD_GUESS_MANIFEST." >&2
  exit 2
fi

write_run_info "$RUN_ROOT"

args=(
  "$EWALD_REPO_ROOT/data_training/scripts/export_structure_guesses.py"
  --model "$MODEL"
  --manifest "$EVAL_MANIFEST"
  --root "$EVAL_ROOT"
  --output-root "$OUTPUT_ROOT"
  --top-k "$TOP_K"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi

run_python "${args[@]}" | tee "$RUN_ROOT/export_guesses.log"
echo "$OUTPUT_ROOT" > "$RUN_ROOT/guesses_path.txt"
echo "guesses=$OUTPUT_ROOT"
