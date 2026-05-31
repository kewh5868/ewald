#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"

RUN_ROOT="${EWALD_FEEDBACK_RUN_ROOT:-$EWALD_RUN_BASE/04_feedback_evaluate_$EWALD_RUN_ID}"
MODEL="${EWALD_MODEL_PATH:-}"
EVAL_ROOT="${EWALD_EVAL_ROOT:-}"
EVAL_MANIFEST="${EWALD_EVAL_MANIFEST:-}"
OUTPUT="${EWALD_FEEDBACK_OUTPUT:-$RUN_ROOT/feedback_metrics.json}"
HISTORY="${EWALD_FEEDBACK_HISTORY:-$RUN_ROOT/feedback_history.jsonl}"
TOP_K="${EWALD_FEEDBACK_TOP_K:-5}"
MAX_SAMPLES="${EWALD_FEEDBACK_MAX_SAMPLES:-}"

if [[ -z "$MODEL" || -z "$EVAL_ROOT" || -z "$EVAL_MANIFEST" ]]; then
  echo "Set EWALD_MODEL_PATH, EWALD_EVAL_ROOT, and EWALD_EVAL_MANIFEST." >&2
  exit 2
fi

write_run_info "$RUN_ROOT"

args=(
  "$EWALD_REPO_ROOT/data_training/scripts/feedback_evaluate.py"
  --model "$MODEL"
  --manifest "$EVAL_MANIFEST"
  --root "$EVAL_ROOT"
  --output "$OUTPUT"
  --history "$HISTORY"
  --top-k "$TOP_K"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi

run_python "${args[@]}" | tee "$RUN_ROOT/feedback.log"
echo "$OUTPUT" > "$RUN_ROOT/metrics_path.txt"
echo "metrics=$OUTPUT"

