#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"

RUN_ROOT="${EWALD_SIM_RUN_ROOT:-$EWALD_RUN_BASE/01_generate_simulations_$EWALD_RUN_ID}"
OUTPUT_ROOT="${EWALD_SIM_OUTPUT_ROOT:-$RUN_ROOT/simulations}"
PLAN="${EWALD_SIM_PLAN:-$EWALD_REPO_ROOT/data_training/configs/simulation_clean.example.yaml}"
STRUCTURES="${EWALD_STRUCTURE_CATALOG:-$EWALD_REPO_ROOT/data_training/catalog/structures.example.yaml}"
ARTIFACTS="${EWALD_ARTIFACT_CONFIG:-$EWALD_REPO_ROOT/data_training/configs/artifacts.example.yaml}"
MANIFEST="${EWALD_SIM_MANIFEST:-$OUTPUT_ROOT/manifest.jsonl}"
DRY_RUN="${EWALD_SIM_DRY_RUN:-0}"

write_run_info "$RUN_ROOT"
mkdir -p "$OUTPUT_ROOT"

args=(
  "$EWALD_REPO_ROOT/data_training/scripts/generate_dataset.py"
  --plan "$PLAN"
  --structures "$STRUCTURES"
  --artifacts "$ARTIFACTS"
  --output-root "$OUTPUT_ROOT"
  --manifest "$MANIFEST"
)

if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

run_python "${args[@]}" | tee "$RUN_ROOT/generate.log"
echo "$MANIFEST" > "$RUN_ROOT/manifest_path.txt"
echo "manifest=$MANIFEST"

