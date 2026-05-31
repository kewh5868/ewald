#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"

RUN_ROOT="${EWALD_FETCH_RUN_ROOT:-$EWALD_RUN_BASE/00_fetch_hybrid3_structures_$EWALD_RUN_ID}"
OUTPUT_ROOT="${EWALD_HYBRID3_OUTPUT_ROOT:-$RUN_ROOT/hybrid3}"
LIMIT="${EWALD_HYBRID3_LIMIT:-2}"
PAGE_SIZE="${EWALD_HYBRID3_PAGE_SIZE:-200}"
TIMEOUT="${EWALD_HYBRID3_TIMEOUT:-20}"
MODE="${EWALD_HYBRID3_MODE:-fixture}"
FIXTURE_ROOT="${EWALD_HYBRID3_FIXTURE_ROOT:-$EWALD_REPO_ROOT/data_training/fixtures/hybrid3}"

write_run_info "$RUN_ROOT"

args=(
  "$EWALD_REPO_ROOT/data_training/scripts/fetch_hybrid3_structures.py"
  --output-root "$OUTPUT_ROOT"
  --page-size "$PAGE_SIZE"
  --timeout "$TIMEOUT"
)

if [[ "$LIMIT" != "0" ]]; then
  args+=(--limit "$LIMIT")
fi

if [[ "$MODE" == "fixture" ]]; then
  args+=(--fixture-root "$FIXTURE_ROOT")
fi

run_python "${args[@]}" | tee "$RUN_ROOT/fetch.log"
echo "$OUTPUT_ROOT/hybrid3_structure_catalog.yaml" > "$RUN_ROOT/catalog_path.txt"
echo "catalog=$OUTPUT_ROOT/hybrid3_structure_catalog.yaml"
