#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"

RUN_ROOT="${EWALD_ART_RUN_ROOT:-$EWALD_RUN_BASE/02_apply_artifacts_$EWALD_RUN_ID}"
SOURCE_ROOT="${EWALD_CLEAN_ROOT:-}"
SOURCE_MANIFEST="${EWALD_CLEAN_MANIFEST:-}"
OUTPUT_ROOT="${EWALD_ARTIFACT_OUTPUT_ROOT:-$RUN_ROOT/artifacts}"
OUTPUT_MANIFEST="${EWALD_ARTIFACT_MANIFEST:-$OUTPUT_ROOT/artifact_manifest.jsonl}"
PROFILES="${EWALD_ARTIFACT_CONFIG:-$EWALD_REPO_ROOT/data_training/configs/artifacts.example.yaml}"
VARIANTS="${EWALD_ARTIFACT_VARIANTS:-1}"
SEED="${EWALD_ARTIFACT_SEED:-2000}"

if [[ -z "$SOURCE_MANIFEST" || -z "$SOURCE_ROOT" ]]; then
  echo "Set EWALD_CLEAN_MANIFEST and EWALD_CLEAN_ROOT." >&2
  exit 2
fi

write_run_info "$RUN_ROOT"

run_python "$EWALD_REPO_ROOT/data_training/scripts/apply_artifact_variants.py" \
  --manifest "$SOURCE_MANIFEST" \
  --root "$SOURCE_ROOT" \
  --profiles "$PROFILES" \
  --output-root "$OUTPUT_ROOT" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --variants-per-profile "$VARIANTS" \
  --seed "$SEED" | tee "$RUN_ROOT/artifacts.log"

echo "$OUTPUT_MANIFEST" > "$RUN_ROOT/manifest_path.txt"
echo "manifest=$OUTPUT_MANIFEST"

