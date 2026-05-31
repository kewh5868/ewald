#!/usr/bin/env bash
#SBATCH --job-name=ewald-merge-manifests
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

: "${EWALD_SCRATCH_DIR:?Set EWALD_SCRATCH_DIR before submitting}"

MANIFEST_DIR="${EWALD_MANIFEST_DIR:-$EWALD_SCRATCH_DIR/manifests}"
OUTPUT_MANIFEST="${EWALD_TRAINING_MANIFEST:-$MANIFEST_DIR/training-manifest.jsonl}"
TMP_MANIFEST="$OUTPUT_MANIFEST.tmp"

mkdir -p "$MANIFEST_DIR"
: > "$TMP_MANIFEST"

matched=0
while IFS= read -r manifest_file; do
  cat "$manifest_file" >> "$TMP_MANIFEST"
  matched=$((matched + 1))
done < <(find "$MANIFEST_DIR" -maxdepth 1 -type f -name "generate-*.jsonl" | sort)

if [[ "$matched" -eq 0 ]]; then
  echo "No generation manifests found in $MANIFEST_DIR" >&2
  rm -f "$TMP_MANIFEST"
  exit 2
fi

mv "$TMP_MANIFEST" "$OUTPUT_MANIFEST"
echo "Merged $matched generation manifests into $OUTPUT_MANIFEST"
