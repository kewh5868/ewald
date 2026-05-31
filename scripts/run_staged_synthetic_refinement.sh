#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-ewald-py312}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
HISTORY_ROOT="${HISTORY_ROOT:-example/projects/synthetic_refinement_history/$RUN_DATE}"
OUTPUT_DIR="${OUTPUT_DIR:-$HISTORY_ROOT/runs/staged}"
LOG_DIR="${LOG_DIR:-$HISTORY_ROOT/background_logs/staged}"
STRUCTURES_DIR="${STRUCTURES_DIR:-example/structures}"
MANIFEST="${MANIFEST:-}"
STRUCTURE_LIMIT="${STRUCTURE_LIMIT:-}"

# Heavy staged defaults. Override any value as an environment variable.
SIMULATIONS_PER_STRUCTURE="${SIMULATIONS_PER_STRUCTURE:-4}"
DETECTOR_SHAPE="${DETECTOR_SHAPE:-512x768}"
HKL_EXTENT="${HKL_EXTENT:-7}"
PEAK_MAX_PEAKS="${PEAK_MAX_PEAKS:-500}"
CANDIDATE_MAX="${CANDIDATE_MAX:-24}"
MAX_GENERATED_CIFS="${MAX_GENERATED_CIFS:-16}"
MAX_SCAFFOLDS="${MAX_SCAFFOLDS:-24}"
MAX_ORGANIC_PROXIES="${MAX_ORGANIC_PROXIES:-16}"
MAX_ORGANIC_REPLACEMENTS="${MAX_ORGANIC_REPLACEMENTS:-12}"
STAGE_SIMULATION_MAX_CIFS="${STAGE_SIMULATION_MAX_CIFS:-8}"
ORGANIC_RMC_STEPS="${ORGANIC_RMC_STEPS:-24}"
ORGANIC_RMC_TRANSLATION_STEP="${ORGANIC_RMC_TRANSLATION_STEP:-0.025}"
ORGANIC_RMC_ROTATION_STEP_DEG="${ORGANIC_RMC_ROTATION_STEP_DEG:-8.0}"

# Keep the q-space window focused where the current synthetic tests are useful.
QXY_RANGE="${QXY_RANGE:--3.0,3.0}"
QZ_RANGE="${QZ_RANGE:-0.0,3.0}"

# Retain simple fibril texture variations instead of fully random orientation.
TEXTURE_MODES="${TEXTURE_MODES:-out_of_plane_stack,in_plane_stack,tilted_out_of_plane_stack,tilted_in_plane_stack}"
TEXTURE_AZIMUTH_JITTER_DEG="${TEXTURE_AZIMUTH_JITTER_DEG:-6.0}"

# Use detected ROI-integrated intensities when comparing to simulated Bragg rows.
BRAGG_INTENSITY_WEIGHT="${BRAGG_INTENSITY_WEIGHT:-0.40}"
BRAGG_INTENSITY_TOLERANCE="${BRAGG_INTENSITY_TOLERANCE:-0.08}"
BRAGG_INTENSITY_MAX_PEAKS="${BRAGG_INTENSITY_MAX_PEAKS:-100}"

SEED="${SEED:-$(date +%Y%m%d)}"
IMAGE_RERANK="${IMAGE_RERANK:-0}"
ORGANIC_PROXIES="${ORGANIC_PROXIES:-1}"
ORGANIC_REPLACEMENTS="${ORGANIC_REPLACEMENTS:-1}"
UNIT_CELL_SYMMETRY="${UNIT_CELL_SYMMETRY:-1}"

mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/staged_synthetic_refinement_${STAMP}.log"
PID_FILE="$LOG_DIR/staged_synthetic_refinement_${STAMP}.pid"
LATEST_ENV="$LOG_DIR/latest.env"

cmd=(
  conda run --no-capture-output -n "$CONDA_ENV"
  python -u -m ewald.app.synthetic_refine
  --structures-dir "$STRUCTURES_DIR"
  --output-dir "$OUTPUT_DIR"
  --seed "$SEED"
  --simulations-per-structure "$SIMULATIONS_PER_STRUCTURE"
  --detector-shape "$DETECTOR_SHAPE"
  --hkl-extent "$HKL_EXTENT"
  --peak-max-peaks "$PEAK_MAX_PEAKS"
  --candidate-max "$CANDIDATE_MAX"
  --max-generated-cifs "$MAX_GENERATED_CIFS"
  --max-scaffolds "$MAX_SCAFFOLDS"
  --max-organic-proxies "$MAX_ORGANIC_PROXIES"
  --max-organic-replacements "$MAX_ORGANIC_REPLACEMENTS"
  --stage-simulation-max-cifs "$STAGE_SIMULATION_MAX_CIFS"
  --organic-rmc-steps "$ORGANIC_RMC_STEPS"
  --organic-rmc-translation-step "$ORGANIC_RMC_TRANSLATION_STEP"
  --organic-rmc-rotation-step-deg "$ORGANIC_RMC_ROTATION_STEP_DEG"
  --bragg-intensity-weight "$BRAGG_INTENSITY_WEIGHT"
  --bragg-intensity-tolerance "$BRAGG_INTENSITY_TOLERANCE"
  --bragg-intensity-max-peaks "$BRAGG_INTENSITY_MAX_PEAKS"
  "--qxy-range=$QXY_RANGE"
  "--qz-range=$QZ_RANGE"
  --texture-modes "$TEXTURE_MODES"
  --texture-azimuth-jitter-deg "$TEXTURE_AZIMUTH_JITTER_DEG"
)

if [[ -n "$STRUCTURE_LIMIT" ]]; then
  cmd+=(--structure-limit "$STRUCTURE_LIMIT")
fi

if [[ -n "$MANIFEST" ]]; then
  cmd+=(--manifest "$MANIFEST")
fi

if [[ "$IMAGE_RERANK" == "1" || "$IMAGE_RERANK" == "true" ]]; then
  cmd+=(--image-rerank)
fi

if [[ "$ORGANIC_PROXIES" == "0" || "$ORGANIC_PROXIES" == "false" ]]; then
  cmd+=(--no-organic-proxies)
fi

if [[ "$ORGANIC_REPLACEMENTS" == "0" || "$ORGANIC_REPLACEMENTS" == "false" ]]; then
  cmd+=(--no-organic-replacements)
fi

if [[ "$UNIT_CELL_SYMMETRY" == "0" || "$UNIT_CELL_SYMMETRY" == "false" ]]; then
  cmd+=(--no-unit-cell-symmetry)
fi

{
  echo "Started: $(date)"
  echo "Repository: $ROOT_DIR"
  echo "Output directory root: $OUTPUT_DIR"
  echo "Log file: $LOG_FILE"
  echo "Staged refinement:"
  echo "  scaffold -> organic electron proxy -> full organic replacement -> organic RMC"
  echo "Fibril texture modes: $TEXTURE_MODES"
  echo "Texture azimuth jitter deg: $TEXTURE_AZIMUTH_JITTER_DEG"
  echo "Command:"
  printf '  %q' "${cmd[@]}"
  echo
  echo
  "${cmd[@]}"
  status=$?
  echo
  echo "Finished: $(date)"
  echo "Exit status: $status"
  exit "$status"
} >"$LOG_FILE" 2>&1 &

pid=$!
echo "$pid" >"$PID_FILE"
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/latest.log"
ln -sf "$(basename "$PID_FILE")" "$LOG_DIR/latest.pid"

cat >"$LATEST_ENV" <<EOF
PID=$pid
PID_FILE=$PID_FILE
LOG_FILE=$LOG_FILE
OUTPUT_DIR=$OUTPUT_DIR
RUN_KIND=staged
RUN_DATE=$RUN_DATE
HISTORY_ROOT=$HISTORY_ROOT
STARTED_AT=$STAMP
EOF

echo "Started staged synthetic refinement in the background."
echo "PID: $pid"
echo "Log: $LOG_FILE"
echo "Latest metadata: $LATEST_ENV"
echo
echo "Useful checks:"
echo "  tail -f \"$LOG_FILE\""
echo "  ps -p $pid"
echo "  grep -E 'Synthetic refinement run|Output directory|Filesets solved|Exit status' \"$LOG_FILE\""
