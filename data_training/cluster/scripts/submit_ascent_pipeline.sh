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

if [[ -z "${EWALD_ALPINE_ACCOUNT:-}" ]]; then
  echo "Set EWALD_ALPINE_ACCOUNT to your CU Boulder Alpine Ascent Slurm account." >&2
  exit 2
fi

RUN_ID="${EWALD_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE="${EWALD_CLUSTER_USER}@${EWALD_CLUSTER_HOST}"

ssh "$REMOTE" "bash -s" -- \
  "$EWALD_RUNTIME_DIR" \
  "$EWALD_SCRATCH_DIR" \
  "$EWALD_CONDA_ENV" \
  "$EWALD_CONDA_SH" \
  "${EWALD_MODULES:-}" \
  "$EWALD_ALPINE_ACCOUNT" \
  "${EWALD_ALPINE_PARTITION:-amilan}" \
  "${EWALD_ALPINE_QOS:-normal}" \
  "$RUN_ID" \
  "${EWALD_HYBRID3_MODE:-live}" \
  "${EWALD_HYBRID3_LIMIT:-25}" \
  "${EWALD_HYBRID3_PAGE_SIZE:-200}" \
  "${EWALD_HYBRID3_TIMEOUT:-60}" \
  "${EWALD_SIM_PLAN:-data_training/configs/simulation_alpine_fibril_training.example.yaml}" \
  "${EWALD_ARTIFACT_CONFIG:-data_training/configs/artifacts.example.yaml}" \
  "${EWALD_ARTIFACT_VARIANTS:-2}" \
  "${EWALD_FEEDBACK_TOP_K:-5}" \
  "${EWALD_FEEDBACK_MAX_SAMPLES:-}" \
  "${EWALD_FETCH_TIME:-04:00:00}" \
  "${EWALD_FETCH_CPUS:-4}" \
  "${EWALD_FETCH_MEM:-16G}" \
  "${EWALD_SIM_TIME:-24:00:00}" \
  "${EWALD_SIM_CPUS:-16}" \
  "${EWALD_SIM_MEM:-64G}" \
  "${EWALD_ART_TIME:-08:00:00}" \
  "${EWALD_ART_CPUS:-8}" \
  "${EWALD_ART_MEM:-32G}" \
  "${EWALD_TRAIN_TIME:-24:00:00}" \
  "${EWALD_TRAIN_CPUS:-16}" \
  "${EWALD_TRAIN_MEM:-96G}" \
  "${EWALD_TRAIN_GRES:-}" \
  "${EWALD_FEEDBACK_TIME:-04:00:00}" \
  "${EWALD_FEEDBACK_CPUS:-8}" \
  "${EWALD_FEEDBACK_MEM:-32G}" <<'REMOTE_SCRIPT'
set -euo pipefail

EWALD_RUNTIME_DIR="$1"
EWALD_SCRATCH_DIR="$2"
EWALD_CONDA_ENV="$3"
EWALD_CONDA_SH="$4"
EWALD_MODULES="$5"
EWALD_ALPINE_ACCOUNT="$6"
EWALD_ALPINE_PARTITION="$7"
EWALD_ALPINE_QOS="$8"
EWALD_RUN_ID="$9"
EWALD_HYBRID3_MODE="${10}"
EWALD_HYBRID3_LIMIT="${11}"
EWALD_HYBRID3_PAGE_SIZE="${12}"
EWALD_HYBRID3_TIMEOUT="${13}"
EWALD_SIM_PLAN="${14}"
EWALD_ARTIFACT_CONFIG="${15}"
EWALD_ARTIFACT_VARIANTS="${16}"
EWALD_FEEDBACK_TOP_K="${17}"
EWALD_FEEDBACK_MAX_SAMPLES="${18}"
EWALD_FETCH_TIME="${19}"
EWALD_FETCH_CPUS="${20}"
EWALD_FETCH_MEM="${21}"
EWALD_SIM_TIME="${22}"
EWALD_SIM_CPUS="${23}"
EWALD_SIM_MEM="${24}"
EWALD_ART_TIME="${25}"
EWALD_ART_CPUS="${26}"
EWALD_ART_MEM="${27}"
EWALD_TRAIN_TIME="${28}"
EWALD_TRAIN_CPUS="${29}"
EWALD_TRAIN_MEM="${30}"
EWALD_TRAIN_GRES="${31}"
EWALD_FEEDBACK_TIME="${32}"
EWALD_FEEDBACK_CPUS="${33}"
EWALD_FEEDBACK_MEM="${34}"

REPO_ROOT="$EWALD_RUNTIME_DIR/repo"
RUN_BASE="$EWALD_SCRATCH_DIR/runs/$EWALD_RUN_ID"
LIB_ROOT="$EWALD_SCRATCH_DIR/libraries/hybrid3_$EWALD_RUN_ID"
SIM_ROOT="$EWALD_SCRATCH_DIR/datasets/$EWALD_RUN_ID/simulations"
ART_ROOT="$EWALD_SCRATCH_DIR/datasets/$EWALD_RUN_ID/artifacts"
MODEL_PATH="$EWALD_SCRATCH_DIR/models/$EWALD_RUN_ID/vector_ranker.json"
METRICS_ROOT="$EWALD_SCRATCH_DIR/metrics/$EWALD_RUN_ID"

mkdir -p "$RUN_BASE" "$LIB_ROOT" "$SIM_ROOT" "$ART_ROOT" \
  "$(dirname "$MODEL_PATH")" "$METRICS_ROOT" "$EWALD_RUNTIME_DIR/logs"

cd "$REPO_ROOT"

base_sbatch_opts=(
  --parsable
  --account="$EWALD_ALPINE_ACCOUNT"
  --partition="$EWALD_ALPINE_PARTITION"
  --output="$EWALD_RUNTIME_DIR/logs/%x-%j.out"
  --error="$EWALD_RUNTIME_DIR/logs/%x-%j.err"
)
if [[ -n "$EWALD_ALPINE_QOS" ]]; then
  base_sbatch_opts+=(--qos="$EWALD_ALPINE_QOS")
fi

submit_trigger() {
  local name="$1"
  local trigger_dir="$2"
  local dependency="$3"
  local walltime="$4"
  local cpus="$5"
  local mem="$6"
  local gres="$7"
  local extra_exports="$8"
  local opts=("${base_sbatch_opts[@]}")
  opts+=(--job-name="$name" --time="$walltime" --cpus-per-task="$cpus" --mem="$mem")
  if [[ -n "$dependency" ]]; then
    opts+=(--dependency="afterok:$dependency")
  fi
  if [[ -n "$gres" ]]; then
    opts+=(--gres="$gres")
  fi
  local export_arg="ALL,EWALD_RUNTIME_DIR=$EWALD_RUNTIME_DIR,EWALD_SCRATCH_DIR=$EWALD_SCRATCH_DIR,EWALD_CONDA_ENV=$EWALD_CONDA_ENV,EWALD_CONDA_SH=$EWALD_CONDA_SH,EWALD_MODULES=$EWALD_MODULES,EWALD_REPO_ROOT=$REPO_ROOT,EWALD_RUN_ID=$EWALD_RUN_ID,EWALD_RUN_BASE=$RUN_BASE,EWALD_TRIGGER_DIR=$trigger_dir,$extra_exports"
  local job_id
  job_id="$(sbatch "${opts[@]}" --export="$export_arg" data_training/cluster/slurm/run_deploy_trigger.sbatch)"
  job_id="${job_id%%;*}"
  echo "$job_id"
}

fetch_exports="EWALD_HYBRID3_MODE=$EWALD_HYBRID3_MODE,EWALD_HYBRID3_LIMIT=$EWALD_HYBRID3_LIMIT,EWALD_HYBRID3_PAGE_SIZE=$EWALD_HYBRID3_PAGE_SIZE,EWALD_HYBRID3_TIMEOUT=$EWALD_HYBRID3_TIMEOUT,EWALD_HYBRID3_OUTPUT_ROOT=$LIB_ROOT"
fetch_job="$(submit_trigger "ewald-fetch-hybrid3" "data_training/deploy/00_fetch_hybrid3_structures" "" "$EWALD_FETCH_TIME" "$EWALD_FETCH_CPUS" "$EWALD_FETCH_MEM" "" "$fetch_exports")"

sim_plan="$EWALD_SIM_PLAN"
if [[ "$sim_plan" != /* ]]; then
  sim_plan="$REPO_ROOT/$sim_plan"
fi
artifact_config="$EWALD_ARTIFACT_CONFIG"
if [[ "$artifact_config" != /* ]]; then
  artifact_config="$REPO_ROOT/$artifact_config"
fi

sim_exports="EWALD_STRUCTURE_CATALOG=$LIB_ROOT/hybrid3_structure_catalog.yaml,EWALD_SIM_PLAN=$sim_plan,EWALD_ARTIFACT_CONFIG=$artifact_config,EWALD_SIM_OUTPUT_ROOT=$SIM_ROOT,EWALD_SIM_MANIFEST=$SIM_ROOT/manifest.jsonl,EWALD_SIM_DRY_RUN=0"
sim_job="$(submit_trigger "ewald-generate-giwaxs" "data_training/deploy/01_generate_simulations" "$fetch_job" "$EWALD_SIM_TIME" "$EWALD_SIM_CPUS" "$EWALD_SIM_MEM" "" "$sim_exports")"

art_exports="EWALD_CLEAN_ROOT=$SIM_ROOT,EWALD_CLEAN_MANIFEST=$SIM_ROOT/manifest.jsonl,EWALD_ARTIFACT_CONFIG=$artifact_config,EWALD_ARTIFACT_OUTPUT_ROOT=$ART_ROOT,EWALD_ARTIFACT_MANIFEST=$ART_ROOT/artifact_manifest.jsonl,EWALD_ARTIFACT_VARIANTS=$EWALD_ARTIFACT_VARIANTS"
art_job="$(submit_trigger "ewald-augment-artifacts" "data_training/deploy/02_apply_artifacts" "$sim_job" "$EWALD_ART_TIME" "$EWALD_ART_CPUS" "$EWALD_ART_MEM" "" "$art_exports")"

train_exports="EWALD_TRAIN_SOURCE_ROOT=$SIM_ROOT,EWALD_TRAIN_MANIFEST=$SIM_ROOT/manifest.jsonl,EWALD_MODEL_PATH=$MODEL_PATH,EWALD_TRAIN_DRY_RUN=0"
train_job="$(submit_trigger "ewald-train-ranker" "data_training/deploy/03_train_ranker" "$sim_job" "$EWALD_TRAIN_TIME" "$EWALD_TRAIN_CPUS" "$EWALD_TRAIN_MEM" "$EWALD_TRAIN_GRES" "$train_exports")"

feedback_exports="EWALD_MODEL_PATH=$MODEL_PATH,EWALD_EVAL_ROOT=$ART_ROOT,EWALD_EVAL_MANIFEST=$ART_ROOT/artifact_manifest.jsonl,EWALD_FEEDBACK_OUTPUT=$METRICS_ROOT/feedback_metrics.json,EWALD_FEEDBACK_HISTORY=$METRICS_ROOT/feedback_history.jsonl,EWALD_FEEDBACK_TOP_K=$EWALD_FEEDBACK_TOP_K"
if [[ -n "$EWALD_FEEDBACK_MAX_SAMPLES" ]]; then
  feedback_exports="$feedback_exports,EWALD_FEEDBACK_MAX_SAMPLES=$EWALD_FEEDBACK_MAX_SAMPLES"
fi
feedback_job="$(submit_trigger "ewald-feedback-eval" "data_training/deploy/04_feedback_evaluate" "$art_job:$train_job" "$EWALD_FEEDBACK_TIME" "$EWALD_FEEDBACK_CPUS" "$EWALD_FEEDBACK_MEM" "" "$feedback_exports")"

guess_root="$EWALD_SCRATCH_DIR/structure-guesses/$EWALD_RUN_ID"
guess_exports="EWALD_MODEL_PATH=$MODEL_PATH,EWALD_GUESS_ROOT=$ART_ROOT,EWALD_GUESS_MANIFEST=$ART_ROOT/artifact_manifest.jsonl,EWALD_GUESS_OUTPUT_ROOT=$guess_root,EWALD_GUESS_TOP_K=$EWALD_FEEDBACK_TOP_K"
if [[ -n "$EWALD_FEEDBACK_MAX_SAMPLES" ]]; then
  guess_exports="$guess_exports,EWALD_GUESS_MAX_SAMPLES=$EWALD_FEEDBACK_MAX_SAMPLES"
fi
guess_job="$(submit_trigger "ewald-export-guesses" "data_training/deploy/05_export_structure_guesses" "$feedback_job" "$EWALD_FEEDBACK_TIME" "$EWALD_FEEDBACK_CPUS" "$EWALD_FEEDBACK_MEM" "" "$guess_exports")"

cat <<SUMMARY
Submitted EWALD Ascent pipeline:
  run_id:       $EWALD_RUN_ID
  fetch:        $fetch_job
  simulation:   $sim_job
  artifacts:    $art_job
  ranker:       $train_job
  feedback:     $feedback_job
  guesses:      $guess_job
  catalog:      $LIB_ROOT/hybrid3_structure_catalog.yaml
  simulations:  $SIM_ROOT
  artifacts:    $ART_ROOT
  model:        $MODEL_PATH
  metrics:      $METRICS_ROOT/feedback_metrics.json
  guesses_dir:  $guess_root
SUMMARY
REMOTE_SCRIPT
