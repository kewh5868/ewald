#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/control_synthetic_refinement.sh status [staged|high_res|oracle_diagnostic]
  scripts/control_synthetic_refinement.sh suspend [staged|high_res|oracle_diagnostic]
  scripts/control_synthetic_refinement.sh resume [staged|high_res|oracle_diagnostic]

Environment overrides:
  RUN_DATE=YYYYMMDD       Date bucket under example/projects/synthetic_refinement_history.
  HISTORY_ROOT=PATH       Synthetic refinement history root for the date.
  LOG_DIR=PATH            Background log directory containing latest.env/latest.pid.
  PID=12345               Explicit root PID to control.
  LATEST_ENV=PATH         Explicit latest.env metadata file.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACTION="${1:-status}"
RUN_KIND="${2:-staged}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
HISTORY_ROOT="${HISTORY_ROOT:-example/projects/synthetic_refinement_history/$RUN_DATE}"
LOG_DIR="${LOG_DIR:-$HISTORY_ROOT/background_logs/$RUN_KIND}"
LATEST_ENV="${LATEST_ENV:-$LOG_DIR/latest.env}"
LATEST_PID="$LOG_DIR/latest.pid"

if [[ "$ACTION" == "-h" || "$ACTION" == "--help" ]]; then
  usage
  exit 0
fi

case "$ACTION" in
  status | suspend | resume) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

case "$RUN_KIND" in
  staged | high_res | oracle_diagnostic) ;;
  *)
    echo "Unknown run kind: $RUN_KIND" >&2
    usage >&2
    exit 2
    ;;
esac

env_value() {
  local key="$1"
  local file="$2"
  [[ -f "$file" ]] || return 1
  awk -F= -v key="$key" '$1 == key {
    print substr($0, index($0, "=") + 1)
    exit
  }' "$file"
}

resolve_pid() {
  if [[ -n "${PID:-}" ]]; then
    echo "$PID"
    return 0
  fi
  local pid=""
  pid="$(env_value PID "$LATEST_ENV" || true)"
  if [[ -z "$pid" && -e "$LATEST_PID" ]]; then
    pid="$(cat "$LATEST_PID")"
  fi
  if [[ -z "$pid" ]]; then
    echo "Could not find a PID for $RUN_KIND in $LOG_DIR" >&2
    exit 1
  fi
  echo "$pid"
}

collect_process_tree() {
  local root_pid="$1"
  local queue=("$root_pid")
  local seen=" "
  local pids=()
  local pid_count=0
  local pid child children

  while ((${#queue[@]} > 0)); do
    pid="${queue[0]}"
    queue=("${queue[@]:1}")
    [[ "$seen" == *" $pid "* ]] && continue
    seen+="$pid "
    if ! ps -p "$pid" >/dev/null 2>&1; then
      continue
    fi
    pids[$pid_count]="$pid"
    pid_count=$((pid_count + 1))
    children="$(pgrep -P "$pid" || true)"
    for child in $children; do
      queue+=("$child")
    done
  done

  if ((pid_count > 0)); then
    printf '%s\n' "${pids[@]}"
  fi
}

pid_csv() {
  local first=1
  local pid
  for pid in "$@"; do
    if ((first)); then
      printf '%s' "$pid"
      first=0
    else
      printf ',%s' "$pid"
    fi
  done
}

print_status() {
  local root_pid="$1"
  shift
  local pid_count="$#"
  local pids=("$@")
  echo "Run kind: $RUN_KIND"
  echo "Run date: $RUN_DATE"
  echo "Metadata: $LATEST_ENV"
  echo "Root PID: $root_pid"
  if ((pid_count == 0)); then
    echo "Status: not running"
    return 0
  fi
  echo "Process tree:"
  ps -o pid,ppid,state,etime,pcpu,pmem,command -p "$(pid_csv "${pids[@]}")"
}

ROOT_PID="$(resolve_pid)"
PIDS=()
PID_COUNT=0
while IFS= read -r pid; do
  if [[ -n "$pid" ]]; then
    PIDS[$PID_COUNT]="$pid"
    PID_COUNT=$((PID_COUNT + 1))
  fi
done < <(collect_process_tree "$ROOT_PID")

if [[ "$ACTION" == "status" ]]; then
  if ((PID_COUNT > 0)); then
    print_status "$ROOT_PID" "${PIDS[@]}"
  else
    print_status "$ROOT_PID"
  fi
  exit 0
fi

if ((PID_COUNT == 0)); then
  echo "No live process tree found for PID $ROOT_PID."
  exit 0
fi

case "$ACTION" in
  suspend)
    kill -STOP "${PIDS[@]}"
    echo "Suspended $PID_COUNT process(es): ${PIDS[*]}"
    ;;
  resume)
    kill -CONT "${PIDS[@]}"
    echo "Resumed $PID_COUNT process(es): ${PIDS[*]}"
    ;;
esac

print_status "$ROOT_PID" "${PIDS[@]}"
