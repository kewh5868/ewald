#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-data_training/cluster/alpine.paths.env}"
LINES="${2:-80}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing environment file: $ENV_FILE" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

REMOTE="${EWALD_CLUSTER_USER}@${EWALD_CLUSTER_HOST}"

ssh "$REMOTE" "bash -lc '
set -euo pipefail
mapfile -t logs < <(find \"$EWALD_RUNTIME_DIR\" \"$EWALD_SCRATCH_DIR\" -type f \( -name \"*.out\" -o -name \"*.err\" -o -name \"*.log\" \) 2>/dev/null | sort | tail -10)
if [[ \"\${#logs[@]}\" -eq 0 ]]; then
  echo \"No cluster logs found under $EWALD_RUNTIME_DIR or $EWALD_SCRATCH_DIR\"
  exit 0
fi
for log_path in \"\${logs[@]}\"; do
  echo \"===== \$log_path =====\"
  tail -n \"$LINES\" \"\$log_path\" || true
done
'"

