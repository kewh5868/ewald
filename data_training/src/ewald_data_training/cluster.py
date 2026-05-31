"""Cluster staging helpers for EWALD training-data generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class ClusterRunConfig:
    """Runtime mapping for a remote SLURM generation job."""

    cluster_name: str = "alpine"
    login_host: str = "alpine-login.rc.colorado.edu"
    project_dir: str = "$HOME/ewald"
    scratch_dir: str = "$SCRATCH/ewald_training"
    runtime_dir: str = "$SCRATCH/ewald_runtime"
    conda_env: str = "ewald-py312"
    account: str = ""
    partition: str = "amilan"
    walltime: str = "12:00:00"
    cpus_per_task: int = 8
    mem: str = "32G"
    array: str = "0-0"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ClusterRunConfig":
        values = asdict(cls())
        values.update({key: payload[key] for key in values if key in payload})
        values["cpus_per_task"] = int(values["cpus_per_task"])
        return cls(**values)


def render_sbatch(
    config: ClusterRunConfig,
    *,
    plan_path: str = "data_training/configs/simulation_sweep.example.yaml",
    manifest_name: str = "manifest.jsonl",
) -> str:
    """Render a portable SLURM array script."""

    account_line = (
        f"#SBATCH --account={config.account}" if config.account else ""
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=ewald-train-gen
#SBATCH --partition={config.partition}
{account_line}
#SBATCH --time={config.walltime}
#SBATCH --cpus-per-task={config.cpus_per_task}
#SBATCH --mem={config.mem}
#SBATCH --array={config.array}
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {config.conda_env}

cd {config.project_dir}
mkdir -p {config.scratch_dir} {config.runtime_dir}
export EWALD_TRAINING_SCRATCH="{config.scratch_dir}"
export EWALD_TRAINING_RUNTIME="{config.runtime_dir}"
export EWALD_TRAINING_SHARD="${{SLURM_ARRAY_TASK_ID:-0}}"

python data_training/scripts/generate_dataset.py \\
  --plan {plan_path} \\
  --output-root "{config.scratch_dir}/shard_${{EWALD_TRAINING_SHARD}}" \\
  --manifest "{config.scratch_dir}/shard_${{EWALD_TRAINING_SHARD}}/{manifest_name}"
"""


def render_sync_commands(
    config: ClusterRunConfig,
    *,
    local_root: str | Path,
) -> dict[str, str]:
    """Return rsync commands for staging code and fetching outputs."""

    local = Path(local_root).expanduser().resolve()
    remote = f"{config.login_host}:{config.project_dir}/"
    scratch = f"{config.login_host}:{config.scratch_dir}/"
    return {
        "push_code": (
            f"rsync -az --delete --exclude .git --exclude .venv "
            f"{local}/ {remote}"
        ),
        "pull_results": f"rsync -az {scratch} {local}/data_training/runs/",
    }
