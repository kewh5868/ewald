"""Runtime and cluster path configuration for training-data
generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config_io import load_config, tuple_of_strings


@dataclass(frozen=True, slots=True)
class ClusterRuntimeConfig:
    """Cluster execution settings without opening network
    connections."""

    cluster_name: str
    scheduler: str = "slurm"
    remote_host: str = ""
    remote_user: str = ""
    local_repo_root: str = "."
    remote_repo_root: str = "$SCRATCH/ewald"
    remote_scratch_root: str = "$SCRATCH/ewald_training"
    remote_dataset_root: str = "$SCRATCH/ewald_training/datasets"
    remote_structure_root: str = "$SCRATCH/ewald_training/structures"
    conda_env: str = "ewald-py312"
    modules: tuple[str, ...] = ()
    account: str = ""
    partition: str = ""
    qos: str = ""
    walltime: str = "04:00:00"
    cpus_per_task: int = 4
    memory: str = "16G"
    array_chunk_size: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ClusterRuntimeConfig":
        return cls(
            cluster_name=str(payload.get("cluster_name") or "cluster"),
            scheduler=str(payload.get("scheduler") or "slurm"),
            remote_host=str(payload.get("remote_host") or ""),
            remote_user=str(payload.get("remote_user") or ""),
            local_repo_root=str(payload.get("local_repo_root") or "."),
            remote_repo_root=str(
                payload.get("remote_repo_root") or "$SCRATCH/ewald"
            ),
            remote_scratch_root=str(
                payload.get("remote_scratch_root") or "$SCRATCH/ewald_training"
            ),
            remote_dataset_root=str(
                payload.get("remote_dataset_root")
                or "$SCRATCH/ewald_training/datasets"
            ),
            remote_structure_root=str(
                payload.get("remote_structure_root")
                or "$SCRATCH/ewald_training/structures"
            ),
            conda_env=str(payload.get("conda_env") or "ewald-py312"),
            modules=tuple_of_strings(payload.get("modules")),
            account=str(payload.get("account") or ""),
            partition=str(payload.get("partition") or ""),
            qos=str(payload.get("qos") or ""),
            walltime=str(payload.get("walltime") or "04:00:00"),
            cpus_per_task=int(payload.get("cpus_per_task", 4)),
            memory=str(payload.get("memory") or "16G"),
            array_chunk_size=int(payload.get("array_chunk_size", 100)),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ClusterRuntimeConfig":
        return cls.from_mapping(load_config(path))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cluster_name": self.cluster_name,
            "scheduler": self.scheduler,
            "remote_host": self.remote_host,
            "remote_user": self.remote_user,
            "local_repo_root": self.local_repo_root,
            "remote_repo_root": self.remote_repo_root,
            "remote_scratch_root": self.remote_scratch_root,
            "remote_dataset_root": self.remote_dataset_root,
            "remote_structure_root": self.remote_structure_root,
            "conda_env": self.conda_env,
            "modules": list(self.modules),
            "account": self.account,
            "partition": self.partition,
            "qos": self.qos,
            "walltime": self.walltime,
            "cpus_per_task": self.cpus_per_task,
            "memory": self.memory,
            "array_chunk_size": self.array_chunk_size,
            "metadata": dict(self.metadata),
        }

    def link_commands(self) -> list[str]:
        """Return shell commands that establish remote runtime
        folders."""

        runtime_link_root = f"{self.remote_repo_root}/data_training/runtime"
        return [
            f"mkdir -p {self.remote_scratch_root}",
            f"mkdir -p {self.remote_dataset_root}",
            f"mkdir -p {self.remote_structure_root}",
            f"mkdir -p {runtime_link_root}",
            (
                "ln -sfn "
                f"{self.remote_structure_root} "
                f"{runtime_link_root}/structures"
            ),
            (
                "ln -sfn "
                f"{self.remote_dataset_root} "
                f"{runtime_link_root}/datasets"
            ),
        ]

    def render_slurm_array_stub(
        self,
        *,
        plan_path: str,
        job_count: int,
        runner: str = "python -m data_training.orchestration run-one",
    ) -> str:
        """Render a starter Slurm array script for future executors."""

        if self.scheduler.lower() != "slurm":
            raise ValueError("Only Slurm script rendering is scaffolded.")
        job_max = max(0, job_count - 1)
        header = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name=ewald-train-{self.cluster_name}",
            f"#SBATCH --time={self.walltime}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --mem={self.memory}",
            f"#SBATCH --array=0-{job_max}%{self.array_chunk_size}",
        ]
        if self.account:
            header.append(f"#SBATCH --account={self.account}")
        if self.partition:
            header.append(f"#SBATCH --partition={self.partition}")
        if self.qos:
            header.append(f"#SBATCH --qos={self.qos}")
        body = [
            "",
            "set -euo pipefail",
            *(f"module load {module}" for module in self.modules),
            f"conda activate {self.conda_env}",
            f"cd {self.remote_repo_root}",
            *self.link_commands(),
            (f"{runner} --plan {plan_path} " "--index ${SLURM_ARRAY_TASK_ID}"),
        ]
        return "\n".join(header + body) + "\n"
