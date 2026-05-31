"""CLI-like planning skeleton for EWALD training data generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import artifact_profile_index, read_artifact_profiles
from .catalog import StructureCatalog, read_structure_catalog
from .conditions import SimulationCondition, read_simulation_sweep
from .config_io import write_json
from .manifest import DatasetImageRecord, DatasetManifest
from .runtime import ClusterRuntimeConfig


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """Expanded jobs for a future local or cluster executor."""

    plan_id: str
    catalog_id: str
    sweep_id: str
    records: tuple[DatasetImageRecord, ...]
    conditions: tuple[SimulationCondition, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "catalog_id": self.catalog_id,
            "sweep_id": self.sweep_id,
            "metadata": dict(self.metadata),
            "records": [record.to_mapping() for record in self.records],
            "conditions": [item.to_mapping() for item in self.conditions],
        }

    def summary(self) -> dict[str, Any]:
        split_counts: dict[str, int] = {}
        structure_counts: dict[str, int] = {}
        artifact_counts: dict[str, int] = {}
        for record in self.records:
            split_counts[record.split] = split_counts.get(record.split, 0) + 1
            structure_counts[record.structure_id] = (
                structure_counts.get(record.structure_id, 0) + 1
            )
            artifact_counts[record.artifact_profile_id] = (
                artifact_counts.get(record.artifact_profile_id, 0) + 1
            )
        return {
            "plan_id": self.plan_id,
            "catalog_id": self.catalog_id,
            "sweep_id": self.sweep_id,
            "job_count": len(self.records),
            "split_counts": split_counts,
            "structure_counts": structure_counts,
            "artifact_counts": artifact_counts,
        }


def build_training_plan(
    *,
    catalog_path: str | Path,
    sweep_path: str | Path,
    artifact_path: str | Path,
) -> TrainingPlan:
    """Expand configs into a deterministic generation plan."""

    catalog = read_structure_catalog(catalog_path)
    sweep = read_simulation_sweep(sweep_path)
    artifacts = artifact_profile_index(read_artifact_profiles(artifact_path))
    conditions = sweep.expand(catalog)
    missing_artifacts = sorted(
        {
            condition.artifact_profile_id
            for condition in conditions
            if condition.artifact_profile_id not in artifacts
        }
    )
    if missing_artifacts:
        raise KeyError(
            "Unknown artifact profile(s): " + ", ".join(missing_artifacts)
        )

    records = tuple(
        _record_for_condition(
            catalog, condition, sweep.split_policy, artifacts
        )
        for condition in conditions
    )
    plan_id = _stable_id(
        {
            "catalog_id": catalog.catalog_id,
            "sweep_id": sweep.sweep_id,
            "record_ids": [record.sample_id for record in records],
        },
        length=12,
    )
    return TrainingPlan(
        plan_id=plan_id,
        catalog_id=catalog.catalog_id,
        sweep_id=sweep.sweep_id,
        records=records,
        conditions=conditions,
        metadata={
            "catalog_path": str(catalog_path),
            "sweep_path": str(sweep_path),
            "artifact_path": str(artifact_path),
            "structure_root": catalog.default_structure_root,
            "note": "Plan only; execution is implemented by future runners.",
        },
    )


def manifest_from_plan(plan: TrainingPlan) -> DatasetManifest:
    """Create a dataset manifest shell from a training plan."""

    return DatasetManifest(
        dataset_id=f"ewald_training_{plan.plan_id}",
        description="Generated from an EWALD data_training plan.",
        records=plan.records,
        metadata={
            "plan_id": plan.plan_id,
            "catalog_id": plan.catalog_id,
            "sweep_id": plan.sweep_id,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan EWALD structure-recognition training datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    _add_plan_args(plan_parser)
    plan_parser.add_argument("--output", type=Path)
    plan_parser.add_argument("--limit", type=int, default=5)

    manifest_parser = subparsers.add_parser("manifest-template")
    _add_plan_args(manifest_parser)
    manifest_parser.add_argument("--output", type=Path, required=True)

    cluster_parser = subparsers.add_parser("cluster-script")
    _add_plan_args(cluster_parser)
    cluster_parser.add_argument("--runtime", type=Path, required=True)
    cluster_parser.add_argument("--plan-path", default="training_plan.json")
    cluster_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate")
    _add_plan_args(validate_parser)
    validate_parser.add_argument("--runtime", type=Path)

    run_one_parser = subparsers.add_parser("run-one")
    run_one_parser.add_argument("--plan", type=Path, required=True)
    run_one_parser.add_argument("--index", type=int, required=True)

    args = parser.parse_args(argv)

    if args.command == "run-one":
        return _run_one_placeholder(args.plan, args.index)

    plan = build_training_plan(
        catalog_path=args.catalog,
        sweep_path=args.sweep,
        artifact_path=args.artifacts,
    )

    if args.command == "plan":
        payload = {
            "summary": plan.summary(),
            "preview_records": [
                record.to_mapping() for record in plan.records[: args.limit]
            ],
        }
        if args.output:
            write_json(args.output, plan.to_mapping())
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "manifest-template":
        manifest_from_plan(plan).write_json(args.output)
        print(f"Wrote {args.output}")
        return 0

    if args.command == "cluster-script":
        runtime = ClusterRuntimeConfig.from_file(args.runtime)
        script = runtime.render_slurm_array_stub(
            plan_path=args.plan_path,
            job_count=len(plan.records),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(script, encoding="utf-8")
        print(f"Wrote {args.output}")
        return 0

    if args.command == "validate":
        runtime = (
            ClusterRuntimeConfig.from_file(args.runtime)
            if args.runtime
            else None
        )
        payload = {"summary": plan.summary()}
        if runtime is not None:
            payload["runtime"] = runtime.to_mapping()
            payload["link_commands"] = runtime.link_commands()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    return 1


def _add_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data_training/configs/structure_catalog.yaml"),
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("data_training/configs/simulation_sweep.yaml"),
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("data_training/configs/artifact_profiles.yaml"),
    )


def _record_for_condition(
    catalog: StructureCatalog,
    condition: SimulationCondition,
    split_policy: dict[str, float],
    artifacts: dict[str, Any],
) -> DatasetImageRecord:
    structure = catalog.by_id()[condition.structure_id]
    sample_id = _stable_id(condition.to_mapping(), length=18)
    split = _split_for_sample(sample_id, split_policy)
    relative_root = f"{split}/{structure.structure_id}/{sample_id}"
    artifact_profile = artifacts[condition.artifact_profile_id]
    return DatasetImageRecord(
        sample_id=sample_id,
        split=split,
        structure_id=structure.structure_id,
        condition_id=condition.condition_id,
        artifact_profile_id=condition.artifact_profile_id,
        image_path=f"images/{relative_root}.tiff",
        peak_table_path=f"labels/{relative_root}.peaks.json",
        label_path=f"labels/{relative_root}.label.json",
        simulator=condition.simulator,
        parameters=condition.parameters,
        texture=condition.texture,
        artifacts=artifact_profile.to_mapping(),
        labels=(),
        metadata={
            "source_path": structure.source_path,
            "peak_table_exporter": condition.peak_table_exporter,
            "axis_values": condition.axis_values,
        },
    )


def _split_for_sample(sample_id: str, split_policy: dict[str, float]) -> str:
    train = float(split_policy.get("train", 0.8))
    validation = float(split_policy.get("validation", 0.1))
    threshold_train = max(0.0, min(1.0, train))
    threshold_validation = max(0.0, min(1.0, train + validation))
    value = int(sample_id[:8], 16) / float(0xFFFFFFFF)
    if value < threshold_train:
        return "train"
    if value < threshold_validation:
        return "validation"
    return "test"


def _stable_id(payload: dict[str, Any], *, length: int = 16) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


def _run_one_placeholder(plan_path: Path, index: int) -> int:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if index < 0 or index >= len(records):
        raise IndexError(
            f"Plan index {index} is outside 0..{len(records) - 1}."
        )
    record = records[index]
    print(
        json.dumps(
            {
                "status": "placeholder",
                "message": (
                    "Executor not implemented yet. Future code should call the "
                    "record simulator, write image/labels, then update manifest."
                ),
                "record": record,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
