"""Command-line entry points for the isolated data-training section."""

from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import load_structure_catalog, validate_catalog_paths
from .conditions import load_generation_plan
from .manifests import (
    read_jsonl_manifest,
    validate_manifest_files,
    write_jsonl_manifest,
)
from .simulator import generate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ewald-data-training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_catalog = subparsers.add_parser("validate-catalog")
    validate_catalog.add_argument("catalog")

    plan = subparsers.add_parser("plan")
    plan.add_argument("plan")

    generate = subparsers.add_parser("generate")
    generate.add_argument("--plan", required=True)
    generate.add_argument("--output-root")
    generate.add_argument("--manifest")
    generate.add_argument("--dry-run", action="store_true")

    validate_manifest = subparsers.add_parser("validate-manifest")
    validate_manifest.add_argument("manifest")
    validate_manifest.add_argument("--root", required=True)

    args = parser.parse_args(argv)
    if args.command == "validate-catalog":
        records = load_structure_catalog(args.catalog)
        errors = validate_catalog_paths(records, args.catalog)
        for error in errors:
            print(error)
        print(f"{len(records)} structures, {len(errors)} errors")
        return 1 if errors else 0
    if args.command == "plan":
        loaded = load_generation_plan(args.plan)
        sample_count = len(loaded["structures"]) * len(loaded["conditions"])
        print(f"dataset: {loaded['dataset']}")
        print(f"structures: {len(loaded['structures'])}")
        print(f"conditions: {len(loaded['conditions'])}")
        print(f"samples: {sample_count}")
        print(f"output_root: {loaded['output_root']}")
        return 0
    if args.command == "generate":
        loaded = load_generation_plan(args.plan)
        output_root = Path(args.output_root or loaded["output_root"])
        samples = generate_dataset(
            structures=loaded["structures"],
            conditions=loaded["conditions"],
            catalog_root=Path(loaded["structures_path"]).parent,
            output_root=output_root,
            artifact_profiles=loaded["artifact_profiles"],
            dry_run=args.dry_run,
        )
        manifest = Path(args.manifest or output_root / "manifest.jsonl")
        write_jsonl_manifest(manifest, samples)
        print(f"wrote {len(samples)} samples to {manifest}")
        return 0
    if args.command == "validate-manifest":
        samples = read_jsonl_manifest(args.manifest)
        errors = validate_manifest_files(samples, root=args.root)
        for error in errors:
            print(error)
        print(f"{len(samples)} samples, {len(errors)} errors")
        return 1 if errors else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
