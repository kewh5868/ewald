#!/usr/bin/env python3
"""Export top-k ranked structure-file guesses for artifact images."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import tifffile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_training" / "src"))

from ewald_data_training.manifests import read_jsonl_manifest  # noqa: E402
from ewald_data_training.artifact_features import (  # noqa: E402
    artifact_weight_map_from_labels,
)
from ewald_data_training.ranking import rank_image_candidates  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--artifact-weight-fraction",
        type=float,
        default=0.5,
        help=(
            "Blend fraction for artifact-weighted overlap. 0 keeps pure image "
            "overlap; 1 uses only artifact-weighted overlap."
        ),
    )
    parser.add_argument(
        "--no-artifact-aware",
        action="store_true",
        help="Disable artifact-derived ranking weights.",
    )
    args = parser.parse_args(argv)

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    model_root = Path(model["root"]).expanduser().resolve()
    candidates, candidate_meta = _load_candidates(model, model_root)
    samples = read_jsonl_manifest(args.manifest)
    source_root = Path(args.root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    exported = 0
    for sample in samples:
        if args.max_samples and exported >= args.max_samples:
            break
        image_path = source_root / sample.image_path
        if not image_path.exists():
            continue
        image = tifffile.imread(image_path)
        labels = _read_label_payload(source_root, sample.label_path)
        quality = labels.get("quality_assessment", {})
        weights = None
        if not args.no_artifact_aware:
            weights = artifact_weight_map_from_labels(labels, image.shape)
        scores = rank_image_candidates(
            image,
            candidates,
            weights=weights,
            artifact_weight_fraction=args.artifact_weight_fraction,
        )
        sample_dir = output_root / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        seen_structures: set[str] = set()
        for score in scores:
            meta = candidate_meta[score.candidate_id]
            structure_id = str(meta.get("structure_id", ""))
            if structure_id in seen_structures:
                continue
            seen_structures.add(structure_id)
            structure_rank = len(rows) + 1
            copied = _copy_structure_file(meta, sample_dir, structure_rank)
            rows.append(
                {
                    "rank": structure_rank,
                    "candidate_rank": score.rank,
                    "score": score.score,
                    "candidate_id": score.candidate_id,
                    "structure_id": structure_id,
                    "condition_id": meta.get("condition_id", ""),
                    "structure_file": copied,
                    "source_structure_file": meta.get(
                        "structure_file_path", ""
                    ),
                    "structure_name": meta.get("structure_name", ""),
                    "artifact_aware_score": not args.no_artifact_aware,
                    "artifact_weight_fraction": args.artifact_weight_fraction,
                }
            )
            if len(rows) >= args.top_k:
                break
        (sample_dir / "ranked_guesses.json").write_text(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "true_structure_id": sample.structure_id,
                    "top_k": args.top_k,
                    "artifact_aware": not args.no_artifact_aware,
                    "artifact_weight_fraction": args.artifact_weight_fraction,
                    "mean_artifact_weight": (
                        float(weights.mean()) if weights is not None else 1.0
                    ),
                    "quality_assessment": quality,
                    "guesses": rows,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        exported += 1
    print(f"exported ranked structure guesses for {exported} samples")
    return 0 if exported else 1


def _read_label_payload(root: Path, label_path: str) -> dict[str, object]:
    if not label_path:
        return {}
    resolved = root / label_path
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _load_candidates(
    model: dict[str, object],
    model_root: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    candidates = {}
    candidate_meta = {}
    for row in model.get("candidates", []):
        if not isinstance(row, dict):
            continue
        image_path = model_root / str(row.get("image_path", ""))
        if not image_path.exists():
            continue
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            continue
        candidates[candidate_id] = tifffile.imread(image_path)
        candidate_meta[candidate_id] = row
    return candidates, candidate_meta


def _copy_structure_file(
    metadata: dict[str, object],
    output_dir: Path,
    rank: int,
) -> str:
    source = Path(str(metadata.get("structure_file_path") or ""))
    if not source.exists():
        return ""
    target = output_dir / (
        f"rank_{rank:03d}_{metadata.get('structure_id', 'structure')}_"
        f"{source.name}"
    )
    shutil.copy2(source, target)
    return target.name


if __name__ == "__main__":
    raise SystemExit(main())
