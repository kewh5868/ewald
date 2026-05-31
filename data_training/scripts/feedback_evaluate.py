#!/usr/bin/env python3
"""Evaluate ranker feedback metrics against an artifact manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tifffile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_training" / "src"))

from ewald_data_training.artifact_features import (  # noqa: E402
    artifact_weight_map_from_labels,
)
from ewald_data_training.manifests import read_jsonl_manifest  # noqa: E402
from ewald_data_training.ranking import rank_image_candidates  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history")
    parser.add_argument("--epoch", type=int, default=0)
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
    candidates = {}
    candidate_meta = {}
    for row in model.get("candidates", []):
        image_path = model_root / row["image_path"]
        if not image_path.exists():
            continue
        candidates[row["candidate_id"]] = tifffile.imread(image_path)
        candidate_meta[row["candidate_id"]] = row
    samples = read_jsonl_manifest(args.manifest)
    root = Path(args.root).expanduser().resolve()
    rows = []
    top1 = 0
    topk = 0
    evaluated = 0
    for sample in samples:
        if args.max_samples and evaluated >= args.max_samples:
            break
        image_path = root / sample.image_path
        if not image_path.exists():
            continue
        image = tifffile.imread(image_path)
        labels = _read_label_payload(root, sample.label_path)
        weights = None
        if not args.no_artifact_aware:
            weights = artifact_weight_map_from_labels(labels, image.shape)
        quality = labels.get("quality_assessment", {})
        scores = rank_image_candidates(
            image,
            candidates,
            weights=weights,
            artifact_weight_fraction=args.artifact_weight_fraction,
        )
        ranked_structures = [
            candidate_meta[item.candidate_id]["structure_id"]
            for item in scores[: args.top_k]
        ]
        hit1 = bool(
            ranked_structures and ranked_structures[0] == sample.structure_id
        )
        hitk = sample.structure_id in ranked_structures
        top1 += int(hit1)
        topk += int(hitk)
        evaluated += 1
        rows.append(
            {
                "sample_id": sample.sample_id,
                "true_structure_id": sample.structure_id,
                "top_candidate_id": scores[0].candidate_id if scores else "",
                "top_structure_id": (
                    ranked_structures[0] if ranked_structures else ""
                ),
                "top_score": scores[0].score if scores else 0.0,
                "top1_hit": hit1,
                "topk_hit": hitk,
                "artifact_aware": not args.no_artifact_aware,
                "artifact_weight_fraction": args.artifact_weight_fraction,
                "artifact_operations": (
                    labels.get("artifact_assessment", {}).get("operations", [])
                    if isinstance(labels.get("artifact_assessment"), dict)
                    else []
                ),
                "mean_artifact_weight": (
                    float(weights.mean()) if weights is not None else 1.0
                ),
                "quality_solvable": (
                    bool(quality.get("solvable", True))
                    if isinstance(quality, dict)
                    else True
                ),
                "quality_signal_to_noise": (
                    float(quality.get("signal_to_noise", 0.0))
                    if isinstance(quality, dict)
                    else 0.0
                ),
            }
        )
    metrics = {
        "epoch": args.epoch,
        "evaluated": evaluated,
        "candidate_count": len(candidates),
        "artifact_aware": not args.no_artifact_aware,
        "artifact_weight_fraction": args.artifact_weight_fraction,
        "solvable_sample_fraction": (
            sum(1 for row in rows if row.get("quality_solvable")) / evaluated
            if evaluated
            else 0.0
        ),
        "top1_accuracy": top1 / evaluated if evaluated else 0.0,
        f"top{args.top_k}_accuracy": topk / evaluated if evaluated else 0.0,
        "rows": rows,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.history:
        history = Path(args.history).expanduser().resolve()
        history.parent.mkdir(parents=True, exist_ok=True)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({k: v for k, v in metrics.items() if k != "rows"})
                + "\n"
            )
    print(
        f"evaluated={evaluated} top1={metrics['top1_accuracy']:.3f} "
        f"top{args.top_k}={metrics[f'top{args.top_k}_accuracy']:.3f}"
    )
    return 0 if evaluated else 1


def _read_label_payload(root: Path, label_path: str) -> dict[str, object]:
    if not label_path:
        return {}
    resolved = root / label_path
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
