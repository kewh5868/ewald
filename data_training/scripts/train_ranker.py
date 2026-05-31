#!/usr/bin/env python3
"""Build a lightweight vector-ranker checkpoint from clean
simulations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_training" / "src"))

from ewald_data_training.artifact_features import (  # noqa: E402
    ARTIFACT_ASSESSMENT_SCHEMA,
)
from ewald_data_training.manifests import read_jsonl_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    samples = read_jsonl_manifest(args.manifest)
    candidates = []
    for sample in samples:
        if not sample.clean_image_path:
            continue
        clean_path = root / sample.clean_image_path
        if not args.dry_run and not clean_path.exists():
            continue
        label_payload = _read_label_payload(root, sample.label_path)
        structure_payload = label_payload.get("structure") or {}
        source_structure_path = str(
            label_payload.get("source_structure_path") or ""
        )
        candidates.append(
            {
                "candidate_id": sample.sample_id,
                "structure_id": sample.structure_id,
                "condition_id": sample.condition_id,
                "image_path": sample.clean_image_path,
                "peak_table_path": sample.peak_table_path,
                "artifact_profile_id": sample.artifact_profile_id,
                "structure_name": structure_payload.get("name", ""),
                "structure_file_path": source_structure_path,
                "structure_file_name": (
                    Path(source_structure_path).name
                    if source_structure_path
                    else ""
                ),
                "structure_metadata": structure_payload.get("metadata", {}),
            }
        )
        if args.max_candidates and len(candidates) >= args.max_candidates:
            break
    model = {
        "model_type": "ewald_vector_ranker_v1_artifact_aware",
        "status": "dry_run" if args.dry_run else "ready",
        "root": str(root),
        "artifact_assessment_schema": ARTIFACT_ASSESSMENT_SCHEMA,
        "artifact_aware_scoring": (
            "feedback_evaluate.py and export_structure_guesses.py use "
            "artifact_assessment labels to build ranker weights for direct "
            "beam, Yoneda, substrate horizon, footprint-spillage, "
            "critical-angle, beamstop, and detector-mask regions. The default "
            "feedback score blends clean image overlap with artifact-weighted "
            "overlap so the baseline keeps Bragg retrieval sensitivity while "
            "tracking non-Bragg aberrations."
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "notes": (
            "Baseline checkpoint for deployment testing. It stores clean "
            "simulation candidates for normalized image-overlap ranking, while "
            "artifact labels on observed samples provide peak-assessment masks "
            "and weights for aberration-aware retrieval."
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(model, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"wrote vector-ranker checkpoint with {len(candidates)} candidates to {output}"
    )
    return 0 if candidates else 1


def _read_label_payload(root: Path, label_path: str) -> dict[str, object]:
    if not label_path:
        return {}
    resolved = root / label_path
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
