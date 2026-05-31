#!/usr/bin/env python3
"""Apply random detector-artifact variants to clean generated samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tifffile
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_training" / "src"))

from ewald_data_training.artifacts import apply_artifacts  # noqa: E402
from ewald_data_training.artifact_features import (  # noqa: E402
    build_artifact_assessment,
    estimate_retrieval_quality,
)
from ewald_data_training.manifests import (  # noqa: E402
    read_jsonl_manifest,
    write_jsonl_manifest,
)
from ewald_data_training.schemas import (
    ArtifactProfile,
    DatasetSample,
    DetectorGeometry,
    stable_id,
)  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest")
    parser.add_argument("--variants-per-profile", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args(argv)

    source_root = Path(args.root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    profiles = _load_profiles(Path(args.profiles))
    samples = read_jsonl_manifest(args.manifest)
    augmented: list[DatasetSample] = []
    for sample in samples:
        clean_path = source_root / sample.clean_image_path
        image = tifffile.imread(clean_path)
        labels = _sample_labels(source_root, sample)
        detector = _sample_detector(labels)
        sample_context = dict(labels.get("sample_scattering", {}) or {})
        for profile in profiles:
            for variant in range(args.variants_per_profile):
                seed = args.seed + len(augmented)
                artifact_image, metadata = apply_artifacts(
                    image,
                    profile,
                    seed=seed,
                    detector=detector,
                    sample_context=sample_context,
                )
                artifact_assessment = build_artifact_assessment(
                    artifact_metadata=metadata,
                    artifact_profile=profile,
                    detector=detector,
                    image_shape=image.shape,
                )
                quality_assessment = estimate_retrieval_quality(
                    image,
                    artifact_image,
                    artifact_assessment=artifact_assessment,
                    detector=detector,
                )
                variant_id = stable_id(
                    {
                        "sample_id": sample.sample_id,
                        "profile": profile.profile_id,
                        "variant": variant,
                        "seed": seed,
                    },
                    "art_",
                )
                sample_dir = output_root / variant_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                image_path = sample_dir / "artifact.tiff"
                label_path = sample_dir / "labels.json"
                tifffile.imwrite(image_path, artifact_image)
                label_path.write_text(
                    json.dumps(
                        {
                            "source_sample": sample.as_dict(),
                            "structure": labels.get("structure", {}),
                            "condition": labels.get("condition", {}),
                            "sample_scattering": sample_context,
                            "peak_table_path": sample.peak_table_path,
                            "artifact_profile": profile.as_dict(),
                            "artifact_metadata": metadata,
                            "artifact_assessment": artifact_assessment,
                            "quality_assessment": quality_assessment,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                augmented.append(
                    DatasetSample(
                        sample_id=variant_id,
                        structure_id=sample.structure_id,
                        condition_id=sample.condition_id,
                        image_path=str(image_path.relative_to(output_root)),
                        label_path=str(label_path.relative_to(output_root)),
                        clean_image_path=str(
                            clean_path.relative_to(source_root)
                        ),
                        peak_table_path=sample.peak_table_path,
                        artifact_profile_id=profile.profile_id,
                        seed=seed,
                        metadata={
                            "source_sample_id": sample.sample_id,
                            "variant": variant,
                        },
                    )
                )
    output_manifest = Path(
        args.output_manifest or output_root / "artifact_manifest.jsonl"
    )
    write_jsonl_manifest(output_manifest, augmented)
    print(f"wrote {len(augmented)} artifact variants to {output_manifest}")
    return 0


def _load_profiles(path: Path) -> list[ArtifactProfile]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = payload.get(
        "artifact_profiles", payload.get("profiles", payload)
    )
    if isinstance(entries, dict):
        entries = [
            {"profile_id": key, **(value or {})}
            for key, value in entries.items()
        ]
    return [ArtifactProfile.from_mapping(entry) for entry in entries]


def _sample_labels(source_root: Path, sample: DatasetSample) -> dict:
    if not sample.label_path:
        return {}
    label_path = source_root / sample.label_path
    if not label_path.exists():
        return {}
    try:
        return json.loads(label_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _sample_detector(labels: dict) -> DetectorGeometry | None:
    if not labels:
        return None
    condition = labels.get("condition") or {}
    detector = condition.get("detector")
    if not detector:
        return None
    return DetectorGeometry.from_mapping(detector)


if __name__ == "__main__":
    raise SystemExit(main())
