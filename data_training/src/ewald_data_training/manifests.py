"""Manifest read/write helpers for generated training datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import DatasetSample


def write_jsonl_manifest(
    path: str | Path,
    samples: Iterable[DatasetSample],
) -> None:
    """Write dataset sample rows as JSON Lines."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.as_dict(), sort_keys=True) + "\n")


def read_jsonl_manifest(path: str | Path) -> list[DatasetSample]:
    """Read a JSON Lines sample manifest."""

    manifest_path = Path(path)
    samples: list[DatasetSample] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{manifest_path}:{line_number}: invalid JSON"
            ) from exc
        samples.append(DatasetSample.from_mapping(payload))
    return samples


def validate_manifest_files(
    samples: Iterable[DatasetSample],
    *,
    root: str | Path,
) -> list[str]:
    """Return missing-file errors for sample paths."""

    base = Path(root).expanduser().resolve()
    errors: list[str] = []
    for sample in samples:
        for field_name in (
            "image_path",
            "label_path",
            "clean_image_path",
            "peak_table_path",
        ):
            raw_path = getattr(sample, field_name)
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = base / path
            if not path.exists():
                errors.append(
                    f"{sample.sample_id}: missing {field_name} {path}"
                )
    return errors
