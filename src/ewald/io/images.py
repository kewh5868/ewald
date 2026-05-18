"""Detector image loading facade."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ewald.data.dataset import DetectorImageSet, open_detector_images

SUPPORTED_IMAGE_EXTENSIONS = {".tif", ".tiff"}


def validate_detector_paths(paths: Iterable[str | Path]) -> list[Path]:
    valid_paths = [Path(path) for path in paths]
    invalid = [
        path
        for path in valid_paths
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if invalid:
        formatted = ", ".join(str(path) for path in invalid)
        raise ValueError(f"Unsupported detector image extension: {formatted}")
    return valid_paths


def load_detector_images(paths: Iterable[str | Path]) -> DetectorImageSet:
    return open_detector_images(validate_detector_paths(paths))
