"""Tests that preserve the source-checkout example dataset contract."""

from importlib.util import find_spec
from pathlib import Path

import pytest

from ewald.io.example_manifest import (
    calibration_from_example_manifest,
    load_example_manifest,
    project_from_example_manifest,
)


def test_example_manifest_points_to_existing_files(example_manifest_path):
    manifest = load_example_manifest(example_manifest_path)

    data_path = manifest["data"][0]["path"]
    calibration = manifest["calibration"]

    assert data_path.endswith(".tiff")
    assert calibration["poni_file"].endswith(".poni")
    assert calibration["mask_file"].endswith(".edf")
    assert all(
        path.exists()
        for path in [
            example_manifest_path.parent / "calib.poni",
            example_manifest_path.parent / "mask.edf",
            Path(manifest["data"][0]["path"]),
        ]
    )


def test_example_manifest_builds_project_and_calibration(
    example_manifest_path,
):
    project = project_from_example_manifest(example_manifest_path)
    calibration = calibration_from_example_manifest(example_manifest_path)

    assert project.name == "EWALD single-image GIWAXS example"
    assert project.data_files[0].data_id == "sam22_1MAI1PbI2_giwaxs"
    assert project.data_files[0].metadata["incident_angle_deg"] == 0.3
    assert calibration.sample_orientation == 4
    assert calibration.poni_file.name == "calib.poni"


@pytest.mark.skipif(find_spec("tifffile") is None, reason="tifffile missing")
def test_example_detector_image_can_be_read(example_manifest_path):
    import tifffile

    manifest = load_example_manifest(example_manifest_path)
    image = tifffile.imread(manifest["data"][0]["path"])

    assert image.ndim == 2
    assert image.size > 0
