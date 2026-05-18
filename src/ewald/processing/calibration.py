"""Calibration metadata for detector image processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CalibrationInputs:
    """Files and scalar settings needed to move detector data into q-space."""

    poni_file: Path | None = None
    mask_file: Path | None = None
    incident_angle_deg: float = 0.0
    tilt_angle_deg: float = 0.0
    sample_orientation: int = 4
    polarization: float = 0.95
    solid_angle: bool = True

    def validate(self) -> None:
        if self.poni_file and self.poni_file.suffix.lower() != ".poni":
            raise ValueError(
                "PONI calibration files must use the .poni suffix."
            )
        if self.mask_file and self.mask_file.suffix.lower() not in {
            ".edf",
            ".json",
            ".npy",
            ".npz",
            ".tif",
            ".tiff",
        }:
            raise ValueError(
                "Mask files must use .edf, .json, .npy, .npz, .tif, or "
                ".tiff suffixes."
            )
        if self.sample_orientation not in range(1, 9):
            raise ValueError("sample_orientation must be between 1 and 8.")

    def as_metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "poni_file": str(self.poni_file) if self.poni_file else None,
            "mask_file": str(self.mask_file) if self.mask_file else None,
            "incident_angle_deg": self.incident_angle_deg,
            "tilt_angle_deg": self.tilt_angle_deg,
            "sample_orientation": self.sample_orientation,
            "polarization": self.polarization,
            "solid_angle": self.solid_angle,
        }
