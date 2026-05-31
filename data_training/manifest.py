"""Dataset manifest records for generated GIWAXS training images."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config_io import load_config, write_json


@dataclass(frozen=True, slots=True)
class PeakLabel:
    """One labeled Bragg peak in q-space and detector-image
    coordinates."""

    h: int
    k: int
    l: int
    qxy: float
    qz: float
    intensity: float
    amplitude: float
    pixel_x: float | None = None
    pixel_y: float | None = None
    family_id: str = ""

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "PeakLabel":
        return cls(
            h=int(payload.get("h", 0)),
            k=int(payload.get("k", 0)),
            l=int(payload.get("l", 0)),
            qxy=float(payload.get("qxy", 0.0)),
            qz=float(payload.get("qz", 0.0)),
            intensity=float(payload.get("intensity", 0.0)),
            amplitude=float(payload.get("amplitude", 0.0)),
            pixel_x=_optional_float(payload.get("pixel_x")),
            pixel_y=_optional_float(payload.get("pixel_y")),
            family_id=str(payload.get("family_id") or ""),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "h": self.h,
            "k": self.k,
            "l": self.l,
            "qxy": self.qxy,
            "qz": self.qz,
            "intensity": self.intensity,
            "amplitude": self.amplitude,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "family_id": self.family_id,
        }


@dataclass(frozen=True, slots=True)
class DatasetImageRecord:
    """One generated image and its supervised-learning labels."""

    sample_id: str
    split: str
    structure_id: str
    condition_id: str
    artifact_profile_id: str
    image_path: str
    peak_table_path: str
    label_path: str
    simulator: str
    parameters: dict[str, Any] = field(default_factory=dict)
    texture: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    labels: tuple[PeakLabel, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DatasetImageRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            split=str(payload.get("split") or "train"),
            structure_id=str(payload["structure_id"]),
            condition_id=str(payload["condition_id"]),
            artifact_profile_id=str(
                payload.get("artifact_profile_id") or "clean"
            ),
            image_path=str(payload["image_path"]),
            peak_table_path=str(payload.get("peak_table_path") or ""),
            label_path=str(payload.get("label_path") or ""),
            simulator=str(payload.get("simulator") or ""),
            parameters=dict(payload.get("parameters") or {}),
            texture=dict(payload.get("texture") or {}),
            artifacts=dict(payload.get("artifacts") or {}),
            labels=tuple(
                PeakLabel.from_mapping(item)
                for item in payload.get("labels", [])
            ),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "structure_id": self.structure_id,
            "condition_id": self.condition_id,
            "artifact_profile_id": self.artifact_profile_id,
            "image_path": self.image_path,
            "peak_table_path": self.peak_table_path,
            "label_path": self.label_path,
            "simulator": self.simulator,
            "parameters": dict(self.parameters),
            "texture": dict(self.texture),
            "artifacts": dict(self.artifacts),
            "labels": [label.to_mapping() for label in self.labels],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Top-level manifest for a generated training dataset."""

    dataset_id: str
    records: tuple[DatasetImageRecord, ...] = ()
    schema_version: str = "1"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DatasetManifest":
        return cls(
            dataset_id=str(payload.get("dataset_id") or "training_dataset"),
            schema_version=str(payload.get("schema_version") or "1"),
            description=str(payload.get("description") or ""),
            metadata=dict(payload.get("metadata") or {}),
            records=tuple(
                DatasetImageRecord.from_mapping(item)
                for item in payload.get("records", [])
            ),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetManifest":
        return cls.from_mapping(load_config(path))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "description": self.description,
            "metadata": dict(self.metadata),
            "records": [record.to_mapping() for record in self.records],
        }

    def write_json(self, path: str | Path) -> Path:
        return write_json(path, self.to_mapping())


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
