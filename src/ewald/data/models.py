"""Serializable data models for EWALD project state."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_EXTENSION = ".ewld"
PROJECT_SCHEMA_VERSION = "0.1"
ROI_HKL_METADATA_KEY = "hkl"
ROI_INTENSITY_METADATA_KEY = "integrated_intensity"
ROI_POLE_FIGURE_METADATA_KEY = "pole_figure"
PEAK_POINT_KIND_RAW = "raw-point"
PEAK_POINT_KIND_TEMPORARY_CHANNEL = "temporary-channel-point"
PEAK_POINT_KIND_COMMITTED = "committed-peak-point"
PEAK_POINT_KIND_FITTED_CENTER = "fitted-peak-center"
PEAK_POINT_KIND_GAP_ESTIMATED = "gap-estimated-peak"
PEAK_HKL_METADATA_KEY = "hkl"
PEAK_PHASE_METADATA_KEY = "phase_tag"
STRUCTURE_ANALYSIS_KEY = "structure_analysis"
STRUCTURE_ANALYSIS_PEAKS_KEY = "peaks"


@dataclass(slots=True)
class DataFileRef:
    """Reference to one detector image or a manifest of detector
    images."""

    path: Path
    data_id: str | None = None
    name: str | None = None
    kind: str = "detector-image"
    archive_path: str | None = None
    local_path: Path | None = field(default=None, repr=False, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.local_path is not None:
            self.local_path = Path(self.local_path)
        if self.data_id is None:
            self.data_id = self.path.stem
        if self.name is None:
            self.name = self.path.stem
        self.metadata["original_file_name"] = self.path.name

    @property
    def usable_path(self) -> Path:
        """Return the extracted archive copy when available."""

        return self.local_path or self.path

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "data_id": self.data_id,
            "name": self.name,
            "kind": self.kind,
            "archive_path": self.archive_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataFileRef":
        return cls(
            path=Path(payload["path"]),
            data_id=payload.get("data_id"),
            name=payload.get("name"),
            kind=payload.get("kind", "detector-image"),
            archive_path=payload.get("archive_path"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class DataGroupRef:
    """A folder or user-defined set of experimental detector images."""

    name: str
    group_id: str | None = None
    path: Path | None = None
    import_kind: str = "file"
    metadata_type: str = "filename"
    delimiter: str = "_"
    data_files: list[DataFileRef] = field(default_factory=list)
    parse_report: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.group_id is None:
            self.group_id = self.name.replace(" ", "_").lower()
        if self.path is not None:
            self.path = Path(self.path)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group_id": self.group_id,
            "path": str(self.path) if self.path is not None else None,
            "import_kind": self.import_kind,
            "metadata_type": self.metadata_type,
            "delimiter": self.delimiter,
            "data_files": [ref.as_dict() for ref in self.data_files],
            "parse_report": self.parse_report,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataGroupRef":
        path = payload.get("path")
        return cls(
            name=payload["name"],
            group_id=payload.get("group_id"),
            path=Path(path) if path else None,
            import_kind=payload.get("import_kind", "file"),
            metadata_type=payload.get("metadata_type", "filename"),
            delimiter=payload.get("delimiter", "_"),
            data_files=[
                DataFileRef.from_dict(item)
                for item in payload.get("data_files", [])
            ],
            parse_report=dict(payload.get("parse_report", {})),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class CorrectionAssetRef:
    """Mask or PONI calibrant that can be shared across data targets."""

    kind: str
    name: str
    asset_id: str | None = None
    path: Path | None = None
    archive_path: str | None = None
    local_path: Path | None = field(default=None, repr=False, compare=False)
    source: str = "loaded"
    target_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.asset_id is None:
            self.asset_id = _slug_name(f"{self.kind}_{self.name}")
        if self.path is not None:
            self.path = Path(self.path)
        if self.local_path is not None:
            self.local_path = Path(self.local_path)
        self.target_ids = list(dict.fromkeys(self.target_ids))

    @property
    def usable_path(self) -> Path | None:
        """Return the extracted archive copy when available."""

        return self.local_path or self.path

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "asset_id": self.asset_id,
            "path": str(self.path) if self.path is not None else None,
            "archive_path": self.archive_path,
            "source": self.source,
            "target_ids": self.target_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CorrectionAssetRef":
        path = payload.get("path")
        return cls(
            kind=payload["kind"],
            name=payload["name"],
            asset_id=payload.get("asset_id"),
            path=Path(path) if path else None,
            archive_path=payload.get("archive_path"),
            source=payload.get("source", "loaded"),
            target_ids=list(payload.get("target_ids", [])),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ImageCorrectionState:
    """Correction state for one imported detector image."""

    target_id: str
    mask_asset_id: str | None = None
    calibrant_asset_id: str | None = None
    xray_energy_kev: float | None = None
    image_rotation_deg: int = 0
    image_mirrored_y: bool = False
    pyfai_sample_orientation: int = 1
    correct_solid_angle: bool = True
    polarization_factor: float | None = 0.95
    normalization_factor: float = 1.0
    dummy: float | None = None
    delta_dummy: float | None = None
    reflected_beam_x_px: float | None = None
    reflected_beam_y_px: float | None = None
    critical_angle_deg: float | None = None
    sample_stoichiometry: str | None = None
    sample_density_g_cm3: float | None = None
    refractive_index_delta: float | None = None
    artifact_regions: list[dict[str, Any]] = field(default_factory=list)
    confirmed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "mask_asset_id": self.mask_asset_id,
            "calibrant_asset_id": self.calibrant_asset_id,
            "xray_energy_kev": self.xray_energy_kev,
            "image_rotation_deg": self.image_rotation_deg,
            "image_mirrored_y": self.image_mirrored_y,
            "pyfai_sample_orientation": self.pyfai_sample_orientation,
            "correct_solid_angle": self.correct_solid_angle,
            "polarization_factor": self.polarization_factor,
            "normalization_factor": self.normalization_factor,
            "dummy": self.dummy,
            "delta_dummy": self.delta_dummy,
            "reflected_beam_x_px": self.reflected_beam_x_px,
            "reflected_beam_y_px": self.reflected_beam_y_px,
            "critical_angle_deg": self.critical_angle_deg,
            "sample_stoichiometry": self.sample_stoichiometry,
            "sample_density_g_cm3": self.sample_density_g_cm3,
            "refractive_index_delta": self.refractive_index_delta,
            "artifact_regions": self.artifact_regions,
            "confirmed": self.confirmed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImageCorrectionState":
        return cls(
            target_id=payload["target_id"],
            mask_asset_id=payload.get("mask_asset_id"),
            calibrant_asset_id=payload.get("calibrant_asset_id"),
            xray_energy_kev=payload.get("xray_energy_kev"),
            image_rotation_deg=int(payload.get("image_rotation_deg", 0)),
            image_mirrored_y=bool(payload.get("image_mirrored_y", False)),
            pyfai_sample_orientation=int(
                payload.get("pyfai_sample_orientation", 1)
            ),
            correct_solid_angle=bool(payload.get("correct_solid_angle", True)),
            polarization_factor=payload.get("polarization_factor", 0.95),
            normalization_factor=float(
                payload.get("normalization_factor", 1.0)
            ),
            dummy=payload.get("dummy"),
            delta_dummy=payload.get("delta_dummy"),
            reflected_beam_x_px=payload.get("reflected_beam_x_px"),
            reflected_beam_y_px=payload.get("reflected_beam_y_px"),
            critical_angle_deg=payload.get("critical_angle_deg"),
            sample_stoichiometry=payload.get("sample_stoichiometry"),
            sample_density_g_cm3=payload.get("sample_density_g_cm3"),
            refractive_index_delta=payload.get("refractive_index_delta"),
            artifact_regions=list(payload.get("artifact_regions", [])),
            confirmed=bool(payload.get("confirmed", False)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ROIRegion:
    """A user-defined integration region on a corrected q-space
    image."""

    target_id: str
    kind: str = "box"
    roi_id: str | None = None
    name: str | None = None
    qxy_min: float | None = None
    qxy_max: float | None = None
    qz_min: float | None = None
    qz_max: float | None = None
    qxy_center: float = 0.0
    qz_center: float = 0.0
    qr_min: float | None = None
    qr_max: float | None = None
    chi_min: float | None = None
    chi_max: float | None = None
    integration_axis: str = "qz"
    integration_direction: str = "vertical"
    source: str = "drawn"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = self.kind.lower()
        if self.name is None:
            self.name = f"{self.kind.title()} ROI"
        if self.roi_id is None:
            self.roi_id = _slug_name(
                f"{self.target_id}_{self.kind}_{self.name}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "kind": self.kind,
            "roi_id": self.roi_id,
            "name": self.name,
            "qxy_min": self.qxy_min,
            "qxy_max": self.qxy_max,
            "qz_min": self.qz_min,
            "qz_max": self.qz_max,
            "qxy_center": self.qxy_center,
            "qz_center": self.qz_center,
            "qr_min": self.qr_min,
            "qr_max": self.qr_max,
            "chi_min": self.chi_min,
            "chi_max": self.chi_max,
            "integration_axis": self.integration_axis,
            "integration_direction": self.integration_direction,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ROIRegion":
        return cls(
            target_id=payload["target_id"],
            kind=payload.get("kind", "box"),
            roi_id=payload.get("roi_id"),
            name=payload.get("name"),
            qxy_min=payload.get("qxy_min"),
            qxy_max=payload.get("qxy_max"),
            qz_min=payload.get("qz_min"),
            qz_max=payload.get("qz_max"),
            qxy_center=float(payload.get("qxy_center", 0.0)),
            qz_center=float(payload.get("qz_center", 0.0)),
            qr_min=payload.get("qr_min"),
            qr_max=payload.get("qr_max"),
            chi_min=payload.get("chi_min"),
            chi_max=payload.get("chi_max"),
            integration_axis=payload.get("integration_axis", "qz"),
            integration_direction=payload.get(
                "integration_direction", "vertical"
            ),
            source=payload.get("source", "drawn"),
            metadata=dict(payload.get("metadata", {})),
        )


def roi_hkl_metadata(roi: ROIRegion) -> dict[str, Any]:
    """Return normalized optional hkl metadata for an ROI."""

    raw = roi.metadata.get(ROI_HKL_METADATA_KEY, {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "h": _optional_int(raw.get("h")),
        "k": _optional_int(raw.get("k")),
        "l": _optional_int(raw.get("l")),
        "label": _clean_optional_label(raw.get("label")),
    }


def set_roi_hkl_metadata(
    roi: ROIRegion,
    *,
    h: Any | None = None,
    k: Any | None = None,
    l: Any | None = None,
    label: Any | None = None,
) -> dict[str, Any]:
    """Validate and store optional integer hkl fields plus a custom
    label."""

    normalized = {
        "h": _optional_int(h),
        "k": _optional_int(k),
        "l": _optional_int(l),
        "label": _clean_optional_label(label),
    }
    if any(value is not None for value in normalized.values()):
        roi.metadata[ROI_HKL_METADATA_KEY] = normalized
    else:
        roi.metadata.pop(ROI_HKL_METADATA_KEY, None)
    return normalized


def roi_hkl_label(roi: ROIRegion) -> str:
    """Return a display hkl label, preferring a user custom label."""

    hkl = roi_hkl_metadata(roi)
    if hkl["label"]:
        return str(hkl["label"])
    if all(hkl[key] is not None for key in ("h", "k", "l")):
        return f"({hkl['h']} {hkl['k']} {hkl['l']})"
    return ""


def roi_intensity_metadata(roi: ROIRegion) -> dict[str, Any]:
    """Return normalized optional integrated-intensity metadata for an ROI."""

    raw = roi.metadata.get(ROI_INTENSITY_METADATA_KEY, {})
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        if key in {
            "integrated_intensity",
            "mean_intensity",
            "max_intensity",
            "min_intensity",
            "pixel_area",
            "qspace_area",
            "area_scaled_integrated_intensity",
        }:
            parsed = _optional_float(value)
            if parsed is not None:
                normalized[key] = parsed
        elif key in {"pixel_count", "finite_pixel_count"}:
            try:
                parsed_int = _optional_int(value)
            except ValueError:
                parsed_int = None
            if parsed_int is not None:
                normalized[key] = parsed_int
        elif key in {"geometry_signature", "method", "coordinate_space"}:
            normalized[key] = _clean_optional_label(value)
        else:
            normalized[key] = value
    return normalized


def set_roi_intensity_metadata(
    roi: ROIRegion,
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach or clear computed integrated-intensity metadata for an ROI."""

    if not record:
        roi.metadata.pop(ROI_INTENSITY_METADATA_KEY, None)
        return {}
    normalized = dict(record)
    integrated = _optional_float(normalized.get("integrated_intensity"))
    if integrated is None:
        roi.metadata.pop(ROI_INTENSITY_METADATA_KEY, None)
        return {}
    normalized["integrated_intensity"] = integrated
    normalized.setdefault("geometry_signature", roi_geometry_signature(roi))
    roi.metadata[ROI_INTENSITY_METADATA_KEY] = normalized
    return normalized


def roi_geometry_signature(roi: ROIRegion) -> str:
    """Return a stable signature for geometry fields that affect
    reduction."""

    payload = {
        "kind": roi.kind,
        "qxy_min": _rounded_float(roi.qxy_min),
        "qxy_max": _rounded_float(roi.qxy_max),
        "qz_min": _rounded_float(roi.qz_min),
        "qz_max": _rounded_float(roi.qz_max),
        "qxy_center": _rounded_float(roi.qxy_center),
        "qz_center": _rounded_float(roi.qz_center),
        "qr_min": _rounded_float(roi.qr_min),
        "qr_max": _rounded_float(roi.qr_max),
        "chi_min": _rounded_float(roi.chi_min),
        "chi_max": _rounded_float(roi.chi_max),
        "integration_axis": roi.integration_axis,
        "integration_direction": roi.integration_direction,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def roi_pole_figure_record(roi: ROIRegion) -> dict[str, Any] | None:
    """Return linked pole-figure metadata when present."""

    record = roi.metadata.get(ROI_POLE_FIGURE_METADATA_KEY)
    return record if isinstance(record, dict) else None


def set_roi_pole_figure_record(
    roi: ROIRegion,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Attach a generated pole-figure record to an ROI."""

    stored = dict(record)
    stored["roi_id"] = roi.roi_id
    stored["hkl_tag"] = roi_hkl_metadata(roi)
    stored["custom_label"] = roi_hkl_label(roi)
    stored["roi_geometry_signature"] = roi_geometry_signature(roi)
    stored["current"] = True
    stored.pop("stale_reason", None)
    roi.metadata[ROI_POLE_FIGURE_METADATA_KEY] = stored
    return stored


def roi_pole_figure_is_current(roi: ROIRegion) -> bool:
    """Return whether linked pole-figure metadata matches ROI
    geometry."""

    record = roi_pole_figure_record(roi)
    if record is None:
        return False
    return bool(record.get("current", False)) and record.get(
        "roi_geometry_signature"
    ) == roi_geometry_signature(roi)


def roi_pole_figure_status(roi: ROIRegion) -> str:
    """Return a compact table label for the linked pole-figure state."""

    record = roi_pole_figure_record(roi)
    if record is None:
        return ""
    return "Current" if roi_pole_figure_is_current(roi) else "Stale"


def mark_roi_pole_figure_stale(
    roi: ROIRegion,
    *,
    reason: str = "ROI geometry changed",
) -> None:
    """Mark a linked pole figure stale without removing saved
    metadata."""

    record = roi_pole_figure_record(roi)
    if record is None:
        return
    record["current"] = False
    record["stale_reason"] = reason


def _optional_int(value: Any | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("hkl values must be integers or blank.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("hkl values must be integers or blank.")
    return numeric


def _optional_float(value: Any | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _clean_optional_label(value: Any | None) -> str:
    if value is None:
        return ""
    label = " ".join(str(value).strip().split())
    if len(label) > 80:
        raise ValueError("hkl labels must be 80 characters or fewer.")
    return label


def _rounded_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


@dataclass(slots=True)
class ProcessingRecord:
    """One processing, fitting, structure, or simulation step."""

    stage: str
    parameters: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "parameters": self.parameters,
            "outputs": self.outputs,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProcessingRecord":
        return cls(
            stage=payload["stage"],
            parameters=dict(payload.get("parameters", {})),
            outputs=dict(payload.get("outputs", {})),
            notes=payload.get("notes", ""),
        )


@dataclass(slots=True)
class ProjectState:
    """In-memory representation of an EWALD project file."""

    name: str = "Untitled EWALD Project"
    schema_version: str = PROJECT_SCHEMA_VERSION
    data_groups: list[DataGroupRef] = field(default_factory=list)
    data_files: list[DataFileRef] = field(default_factory=list)
    masks: list[CorrectionAssetRef] = field(default_factory=list)
    calibrants: list[CorrectionAssetRef] = field(default_factory=list)
    processed_products: dict[str, str] = field(default_factory=dict)
    integration_regions: dict[str, Any] = field(default_factory=dict)
    roi_regions: dict[str, list[ROIRegion]] = field(default_factory=dict)
    image_corrections: dict[str, ImageCorrectionState] = field(
        default_factory=dict
    )
    film_material_memory: list[dict[str, Any]] = field(default_factory=list)
    peak_sets: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    fits: dict[str, Any] = field(default_factory=dict)
    analysis_results: dict[str, Any] = field(default_factory=dict)
    structures: dict[str, Any] = field(default_factory=dict)
    reference_cifs: dict[str, Any] = field(default_factory=dict)
    simulations: dict[str, Any] = field(default_factory=dict)
    processing_history: list[ProcessingRecord] = field(default_factory=list)

    def add_data_file(self, path: str | Path, **metadata: Any) -> DataFileRef:
        ref = DataFileRef(path=Path(path), metadata=metadata)
        self.data_files.append(ref)
        self.processing_history.append(
            ProcessingRecord(
                stage="data.load.requested",
                parameters={"path": str(ref.path), "data_id": ref.data_id},
            )
        )
        return ref

    def add_data_group(self, group: DataGroupRef) -> DataGroupRef:
        self.data_groups.append(group)
        self.processing_history.append(
            ProcessingRecord(
                stage="data.group.loaded",
                parameters={
                    "group_id": group.group_id,
                    "name": group.name,
                    "file_count": len(group.data_files),
                },
            )
        )
        return group

    def add_mask(
        self,
        path: str | Path | None = None,
        *,
        name: str | None = None,
        source: str = "loaded",
        target_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionAssetRef:
        """Add a mask asset and optionally apply it to data targets."""

        return self.add_correction_asset(
            "mask",
            path=path,
            name=name,
            source=source,
            target_ids=target_ids,
            metadata=metadata,
        )

    def add_calibrant(
        self,
        path: str | Path | None = None,
        *,
        name: str | None = None,
        source: str = "loaded",
        target_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionAssetRef:
        """Add a PONI calibrant and optionally apply it to data
        targets."""

        return self.add_correction_asset(
            "calibrant",
            path=path,
            name=name,
            source=source,
            target_ids=target_ids,
            metadata=metadata,
        )

    def add_correction_asset(
        self,
        kind: str,
        path: str | Path | None = None,
        *,
        name: str | None = None,
        source: str = "loaded",
        target_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CorrectionAssetRef:
        """Add a reusable correction asset to the project."""

        asset_path = Path(path) if path is not None else None
        asset_name = name or (asset_path.stem if asset_path else kind.title())
        asset = CorrectionAssetRef(
            kind=kind,
            name=asset_name,
            asset_id=self._unique_asset_id(kind, asset_name),
            path=asset_path,
            source=source,
            target_ids=target_ids or [],
            metadata=metadata or {},
        )
        self._asset_collection(kind).append(asset)
        self.processing_history.append(
            ProcessingRecord(
                stage=f"correction.{kind}.added",
                parameters={
                    "asset_id": asset.asset_id,
                    "path": str(asset.path) if asset.path else None,
                    "target_ids": asset.target_ids,
                },
            )
        )
        return asset

    def apply_correction_asset(
        self,
        kind: str,
        asset_id: str,
        target_id: str,
    ) -> CorrectionAssetRef:
        """Apply a mask or calibrant to a data file or data group."""

        asset = self.get_correction_asset(kind, asset_id)
        if target_id not in asset.target_ids:
            asset.target_ids.append(target_id)
            self.processing_history.append(
                ProcessingRecord(
                    stage=f"correction.{kind}.applied",
                    parameters={
                        "asset_id": asset.asset_id,
                        "target_id": target_id,
                    },
                )
            )
        return asset

    def set_correction_asset_assignment(
        self,
        kind: str,
        asset_id: str,
        target_id: str,
    ) -> CorrectionAssetRef:
        """Set the active correction asset for a data target."""

        for asset in self._asset_collection(kind):
            if asset.asset_id != asset_id and target_id in asset.target_ids:
                asset.target_ids.remove(target_id)
        return self.apply_correction_asset(kind, asset_id, target_id)

    def apply_mask(
        self,
        asset_id: str,
        target_id: str,
    ) -> CorrectionAssetRef:
        """Apply a mask asset to a data target."""

        return self.apply_correction_asset("mask", asset_id, target_id)

    def apply_calibrant(
        self,
        asset_id: str,
        target_id: str,
    ) -> CorrectionAssetRef:
        """Apply a PONI calibrant to a data target."""

        return self.apply_correction_asset("calibrant", asset_id, target_id)

    def get_correction_asset(
        self,
        kind: str,
        asset_id: str,
    ) -> CorrectionAssetRef:
        """Return one correction asset by kind and id."""

        for asset in self._asset_collection(kind):
            if asset.asset_id == asset_id:
                return asset
        raise KeyError(f"No {kind} asset with id {asset_id!r}")

    def assigned_assets(
        self,
        kind: str,
        target_id: str | None,
    ) -> list[CorrectionAssetRef]:
        """Return correction assets applied directly to a data
        target."""

        if target_id is None:
            return []
        return [
            asset
            for asset in self._asset_collection(kind)
            if target_id in asset.target_ids
        ]

    def set_image_corrections(
        self,
        state: ImageCorrectionState,
    ) -> ImageCorrectionState:
        """Store correction confirmation state for one detector
        image."""

        self.image_corrections[state.target_id] = state
        self.processing_history.append(
            ProcessingRecord(
                stage="image.corrections.confirmed",
                parameters=state.as_dict(),
            )
        )
        return state

    def remember_film_material(
        self,
        stoichiometry: str,
        density_g_cm3: float,
        *,
        label: str | None = None,
        memory_id: str | None = None,
        refractive_index_delta: float | None = None,
        critical_angle_deg: float | None = None,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add or update a remembered film composition/density pair."""

        formula = " ".join(str(stoichiometry).strip().split())
        if not formula:
            raise ValueError("Film stoichiometry is required.")
        density = float(density_g_cm3)
        if density <= 0:
            raise ValueError("Film density must be positive.")
        item_id = memory_id or _unique_memory_id(
            self.film_material_memory,
            f"{formula}_{density:g}",
        )
        payload = {
            "memory_id": item_id,
            "label": label or f"{formula} ({density:g} g/cm3)",
            "stoichiometry": formula,
            "density_g_cm3": density,
            "refractive_index_delta": refractive_index_delta,
            "critical_angle_deg": critical_angle_deg,
            "source": source,
            "metadata": dict(metadata or {}),
        }
        for index, item in enumerate(self.film_material_memory):
            if item.get("memory_id") == item_id:
                self.film_material_memory[index] = payload
                break
        else:
            self.film_material_memory.append(payload)
        return payload

    def remove_film_material_memory(self, memory_id: str) -> bool:
        """Remove one remembered film material by id."""

        before = len(self.film_material_memory)
        self.film_material_memory = [
            item
            for item in self.film_material_memory
            if item.get("memory_id") != memory_id
        ]
        return len(self.film_material_memory) != before

    def clear_film_material_memory(self) -> None:
        """Clear all remembered film material entries."""

        self.film_material_memory.clear()

    def remember_loaded_cif(
        self,
        path: str | Path,
        *,
        cif_text: str,
        lattice: dict[str, Any],
        crystal_system: str | None = None,
        label: str | None = None,
        target_id: str | None = None,
    ) -> dict[str, Any]:
        """Store a user-loaded CIF in the project-scoped CIF library."""

        cif_path = Path(path)
        cif_id = _loaded_cif_id(cif_path, cif_text)
        loaded = self.reference_cifs.setdefault("loaded", {})
        if not isinstance(loaded, dict):
            loaded = {}
            self.reference_cifs["loaded"] = loaded
        target_ids: list[str] = []
        existing = loaded.get(cif_id)
        if isinstance(existing, dict):
            target_ids = [str(item) for item in existing.get("target_ids", [])]
        if target_id:
            target_text = str(target_id)
            if target_text not in target_ids:
                target_ids.append(target_text)
        record = {
            "cif_id": cif_id,
            "label": label or cif_path.stem,
            "path": str(cif_path),
            "structure_path": str(cif_path),
            "source": "loaded",
            "cif_text": cif_text,
            "lattice": {
                key: float(lattice[key])
                for key in ("a", "b", "c", "alpha", "beta", "gamma")
            },
            "crystal_system": crystal_system or "Triclinic",
            "target_ids": target_ids,
        }
        loaded[cif_id] = record
        self.processing_history.append(
            ProcessingRecord(
                stage="reference_cif.loaded",
                parameters={
                    "cif_id": cif_id,
                    "path": str(cif_path),
                    "target_id": target_id,
                    "lattice": record["lattice"],
                    "crystal_system": record["crystal_system"],
                },
            )
        )
        return record

    def image_corrections_confirmed(self, target_id: str | None) -> bool:
        """Return True when corrections are permanently confirmed."""

        if target_id is None:
            return False
        state = self.image_corrections.get(target_id)
        return bool(state and state.confirmed)

    def add_roi_region(self, roi: ROIRegion) -> ROIRegion:
        """Store a user-drawn ROI for one corrected data target."""

        regions = self.roi_regions.setdefault(roi.target_id, [])
        roi.roi_id = self._unique_roi_id(roi.target_id, roi.roi_id)
        if roi.name is None:
            roi.name = f"{roi.kind.title()} ROI {len(regions) + 1}"
        regions.append(roi)
        self.processing_history.append(
            ProcessingRecord(
                stage="roi.region.added",
                parameters=roi.as_dict(),
            )
        )
        return roi

    def set_roi_regions(
        self,
        target_id: str,
        regions: list[ROIRegion],
    ) -> None:
        """Replace the ROI list for a target."""

        existing: set[str] = set()
        normalized: list[ROIRegion] = []
        for region in regions:
            region.target_id = target_id
            region.roi_id = self._unique_roi_id(
                target_id,
                region.roi_id,
                existing=existing,
            )
            existing.add(region.roi_id)
            normalized.append(region)
        self.roi_regions[target_id] = normalized

    def rois_for_target(self, target_id: str | None) -> list[ROIRegion]:
        """Return saved ROIs for a target."""

        if target_id is None:
            return []
        return list(self.roi_regions.get(target_id, []))

    def roi_by_id(
        self,
        target_id: str | None,
        roi_id: str | None,
    ) -> ROIRegion | None:
        """Return one ROI by target and id."""

        if target_id is None or roi_id is None:
            return None
        for roi in self.roi_regions.get(target_id, []):
            if roi.roi_id == roi_id:
                return roi
        return None

    def mark_roi_pole_figures_stale(
        self,
        target_id: str,
        roi_id: str,
        *,
        reason: str = "ROI geometry changed",
    ) -> None:
        """Mark pole figures linked to an ROI or its coupled pair
        stale."""

        linked_ids = {roi_id}
        for roi in self.roi_regions.get(target_id, []):
            coupled = _metadata_id_set(roi.metadata.get("coupled_roi_ids"))
            coupled.add(str(roi.metadata.get("coupled_roi_id") or ""))
            if roi.roi_id == roi_id:
                linked_ids.update(coupled)
            elif roi_id in coupled and roi.roi_id:
                linked_ids.add(roi.roi_id)
        linked_ids.discard("")
        for roi in self.roi_regions.get(target_id, []):
            if roi.roi_id in linked_ids:
                mark_roi_pole_figure_stale(roi, reason=reason)
        pole_figures = self.analysis_results.get("pole_figures", {})
        if not isinstance(pole_figures, dict):
            return
        target_pole_figures = pole_figures.get(target_id, {})
        if not isinstance(target_pole_figures, dict):
            return
        for linked_id in linked_ids:
            record = target_pole_figures.get(linked_id)
            if isinstance(record, dict):
                record["current"] = False
                record["stale_reason"] = reason

    def set_roi_hkl_tag(
        self,
        target_id: str,
        roi_id: str,
        *,
        h: Any | None = None,
        k: Any | None = None,
        l: Any | None = None,
        label: Any | None = None,
    ) -> dict[str, Any]:
        """Set an ROI hkl tag and refresh linked pole-figure labels."""

        roi = self.roi_by_id(target_id, roi_id)
        if roi is None:
            raise KeyError(f"No ROI {roi_id!r} for target {target_id!r}")
        tag = set_roi_hkl_metadata(roi, h=h, k=k, l=l, label=label)
        pole_record = roi_pole_figure_record(roi)
        if pole_record is not None:
            pole_record["hkl_tag"] = tag
            pole_record["custom_label"] = roi_hkl_label(roi)
        pole_figures = self.analysis_results.get("pole_figures", {})
        if isinstance(pole_figures, dict):
            target_pole_figures = pole_figures.get(target_id, {})
            if isinstance(target_pole_figures, dict):
                stored_pole_record = target_pole_figures.get(roi_id)
                if isinstance(stored_pole_record, dict):
                    stored_pole_record["hkl_tag"] = tag
                    stored_pole_record["custom_label"] = roi_hkl_label(roi)
        has_tag = any(
            value is not None and value != "" for value in tag.values()
        )
        for record in self.peak_sets.get(target_id, []):
            if record.get("roi_id") == roi_id:
                metadata = record.setdefault("metadata", {})
                if has_tag:
                    record[PEAK_HKL_METADATA_KEY] = dict(tag)
                    metadata[PEAK_HKL_METADATA_KEY] = dict(tag)
                else:
                    record.pop(PEAK_HKL_METADATA_KEY, None)
                    metadata.pop(PEAK_HKL_METADATA_KEY, None)
                self.sync_structure_analysis_peak_from_fit(
                    target_id,
                    str(record.get("peak_id") or record.get("id") or ""),
                )
        return tag

    def set_roi_pole_figure_metadata(
        self,
        target_id: str,
        roi_id: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach generated pole-figure metadata to an ROI."""

        roi = self.roi_by_id(target_id, roi_id)
        if roi is None:
            raise KeyError(f"No ROI {roi_id!r} for target {target_id!r}")
        stored = set_roi_pole_figure_record(roi, record)
        pole_figures_by_target = self.analysis_results.setdefault(
            "pole_figures",
            {},
        )
        if not isinstance(pole_figures_by_target, dict):
            pole_figures_by_target = {}
            self.analysis_results["pole_figures"] = pole_figures_by_target
        pole_figures = pole_figures_by_target.setdefault(target_id, {})
        if isinstance(pole_figures, dict):
            pole_figures[roi_id] = stored
        return stored

    def peak_by_id(
        self,
        target_id: str,
        peak_id: str | None,
    ) -> dict[str, Any] | None:
        """Return one committed peak record by id."""

        if peak_id is None:
            return None
        for record in self.peak_sets.get(target_id, []):
            if _peak_record_id(record) == peak_id:
                _normalize_peak_record(record)
                return record
        return None

    def set_peak_phase_tag(
        self,
        target_id: str,
        peak_id: str,
        phase_tag: str | None,
    ) -> dict[str, Any]:
        """Set a peak phase tag and mark structure candidates for
        refresh."""

        record = self.peak_by_id(target_id, peak_id)
        if record is None:
            raise KeyError(f"No peak {peak_id!r} for target {target_id!r}")
        phase = " ".join(str(phase_tag or "").strip().split())
        if phase:
            record[PEAK_PHASE_METADATA_KEY] = phase
            record.setdefault("metadata", {})[PEAK_PHASE_METADATA_KEY] = phase
        else:
            record.pop(PEAK_PHASE_METADATA_KEY, None)
            record.setdefault("metadata", {}).pop(
                PEAK_PHASE_METADATA_KEY,
                None,
            )
        entry = self.sync_structure_analysis_peak_from_fit(target_id, peak_id)
        structure = self._structure_analysis(target_id)
        structure["candidate_selection_stale"] = True
        structure["phase_tags"] = sorted(
            {
                str(item.get(PEAK_PHASE_METADATA_KEY))
                for item in self.peak_sets.get(target_id, [])
                if item.get(PEAK_PHASE_METADATA_KEY)
            }
        )
        return entry or {}

    def set_peak_hkl_tag(
        self,
        target_id: str,
        peak_id: str,
        *,
        h: Any | None = None,
        k: Any | None = None,
        l: Any | None = None,
        label: Any | None = None,
    ) -> dict[str, Any]:
        """Set a peak hkl tag and mirror it to the linked ROI when
        present."""

        record = self.peak_by_id(target_id, peak_id)
        if record is None:
            raise KeyError(f"No peak {peak_id!r} for target {target_id!r}")
        tag = {
            "h": _optional_int(h),
            "k": _optional_int(k),
            "l": _optional_int(l),
            "label": _clean_optional_label(label),
        }
        if any(value is not None and value != "" for value in tag.values()):
            record[PEAK_HKL_METADATA_KEY] = dict(tag)
            record.setdefault("metadata", {})[PEAK_HKL_METADATA_KEY] = dict(
                tag
            )
        else:
            record.pop(PEAK_HKL_METADATA_KEY, None)
            record.setdefault("metadata", {}).pop(PEAK_HKL_METADATA_KEY, None)
        roi_id = record.get("roi_id")
        if roi_id:
            self.set_roi_hkl_tag(target_id, str(roi_id), **tag)
        self.sync_structure_analysis_peak_from_fit(target_id, peak_id)
        return tag

    def set_peak_fit_result(
        self,
        target_id: str,
        peak_id: str,
        fit_2d: dict[str, Any],
        *,
        roi_id: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store the fitted center source of truth and sync analysis
        rows."""

        fit_store = self._peak_fit_store(target_id, create=True)
        peak_store = fit_store.setdefault(
            peak_id,
            {"peak_id": peak_id, "label": peak_id},
        )
        if not isinstance(peak_store, dict):
            peak_store = {"peak_id": peak_id, "label": peak_id}
            fit_store[peak_id] = peak_store
        peak_store["fit_2d"] = dict(fit_2d)
        if roi_id:
            peak_store["roi_id"] = roi_id
        if metrics:
            statistics = peak_store["fit_2d"].setdefault("statistics", {})
            if isinstance(statistics, dict):
                statistics.update(metrics)
        return (
            self.sync_structure_analysis_peak_from_fit(
                target_id,
                peak_id,
            )
            or {}
        )

    def sync_structure_analysis_peak_from_fit(
        self,
        target_id: str,
        peak_id: str,
    ) -> dict[str, Any] | None:
        """Refresh one Structure Analysis entry from peak and fit
        state."""

        if not peak_id:
            return None
        record = self.peak_by_id(target_id, peak_id)
        fit_store = self._peak_fit_store(target_id, create=False)
        fit_record = fit_store.get(peak_id, {})
        if not isinstance(fit_record, dict):
            fit_record = {}
        fit_2d = fit_record.get("fit_2d")
        if not isinstance(fit_2d, dict) and record is None:
            return None
        peaks = self._structure_analysis_peaks(target_id)
        entry = _structure_peak_entry(peaks, peak_id)
        user_edited = bool(entry.get("metadata", {}).get("user_edited_center"))
        center_qxy = (
            fit_2d.get("center_qxy") if isinstance(fit_2d, dict) else None
        )
        center_qz = (
            fit_2d.get("center_qz") if isinstance(fit_2d, dict) else None
        )
        if center_qxy is None and record is not None:
            center_qxy = _peak_qxy(record)
        if center_qz is None and record is not None:
            center_qz = _peak_qz(record)
        roi_id = (
            fit_record.get("roi_id")
            or (record or {}).get("roi_id")
            or (record or {}).get("roi", {}).get("roi_id")
        )
        source = (
            PEAK_POINT_KIND_FITTED_CENTER
            if isinstance(fit_2d, dict)
            else (record or {}).get("point_kind", PEAK_POINT_KIND_COMMITTED)
        )
        if _record_is_gap_estimated(record):
            source = PEAK_POINT_KIND_GAP_ESTIMATED
        entry.update(
            {
                "peak_id": peak_id,
                "label": (record or {}).get("label", peak_id),
                "source": source,
                "source_peak_id": peak_id,
                "source_fit_key": "fit_2d" if isinstance(fit_2d, dict) else "",
                "roi_id": roi_id,
                "phase_tag": _peak_phase_tag(record),
                "hkl_label": _hkl_label_from_tag(_peak_hkl_tag(record)),
                "hkl_tag": _peak_hkl_tag(record),
                "gap_estimated": _record_is_gap_estimated(record),
                "estimate_method": _gap_estimate_method(record),
                "fit_quality": (
                    fit_2d.get("statistics", {}).get("r_squared")
                    if isinstance(fit_2d, dict)
                    else None
                ),
                "fit_status": (
                    fit_2d.get("status", "")
                    if isinstance(fit_2d, dict)
                    else ""
                ),
                "fit_metrics": (
                    dict(fit_2d.get("statistics", {}))
                    if isinstance(fit_2d, dict)
                    else {}
                ),
                "metadata": {
                    **dict(entry.get("metadata", {})),
                    "peak_record": dict(record or {}),
                    "fit_record": dict(fit_record),
                },
            }
        )
        if not user_edited:
            entry["qxy"] = center_qxy
            entry["qz"] = center_qz
            entry["center_qxy"] = center_qxy
            entry["center_qz"] = center_qz
        else:
            entry["source"] = "structure-analysis-manual"
        return entry

    def update_structure_analysis_peak(
        self,
        target_id: str,
        peak_id: str,
        **updates: Any,
    ) -> dict[str, Any]:
        """Apply a manual Structure Analysis table edit."""

        peaks = self._structure_analysis_peaks(target_id)
        entry = _structure_peak_entry(peaks, peak_id)
        if "qxy" in updates and "center_qxy" not in updates:
            updates["center_qxy"] = updates["qxy"]
        if "qz" in updates and "center_qz" not in updates:
            updates["center_qz"] = updates["qz"]
        if "center_qxy" in updates and "qxy" not in updates:
            updates["qxy"] = updates["center_qxy"]
        if "center_qz" in updates and "qz" not in updates:
            updates["qz"] = updates["center_qz"]
        entry.update(updates)
        if {"qxy", "qz", "center_qxy", "center_qz"} & updates.keys():
            entry["source"] = "structure-analysis-manual"
            entry.setdefault("metadata", {})["user_edited_center"] = True
        return entry

    def link_simulation_to_data_file(
        self,
        simulation_id: str,
        data_id: str | None,
    ) -> dict[str, Any]:
        """Associate a stored simulation with one detector data file."""

        record = self.simulations.get(simulation_id)
        if not isinstance(record, dict):
            raise KeyError(f"No simulation with id {simulation_id!r}")
        if data_id is not None and self.data_file_by_id(data_id) is None:
            raise KeyError(f"No data file with id {data_id!r}")

        previous_data_id = record.get("data_id")
        if data_id:
            record["data_id"] = data_id
        else:
            record.pop("data_id", None)
        self.processing_history.append(
            ProcessingRecord(
                stage="simulation.data_file.linked",
                parameters={
                    "simulation_id": simulation_id,
                    "data_id": data_id,
                    "previous_data_id": previous_data_id,
                },
            )
        )
        return record

    def simulations_for_data_file(
        self,
        data_id: str | None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return stored simulations linked to a detector data file."""

        if data_id is None:
            return []
        return [
            (simulation_id, record)
            for simulation_id, record in self.simulations.items()
            if isinstance(record, dict) and record.get("data_id") == data_id
        ]

    def data_file_by_id(self, data_id: str | None) -> DataFileRef | None:
        """Return a data file reference from grouped or ungrouped
        data."""

        if data_id is None:
            return None
        for group in self.data_groups:
            for data_file in group.data_files:
                if data_file.data_id == data_id:
                    return data_file
        for data_file in self.data_files:
            if data_file.data_id == data_id:
                return data_file
        return None

    def _target_analysis(self, target_id: str) -> dict[str, Any]:
        analysis = self.analysis_results.setdefault(target_id, {})
        if not isinstance(analysis, dict):
            analysis = {"legacy": analysis}
            self.analysis_results[target_id] = analysis
        return analysis

    def _structure_analysis(self, target_id: str) -> dict[str, Any]:
        analyses = self.analysis_results.setdefault(STRUCTURE_ANALYSIS_KEY, {})
        if not isinstance(analyses, dict):
            analyses = {}
            self.analysis_results[STRUCTURE_ANALYSIS_KEY] = analyses
        structure = analyses.setdefault(
            target_id,
            {
                STRUCTURE_ANALYSIS_PEAKS_KEY: [],
                "candidates": [],
                "families": [],
                "wyckoff": {},
            },
        )
        if not isinstance(structure, dict):
            structure = {"legacy": structure, STRUCTURE_ANALYSIS_PEAKS_KEY: []}
            analyses[target_id] = structure
        return structure

    def _structure_analysis_peaks(
        self, target_id: str
    ) -> list[dict[str, Any]]:
        structure = self._structure_analysis(target_id)
        peaks = structure.setdefault(STRUCTURE_ANALYSIS_PEAKS_KEY, [])
        if not isinstance(peaks, list):
            peaks = []
            structure[STRUCTURE_ANALYSIS_PEAKS_KEY] = peaks
        return peaks

    def _peak_fit_store(
        self,
        target_id: str,
        *,
        create: bool,
    ) -> dict[str, Any]:
        container = self.fits.get(target_id)
        if container is None:
            if not create:
                return {}
            container = {}
            self.fits[target_id] = container
        if not isinstance(container, dict):
            if not create:
                return {}
            container = {"legacy": container}
            self.fits[target_id] = container
        peak_fit = container.get("peak_fit")
        if peak_fit is None:
            if not create:
                return {}
            peak_fit = {}
            container["peak_fit"] = peak_fit
        if not isinstance(peak_fit, dict):
            if not create:
                return {}
            peak_fit = {}
            container["peak_fit"] = peak_fit
        return peak_fit

    def _asset_collection(self, kind: str) -> list[CorrectionAssetRef]:
        if kind == "mask":
            return self.masks
        if kind == "calibrant":
            return self.calibrants
        raise ValueError(f"Unsupported correction asset kind: {kind}")

    def _unique_asset_id(self, kind: str, name: str) -> str:
        existing = {asset.asset_id for asset in self._asset_collection(kind)}
        base = _slug_name(f"{kind}_{name}")
        candidate = base
        index = 2
        while candidate in existing:
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    def _unique_roi_id(
        self,
        target_id: str,
        roi_id: str | None,
        *,
        existing: set[str] | None = None,
    ) -> str:
        target_regions = self.roi_regions.get(target_id, [])
        used = {region.roi_id for region in target_regions}
        if existing:
            used.update(existing)
        base = _slug_name(roi_id or f"{target_id}_roi")
        candidate = base
        index = 2
        while candidate in used:
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "data_groups": [group.as_dict() for group in self.data_groups],
            "data_files": [ref.as_dict() for ref in self.data_files],
            "masks": [asset.as_dict() for asset in self.masks],
            "calibrants": [asset.as_dict() for asset in self.calibrants],
            "processed_products": self.processed_products,
            "integration_regions": self.integration_regions,
            "roi_regions": {
                target_id: [region.as_dict() for region in regions]
                for target_id, regions in self.roi_regions.items()
            },
            "image_corrections": {
                target_id: state.as_dict()
                for target_id, state in self.image_corrections.items()
            },
            "film_material_memory": self.film_material_memory,
            "peak_sets": self.peak_sets,
            "fits": self.fits,
            "analysis_results": self.analysis_results,
            "structures": self.structures,
            "reference_cifs": self.reference_cifs,
            "simulations": self.simulations,
            "processing_history": [
                record.as_dict() for record in self.processing_history
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectState":
        return cls(
            name=payload.get("name", "Untitled EWALD Project"),
            schema_version=payload.get(
                "schema_version", PROJECT_SCHEMA_VERSION
            ),
            data_groups=[
                DataGroupRef.from_dict(item)
                for item in payload.get("data_groups", [])
            ],
            data_files=[
                DataFileRef.from_dict(item)
                for item in payload.get("data_files", [])
            ],
            masks=[
                CorrectionAssetRef.from_dict(item)
                for item in payload.get("masks", [])
            ],
            calibrants=[
                CorrectionAssetRef.from_dict(item)
                for item in payload.get("calibrants", [])
            ],
            processed_products=dict(payload.get("processed_products", {})),
            integration_regions=dict(payload.get("integration_regions", {})),
            roi_regions={
                target_id: [ROIRegion.from_dict(region) for region in regions]
                for target_id, regions in payload.get(
                    "roi_regions", {}
                ).items()
            },
            image_corrections={
                target_id: ImageCorrectionState.from_dict(state)
                for target_id, state in payload.get(
                    "image_corrections", {}
                ).items()
            },
            film_material_memory=list(payload.get("film_material_memory", [])),
            peak_sets=dict(payload.get("peak_sets", {})),
            fits=dict(payload.get("fits", {})),
            analysis_results=dict(payload.get("analysis_results", {})),
            structures=dict(payload.get("structures", {})),
            reference_cifs=dict(payload.get("reference_cifs", {})),
            simulations=dict(payload.get("simulations", {})),
            processing_history=[
                ProcessingRecord.from_dict(item)
                for item in payload.get("processing_history", [])
            ],
        )


def _slug_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "asset"


def _loaded_cif_id(path: Path, cif_text: str) -> str:
    basis = f"{path.expanduser()}::{cif_text}"
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]
    return f"{_slug_name(path.stem)}_{digest}"


def _unique_memory_id(items: list[dict[str, Any]], value: str) -> str:
    used = {str(item.get("memory_id")) for item in items}
    base = _slug_name(f"film_{value}")
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def _metadata_id_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item}
    return set()


def _structure_peak_entry(
    peaks: list[dict[str, Any]],
    peak_id: str,
) -> dict[str, Any]:
    for item in peaks:
        if isinstance(item, dict) and item.get("peak_id") == peak_id:
            return item
    entry = {"peak_id": peak_id}
    peaks.append(entry)
    return entry


def _peak_record_id(record: dict[str, Any]) -> str:
    return str(record.get("peak_id") or record.get("id") or "")


def _normalize_peak_record(record: dict[str, Any]) -> None:
    if not record.get("point_kind"):
        record["point_kind"] = PEAK_POINT_KIND_COMMITTED
    if record.get("gap_estimated"):
        record["point_kind"] = PEAK_POINT_KIND_GAP_ESTIMATED
    if "metadata" not in record or not isinstance(record["metadata"], dict):
        record["metadata"] = {}
    if PEAK_PHASE_METADATA_KEY in record:
        record["metadata"][PEAK_PHASE_METADATA_KEY] = record[
            PEAK_PHASE_METADATA_KEY
        ]
    if PEAK_HKL_METADATA_KEY in record:
        record["metadata"][PEAK_HKL_METADATA_KEY] = record[
            PEAK_HKL_METADATA_KEY
        ]


def _peak_qxy(record: dict[str, Any]) -> float | None:
    value = record.get("qxy", record.get("qx", record.get("x")))
    return float(value) if value is not None else None


def _peak_qz(record: dict[str, Any]) -> float | None:
    value = record.get("qz", record.get("y"))
    return float(value) if value is not None else None


def _peak_phase_tag(record: dict[str, Any] | None) -> str:
    if not isinstance(record, dict):
        return ""
    value = record.get(PEAK_PHASE_METADATA_KEY) or record.get(
        "metadata",
        {},
    ).get(PEAK_PHASE_METADATA_KEY)
    return str(value) if value else ""


def _peak_hkl_tag(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    raw = record.get(PEAK_HKL_METADATA_KEY) or record.get("metadata", {}).get(
        PEAK_HKL_METADATA_KEY,
        {},
    )
    return dict(raw) if isinstance(raw, dict) else {}


def _hkl_label_from_tag(tag: dict[str, Any]) -> str:
    label = str(tag.get("label") or "").strip()
    if label:
        return label
    values = [tag.get(key) for key in ("h", "k", "l")]
    if any(value is None for value in values):
        return ""
    return f"({int(values[0])} {int(values[1])} {int(values[2])})"


def _record_is_gap_estimated(record: dict[str, Any] | None) -> bool:
    if not isinstance(record, dict):
        return False
    metadata = record.get("metadata", {})
    return bool(
        record.get("gap_estimated")
        or record.get("point_kind") == PEAK_POINT_KIND_GAP_ESTIMATED
        or (isinstance(metadata, dict) and metadata.get("gap_estimate"))
        or "gap" in str(record.get("source", "")).lower()
    )


def _gap_estimate_method(record: dict[str, Any] | None) -> str:
    if not isinstance(record, dict):
        return ""
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        method = metadata.get("estimate_method") or metadata.get(
            "gap_estimate_method"
        )
        if method:
            return str(method)
    return str(record.get("estimate_method") or "")
