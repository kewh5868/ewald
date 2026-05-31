"""Typed records used by the EWALD training-data pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a training-data configuration cannot be
    interpreted."""


def stable_id(payload: Mapping[str, Any], prefix: str = "") -> str:
    """Return a stable short id for JSON-serializable metadata."""

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:12]
    return f"{prefix}{digest}" if prefix else digest


def _tuple_float(
    value: Any,
    *,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    if value is None:
        raise ConfigError(f"Missing required field: {field_name}")
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ConfigError(f"{field_name} must contain {length} numbers")
    return tuple(float(item) for item in value)


@dataclass(slots=True)
class StructureRecord:
    """Catalog entry for a structure file used as synthetic truth."""

    structure_id: str
    name: str
    path: str
    file_format: str = "cif"
    family: str = ""
    phase_class: str = ""
    source: str = ""
    license: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StructureRecord":
        path = str(payload.get("path", "")).strip()
        name = str(payload.get("name") or Path(path).stem).strip()
        if not path:
            raise ConfigError("Structure entries require a path.")
        if not name:
            raise ConfigError("Structure entries require a name.")
        structure_id = str(
            payload.get("structure_id")
            or stable_id({"name": name, "path": path}, "str_")
        )
        tags = tuple(str(tag) for tag in payload.get("tags", ()) or ())
        metadata = dict(payload.get("metadata", {}) or {})
        return cls(
            structure_id=structure_id,
            name=name,
            path=path,
            file_format=str(payload.get("file_format", "cif")),
            family=str(payload.get("family", "")),
            phase_class=str(payload.get("phase_class", "")),
            source=str(payload.get("source", "")),
            license=str(payload.get("license", "")),
            tags=tags,
            metadata=metadata,
        )

    def resolved_path(self, catalog_root: Path) -> Path:
        """Return an absolute path relative to a catalog file
        directory."""

        candidate = Path(self.path).expanduser()
        if not candidate.is_absolute():
            candidate = catalog_root / candidate
        return candidate.resolve()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DetectorGeometry:
    """Reciprocal-space image extent and sampling."""

    qxy_range: tuple[float, float] = (-4.0, 4.0)
    qz_range: tuple[float, float] = (0.0, 4.0)
    resolution: tuple[int, int] = (384, 256)
    wavelength_angstrom: float | None = None
    incident_angle_deg: float | None = None
    tilt_angle_deg: float = 0.0
    solid_angle_correction: bool = True
    missing_wedge_correction: bool = True
    detector: str = "reciprocal-space-grid"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DetectorGeometry":
        qxy_range = _tuple_float(
            payload.get("qxy_range", (-3.0, 3.0)),
            length=2,
            field_name="qxy_range",
        )
        qz_range = _tuple_float(
            payload.get("qz_range", (0.0, 3.0)),
            length=2,
            field_name="qz_range",
        )
        resolution_raw = payload.get("resolution", (256, 128))
        if not isinstance(resolution_raw, (list, tuple)):
            raise ConfigError("resolution must contain [x_pixels, z_pixels]")
        if len(resolution_raw) != 2:
            raise ConfigError("resolution must contain [x_pixels, z_pixels]")
        resolution = (int(resolution_raw[0]), int(resolution_raw[1]))
        return cls(
            qxy_range=(float(qxy_range[0]), float(qxy_range[1])),
            qz_range=(float(qz_range[0]), float(qz_range[1])),
            resolution=resolution,
            wavelength_angstrom=_optional_float(
                payload.get("wavelength_angstrom")
            ),
            incident_angle_deg=_optional_float(
                payload.get("incident_angle_deg")
            ),
            tilt_angle_deg=float(payload.get("tilt_angle_deg", 0.0)),
            solid_angle_correction=bool(
                payload.get("solid_angle_correction", True)
            ),
            missing_wedge_correction=bool(
                payload.get("missing_wedge_correction", True)
            ),
            detector=str(payload.get("detector", "reciprocal-space-grid")),
        )

    def as_giwaxs_parameters(self) -> dict[str, Any]:
        """Return EWALD simulation parameter keys for this grid."""

        return {
            "qxy_min": self.qxy_range[0],
            "qxy_max": self.qxy_range[1],
            "qz_min": self.qz_range[0],
            "qz_max": self.qz_range[1],
            "resolution_x": int(self.resolution[0]),
            "resolution_z": int(self.resolution[1]),
            "wavelength_angstrom": self.wavelength_angstrom,
            "incident_angle_deg": self.incident_angle_deg,
            "tilt_angle_deg": self.tilt_angle_deg,
            "solid_angle_correction": self.solid_angle_correction,
            "missing_wedge_correction": self.missing_wedge_correction,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SimulationCondition:
    """One deterministic forward-model condition for one structure."""

    condition_id: str
    theta_x_deg: float = 90.0
    theta_y_deg: float = 0.0
    sigma_theta: float = 0.03
    sigma_phi: float = 0.25
    sigma_r: float = 0.035
    q_dependent_sigma_r: float = 0.0
    q_dependent_sigma_z: float = 0.0
    hkl_extent: int = 4
    orientation_label: str = "fiber"
    texture_model: str = "fiber_gaussian"
    artifact_profile_id: str = "default"
    seed: int = 0
    detector: DetectorGeometry = field(default_factory=DetectorGeometry)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        detector: DetectorGeometry | None = None,
    ) -> "SimulationCondition":
        geometry = detector or DetectorGeometry.from_mapping(
            payload.get("detector", {}) or {}
        )
        base = {
            "theta_x_deg": float(payload.get("theta_x_deg", 90.0)),
            "theta_y_deg": float(payload.get("theta_y_deg", 0.0)),
            "sigma_theta": float(payload.get("sigma_theta", 0.03)),
            "sigma_phi": float(payload.get("sigma_phi", 0.25)),
            "sigma_r": float(payload.get("sigma_r", 0.035)),
            "q_dependent_sigma_r": float(
                payload.get("q_dependent_sigma_r", 0.0)
            ),
            "q_dependent_sigma_z": float(
                payload.get("q_dependent_sigma_z", 0.0)
            ),
            "hkl_extent": int(payload.get("hkl_extent", 4)),
            "artifact_profile_id": str(
                payload.get("artifact_profile_id", "default")
            ),
            "seed": int(payload.get("seed", 0)),
            "detector": geometry.as_dict(),
        }
        condition_id = str(
            payload.get("condition_id") or stable_id(base, "cnd_")
        )
        return cls(
            condition_id=condition_id,
            theta_x_deg=base["theta_x_deg"],
            theta_y_deg=base["theta_y_deg"],
            sigma_theta=base["sigma_theta"],
            sigma_phi=base["sigma_phi"],
            sigma_r=base["sigma_r"],
            q_dependent_sigma_r=base["q_dependent_sigma_r"],
            q_dependent_sigma_z=base["q_dependent_sigma_z"],
            hkl_extent=base["hkl_extent"],
            orientation_label=str(payload.get("orientation_label", "fiber")),
            texture_model=str(payload.get("texture_model", "fiber_gaussian")),
            artifact_profile_id=base["artifact_profile_id"],
            seed=base["seed"],
            detector=geometry,
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def as_giwaxs_parameters(self) -> dict[str, Any]:
        params = self.detector.as_giwaxs_parameters()
        params.update(
            {
                "sigma_theta": self.sigma_theta,
                "sigma_phi": self.sigma_phi,
                "sigma_r": self.sigma_r,
                "q_dependent_sigma_r": self.q_dependent_sigma_r,
                "q_dependent_sigma_z": self.q_dependent_sigma_z,
                "hkl_extent": self.hkl_extent,
                "theta_x_deg": self.theta_x_deg,
                "theta_y_deg": self.theta_y_deg,
            }
        )
        return params

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["detector"] = self.detector.as_dict()
        return payload


@dataclass(slots=True)
class ArtifactProfile:
    """Stochastic detector and sample artifact controls."""

    profile_id: str = "default"
    enabled: bool = True
    poisson_counts: float = 2500.0
    gaussian_read_noise: float = 0.015
    background_level: float = 0.02
    background_gradient: tuple[float, float] = (0.0, 0.06)
    beamstop: bool = True
    beamstop_width_fraction: float = 0.035
    beamstop_height_fraction: float = 0.42
    detector_gap_fraction: float = 0.018
    detector_layout: str = "random_common"
    detector_gap_jitter_pixels: int = 1
    dead_pixel_fraction: float = 0.0004
    dead_pixel_cluster_count: int = 0
    hot_pixel_fraction: float = 0.0002
    hot_pixel_intensity: float = 1.0
    saturation_level: float = 1.0
    flat_field_strength: float = 0.08
    parasitic_streaks: int = 0
    diffuse_ring_count: int = 1
    diffuse_ring_strength: float = 0.06
    diffuse_ring_width_range: tuple[float, float] = (0.035, 0.12)
    q_dependent_background: float = 0.015
    surface_scattering: bool = True
    direct_beam: bool = True
    direct_beam_strength: float = 0.35
    direct_beam_width_qxy: float = 0.045
    direct_beam_width_qz: float = 0.03
    specular_reflection_strength: float = 0.12
    yoneda_peak: bool = True
    yoneda_strength: float = 0.08
    yoneda_width_qz: float = 0.035
    yoneda_qxy_decay: float = 3.5
    critical_peak_splitting: bool = True
    critical_peak_split_strength: float = 0.12
    critical_peak_split_quantile: float = 0.992
    critical_angle_deg: float | None = None
    substrate_horizon: bool = True
    substrate_horizon_strength: float = 0.12
    substrate_horizon_width_qz: float = 0.025
    substrate_horizon_slope: float = 0.0
    substrate_horizon_roughness: float = 0.005
    below_horizon_transmission: float = 0.08
    substrate_length_mm: float = 10.0
    substrate_width_mm: float = 10.0
    beam_height_um: float = 35.0
    beam_width_mm: float = 0.25
    spillage_horizon_gain: float = 1.0
    spillage_broadening: bool = True
    spillage_broadening_strength: float = 0.25
    spillage_broadening_max_qz: float = 0.08

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactProfile":
        gradient = _tuple_float(
            payload.get("background_gradient", (0.0, 0.06)),
            length=2,
            field_name="background_gradient",
        )
        ring_width = _tuple_float(
            payload.get("diffuse_ring_width_range", (0.035, 0.12)),
            length=2,
            field_name="diffuse_ring_width_range",
        )
        return cls(
            profile_id=str(payload.get("profile_id", "default")),
            enabled=bool(payload.get("enabled", True)),
            poisson_counts=float(payload.get("poisson_counts", 2500.0)),
            gaussian_read_noise=float(
                payload.get("gaussian_read_noise", 0.015)
            ),
            background_level=float(payload.get("background_level", 0.02)),
            background_gradient=(float(gradient[0]), float(gradient[1])),
            beamstop=bool(payload.get("beamstop", True)),
            beamstop_width_fraction=float(
                payload.get("beamstop_width_fraction", 0.035)
            ),
            beamstop_height_fraction=float(
                payload.get("beamstop_height_fraction", 0.42)
            ),
            detector_gap_fraction=float(
                payload.get("detector_gap_fraction", 0.018)
            ),
            detector_layout=str(
                payload.get("detector_layout", "random_common")
            ),
            detector_gap_jitter_pixels=int(
                payload.get("detector_gap_jitter_pixels", 1)
            ),
            dead_pixel_fraction=float(
                payload.get("dead_pixel_fraction", 0.0004)
            ),
            dead_pixel_cluster_count=int(
                payload.get("dead_pixel_cluster_count", 0)
            ),
            hot_pixel_fraction=float(
                payload.get("hot_pixel_fraction", 0.0002)
            ),
            hot_pixel_intensity=float(payload.get("hot_pixel_intensity", 1.0)),
            saturation_level=float(payload.get("saturation_level", 1.0)),
            flat_field_strength=float(
                payload.get("flat_field_strength", 0.08)
            ),
            parasitic_streaks=int(payload.get("parasitic_streaks", 0)),
            diffuse_ring_count=int(payload.get("diffuse_ring_count", 1)),
            diffuse_ring_strength=float(
                payload.get("diffuse_ring_strength", 0.06)
            ),
            diffuse_ring_width_range=(
                float(ring_width[0]),
                float(ring_width[1]),
            ),
            q_dependent_background=float(
                payload.get("q_dependent_background", 0.015)
            ),
            surface_scattering=bool(payload.get("surface_scattering", True)),
            direct_beam=bool(payload.get("direct_beam", True)),
            direct_beam_strength=float(
                payload.get("direct_beam_strength", 0.35)
            ),
            direct_beam_width_qxy=float(
                payload.get("direct_beam_width_qxy", 0.045)
            ),
            direct_beam_width_qz=float(
                payload.get("direct_beam_width_qz", 0.03)
            ),
            specular_reflection_strength=float(
                payload.get("specular_reflection_strength", 0.12)
            ),
            yoneda_peak=bool(payload.get("yoneda_peak", True)),
            yoneda_strength=float(payload.get("yoneda_strength", 0.08)),
            yoneda_width_qz=float(payload.get("yoneda_width_qz", 0.035)),
            yoneda_qxy_decay=float(payload.get("yoneda_qxy_decay", 3.5)),
            critical_peak_splitting=bool(
                payload.get("critical_peak_splitting", True)
            ),
            critical_peak_split_strength=float(
                payload.get("critical_peak_split_strength", 0.12)
            ),
            critical_peak_split_quantile=float(
                payload.get("critical_peak_split_quantile", 0.992)
            ),
            critical_angle_deg=_optional_float(
                payload.get("critical_angle_deg")
            ),
            substrate_horizon=bool(payload.get("substrate_horizon", True)),
            substrate_horizon_strength=float(
                payload.get("substrate_horizon_strength", 0.12)
            ),
            substrate_horizon_width_qz=float(
                payload.get("substrate_horizon_width_qz", 0.025)
            ),
            substrate_horizon_slope=float(
                payload.get("substrate_horizon_slope", 0.0)
            ),
            substrate_horizon_roughness=float(
                payload.get("substrate_horizon_roughness", 0.005)
            ),
            below_horizon_transmission=float(
                payload.get("below_horizon_transmission", 0.08)
            ),
            substrate_length_mm=float(
                payload.get("substrate_length_mm", 10.0)
            ),
            substrate_width_mm=float(payload.get("substrate_width_mm", 10.0)),
            beam_height_um=float(payload.get("beam_height_um", 35.0)),
            beam_width_mm=float(payload.get("beam_width_mm", 0.25)),
            spillage_horizon_gain=float(
                payload.get("spillage_horizon_gain", 1.0)
            ),
            spillage_broadening=bool(payload.get("spillage_broadening", True)),
            spillage_broadening_strength=float(
                payload.get("spillage_broadening_strength", 0.25)
            ),
            spillage_broadening_max_qz=float(
                payload.get("spillage_broadening_max_qz", 0.08)
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetSample:
    """Manifest row for one generated training example."""

    sample_id: str
    structure_id: str
    condition_id: str
    image_path: str
    label_path: str
    clean_image_path: str = ""
    peak_table_path: str = ""
    artifact_profile_id: str = ""
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetSample":
        required = ("sample_id", "structure_id", "condition_id", "image_path")
        missing = [key for key in required if not payload.get(key)]
        if missing:
            raise ConfigError(f"Manifest sample missing: {', '.join(missing)}")
        return cls(
            sample_id=str(payload["sample_id"]),
            structure_id=str(payload["structure_id"]),
            condition_id=str(payload["condition_id"]),
            image_path=str(payload["image_path"]),
            label_path=str(payload.get("label_path", "")),
            clean_image_path=str(payload.get("clean_image_path", "")),
            peak_table_path=str(payload.get("peak_table_path", "")),
            artifact_profile_id=str(payload.get("artifact_profile_id", "")),
            seed=int(payload.get("seed", 0)),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
