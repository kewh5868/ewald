"""Artifact-aware labels and ranking masks for synthetic GIWAXS training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .detectors import (
    detector_q_axes,
    resolve_detector_preset,
    scaled_detector_gap_mask,
)
from .schemas import ArtifactProfile, DetectorGeometry

ARTIFACT_ASSESSMENT_SCHEMA = "ewald_artifact_assessment_v1"


def build_artifact_assessment(
    *,
    artifact_metadata: Mapping[str, Any] | None,
    artifact_profile: Mapping[str, Any] | ArtifactProfile | None = None,
    detector: DetectorGeometry | Mapping[str, Any] | None = None,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Return compact artifact labels for model training and peak assessment."""

    metadata = dict(artifact_metadata or {})
    profile = _profile_dict(artifact_profile)
    geometry = _detector_geometry(detector)
    operations = tuple(str(item) for item in metadata.get("operations", ()))
    surface = dict(metadata.get("surface_scattering", {}) or {})
    regions: list[dict[str, Any]] = []

    if "direct_beam_specular" in operations and surface:
        width_qxy = float(profile.get("direct_beam_width_qxy", 0.045))
        width_qz = float(profile.get("direct_beam_width_qz", 0.03))
        regions.append(
            {
                "kind": "direct_beam",
                "operation": "direct_beam_specular",
                "qxy_center": 0.0,
                "qz_center": float(surface.get("direct_beam_qz", 0.0)),
                "sigma_qxy": width_qxy,
                "sigma_qz": width_qz,
                "training_weight": 0.03,
            }
        )
        regions.append(
            {
                "kind": "specular_reflection",
                "operation": "direct_beam_specular",
                "qxy_center": 0.0,
                "qz_center": float(surface.get("specular_qz", 0.0)),
                "sigma_qxy": width_qxy * 1.4,
                "sigma_qz": width_qz * 1.8,
                "training_weight": 0.12,
            }
        )

    if "yoneda_band" in operations and surface:
        regions.append(
            {
                "kind": "yoneda_band",
                "operation": "yoneda_band",
                "qz_center": float(surface.get("yoneda_qz", 0.0)),
                "sigma_qz": float(profile.get("yoneda_width_qz", 0.035)),
                "qxy_decay": float(profile.get("yoneda_qxy_decay", 3.5)),
                "training_weight": 0.35,
            }
        )

    if "substrate_horizon_shadow" in operations and surface:
        regions.append(
            {
                "kind": "substrate_horizon",
                "operation": "substrate_horizon_shadow",
                "qz_center": float(
                    surface.get(
                        "substrate_horizon_qz",
                        surface.get("horizon_qz", 0.0),
                    )
                ),
                "width_qz": float(
                    surface.get(
                        "substrate_horizon_width_qz",
                        profile.get("substrate_horizon_width_qz", 0.025),
                    )
                ),
                "slope": float(surface.get("substrate_horizon_slope", 0.0)),
                "roughness": float(
                    surface.get("substrate_horizon_roughness", 0.0)
                ),
                "below_horizon_transmission": float(
                    surface.get("below_horizon_transmission", 0.08)
                ),
                "footprint_spillage_fraction": float(
                    surface.get("footprint_spillage_fraction", 0.0)
                ),
                "training_weight": 0.25,
            }
        )

    if "critical_angle_peak_splitting" in operations and surface:
        regions.append(
            {
                "kind": "critical_angle_peak_split",
                "operation": "critical_angle_peak_splitting",
                "critical_angle_deg": float(
                    surface.get("critical_angle_deg", 0.0)
                ),
                "critical_q_shift": float(
                    surface.get("critical_q_shift", 0.0)
                ),
                "training_weight": 0.65,
            }
        )

    if metadata.get("detector_mask_spans"):
        spans = dict(metadata.get("detector_mask_spans") or {})
        regions.append(
            {
                "kind": "detector_module_gap_mask",
                "operation": "detector_module_gap_mask",
                "pixel_spans": spans,
                "masked_fraction": float(
                    metadata.get("detector_masked_fraction", 0.0)
                ),
                "training_weight": 0.0,
            }
        )
    elif "detector_module_gap_mask" in operations:
        preset = metadata.get("detector_preset", {}) or {}
        regions.append(
            {
                "kind": "detector_module_gap_mask",
                "operation": "detector_module_gap_mask",
                "detector_preset_id": str(preset.get("preset_id", "")),
                "masked_fraction": float(
                    metadata.get("detector_masked_fraction", 0.0)
                ),
                "training_weight": 0.0,
            }
        )

    if metadata.get("beamstop_shadow"):
        regions.append(
            {
                "kind": "beamstop_shadow",
                "operation": "beamstop_shadow",
                "pixel_spans": metadata["beamstop_shadow"],
                "training_weight": 0.0,
            }
        )

    if "hot_dead_pixels" in operations:
        regions.append(
            {
                "kind": "hot_dead_pixels",
                "operation": "hot_dead_pixels",
                "dead_pixel_count": int(metadata.get("dead_pixel_count", 0)),
                "hot_pixel_count": int(metadata.get("hot_pixel_count", 0)),
                "dead_pixel_cluster_count": int(
                    metadata.get("dead_pixel_cluster_count", 0)
                ),
                "training_weight": 0.2,
            }
        )

    scalar_features = _artifact_scalar_features(metadata, surface)
    assessment = {
        "schema": ARTIFACT_ASSESSMENT_SCHEMA,
        "operations": list(operations),
        "regions": regions,
        "scalar_features": scalar_features,
        "intended_use": (
            "Downweight non-Bragg detector and surface-scattering regions "
            "during image overlap, peak assessment, and indexing."
        ),
    }
    if image_shape is not None:
        weights = artifact_weight_map_from_assessment(
            assessment,
            detector=geometry,
            image_shape=image_shape,
        )
        assessment["image_shape"] = [int(image_shape[0]), int(image_shape[1])]
        assessment["usable_fraction"] = float(np.mean(weights > 0.5))
        assessment["mean_training_weight"] = float(np.mean(weights))
    return assessment


def estimate_retrieval_quality(
    clean_image: np.ndarray,
    artifact_image: np.ndarray,
    *,
    artifact_assessment: Mapping[str, Any] | None = None,
    detector: DetectorGeometry | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate whether an augmented image still contains retrievable signal."""

    clean = _normalize_quality_image(clean_image)
    artifact = _normalize_quality_image(artifact_image)
    if clean.shape != artifact.shape:
        raise ValueError("clean and artifact image shapes must match.")
    if artifact_assessment is None:
        weights = np.ones(clean.shape, dtype=np.float32)
    else:
        weights = artifact_weight_map_from_assessment(
            artifact_assessment,
            detector=detector,
            image_shape=clean.shape,
        )

    positive_clean = clean[clean > 0.0]
    if positive_clean.size:
        signal_threshold = float(np.nanquantile(positive_clean, 0.965))
    else:
        signal_threshold = 1.0
    signal_mask = (clean >= signal_threshold) & (weights > 0.2)
    if not np.any(signal_mask):
        signal_mask = clean >= float(np.nanmax(clean))
    background_threshold = float(np.nanquantile(clean, 0.55))
    background_mask = (clean <= background_threshold) & (weights > 0.5)
    if np.count_nonzero(background_mask) < 16:
        background_mask = clean <= background_threshold

    background_values = artifact[background_mask]
    signal_values = artifact[signal_mask]
    background_median = float(np.nanmedian(background_values))
    background_mad = float(
        np.nanmedian(np.abs(background_values - background_median))
    )
    noise_sigma = max(1.4826 * background_mad, 1.0e-6)
    signal_median = float(np.nanmedian(signal_values))
    signal_to_noise = (signal_median - background_median) / noise_sigma
    detectable_signal = signal_values > (background_median + 3.0 * noise_sigma)
    retrievable_signal_fraction = float(np.mean(detectable_signal))
    usable_fraction = float(np.mean(weights > 0.5))
    saturated_fraction = float(np.mean(artifact >= 0.995))
    weighted_overlap = _quality_overlap(clean, artifact, weights)
    full_overlap = _quality_overlap(clean, artifact, np.ones_like(weights))
    solvable = bool(
        signal_to_noise >= 2.5
        and retrievable_signal_fraction >= 0.45
        and usable_fraction >= 0.65
        and saturated_fraction <= 0.18
        and weighted_overlap >= 0.25
    )
    warning_reasons: list[str] = []
    if signal_to_noise < 2.5:
        warning_reasons.append("low_signal_to_noise")
    if retrievable_signal_fraction < 0.45:
        warning_reasons.append("low_retrievable_signal_fraction")
    if usable_fraction < 0.65:
        warning_reasons.append("too_much_masked_or_downweighted_area")
    if saturated_fraction > 0.18:
        warning_reasons.append("excessive_saturation")
    if weighted_overlap < 0.25:
        warning_reasons.append("low_clean_artifact_overlap")
    return {
        "schema": "ewald_retrieval_quality_v1",
        "solvable": solvable,
        "warning_reasons": warning_reasons,
        "signal_to_noise": float(signal_to_noise),
        "retrievable_signal_fraction": retrievable_signal_fraction,
        "usable_fraction": usable_fraction,
        "saturated_fraction": saturated_fraction,
        "weighted_clean_artifact_overlap": float(weighted_overlap),
        "full_clean_artifact_overlap": float(full_overlap),
        "signal_pixel_fraction": float(np.mean(signal_mask)),
        "background_noise_sigma": float(noise_sigma),
    }


def artifact_weight_map_from_labels(
    labels: Mapping[str, Any],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Return a ranker weight map from sample labels."""

    assessment = labels.get("artifact_assessment")
    detector = _detector_from_labels(labels)
    if isinstance(assessment, Mapping):
        return artifact_weight_map_from_assessment(
            assessment,
            detector=detector,
            image_shape=image_shape,
        )
    return np.ones(image_shape, dtype=np.float32)


def _normalize_quality_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = values - float(np.nanmin(values))
    scale = float(np.nanmax(values))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(values)
    return values / scale


def _quality_overlap(
    clean: np.ndarray,
    artifact: np.ndarray,
    weights: np.ndarray,
) -> float:
    lhs = np.asarray(clean, dtype=np.float64) * weights
    rhs = np.asarray(artifact, dtype=np.float64) * weights
    lhs = lhs.ravel() - float(np.mean(lhs))
    rhs = rhs.ravel() - float(np.mean(rhs))
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.dot(lhs, rhs) / denom)


def artifact_weight_map_from_assessment(
    assessment: Mapping[str, Any],
    *,
    detector: DetectorGeometry | Mapping[str, Any] | None = None,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize artifact labels into multiplicative training weights."""

    geometry = _detector_geometry(detector)
    weights = np.ones(image_shape, dtype=np.float32)
    qxy_axis, qz_axis = detector_q_axes(geometry, image_shape)
    qxy_grid, qz_grid = np.meshgrid(qxy_axis, qz_axis)
    qxy_scale = max(float(np.nanmax(np.abs(qxy_grid))), 1.0e-6)

    for region in assessment.get("regions", ()) or ():
        if not isinstance(region, Mapping):
            continue
        kind = str(region.get("kind", ""))
        target_weight = float(region.get("training_weight", 1.0))
        target_weight = float(np.clip(target_weight, 0.0, 1.0))
        if kind in {"direct_beam", "specular_reflection"}:
            local = _gaussian_region(qxy_grid, qz_grid, region)
            weights = np.minimum(
                weights,
                1.0 - (1.0 - target_weight) * local.astype(np.float32),
            )
        elif kind == "yoneda_band":
            sigma_qz = max(float(region.get("sigma_qz", 0.035)), 1.0e-6)
            qxy_decay = max(float(region.get("qxy_decay", 3.5)), 1.0e-6)
            local = np.exp(
                -0.5
                * ((qz_grid - float(region.get("qz_center", 0.0))) / sigma_qz)
                ** 2
            )
            local *= np.exp(-np.abs(qxy_grid) / qxy_decay)
            weights = np.minimum(
                weights,
                1.0 - (1.0 - target_weight) * local.astype(np.float32),
            )
        elif kind == "substrate_horizon":
            width = max(float(region.get("width_qz", 0.025)), 1.0e-6)
            horizon = float(region.get("qz_center", 0.0))
            horizon += float(region.get("slope", 0.0)) * qxy_grid
            roughness = max(float(region.get("roughness", 0.0)), 0.0)
            if roughness > 0.0:
                horizon += roughness * np.sin(
                    2.0 * np.pi * qxy_grid / qxy_scale
                )
            transition = 1.0 / (1.0 + np.exp(-(qz_grid - horizon) / width))
            below = float(
                region.get("below_horizon_transmission", target_weight)
            )
            below = float(np.clip(min(below, target_weight), 0.0, 1.0))
            local_weights = below + (1.0 - below) * transition
            edge = np.exp(-0.5 * ((qz_grid - horizon) / width) ** 2)
            local_weights = np.minimum(
                local_weights,
                1.0 - (1.0 - target_weight) * edge,
            )
            weights = np.minimum(weights, local_weights.astype(np.float32))
        elif kind in {"detector_module_gap_mask", "beamstop_shadow"}:
            _apply_pixel_span_weights(weights, region, target_weight)
            if kind == "detector_module_gap_mask" and not region.get(
                "pixel_spans"
            ):
                _apply_detector_preset_mask(weights, region, geometry)

    return np.clip(weights, 0.0, 1.0).astype(np.float32, copy=False)


def annotate_peaks_with_artifacts(
    peaks: Sequence[Mapping[str, Any]],
    assessment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Attach artifact-overlap labels to simulated Bragg peak rows."""

    annotated: list[dict[str, Any]] = []
    regions = [
        region
        for region in assessment.get("regions", ()) or ()
        if isinstance(region, Mapping)
    ]
    for peak in peaks:
        row = dict(peak)
        if _peak_excluded_from_indexing(row):
            row["artifact_overlap"] = []
            row["bragg_training_weight"] = 0.0
            row["training_excluded"] = True
            annotated.append(row)
            continue
        try:
            qxy = float(row["qxy"])
            qz = float(row["qz"])
        except (KeyError, TypeError, ValueError):
            annotated.append(row)
            continue
        overlaps = [
            str(region.get("kind", ""))
            for region in regions
            if _q_point_overlaps_region(qxy, qz, region)
        ]
        row["artifact_overlap"] = [item for item in overlaps if item]
        row["bragg_training_weight"] = min(
            (
                float(region.get("training_weight", 1.0))
                for region in regions
                if str(region.get("kind", "")) in overlaps
            ),
            default=1.0,
        )
        annotated.append(row)
    return annotated


def _peak_excluded_from_indexing(peak: Mapping[str, Any]) -> bool:
    return bool(
        peak.get("forbidden_reflection")
        or peak.get("excluded_from_indexing")
        or str(peak.get("reflection_status", "")).lower() == "forbidden"
    )


def _artifact_scalar_features(
    metadata: Mapping[str, Any],
    surface: Mapping[str, Any],
) -> dict[str, float]:
    keys = (
        "incident_angle_deg",
        "critical_angle_deg",
        "wavelength_angstrom",
        "direct_beam_qz",
        "horizon_qz",
        "specular_qz",
        "yoneda_qz",
        "critical_q_shift",
        "footprint_spillage_fraction",
        "beam_footprint_length_mm",
        "beam_footprint_area_mm2",
        "substrate_horizon_qz",
        "substrate_horizon_width_qz",
        "below_horizon_transmission",
    )
    scalars: dict[str, float] = {}
    for key in keys:
        if key in surface:
            scalars[key] = float(surface[key])
    for key in (
        "detector_masked_fraction",
        "dead_pixel_count",
        "hot_pixel_count",
        "dead_pixel_cluster_count",
    ):
        if key in metadata:
            scalars[key] = float(metadata[key])
    return scalars


def _gaussian_region(
    qxy_grid: np.ndarray,
    qz_grid: np.ndarray,
    region: Mapping[str, Any],
) -> np.ndarray:
    sigma_qxy = max(float(region.get("sigma_qxy", 0.045)), 1.0e-6)
    sigma_qz = max(float(region.get("sigma_qz", 0.03)), 1.0e-6)
    return np.exp(
        -0.5
        * (
            ((qxy_grid - float(region.get("qxy_center", 0.0))) / sigma_qxy)
            ** 2
            + ((qz_grid - float(region.get("qz_center", 0.0))) / sigma_qz) ** 2
        )
    )


def _apply_pixel_span_weights(
    weights: np.ndarray,
    region: Mapping[str, Any],
    target_weight: float,
) -> None:
    spans = region.get("pixel_spans") or {}
    if not isinstance(spans, Mapping):
        return
    for start, stop in _span_pairs(spans.get("columns", ())):
        weights[:, max(0, start) : min(weights.shape[1], stop)] = np.minimum(
            weights[:, max(0, start) : min(weights.shape[1], stop)],
            target_weight,
        )
    for start, stop in _span_pairs(spans.get("rows", ())):
        weights[max(0, start) : min(weights.shape[0], stop), :] = np.minimum(
            weights[max(0, start) : min(weights.shape[0], stop), :],
            target_weight,
        )
    if "row_start" in spans and "col_start" in spans:
        row_start = max(0, int(spans.get("row_start", 0)))
        row_stop = min(weights.shape[0], int(spans.get("row_stop", row_start)))
        col_start = max(0, int(spans.get("col_start", 0)))
        col_stop = min(weights.shape[1], int(spans.get("col_stop", col_start)))
        weights[row_start:row_stop, col_start:col_stop] = np.minimum(
            weights[row_start:row_stop, col_start:col_stop],
            target_weight,
        )


def _apply_detector_preset_mask(
    weights: np.ndarray,
    region: Mapping[str, Any],
    detector: DetectorGeometry | None,
) -> None:
    preset_id = str(region.get("detector_preset_id", ""))
    if not preset_id and detector is not None:
        preset_id = detector.detector
    preset = resolve_detector_preset(preset_id)
    if preset is None:
        return
    mask = scaled_detector_gap_mask(preset, weights.shape)
    weights[mask] = 0.0


def _q_point_overlaps_region(
    qxy: float,
    qz: float,
    region: Mapping[str, Any],
) -> bool:
    kind = str(region.get("kind", ""))
    if kind in {"direct_beam", "specular_reflection"}:
        sigma_qxy = max(float(region.get("sigma_qxy", 0.045)), 1.0e-6)
        sigma_qz = max(float(region.get("sigma_qz", 0.03)), 1.0e-6)
        qxy_term = abs(qxy - float(region.get("qxy_center", 0.0))) / sigma_qxy
        qz_term = abs(qz - float(region.get("qz_center", 0.0))) / sigma_qz
        return qxy_term <= 3.0 and qz_term <= 3.0
    if kind == "yoneda_band":
        sigma_qz = max(float(region.get("sigma_qz", 0.035)), 1.0e-6)
        return abs(qz - float(region.get("qz_center", 0.0))) <= 3.0 * sigma_qz
    if kind == "substrate_horizon":
        width = max(float(region.get("width_qz", 0.025)), 1.0e-6)
        horizon = float(region.get("qz_center", 0.0))
        horizon += float(region.get("slope", 0.0)) * qxy
        return qz <= horizon + 3.0 * width
    return False


def _profile_dict(
    profile: Mapping[str, Any] | ArtifactProfile | None,
) -> dict[str, Any]:
    if profile is None:
        return {}
    if isinstance(profile, ArtifactProfile):
        return profile.as_dict()
    return dict(profile)


def _detector_geometry(
    detector: DetectorGeometry | Mapping[str, Any] | None,
) -> DetectorGeometry | None:
    if detector is None or isinstance(detector, DetectorGeometry):
        return detector
    return DetectorGeometry.from_mapping(detector)


def _detector_from_labels(labels: Mapping[str, Any]) -> DetectorGeometry | None:
    condition = labels.get("condition") or {}
    if not isinstance(condition, Mapping):
        return None
    detector = condition.get("detector")
    if not isinstance(detector, Mapping):
        return None
    return DetectorGeometry.from_mapping(detector)


def _span_pairs(value: Any) -> list[tuple[int, int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    pairs: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, Sequence) or len(item) != 2:
            continue
        pairs.append((int(item[0]), int(item[1])))
    return pairs
