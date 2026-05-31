"""Detector and sample artifact augmentation for synthetic GIWAXS images."""

from __future__ import annotations

from typing import Any

import numpy as np

from .detectors import (
    choose_detector_preset,
    detector_q_axes,
    q_radius_grid,
    scaled_detector_gap_mask,
)
from .schemas import ArtifactProfile, DetectorGeometry


def apply_artifacts(
    image: np.ndarray,
    profile: ArtifactProfile,
    *,
    seed: int = 0,
    detector: DetectorGeometry | None = None,
    sample_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply reproducible detector-like artifacts to a simulated image."""

    base = _as_float_image(image)
    if not profile.enabled:
        return base.copy(), {
            "profile_id": profile.profile_id,
            "enabled": False,
        }

    rng = np.random.default_rng(seed)
    augmented = _normalize(base)
    metadata: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "enabled": True,
        "seed": int(seed),
        "operations": [],
    }

    augmented = _add_background(augmented, profile, rng, metadata, detector)
    augmented = _add_diffuse_rings(augmented, profile, rng, metadata, detector)
    augmented = _add_surface_scattering(
        augmented,
        profile,
        metadata,
        detector,
        sample_context,
    )
    augmented = _add_streaks(augmented, profile, rng, metadata)
    augmented = _apply_flat_field(augmented, profile, rng, metadata)
    augmented = _apply_poisson_noise(augmented, profile, rng, metadata)
    augmented = _apply_read_noise(augmented, profile, rng, metadata)
    augmented = _apply_hot_dead_pixels(augmented, profile, rng, metadata)
    augmented = _apply_detector_gaps(augmented, profile, rng, metadata, detector)
    augmented = _apply_beamstop(augmented, profile, metadata)
    augmented = np.clip(augmented, 0.0, profile.saturation_level)
    metadata["operations"].append("saturation_clip")
    return augmented.astype(np.float32, copy=False), metadata


def _as_float_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Artifact augmentation expects a 2D image.")
    return values


def _normalize(image: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(np.min(finite))
    shifted = finite - min_value
    max_value = float(np.max(shifted))
    if max_value <= 0.0:
        return shifted
    return shifted / max_value


def _add_background(
    image: np.ndarray,
    profile: ArtifactProfile,
    rng: np.random.Generator,
    metadata: dict[str, Any],
    detector: DetectorGeometry | None,
) -> np.ndarray:
    rows, cols = image.shape
    y = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
    slope_x, slope_y = profile.background_gradient
    background = profile.background_level + slope_x * x + slope_y * y
    background += rng.uniform(0.0, profile.background_level * 0.25)
    q_background = max(0.0, profile.q_dependent_background)
    if q_background > 0.0:
        radius = q_radius_grid(detector, image.shape)
        finite_radius = radius[np.isfinite(radius)]
        scale = float(np.nanmax(finite_radius)) if finite_radius.size else 1.0
        if scale > 0.0:
            background += q_background * np.exp(-2.2 * radius / scale)
    metadata["operations"].append("background_gradient")
    return image + background.astype(np.float32)


def _add_diffuse_rings(
    image: np.ndarray,
    profile: ArtifactProfile,
    rng: np.random.Generator,
    metadata: dict[str, Any],
    detector: DetectorGeometry | None,
) -> np.ndarray:
    if profile.diffuse_ring_count <= 0:
        return image
    radius = q_radius_grid(detector, image.shape)
    centers = _structure_correlated_ring_centers(
        image,
        radius,
        max_count=profile.diffuse_ring_count,
        rng=rng,
    )
    if not centers:
        centers = _fallback_ring_centers(radius, profile.diffuse_ring_count, rng)
    width_min, width_max = sorted(profile.diffuse_ring_width_range)
    out = image.copy()
    ring_metadata: list[dict[str, float]] = []
    for center in centers:
        width = rng.uniform(max(width_min, 1.0e-4), max(width_max, width_min))
        amplitude = rng.uniform(0.45, 1.0) * profile.diffuse_ring_strength
        out += amplitude * np.exp(-0.5 * ((radius - center) / width) ** 2)
        ring_metadata.append(
            {
                "q_center": float(center),
                "q_width": float(width),
                "amplitude": float(amplitude),
            }
        )
    metadata["operations"].append("structure_correlated_diffuse_rings")
    metadata["diffuse_rings"] = ring_metadata
    return out


def _add_surface_scattering(
    image: np.ndarray,
    profile: ArtifactProfile,
    metadata: dict[str, Any],
    detector: DetectorGeometry | None,
    sample_context: dict[str, Any] | None,
) -> np.ndarray:
    if not profile.surface_scattering:
        return image
    geometry = _surface_geometry(profile, detector, sample_context)
    qxy_axis, qz_axis = detector_q_axes(detector, image.shape)
    qxy_grid, qz_grid = np.meshgrid(qxy_axis, qz_axis)
    out = image.copy()
    operations: list[str] = []
    if profile.spillage_broadening:
        broadened = _apply_spillage_broadening(
            out,
            profile,
            geometry,
            qz_axis,
        )
        if broadened is not out:
            out = broadened
            operations.append("footprint_spillage_peak_broadening")

    if profile.direct_beam:
        qxy_width = max(float(profile.direct_beam_width_qxy), 1.0e-6)
        qz_width = max(float(profile.direct_beam_width_qz), 1.0e-6)
        direct = _gaussian_2d(
            qxy_grid,
            qz_grid,
            center_qxy=0.0,
            center_qz=geometry["direct_beam_qz"],
            sigma_qxy=qxy_width,
            sigma_qz=qz_width,
        )
        specular = _gaussian_2d(
            qxy_grid,
            qz_grid,
            center_qxy=0.0,
            center_qz=geometry["specular_qz"],
            sigma_qxy=qxy_width * 1.4,
            sigma_qz=qz_width * 1.8,
        )
        out += float(profile.direct_beam_strength) * direct
        out += float(profile.specular_reflection_strength) * specular
        operations.append("direct_beam_specular")

    if profile.yoneda_peak:
        qz_width = max(float(profile.yoneda_width_qz), 1.0e-6)
        qxy_scale = max(float(profile.yoneda_qxy_decay), 1.0e-6)
        yoneda = np.exp(
            -0.5 * ((qz_grid - geometry["yoneda_qz"]) / qz_width) ** 2
        )
        yoneda *= np.exp(-np.abs(qxy_grid) / qxy_scale)
        out += float(profile.yoneda_strength) * yoneda
        operations.append("yoneda_band")

    if profile.critical_peak_splitting:
        split = _critical_angle_peak_split(image, profile, geometry, qz_axis)
        if split is not None:
            out += split
            operations.append("critical_angle_peak_splitting")

    if profile.substrate_horizon:
        out, horizon_metadata = _apply_substrate_horizon(
            out,
            profile,
            geometry,
            qxy_grid,
            qz_grid,
        )
        operations.append("substrate_horizon_shadow")
        geometry.update(horizon_metadata)

    if operations:
        metadata["operations"].extend(operations)
        metadata["surface_scattering"] = geometry
    return out


def _surface_geometry(
    profile: ArtifactProfile,
    detector: DetectorGeometry | None,
    sample_context: dict[str, Any] | None,
) -> dict[str, float | str]:
    wavelength = 1.0
    wavelength_source = "default"
    if detector is not None and detector.wavelength_angstrom:
        wavelength = float(detector.wavelength_angstrom)
        wavelength_source = "detector_geometry"
    incident_angle_deg = 0.2
    incident_source = "default"
    if detector is not None and detector.incident_angle_deg is not None:
        incident_angle_deg = float(detector.incident_angle_deg)
        incident_source = "detector_geometry"
    critical_angle_deg = profile.critical_angle_deg
    critical_source = "artifact_profile"
    if critical_angle_deg is None and sample_context:
        raw_critical = sample_context.get("critical_angle_deg")
        if raw_critical is not None:
            critical_angle_deg = float(raw_critical)
            critical_source = "sample_electron_density"
    if critical_angle_deg is None:
        critical_angle_deg = 0.12
        critical_source = "default"

    k0 = 2.0 * np.pi / max(wavelength, 1.0e-9)
    incident_rad = np.radians(incident_angle_deg)
    critical_rad = np.radians(float(critical_angle_deg))
    direct_qz = k0 * (
        np.sin(incident_rad) + np.sin(-incident_rad)
    )
    specular_qz = k0 * (np.sin(incident_rad) + np.sin(incident_rad))
    yoneda_qz = k0 * (np.sin(incident_rad) + np.sin(critical_rad))
    critical_q_shift = k0 * abs(
        np.sin(critical_rad) - np.sin(incident_rad)
    )
    horizon_qz = k0 * np.sin(incident_rad)
    return {
        "incident_angle_deg": float(incident_angle_deg),
        "incident_angle_source": incident_source,
        "critical_angle_deg": float(critical_angle_deg),
        "critical_angle_source": critical_source,
        "wavelength_angstrom": float(wavelength),
        "wavelength_source": wavelength_source,
        "direct_beam_qz": float(direct_qz),
        "horizon_qz": float(horizon_qz),
        "specular_qz": float(specular_qz),
        "yoneda_qz": float(yoneda_qz),
        "critical_q_shift": float(critical_q_shift),
        **_beam_footprint_geometry(profile, incident_angle_deg),
    }


def _gaussian_2d(
    qxy_grid: np.ndarray,
    qz_grid: np.ndarray,
    *,
    center_qxy: float,
    center_qz: float,
    sigma_qxy: float,
    sigma_qz: float,
) -> np.ndarray:
    return np.exp(
        -0.5
        * (
            ((qxy_grid - center_qxy) / sigma_qxy) ** 2
            + ((qz_grid - center_qz) / sigma_qz) ** 2
        )
    )


def _critical_angle_peak_split(
    image: np.ndarray,
    profile: ArtifactProfile,
    geometry: dict[str, float | str],
    qz_axis: np.ndarray,
) -> np.ndarray | None:
    finite = image[np.isfinite(image)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return None
    threshold = float(
        np.nanquantile(
            finite,
            min(max(float(profile.critical_peak_split_quantile), 0.0), 1.0),
        )
    )
    peak_component = np.clip(image - threshold, 0.0, None)
    if not np.any(peak_component > 0.0):
        return None
    if qz_axis.size < 2:
        return None
    qz_step = float(np.nanmedian(np.diff(qz_axis)))
    if not np.isfinite(qz_step) or abs(qz_step) <= 1.0e-9:
        return None
    shift_q = max(float(geometry["critical_q_shift"]), abs(qz_step))
    shift_pixels = max(1, int(round(shift_q / abs(qz_step))))
    if shift_pixels >= image.shape[0]:
        return None
    split = np.zeros_like(image)
    split[shift_pixels:, :] += peak_component[:-shift_pixels, :]
    split[:-shift_pixels, :] += 0.6 * peak_component[shift_pixels:, :]
    return float(profile.critical_peak_split_strength) * split


def _apply_substrate_horizon(
    image: np.ndarray,
    profile: ArtifactProfile,
    geometry: dict[str, float | str],
    qxy_grid: np.ndarray,
    qz_grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    horizon_qz = float(geometry["horizon_qz"])
    spillage = min(
        max(float(geometry.get("footprint_spillage_fraction", 0.0)), 0.0),
        1.0,
    )
    gain = max(float(profile.spillage_horizon_gain), 0.0)
    width = max(
        float(profile.substrate_horizon_width_qz) * (1.0 + gain * spillage),
        1.0e-6,
    )
    slope = float(profile.substrate_horizon_slope)
    roughness = max(
        float(profile.substrate_horizon_roughness) * (1.0 + 2.0 * spillage),
        0.0,
    )
    qxy_scale = max(float(np.nanmax(np.abs(qxy_grid))), 1.0e-6)
    horizon = horizon_qz + slope * qxy_grid
    if roughness > 0.0:
        horizon += roughness * np.sin(2.0 * np.pi * qxy_grid / qxy_scale)
    transition = 1.0 / (1.0 + np.exp(-(qz_grid - horizon) / width))
    base_below = min(max(float(profile.below_horizon_transmission), 0.0), 1.0)
    below = min(1.0, base_below + (1.0 - base_below) * 0.45 * spillage)
    attenuation = below + (1.0 - below) * transition
    edge = np.exp(-0.5 * ((qz_grid - horizon) / width) ** 2)
    out = image * attenuation
    out += float(profile.substrate_horizon_strength) * (1.0 + gain * spillage) * edge
    return out, {
        "substrate_horizon_qz": float(horizon_qz),
        "substrate_horizon_width_qz": float(width),
        "substrate_horizon_slope": float(slope),
        "substrate_horizon_roughness": float(roughness),
        "below_horizon_transmission": float(below),
    }


def _beam_footprint_geometry(
    profile: ArtifactProfile,
    incident_angle_deg: float,
) -> dict[str, float]:
    incident_rad = np.radians(max(float(incident_angle_deg), 1.0e-6))
    beam_height_mm = max(float(profile.beam_height_um), 0.0) / 1000.0
    beam_width_mm = max(float(profile.beam_width_mm), 1.0e-9)
    footprint_length_mm = beam_height_mm / max(
        float(np.sin(incident_rad)),
        1.0e-9,
    )
    substrate_length_mm = max(float(profile.substrate_length_mm), 1.0e-9)
    substrate_width_mm = max(float(profile.substrate_width_mm), 1.0e-9)
    covered_length = min(footprint_length_mm, substrate_length_mm)
    covered_width = min(beam_width_mm, substrate_width_mm)
    footprint_area = footprint_length_mm * beam_width_mm
    covered_area = covered_length * covered_width
    spillage = 1.0 - covered_area / max(footprint_area, 1.0e-12)
    return {
        "substrate_length_mm": float(substrate_length_mm),
        "substrate_width_mm": float(substrate_width_mm),
        "beam_height_um": float(profile.beam_height_um),
        "beam_width_mm": float(beam_width_mm),
        "beam_footprint_length_mm": float(footprint_length_mm),
        "beam_footprint_area_mm2": float(footprint_area),
        "footprint_spillage_fraction": float(np.clip(spillage, 0.0, 1.0)),
    }


def _apply_spillage_broadening(
    image: np.ndarray,
    profile: ArtifactProfile,
    geometry: dict[str, float | str],
    qz_axis: np.ndarray,
) -> np.ndarray:
    spillage = min(
        max(float(geometry.get("footprint_spillage_fraction", 0.0)), 0.0),
        1.0,
    )
    if spillage <= 1.0e-6 or qz_axis.size < 2:
        return image
    qz_step = float(np.nanmedian(np.diff(qz_axis)))
    if not np.isfinite(qz_step) or abs(qz_step) <= 1.0e-12:
        return image
    sigma_qz = float(profile.spillage_broadening_max_qz) * spillage
    sigma_pixels = max(sigma_qz / abs(qz_step), 0.0)
    if sigma_pixels <= 0.25:
        return image
    blurred = _vertical_gaussian_blur(image, sigma_pixels)
    strength = min(
        max(float(profile.spillage_broadening_strength) * spillage, 0.0),
        1.0,
    )
    return (1.0 - strength) * image + strength * blurred


def _vertical_gaussian_blur(image: np.ndarray, sigma_pixels: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma_pixels)))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma_pixels) ** 2)
    weights = weights / np.sum(weights)
    padded = np.pad(image, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(image, dtype=float)
    for weight, offset in zip(weights, offsets):
        start = radius + int(offset)
        out += float(weight) * padded[start : start + image.shape[0], :]
    return out


def _add_streaks(
    image: np.ndarray,
    profile: ArtifactProfile,
    rng: np.random.Generator,
    metadata: dict[str, Any],
) -> np.ndarray:
    if profile.parasitic_streaks <= 0:
        return image
    rows, cols = image.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    out = image.copy()
    for _index in range(profile.parasitic_streaks):
        x0 = rng.uniform(cols * 0.05, cols * 0.95)
        y0 = rng.uniform(rows * 0.05, rows * 0.95)
        length = rng.uniform(cols * 0.02, cols * 0.12)
        angle = rng.uniform(0.0, np.pi)
        width = rng.uniform(0.6, 1.8)
        amplitude = rng.uniform(0.12, 0.45)
        dx = np.cos(angle)
        dy = np.sin(angle)
        along = (xx - x0) * dx + (yy - y0) * dy
        across = -(xx - x0) * dy + (yy - y0) * dx
        local = np.exp(-0.5 * (across / width) ** 2)
        local *= np.exp(-0.5 * (along / length) ** 8)
        out += amplitude * local
    metadata["operations"].append("localized_cosmic_ray_trails")
    return out


def _apply_flat_field(
    image: np.ndarray,
    profile: ArtifactProfile,
    rng: np.random.Generator,
    metadata: dict[str, Any],
) -> np.ndarray:
    strength = max(0.0, profile.flat_field_strength)
    if strength <= 0.0:
        return image
    rows, cols = image.shape
    y = np.linspace(-1.0, 1.0, rows, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, cols, dtype=np.float32)[None, :]
    vignette = 1.0 - strength * (x * x + y * y) / 2.0
    stripe = rng.normal(1.0, strength * 0.12, size=(1, cols))
    metadata["operations"].append("flat_field")
    return image * np.clip(vignette * stripe, 0.2, 1.4)


def _apply_poisson_noise(
    image: np.ndarray,
    profile: ArtifactProfile,
    rng: np.random.Generator,
    metadata: dict[str, Any],
) -> np.ndarray:
    counts = max(0.0, profile.poisson_counts)
    if counts <= 0.0:
        return image
    scaled = np.clip(image, 0.0, None) * counts
    noisy = rng.poisson(scaled).astype(np.float32) / counts
    metadata["operations"].append("poisson_noise")
    return noisy


def _apply_read_noise(
    image: np.ndarray,
    profile: ArtifactProfile,
    rng: np.random.Generator,
    metadata: dict[str, Any],
) -> np.ndarray:
    sigma = max(0.0, profile.gaussian_read_noise)
    if sigma <= 0.0:
        return image
    metadata["operations"].append("gaussian_read_noise")
    return image + rng.normal(0.0, sigma, size=image.shape)


def _apply_hot_dead_pixels(
    image: np.ndarray,
    profile: ArtifactProfile,
    rng: np.random.Generator,
    metadata: dict[str, Any],
) -> np.ndarray:
    out = image.copy()
    pixel_count = out.size
    dead_count = int(pixel_count * max(0.0, profile.dead_pixel_fraction))
    hot_count = int(pixel_count * max(0.0, profile.hot_pixel_fraction))
    if dead_count:
        indices = rng.choice(pixel_count, dead_count, replace=False)
        out.ravel()[indices] = 0.0
    if hot_count:
        indices = rng.choice(pixel_count, hot_count, replace=False)
        out.ravel()[indices] = profile.hot_pixel_intensity
    cluster_count = max(0, int(profile.dead_pixel_cluster_count))
    for _index in range(cluster_count):
        height = int(rng.integers(2, max(3, min(12, out.shape[0] // 12))))
        width = int(rng.integers(2, max(3, min(12, out.shape[1] // 12))))
        row = int(rng.integers(0, max(1, out.shape[0] - height)))
        col = int(rng.integers(0, max(1, out.shape[1] - width)))
        out[row : row + height, col : col + width] = 0.0
    if dead_count or hot_count or cluster_count:
        metadata["operations"].append("hot_dead_pixels")
    metadata["dead_pixel_count"] = dead_count
    metadata["hot_pixel_count"] = hot_count
    metadata["dead_pixel_cluster_count"] = cluster_count
    return out


def _apply_detector_gaps(
    image: np.ndarray,
    profile: ArtifactProfile,
    rng: np.random.Generator,
    metadata: dict[str, Any],
    detector: DetectorGeometry | None,
) -> np.ndarray:
    preset = choose_detector_preset(
        profile.detector_layout,
        rng=rng,
        fallback=detector.detector if detector is not None else None,
    )
    if preset is not None:
        mask = scaled_detector_gap_mask(
            preset,
            image.shape,
            jitter_pixels=max(0, int(profile.detector_gap_jitter_pixels)),
            rng=rng,
        )
        if np.any(mask):
            out = image.copy()
            out[mask] = 0.0
            metadata["operations"].append("detector_module_gap_mask")
            metadata["detector_preset"] = preset.as_dict()
            metadata["detector_masked_fraction"] = float(np.mean(mask))
            metadata["detector_mask_spans"] = _mask_axis_spans(mask)
            return out
        metadata["operations"].append("detector_footprint_no_gap")
        metadata["detector_preset"] = preset.as_dict()
        metadata["detector_masked_fraction"] = 0.0
        return image

    width = int(
        round(image.shape[1] * max(0.0, profile.detector_gap_fraction))
    )
    if width <= 0:
        return image
    out = image.copy()
    center = image.shape[1] // 2
    col_start = max(0, center - width)
    col_stop = min(image.shape[1], center + width)
    out[:, col_start:col_stop] = 0.0
    metadata["operations"].append("detector_gap")
    metadata["detector_mask_spans"] = {
        "columns": [[int(col_start), int(col_stop)]],
        "rows": [],
    }
    return out


def _structure_correlated_ring_centers(
    image: np.ndarray,
    radius: np.ndarray,
    *,
    max_count: int,
    rng: np.random.Generator,
) -> list[float]:
    values = np.asarray(image, dtype=float)
    finite = np.isfinite(values) & np.isfinite(radius)
    positive = values[finite]
    if positive.size == 0:
        return []
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return []
    threshold = float(np.nanquantile(positive, 0.985))
    bright = finite & (values >= threshold)
    if not np.any(bright):
        return []
    bright_radius = radius[bright].ravel()
    weights = np.clip(values[bright].ravel(), 0.0, None)
    q_min = float(np.nanmin(radius[finite]))
    q_max = float(np.nanmax(radius[finite]))
    if not np.isfinite(q_max) or q_max <= q_min:
        return []
    bins = np.linspace(q_min, q_max, 96)
    histogram, edges = np.histogram(bright_radius, bins=bins, weights=weights)
    if not np.any(histogram > 0.0):
        return []
    centers = 0.5 * (edges[:-1] + edges[1:])
    order = np.argsort(histogram)[::-1]
    selected: list[float] = []
    min_separation = max((q_max - q_min) / 30.0, 0.04)
    for index in order:
        center = float(centers[index])
        if center <= q_min:
            continue
        if any(abs(center - existing) < min_separation for existing in selected):
            continue
        jitter = rng.normal(0.0, min_separation * 0.12)
        selected.append(float(np.clip(center + jitter, q_min, q_max)))
        if len(selected) >= max_count:
            break
    return selected


def _fallback_ring_centers(
    radius: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[float]:
    finite = radius[np.isfinite(radius)]
    if finite.size == 0:
        return []
    q_max = float(np.nanmax(finite))
    if q_max <= 0.0:
        return []
    return [
        float(rng.uniform(0.25 * q_max, 0.82 * q_max))
        for _index in range(max(0, int(count)))
    ]


def _apply_beamstop(
    image: np.ndarray,
    profile: ArtifactProfile,
    metadata: dict[str, Any],
) -> np.ndarray:
    if not profile.beamstop:
        return image
    rows, cols = image.shape
    half_width = max(1, int(cols * profile.beamstop_width_fraction / 2.0))
    height = max(1, int(rows * profile.beamstop_height_fraction))
    center = cols // 2
    out = image.copy()
    col_start = max(0, center - half_width)
    col_stop = min(cols, center + half_width + 1)
    out[:height, col_start:col_stop] = 0.0
    metadata["operations"].append("beamstop_shadow")
    metadata["beamstop_shadow"] = {
        "row_start": 0,
        "row_stop": int(height),
        "col_start": int(col_start),
        "col_stop": int(col_stop),
    }
    return out


def _mask_axis_spans(mask: np.ndarray) -> dict[str, list[list[int]]]:
    values = np.asarray(mask, dtype=bool)
    return {
        "columns": _contiguous_spans(np.mean(values, axis=0) > 0.85),
        "rows": _contiguous_spans(np.mean(values, axis=1) > 0.85),
    }


def _contiguous_spans(values: np.ndarray) -> list[list[int]]:
    spans: list[list[int]] = []
    start: int | None = None
    for index, enabled in enumerate(np.asarray(values, dtype=bool)):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            spans.append([int(start), int(index)])
            start = None
    if start is not None:
        spans.append([int(start), int(len(values))])
    return spans
