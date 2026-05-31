"""Detector presets and geometry utilities for training-data
augmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .schemas import DetectorGeometry


@dataclass(frozen=True, slots=True)
class DetectorPreset:
    """Physical detector footprint used to mask synthetic detector
    frames."""

    preset_id: str
    aliases: tuple[str, ...]
    manufacturer: str
    family: str
    pixel_size_um: tuple[float, float]
    pixel_array: tuple[int, int]
    module_grid: tuple[int, int] = (1, 1)
    module_size: tuple[int, int] | None = None
    module_gap: tuple[int, int] = (0, 0)
    suggested_resolution: tuple[int, int] = (384, 256)
    suggested_qxy_range: tuple[float, float] = (-4.0, 4.0)
    suggested_qz_range: tuple[float, float] = (0.0, 4.0)
    source: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


DETECTOR_PRESETS: tuple[DetectorPreset, ...] = (
    DetectorPreset(
        preset_id="pilatus1m_pyfai",
        aliases=("pilatus1m", "pilatus_1m", "pilatus 1m", "pilatus"),
        manufacturer="DECTRIS",
        family="PILATUS",
        pixel_size_um=(172.0, 172.0),
        pixel_array=(981, 1043),
        module_grid=(2, 5),
        module_size=(487, 195),
        module_gap=(7, 17),
        suggested_resolution=(288, 320),
        suggested_qxy_range=(-4.0, 4.0),
        suggested_qz_range=(0.0, 4.0),
        source="pyFAI detector definitions: Pilatus1M",
        notes="pyFAI layout with 487 x 195 pixel modules and 7/17 pixel inter-module gaps.",
    ),
    DetectorPreset(
        preset_id="pilatus1m_legacy_psi",
        aliases=("pilatus1m_legacy", "psi_pilatus1m"),
        manufacturer="PSI/DECTRIS",
        family="PILATUS",
        pixel_size_um=(217.0, 217.0),
        pixel_array=(1120, 967),
        module_grid=(3, 6),
        module_size=(366, 157),
        module_gap=(11, 5),
        suggested_resolution=(336, 288),
        suggested_qxy_range=(-3.6, 3.6),
        suggested_qz_range=(0.0, 3.4),
        source="Broennimann et al., J. Synchrotron Rad. 2006, PILATUS 1M",
        notes="Original 18-module PILATUS 1M layout; module gaps are scaled into training masks.",
    ),
    DetectorPreset(
        preset_id="eiger2_x_1m",
        aliases=("eiger1m", "eiger2_1m", "eiger2 x 1m"),
        manufacturer="DECTRIS",
        family="EIGER2",
        pixel_size_um=(75.0, 75.0),
        pixel_array=(1028, 1062),
        module_grid=(1, 2),
        module_size=(1028, 512),
        module_gap=(0, 38),
        suggested_resolution=(320, 320),
        suggested_qxy_range=(-4.0, 4.0),
        suggested_qz_range=(0.0, 4.0),
        source="DECTRIS EIGER2 X/XE specifications",
        notes="Compact high-resolution hybrid-pixel footprint with a horizontal module gap.",
    ),
    DetectorPreset(
        preset_id="eiger2_x_1m_w",
        aliases=("eiger1m_w", "eiger2_1m_w", "eiger2 x 1m-w"),
        manufacturer="DECTRIS",
        family="EIGER2",
        pixel_size_um=(75.0, 75.0),
        pixel_array=(2068, 512),
        module_grid=(2, 1),
        module_size=(1028, 512),
        module_gap=(12, 0),
        suggested_resolution=(512, 160),
        suggested_qxy_range=(-4.5, 4.5),
        suggested_qz_range=(0.0, 2.6),
        source="DECTRIS EIGER2 X/XE specifications",
        notes="Wide WAXS-oriented EIGER2 footprint with a vertical inter-module gap.",
    ),
    DetectorPreset(
        preset_id="eiger2_x_4m",
        aliases=("eiger4m", "eiger2_4m", "eiger2 x 4m"),
        manufacturer="DECTRIS",
        family="EIGER2",
        pixel_size_um=(75.0, 75.0),
        pixel_array=(2068, 2162),
        module_grid=(2, 4),
        module_size=(1028, 512),
        module_gap=(12, 38),
        suggested_resolution=(384, 400),
        suggested_qxy_range=(-4.0, 4.0),
        suggested_qz_range=(0.0, 4.0),
        source="DECTRIS EIGER2 X 4M technical specifications",
        notes="Large EIGER2 detector; downsampled for local training sweeps.",
    ),
    DetectorPreset(
        preset_id="perkin_elmer_xrd_1621",
        aliases=("perkin_elmer", "perkinelmer", "xrd1621", "xrd 1621"),
        manufacturer="PerkinElmer",
        family="XRD",
        pixel_size_um=(200.0, 200.0),
        pixel_array=(2048, 2048),
        module_grid=(1, 1),
        module_size=(2048, 2048),
        module_gap=(0, 0),
        suggested_resolution=(384, 384),
        suggested_qxy_range=(-4.0, 4.0),
        suggested_qz_range=(0.0, 4.0),
        source="Common flat-panel GIWAXS detector footprint",
        notes="Continuous flat-panel detector; augmentation adds bad pixels but no module gaps.",
    ),
)

_PRESET_BY_ID: dict[str, DetectorPreset] = {
    key.lower(): preset
    for preset in DETECTOR_PRESETS
    for key in (preset.preset_id, *preset.aliases)
}

RANDOM_COMMON_DETECTOR_IDS = (
    "pilatus1m_pyfai",
    "eiger2_x_1m",
    "eiger2_x_1m_w",
    "eiger2_x_4m",
    "perkin_elmer_xrd_1621",
)


def detector_catalog() -> list[dict[str, object]]:
    """Return detector presets as serializable records."""

    return [preset.as_dict() for preset in DETECTOR_PRESETS]


def resolve_detector_preset(identifier: str | None) -> DetectorPreset | None:
    """Resolve a preset id, alias, or detector label."""

    if not identifier:
        return None
    return _PRESET_BY_ID.get(str(identifier).strip().lower())


def choose_detector_preset(
    requested: str | None,
    *,
    rng: np.random.Generator,
    fallback: str | None = None,
) -> DetectorPreset | None:
    """Resolve or randomly choose a detector preset for augmentation."""

    if requested and str(requested).lower() in {
        "random",
        "random_common",
        "common",
    }:
        return resolve_detector_preset(
            str(rng.choice(RANDOM_COMMON_DETECTOR_IDS))
        )
    preset = resolve_detector_preset(requested)
    if preset is not None:
        return preset
    return resolve_detector_preset(fallback)


def detector_q_axes(
    detector: DetectorGeometry | None,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return qxy and qz axes for an image shape."""

    rows, cols = image_shape
    if detector is None:
        qxy = np.linspace(-1.0, 1.0, cols, dtype=np.float32)
        qz = np.linspace(0.0, 1.0, rows, dtype=np.float32)
        return qxy, qz
    qxy = np.linspace(
        detector.qxy_range[0], detector.qxy_range[1], cols, dtype=np.float32
    )
    qz = np.linspace(
        detector.qz_range[0], detector.qz_range[1], rows, dtype=np.float32
    )
    return qxy, qz


def q_radius_grid(
    detector: DetectorGeometry | None,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Return a q-magnitude grid matching an image."""

    qxy, qz = detector_q_axes(detector, image_shape)
    qxy_grid, qz_grid = np.meshgrid(qxy, qz)
    return np.sqrt(qxy_grid * qxy_grid + qz_grid * qz_grid)


def scaled_detector_gap_mask(
    preset: DetectorPreset,
    image_shape: tuple[int, int],
    *,
    jitter_pixels: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return True for pixels that fall into scaled module gaps."""

    rows, cols = image_shape
    mask = np.zeros((rows, cols), dtype=bool)
    module_cols, module_rows = preset.module_grid
    gap_x, gap_z = preset.module_gap
    module_size = preset.module_size or _infer_module_size(preset)
    module_x, module_z = module_size
    native_x, native_z = preset.pixel_array

    for start, width in _gap_spans(module_cols, module_x, gap_x):
        _mask_scaled_span(
            mask,
            axis="x",
            start=start,
            width=width,
            native_extent=native_x,
            jitter_pixels=jitter_pixels,
            rng=rng,
        )
    for start, width in _gap_spans(module_rows, module_z, gap_z):
        _mask_scaled_span(
            mask,
            axis="z",
            start=start,
            width=width,
            native_extent=native_z,
            jitter_pixels=jitter_pixels,
            rng=rng,
        )
    return mask


def flat_detector_solid_angle(
    detector: DetectorGeometry,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Approximate flat-detector solid-angle response on a q-space
    grid."""

    wavelength = detector.wavelength_angstrom
    if wavelength is None or wavelength <= 0.0:
        return np.ones(image_shape, dtype=np.float32)
    radius = q_radius_grid(detector, image_shape)
    sin_theta = np.clip(radius * float(wavelength) / (4.0 * np.pi), 0.0, 1.0)
    two_theta = 2.0 * np.arcsin(sin_theta)
    correction = np.cos(two_theta) ** 3
    correction[~np.isfinite(correction)] = 0.0
    return np.clip(correction, 0.0, 1.0).astype(np.float32)


def _infer_module_size(preset: DetectorPreset) -> tuple[int, int]:
    module_cols, module_rows = preset.module_grid
    gap_x, gap_z = preset.module_gap
    native_x, native_z = preset.pixel_array
    module_x = int(
        round((native_x - max(0, module_cols - 1) * gap_x) / module_cols)
    )
    module_z = int(
        round((native_z - max(0, module_rows - 1) * gap_z) / module_rows)
    )
    return max(1, module_x), max(1, module_z)


def _gap_spans(
    module_count: int,
    module_size: int,
    gap_size: int,
) -> Iterable[tuple[int, int]]:
    if module_count <= 1 or gap_size <= 0:
        return ()
    return (
        (
            module_index * module_size + (module_index - 1) * gap_size,
            gap_size,
        )
        for module_index in range(1, module_count)
    )


def _mask_scaled_span(
    mask: np.ndarray,
    *,
    axis: str,
    start: int,
    width: int,
    native_extent: int,
    jitter_pixels: int,
    rng: np.random.Generator | None,
) -> None:
    if width <= 0 or native_extent <= 0:
        return
    extent = mask.shape[1] if axis == "x" else mask.shape[0]
    scaled_start = int(round(start / native_extent * extent))
    scaled_width = max(1, int(round(width / native_extent * extent)))
    if rng is not None and jitter_pixels > 0:
        scaled_start += int(rng.integers(-jitter_pixels, jitter_pixels + 1))
    scaled_start = max(0, min(extent - 1, scaled_start))
    scaled_stop = max(
        scaled_start + 1, min(extent, scaled_start + scaled_width)
    )
    if axis == "x":
        mask[:, scaled_start:scaled_stop] = True
    else:
        mask[scaled_start:scaled_stop, :] = True
