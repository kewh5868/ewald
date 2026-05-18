"""Peak detection helpers for corrected detector images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class PeakCandidate:
    """Candidate peak location in detector or q-space coordinates."""

    x: float
    y: float
    intensity: float
    label: str | None = None
    background: float | None = None
    noise: float | None = None
    snr: float | None = None
    prominence: float | None = None
    score: float | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "x": self.x,
            "y": self.y,
            "intensity": self.intensity,
            "label": self.label,
            "background": self.background,
            "noise": self.noise,
            "snr": self.snr,
            "prominence": self.prominence,
            "score": self.score,
        }


@dataclass(slots=True)
class LocalMaxPeakFinderConfig:
    """Settings for local-maximum peak detection."""

    threshold_percentile: float = 99.5
    adaptive_threshold: bool = False
    adaptive_floor_percentile: float = 94.0
    min_snr: float = 4.5
    min_prominence: float | None = None
    background_radius_px: int = 18
    max_peaks: int = 500
    min_distance_px: int = 8
    neighborhood_radius_px: int = 2
    min_intensity: float | None = None
    ignore_nonpositive: bool = True
    rank_by: str = "score"


def find_local_maxima_peaks(
    image: Any,
    *,
    x_axis: Any | None = None,
    y_axis: Any | None = None,
    valid_mask: Any | None = None,
    config: LocalMaxPeakFinderConfig | None = None,
) -> list[PeakCandidate]:
    """Find separated local maxima in a 2D detector or q-space image.

    The returned ``PeakCandidate.x`` and ``PeakCandidate.y`` values are in
    the supplied axis coordinates when ``x_axis`` and ``y_axis`` are provided;
    otherwise they fall back to pixel-column and pixel-row coordinates.
    """

    cfg = config or LocalMaxPeakFinderConfig()
    array = np.asarray(image, dtype=float)
    if array.ndim != 2:
        raise ValueError("find_local_maxima_peaks expects a 2D image.")

    finite = np.isfinite(array)
    usable = finite.copy()
    if valid_mask is not None:
        usable &= np.asarray(valid_mask, dtype=bool)
    if cfg.ignore_nonpositive:
        usable &= array > 0.0
    if not np.any(usable):
        return []

    usable_values = array[usable]
    cutoff = (
        float(cfg.min_intensity)
        if cfg.min_intensity is not None
        else float(np.nanpercentile(usable_values, cfg.threshold_percentile))
    )
    adaptive_floor = (
        float(cfg.min_intensity)
        if cfg.min_intensity is not None
        else float(
            np.nanpercentile(
                usable_values,
                min(cfg.threshold_percentile, cfg.adaptive_floor_percentile),
            )
        )
    )
    background, residual, noise = _adaptive_background_residual(
        array,
        usable,
        cfg,
    )
    prominence = np.where(usable, residual, -np.inf)
    snr = np.where(usable, residual / max(noise, 1.0e-12), -np.inf)
    min_prominence = (
        float(cfg.min_prominence)
        if cfg.min_prominence is not None
        else max(noise * float(cfg.min_snr), 0.0)
    )
    try:
        from scipy import ndimage

        radius = max(1, int(cfg.neighborhood_radius_px))
        local_max = array == ndimage.maximum_filter(
            np.where(usable, array, -np.inf),
            size=radius * 2 + 1,
            mode="nearest",
        )
    except Exception:
        local_max = _local_maximum_mask(
            array,
            usable,
            max(1, int(cfg.neighborhood_radius_px)),
        )

    bright_enough = array >= cutoff
    if cfg.adaptive_threshold:
        adaptive_enough = (
            (array >= adaptive_floor)
            & (snr >= float(cfg.min_snr))
            & (prominence >= min_prominence)
        )
        candidate_mask = usable & local_max & (bright_enough | adaptive_enough)
    else:
        candidate_mask = usable & local_max & bright_enough

    yy, xx = np.where(candidate_mask)
    if yy.size == 0:
        return []
    intensities = np.asarray(array[yy, xx], dtype=float)
    candidate_scores = _candidate_scores(
        intensities,
        np.asarray(prominence[yy, xx], dtype=float),
        np.asarray(snr[yy, xx], dtype=float),
        usable_values,
        cfg.rank_by,
    )
    order = np.argsort(candidate_scores)[::-1]
    selected: list[int] = []
    min_distance_sq = float(max(0, int(cfg.min_distance_px)) ** 2)
    for index in order:
        if len(selected) >= max(1, int(cfg.max_peaks)):
            break
        y_value = float(yy[index])
        x_value = float(xx[index])
        if any(
            (x_value - float(xx[kept])) ** 2 + (y_value - float(yy[kept])) ** 2
            < min_distance_sq
            for kept in selected
        ):
            continue
        selected.append(int(index))

    x_coords = _axis_values(x_axis, array.shape[1])
    y_coords = _axis_values(y_axis, array.shape[0])
    return [
        PeakCandidate(
            x=float(x_coords[xx[index]]),
            y=float(y_coords[yy[index]]),
            intensity=float(array[yy[index], xx[index]]),
            background=float(background[yy[index], xx[index]]),
            noise=float(noise),
            snr=float(snr[yy[index], xx[index]]),
            prominence=float(prominence[yy[index], xx[index]]),
            score=float(candidate_scores[index]),
        )
        for index in selected
    ]


def _adaptive_background_residual(
    array: np.ndarray,
    usable: np.ndarray,
    cfg: LocalMaxPeakFinderConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not cfg.adaptive_threshold:
        background = np.zeros_like(array, dtype=float)
        residual = np.where(usable, array, 0.0)
        noise = _robust_noise(residual[usable])
        return background, residual, noise
    try:
        from scipy import ndimage

        radius = max(
            int(cfg.background_radius_px),
            int(cfg.neighborhood_radius_px) * 3,
            3,
        )
        safe = np.where(usable, array, np.nan)
        filled = np.where(np.isfinite(safe), safe, np.nanmedian(array[usable]))
        background = ndimage.median_filter(
            filled,
            size=radius * 2 + 1,
            mode="nearest",
        )
    except Exception:
        background = np.full_like(
            array,
            float(np.nanmedian(array[usable])),
            dtype=float,
        )
    residual = np.where(usable, array - background, 0.0)
    positive_residual = residual[usable & np.isfinite(residual)]
    noise = _robust_noise(positive_residual)
    return background, residual, noise


def _robust_noise(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    median = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - median)))
    noise = 1.4826 * mad
    if not np.isfinite(noise) or noise <= 1.0e-12:
        q25, q75 = np.nanpercentile(finite, [25.0, 75.0])
        noise = float((q75 - q25) / 1.349)
    if not np.isfinite(noise) or noise <= 1.0e-12:
        noise = float(np.nanstd(finite))
    return max(noise, 1.0e-12)


def _candidate_scores(
    intensities: np.ndarray,
    prominence: np.ndarray,
    snr: np.ndarray,
    usable_values: np.ndarray,
    rank_by: str,
) -> np.ndarray:
    mode = str(rank_by or "score").lower()
    if mode == "intensity":
        return intensities
    finite = usable_values[np.isfinite(usable_values)]
    if finite.size == 0:
        intensity_scale = 1.0
        intensity_center = 0.0
    else:
        lo, hi = np.nanpercentile(finite, [50.0, 99.5])
        intensity_scale = max(float(hi - lo), 1.0e-12)
        intensity_center = float(np.nanmedian(finite))
    normalized_intensity = np.clip(
        (intensities - intensity_center) / intensity_scale,
        0.0,
        None,
    )
    if mode == "snr":
        return snr
    if mode == "prominence":
        return prominence
    return np.sqrt(normalized_intensity + 1.0) * np.clip(snr, 0.0, None)


def _local_maximum_mask(
    array: np.ndarray,
    usable: np.ndarray,
    radius: int,
) -> np.ndarray:
    padded = np.pad(
        np.where(usable, array, -np.inf),
        radius,
        mode="edge",
    )
    local_max = np.ones(array.shape, dtype=bool)
    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            if row_offset == 0 and col_offset == 0:
                continue
            shifted = padded[
                radius + row_offset : radius + row_offset + array.shape[0],
                radius + col_offset : radius + col_offset + array.shape[1],
            ]
            local_max &= array >= shifted
    return local_max


def _axis_values(axis: Any | None, size: int) -> np.ndarray:
    if axis is None:
        return np.arange(size, dtype=float)
    values = np.asarray(axis, dtype=float)
    if values.size != size:
        raise ValueError("Axis length does not match image shape.")
    return values


def threshold_peaks(
    image: Any,
    *,
    threshold: float | None = None,
    max_peaks: int = 1000,
) -> list[PeakCandidate]:
    """Return bright-pixel peak candidates from a 2D image.

    This is a deliberately small placeholder for the later 2D peak model. It
    gives the UI and project file a concrete contract while the full detector
    fitting path is built out.
    """

    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("threshold_peaks expects a 2D image.")
    cutoff = float(
        np.nanpercentile(array, 99.5) if threshold is None else threshold
    )
    yy, xx = np.where(array >= cutoff)
    intensities = array[yy, xx]
    if intensities.size > max_peaks:
        keep = np.argpartition(intensities, -max_peaks)[-max_peaks:]
        yy = yy[keep]
        xx = xx[keep]
        intensities = intensities[keep]
    order = np.argsort(intensities)[::-1]
    return [
        PeakCandidate(
            x=float(xx[index]),
            y=float(yy[index]),
            intensity=float(intensities[index]),
        )
        for index in order
    ]
