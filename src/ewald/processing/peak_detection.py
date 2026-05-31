"""Peak detection helpers for corrected detector images."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
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
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class GapAwarePeakInferenceConfig:
    """Settings for detector-gap peak inference.

    Gap bridging is deliberately conservative. It is meant for narrow module
    gaps where two anchor maxima on either side are plausibly clipped halves of
    one Bragg spot, not for broad missing wedges, beamstops, substrate horizon
    shadows, or isotropic powder/ring patterns.
    """

    enabled: bool = True
    sample_texture: str = "auto"
    max_gap_width_px: int = 18
    max_gap_fraction: float = 0.045
    gap_line_coverage: float = 0.82
    side_search_px: int | None = None
    perpendicular_tolerance_px: int = 3
    min_anchor_intensity_ratio: float = 0.25
    min_anisotropy_score: float = 0.22
    max_inferred_peaks: int = 100


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
    gap_inference: GapAwarePeakInferenceConfig | None = field(
        default_factory=GapAwarePeakInferenceConfig
    )


@dataclass(frozen=True, slots=True)
class _GapSpan:
    axis: str
    start: int
    stop: int
    coverage: float

    @property
    def width(self) -> int:
        return max(0, self.stop - self.start)


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

    def candidate_from_detection_index(index: int) -> PeakCandidate:
        return PeakCandidate(
            x=float(x_coords[xx[index]]),
            y=float(y_coords[yy[index]]),
            intensity=float(array[yy[index], xx[index]]),
            background=float(background[yy[index], xx[index]]),
            noise=float(noise),
            snr=float(snr[yy[index], xx[index]]),
            prominence=float(prominence[yy[index], xx[index]]),
            score=float(candidate_scores[index]),
        )

    candidates = [candidate_from_detection_index(index) for index in selected]
    gap_cfg = cfg.gap_inference
    if gap_cfg is not None and gap_cfg.enabled:
        anchor_limit = min(
            order.size,
            max(
                max(1, int(cfg.max_peaks)) * 4,
                max(1, int(cfg.max_peaks))
                + 2 * int(gap_cfg.max_inferred_peaks),
            ),
        )
        anchor_candidates = [
            candidate_from_detection_index(int(index))
            for index in order[:anchor_limit]
        ]
        inferred = infer_detector_gap_peaks(
            array,
            anchor_candidates,
            x_axis=x_coords,
            y_axis=y_coords,
            valid_mask=valid_mask,
            config=gap_cfg,
            min_distance_px=cfg.min_distance_px,
            treat_nonpositive_as_gap=cfg.ignore_nonpositive,
        )
        if inferred:
            candidates = _merge_ranked_candidates(
                candidates,
                inferred,
                x_axis=x_coords,
                y_axis=y_coords,
                max_peaks=max(1, int(cfg.max_peaks)),
                min_distance_px=max(0, int(cfg.min_distance_px)),
            )
    return candidates


def infer_detector_gap_peaks(
    image: Any,
    peaks: Sequence[PeakCandidate],
    *,
    x_axis: Any | None = None,
    y_axis: Any | None = None,
    valid_mask: Any | None = None,
    config: GapAwarePeakInferenceConfig | None = None,
    min_distance_px: int = 8,
    treat_nonpositive_as_gap: bool = True,
) -> list[PeakCandidate]:
    """Infer hidden Bragg peak centers clipped by narrow detector gaps.

    The function only considers mask spans that look like detector module gaps:
    thin rows or columns covering most of the frame. It refuses broad spans,
    which are more likely missing wedges, beamstops, horizon shadows, or
    unusable detector regions. Set ``sample_texture`` to ``"isotropic"`` or
    ``"powder"`` to disable inference for ring-like samples.
    """

    cfg = config or GapAwarePeakInferenceConfig()
    if not cfg.enabled or not peaks:
        return []
    texture = str(cfg.sample_texture or "auto").lower()
    if texture in {"isotropic", "powder", "ring", "rings"}:
        return []

    array = np.asarray(image, dtype=float)
    if array.ndim != 2:
        raise ValueError("infer_detector_gap_peaks expects a 2D image.")
    x_coords = _axis_values(x_axis, array.shape[1])
    y_coords = _axis_values(y_axis, array.shape[0])
    peak_pixels = _candidate_pixel_positions(peaks, x_coords, y_coords)
    anisotropy_score = _peak_anisotropy_score(
        peak_pixels,
        array.shape,
    )
    if (
        texture == "auto"
        and len(peak_pixels) >= 8
        and anisotropy_score < float(cfg.min_anisotropy_score)
    ):
        return []

    gap_mask = _gap_mask_from_image(
        array,
        valid_mask=valid_mask,
        treat_nonpositive_as_gap=treat_nonpositive_as_gap,
    )
    spans = _detector_gap_spans(gap_mask, cfg)
    if not spans:
        return []

    side_search_px = (
        int(cfg.side_search_px)
        if cfg.side_search_px is not None
        else max(int(min_distance_px) * 2, int(cfg.max_gap_width_px))
    )
    side_search_px = max(1, side_search_px)
    perpendicular_tolerance_px = max(
        0,
        int(cfg.perpendicular_tolerance_px),
        int(min_distance_px) // 2,
    )

    inferred: dict[tuple[str, int, int], PeakCandidate] = {}
    for span in spans:
        for candidate in _bridge_candidates_for_span(
            peaks,
            peak_pixels,
            span,
            x_coords=x_coords,
            y_coords=y_coords,
            side_search_px=side_search_px,
            perpendicular_tolerance_px=perpendicular_tolerance_px,
            min_anchor_intensity_ratio=float(cfg.min_anchor_intensity_ratio),
            max_gap_width_px=max(1, int(cfg.max_gap_width_px)),
            anisotropy_score=anisotropy_score,
            sample_texture=texture,
        ):
            cx, cy = _candidate_pixel_position(
                candidate,
                x_coords,
                y_coords,
            )
            key = (span.axis, int(round(cx)), int(round(cy)))
            current = inferred.get(key)
            current_score = (
                -np.inf if current is None else _candidate_rank_value(current)
            )
            if _candidate_rank_value(candidate) > current_score:
                inferred[key] = candidate

    ordered = sorted(
        inferred.values(),
        key=_candidate_rank_value,
        reverse=True,
    )
    return ordered[: max(0, int(cfg.max_inferred_peaks))]


def _merge_ranked_candidates(
    observed: Sequence[PeakCandidate],
    inferred: Sequence[PeakCandidate],
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    max_peaks: int,
    min_distance_px: int,
) -> list[PeakCandidate]:
    combined = sorted(
        [*observed, *inferred],
        key=_candidate_rank_value,
        reverse=True,
    )
    selected: list[PeakCandidate] = []
    selected_pixels: list[tuple[float, float]] = []
    min_distance_sq = float(max(0, min_distance_px) ** 2)
    for candidate in combined:
        if len(selected) >= max_peaks:
            break
        pixel_x, pixel_y = _candidate_pixel_position(
            candidate,
            x_axis,
            y_axis,
        )
        nearby_indices = [
            index
            for index, (kept_x, kept_y) in enumerate(selected_pixels)
            if (pixel_x - kept_x) ** 2 + (pixel_y - kept_y) ** 2
            < min_distance_sq
        ]
        if nearby_indices and not _candidate_is_gap_anchor_bridge(
            candidate,
            [selected_pixels[index] for index in nearby_indices],
        ):
            continue
        selected.append(candidate)
        selected_pixels.append((pixel_x, pixel_y))
    return selected


def _bridge_candidates_for_span(
    peaks: Sequence[PeakCandidate],
    peak_pixels: Sequence[tuple[float, float]],
    span: _GapSpan,
    *,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    side_search_px: int,
    perpendicular_tolerance_px: int,
    min_anchor_intensity_ratio: float,
    max_gap_width_px: int,
    anisotropy_score: float,
    sample_texture: str,
) -> list[PeakCandidate]:
    entries = list(zip(peaks, peak_pixels, strict=False))
    if span.axis == "x":
        minus = [
            (peak, pixel)
            for peak, pixel in entries
            if 0.0 <= span.start - pixel[0] <= side_search_px
        ]
        plus = [
            (peak, pixel)
            for peak, pixel in entries
            if 0.0 <= pixel[0] - (span.stop - 1) <= side_search_px
        ]
    else:
        minus = [
            (peak, pixel)
            for peak, pixel in entries
            if 0.0 <= span.start - pixel[1] <= side_search_px
        ]
        plus = [
            (peak, pixel)
            for peak, pixel in entries
            if 0.0 <= pixel[1] - (span.stop - 1) <= side_search_px
        ]

    candidates: list[PeakCandidate] = []
    for minus_peak, minus_pixel in minus:
        for plus_peak, plus_pixel in plus:
            if span.axis == "x":
                perp_delta = abs(minus_pixel[1] - plus_pixel[1])
                minus_distance = span.start - minus_pixel[0]
                plus_distance = plus_pixel[0] - (span.stop - 1)
                center_x = (span.start + span.stop - 1) / 2.0
                center_y = _weighted_average(
                    minus_pixel[1],
                    plus_pixel[1],
                    minus_peak.intensity,
                    plus_peak.intensity,
                )
            else:
                perp_delta = abs(minus_pixel[0] - plus_pixel[0])
                minus_distance = span.start - minus_pixel[1]
                plus_distance = plus_pixel[1] - (span.stop - 1)
                center_x = _weighted_average(
                    minus_pixel[0],
                    plus_pixel[0],
                    minus_peak.intensity,
                    plus_peak.intensity,
                )
                center_y = (span.start + span.stop - 1) / 2.0

            if perp_delta > perpendicular_tolerance_px:
                continue
            low = min(abs(minus_peak.intensity), abs(plus_peak.intensity))
            high = max(abs(minus_peak.intensity), abs(plus_peak.intensity))
            if high <= 0.0 or low / high < min_anchor_intensity_ratio:
                continue

            confidence = _gap_bridge_confidence(
                span_width=span.width,
                max_gap_width_px=max_gap_width_px,
                perp_delta=perp_delta,
                perpendicular_tolerance_px=perpendicular_tolerance_px,
                minus_distance=minus_distance,
                plus_distance=plus_distance,
                side_search_px=side_search_px,
                intensity_ratio=low / high,
            )
            score = (
                min(
                    _candidate_rank_value(minus_peak),
                    _candidate_rank_value(plus_peak),
                )
                * confidence
            )
            qxy = _axis_interpolated_value(x_coords, center_x)
            qz = _axis_interpolated_value(y_coords, center_y)
            metadata = _gap_bridge_metadata(
                span,
                minus_peak,
                plus_peak,
                minus_pixel,
                plus_pixel,
                confidence=confidence,
                anisotropy_score=anisotropy_score,
                sample_texture=sample_texture,
            )
            candidates.append(
                PeakCandidate(
                    x=qxy,
                    y=qz,
                    intensity=float(
                        np.average([minus_peak.intensity, plus_peak.intensity])
                    ),
                    label="detector-gap-estimate",
                    background=_mean_optional(
                        minus_peak.background,
                        plus_peak.background,
                    ),
                    noise=_mean_optional(minus_peak.noise, plus_peak.noise),
                    snr=_mean_optional(minus_peak.snr, plus_peak.snr),
                    prominence=_mean_optional(
                        minus_peak.prominence,
                        plus_peak.prominence,
                    ),
                    score=float(score),
                    metadata=metadata,
                )
            )
    return candidates


def _gap_bridge_confidence(
    *,
    span_width: int,
    max_gap_width_px: int,
    perp_delta: float,
    perpendicular_tolerance_px: int,
    minus_distance: float,
    plus_distance: float,
    side_search_px: int,
    intensity_ratio: float,
) -> float:
    gap_score = 1.0 - (span_width - 1) / max(float(max_gap_width_px), 1.0)
    perp_score = 1.0 - perp_delta / max(
        float(perpendicular_tolerance_px + 1),
        1.0,
    )
    distance_score = 1.0 - (
        max(minus_distance, 0.0) + max(plus_distance, 0.0)
    ) / max(float(side_search_px * 2 + 1), 1.0)
    confidence = 0.2 + 0.8 * np.clip(gap_score, 0.0, 1.0) * np.clip(
        perp_score, 0.0, 1.0
    ) * np.clip(distance_score, 0.0, 1.0) * np.clip(intensity_ratio, 0.0, 1.0)
    return float(np.clip(confidence, 0.0, 1.0))


def _gap_bridge_metadata(
    span: _GapSpan,
    minus_peak: PeakCandidate,
    plus_peak: PeakCandidate,
    minus_pixel: tuple[float, float],
    plus_pixel: tuple[float, float],
    *,
    confidence: float,
    anisotropy_score: float,
    sample_texture: str,
) -> dict[str, Any]:
    axis_name = "qxy" if span.axis == "x" else "qz"
    return {
        "gap_estimate": True,
        "gap_inferred": True,
        "masked_gap": True,
        "estimate_method": "detector-gap bridge",
        "gap_axis": axis_name,
        "gap_start_px": int(span.start),
        "gap_stop_px": int(span.stop),
        "gap_width_px": int(span.width),
        "gap_line_coverage": float(span.coverage),
        "gap_bridge_confidence": float(confidence),
        "sample_texture": sample_texture,
        "anisotropy_score": float(anisotropy_score),
        "minus_anchor": _anchor_metadata(minus_peak, minus_pixel),
        "plus_anchor": _anchor_metadata(plus_peak, plus_pixel),
    }


def _anchor_metadata(
    peak: PeakCandidate,
    pixel: tuple[float, float],
) -> dict[str, float | str | None]:
    return {
        "x": float(peak.x),
        "y": float(peak.y),
        "pixel_x": float(pixel[0]),
        "pixel_y": float(pixel[1]),
        "intensity": float(peak.intensity),
        "snr": None if peak.snr is None else float(peak.snr),
        "score": None if peak.score is None else float(peak.score),
        "label": peak.label,
    }


def _candidate_is_gap_anchor_bridge(
    candidate: PeakCandidate,
    nearby_pixels: Sequence[tuple[float, float]],
) -> bool:
    metadata = candidate.metadata or {}
    if not metadata.get("gap_estimate"):
        return False
    anchors = []
    for key in ("minus_anchor", "plus_anchor"):
        anchor = metadata.get(key)
        if isinstance(anchor, dict):
            try:
                anchors.append(
                    (
                        float(anchor["pixel_x"]),
                        float(anchor["pixel_y"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not anchors:
        return False
    return all(
        any(
            (near_x - anchor_x) ** 2 + (near_y - anchor_y) ** 2 <= 2.25
            for anchor_x, anchor_y in anchors
        )
        for near_x, near_y in nearby_pixels
    )


def _detector_gap_spans(
    gap_mask: np.ndarray,
    cfg: GapAwarePeakInferenceConfig,
) -> list[_GapSpan]:
    spans: list[_GapSpan] = []
    max_width = max(1, int(cfg.max_gap_width_px))
    line_coverage = float(np.clip(cfg.gap_line_coverage, 0.0, 1.0))
    for axis, coverage, extent in (
        ("x", np.mean(gap_mask, axis=0), gap_mask.shape[1]),
        ("y", np.mean(gap_mask, axis=1), gap_mask.shape[0]),
    ):
        active = coverage >= line_coverage
        fraction_width = max(
            1,
            int(round(max(float(cfg.max_gap_fraction), 0.0) * extent)),
        )
        allowed_width = min(max_width, fraction_width)
        for start, stop in _contiguous_true_spans(active):
            width = stop - start
            if width <= 0 or width > allowed_width:
                continue
            if width >= extent:
                continue
            spans.append(
                _GapSpan(
                    axis=axis,
                    start=start,
                    stop=stop,
                    coverage=float(np.nanmean(coverage[start:stop])),
                )
            )
    return spans


def _gap_mask_from_image(
    array: np.ndarray,
    *,
    valid_mask: Any | None,
    treat_nonpositive_as_gap: bool,
) -> np.ndarray:
    usable = np.isfinite(array)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != array.shape:
            raise ValueError("valid_mask shape must match image shape.")
        usable &= mask
    if treat_nonpositive_as_gap:
        usable &= array > 0.0
    return ~usable


def _contiguous_true_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(values):
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, values.size))
    return spans


def _candidate_pixel_positions(
    peaks: Sequence[PeakCandidate],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> list[tuple[float, float]]:
    return [_candidate_pixel_position(peak, x_axis, y_axis) for peak in peaks]


def _candidate_pixel_position(
    peak: PeakCandidate,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> tuple[float, float]:
    metadata = peak.metadata or {}
    try:
        return float(metadata["pixel_x"]), float(metadata["pixel_y"])
    except (KeyError, TypeError, ValueError):
        pass
    return (
        float(_nearest_axis_index(x_axis, float(peak.x))),
        float(_nearest_axis_index(y_axis, float(peak.y))),
    )


def _nearest_axis_index(axis: np.ndarray, value: float) -> int:
    finite = np.isfinite(axis)
    if not finite.any():
        return 0
    distances = np.where(finite, np.abs(axis - value), np.inf)
    return int(np.nanargmin(distances))


def _axis_interpolated_value(axis: np.ndarray, index: float) -> float:
    if axis.size == 0:
        return float(index)
    left = int(np.floor(index))
    right = int(np.ceil(index))
    left = int(np.clip(left, 0, axis.size - 1))
    right = int(np.clip(right, 0, axis.size - 1))
    if left == right:
        return float(axis[left])
    fraction = float(index - left)
    return float((1.0 - fraction) * axis[left] + fraction * axis[right])


def _peak_anisotropy_score(
    peak_pixels: Sequence[tuple[float, float]],
    shape: tuple[int, int],
) -> float:
    if len(peak_pixels) < 3:
        return 1.0
    height, width = shape
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    coords = np.asarray(peak_pixels, dtype=float)
    x = (coords[:, 0] - center_x) / max(float(width), 1.0)
    y = (coords[:, 1] - center_y) / max(float(height), 1.0)
    radius = np.hypot(x, y)
    valid = radius > 1.0e-6
    if np.count_nonzero(valid) < 3:
        return 1.0
    angles = np.arctan2(y[valid], x[valid])
    order_2 = abs(np.mean(np.exp(2j * angles)))
    order_4 = abs(np.mean(np.exp(4j * angles)))
    return float(max(order_2, order_4))


def _candidate_rank_value(candidate: PeakCandidate) -> float:
    score = candidate.score
    if score is not None and np.isfinite(score):
        return float(score)
    return float(candidate.intensity)


def _weighted_average(
    first: float,
    second: float,
    first_weight: float,
    second_weight: float,
) -> float:
    weights = np.asarray(
        [max(abs(first_weight), 1.0e-12), max(abs(second_weight), 1.0e-12)],
        dtype=float,
    )
    return float(np.average([first, second], weights=weights))


def _mean_optional(
    first: float | None,
    second: float | None,
) -> float | None:
    values = [value for value in (first, second) if value is not None]
    if not values:
        return None
    return float(np.mean(values))


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

    This is a deliberately small placeholder for the later 2D peak
    model. It gives the UI and project file a concrete contract while
    the full detector fitting path is built out.
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
