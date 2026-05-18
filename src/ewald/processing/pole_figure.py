"""Pole-figure generation from EWALD q-space ROIs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ewald.data.models import ROIRegion, roi_hkl_label, roi_hkl_metadata

BACKGROUND_NONE = "none"
BACKGROUND_CONSTANT = "constant"
BACKGROUND_LOCAL_ANNULAR = "local_annular"
BACKGROUND_ROI = "roi"
BACKGROUND_POLYNOMIAL = "polynomial"
BACKGROUND_METHODS = (
    BACKGROUND_NONE,
    BACKGROUND_CONSTANT,
    BACKGROUND_LOCAL_ANNULAR,
    BACKGROUND_ROI,
    BACKGROUND_POLYNOMIAL,
)
INTENSITY_SUM = "sum"
INTENSITY_MEAN = "mean"
NORMALIZE_NONE = "none"
NORMALIZE_MAX = "max"
NORMALIZE_AREA = "area"


@dataclass(frozen=True, slots=True)
class PoleFigureSettings:
    """User-facing controls for reducing one ROI to a pole figure."""

    chi_min_deg: float = -90.0
    chi_max_deg: float = 90.0
    chi_bin_width_deg: float = 1.0
    intensity_mode: str = INTENSITY_SUM
    background_method: str = BACKGROUND_NONE
    background_constant: float = 0.0
    background_roi_id: str | None = None
    local_background_gap: float = 0.02
    local_background_width: float = 0.05
    polynomial_degree: int = 2
    polynomial_percentile: float = 60.0
    normalization: str = NORMALIZE_NONE
    clip_negative: bool = False
    display_label: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable settings."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PoleFigureResult:
    """Generated pole figure profile and provenance arrays."""

    chi_deg: np.ndarray
    intensity: np.ndarray
    raw_intensity: np.ndarray
    background: np.ndarray
    counts: np.ndarray
    settings: PoleFigureSettings
    background_record: dict[str, Any]
    missing_fraction: float
    source_roi_id: str | None
    source_roi_name: str
    hkl_label: str

    def table_rows(self) -> list[dict[str, float | int | None]]:
        """Return rows suitable for CSV or table export."""

        rows: list[dict[str, float | int | None]] = []
        for chi, value, raw, background, count in zip(
            self.chi_deg,
            self.intensity,
            self.raw_intensity,
            self.background,
            self.counts,
            strict=False,
        ):
            rows.append(
                {
                    "chi_deg": _finite_float_or_none(chi),
                    "intensity": _finite_float_or_none(value),
                    "raw_intensity": _finite_float_or_none(raw),
                    "background": _finite_float_or_none(background),
                    "valid_pixel_count": int(count),
                }
            )
        return rows


def generate_pole_figure(
    roi: ROIRegion,
    image_data: np.ndarray | None,
    axis_ranges: tuple[float, float, float, float] | None,
    *,
    settings: PoleFigureSettings | None = None,
    background_roi: ROIRegion | None = None,
) -> PoleFigureResult | None:
    """Reduce one ROI in q-space to intensity as a function of chi."""

    if image_data is None or axis_ranges is None:
        return None
    image = np.asarray(image_data, dtype=float)
    if image.ndim != 2:
        return None
    settings = _normalize_settings(settings or PoleFigureSettings())
    x_axis, y_axis = image_axes(image.shape, axis_ranges)
    qxy_grid, qz_grid = np.meshgrid(x_axis, y_axis)
    roi_mask, chi_grid, radial_grid = roi_reduction_grids(
        roi,
        qxy_grid,
        qz_grid,
    )
    if not np.any(roi_mask):
        return None

    edges = chi_edges(settings)
    raw_intensity, counts = integrate_by_chi(
        image,
        roi_mask,
        chi_grid,
        edges,
        mode=settings.intensity_mode,
    )
    background, background_record = background_by_chi(
        roi,
        image,
        qxy_grid,
        qz_grid,
        roi_mask,
        chi_grid,
        radial_grid,
        raw_intensity,
        counts,
        edges,
        settings,
        background_roi=background_roi,
    )
    intensity = raw_intensity - background
    if settings.clip_negative:
        intensity = np.where(
            np.isfinite(intensity), np.maximum(intensity, 0), np.nan
        )
    intensity = normalize_profile(
        intensity,
        settings.normalization,
        bin_width_deg=settings.chi_bin_width_deg,
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    missing_fraction = float(np.mean(counts == 0)) if counts.size else 1.0
    return PoleFigureResult(
        chi_deg=centers,
        intensity=intensity,
        raw_intensity=raw_intensity,
        background=background,
        counts=counts,
        settings=settings,
        background_record=background_record,
        missing_fraction=missing_fraction,
        source_roi_id=roi.roi_id,
        source_roi_name=roi.name or roi.roi_id or "ROI",
        hkl_label=roi_hkl_label(roi),
    )


def image_axes(
    shape: tuple[int, int],
    axis_ranges: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return qxy and qz coordinate vectors for a remapped image."""

    height, width = shape
    qxy_min, qxy_max, qz_min, qz_max = axis_ranges
    return (
        np.linspace(qxy_min, qxy_max, max(width, 1)),
        np.linspace(qz_min, qz_max, max(height, 1)),
    )


def chi_edges(settings: PoleFigureSettings) -> np.ndarray:
    """Return monotonic chi bin edges."""

    chi_min = float(settings.chi_min_deg)
    chi_max = float(settings.chi_max_deg)
    if chi_max <= chi_min:
        chi_min, chi_max = chi_max, chi_min
    width = max(float(settings.chi_bin_width_deg), 1.0e-6)
    bin_count = max(1, int(np.ceil((chi_max - chi_min) / width)))
    return np.linspace(chi_min, chi_max, bin_count + 1)


def roi_reduction_grids(
    roi: ROIRegion,
    qxy_grid: np.ndarray,
    qz_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ROI mask, chi grid, and radius grid for reduction."""

    if roi.kind == "arch":
        qxy_relative = qxy_grid - float(roi.qxy_center)
        qz_relative = qz_grid - float(roi.qz_center)
        radial_grid = np.hypot(qxy_relative, qz_relative)
        chi_grid = np.degrees(np.arctan2(qxy_relative, qz_relative))
        qr_min, qr_max = sorted(
            (
                float(roi.qr_min or 0.0),
                float(roi.qr_max or 0.0),
            )
        )
        chi_min, chi_max = sorted(
            (
                float(roi.chi_min or -90.0),
                float(roi.chi_max or 90.0),
            )
        )
        mask = (
            (radial_grid >= qr_min)
            & (radial_grid <= qr_max)
            & (chi_grid >= chi_min)
            & (chi_grid <= chi_max)
        )
        return mask, chi_grid, radial_grid

    radial_grid = np.hypot(qxy_grid, qz_grid)
    chi_grid = np.degrees(np.arctan2(qxy_grid, qz_grid))
    qxy_min, qxy_max = sorted(
        (
            float(roi.qxy_min or 0.0),
            float(roi.qxy_max or 0.0),
        )
    )
    qz_min, qz_max = sorted(
        (
            float(roi.qz_min or 0.0),
            float(roi.qz_max or 0.0),
        )
    )
    mask = (
        (qxy_grid >= qxy_min)
        & (qxy_grid <= qxy_max)
        & (qz_grid >= qz_min)
        & (qz_grid <= qz_max)
    )
    return mask, chi_grid, radial_grid


def integrate_by_chi(
    image: np.ndarray,
    mask: np.ndarray,
    chi_grid: np.ndarray,
    edges: np.ndarray,
    *,
    mode: str = INTENSITY_SUM,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate finite masked pixels into chi bins."""

    finite = mask & np.isfinite(image) & np.isfinite(chi_grid)
    bin_count = max(len(edges) - 1, 0)
    if bin_count <= 0:
        return np.array([], dtype=float), np.array([], dtype=int)
    if not np.any(finite):
        return (
            np.full(bin_count, np.nan, dtype=float),
            np.zeros(bin_count, dtype=int),
        )
    bin_index = np.digitize(chi_grid[finite], edges) - 1
    valid = (bin_index >= 0) & (bin_index < bin_count)
    if not np.any(valid):
        return (
            np.full(bin_count, np.nan, dtype=float),
            np.zeros(bin_count, dtype=int),
        )
    weights = np.asarray(image[finite][valid], dtype=float)
    sums = np.bincount(bin_index[valid], weights=weights, minlength=bin_count)
    counts = np.bincount(bin_index[valid], minlength=bin_count).astype(int)
    if mode == INTENSITY_MEAN:
        with np.errstate(divide="ignore", invalid="ignore"):
            values = sums / counts
    else:
        values = sums
    values = np.where(counts > 0, values, np.nan)
    return values.astype(float, copy=False), counts


def background_by_chi(
    roi: ROIRegion,
    image: np.ndarray,
    qxy_grid: np.ndarray,
    qz_grid: np.ndarray,
    roi_mask: np.ndarray,
    chi_grid: np.ndarray,
    radial_grid: np.ndarray,
    raw_intensity: np.ndarray,
    counts: np.ndarray,
    edges: np.ndarray,
    settings: PoleFigureSettings,
    *,
    background_roi: ROIRegion | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return background profile matched to the ROI chi bins."""

    method = settings.background_method
    zeros = np.zeros_like(raw_intensity, dtype=float)
    if method == BACKGROUND_NONE:
        return zeros, {"method": BACKGROUND_NONE}
    if method == BACKGROUND_CONSTANT:
        value = float(settings.background_constant)
        if settings.intensity_mode == INTENSITY_SUM:
            background = value * counts.astype(float)
        else:
            background = np.where(counts > 0, value, np.nan)
        return background, {"method": BACKGROUND_CONSTANT, "value": value}
    if method == BACKGROUND_LOCAL_ANNULAR:
        background_mask = local_annular_background_mask(
            roi_mask,
            radial_grid,
            chi_grid,
            settings=settings,
        )
        background = scaled_background_profile(
            image,
            background_mask,
            chi_grid,
            edges,
            counts,
            mode=settings.intensity_mode,
        )
        return background, {
            "method": BACKGROUND_LOCAL_ANNULAR,
            "gap": settings.local_background_gap,
            "width": settings.local_background_width,
        }
    if method == BACKGROUND_ROI:
        if background_roi is None:
            return zeros, {
                "method": BACKGROUND_ROI,
                "background_roi_id": settings.background_roi_id,
                "warning": "No background ROI was available.",
            }
        background_mask, background_chi, _background_radius = (
            roi_reduction_grids(background_roi, qxy_grid, qz_grid)
        )
        background = scaled_background_profile(
            image,
            background_mask,
            background_chi,
            edges,
            counts,
            mode=settings.intensity_mode,
        )
        return background, {
            "method": BACKGROUND_ROI,
            "background_roi_id": background_roi.roi_id,
            "background_roi_name": background_roi.name,
        }
    if method == BACKGROUND_POLYNOMIAL:
        background = polynomial_baseline(
            (edges[:-1] + edges[1:]) / 2.0,
            raw_intensity,
            degree=settings.polynomial_degree,
            percentile=settings.polynomial_percentile,
        )
        return background, {
            "method": BACKGROUND_POLYNOMIAL,
            "degree": settings.polynomial_degree,
            "percentile": settings.polynomial_percentile,
        }
    return zeros, {"method": BACKGROUND_NONE}


def local_annular_background_mask(
    roi_mask: np.ndarray,
    radial_grid: np.ndarray,
    chi_grid: np.ndarray,
    *,
    settings: PoleFigureSettings,
) -> np.ndarray:
    """Return neighboring radial bands at the same chi range as the
    ROI."""

    if not np.any(roi_mask):
        return np.zeros_like(roi_mask, dtype=bool)
    roi_radius = radial_grid[roi_mask & np.isfinite(radial_grid)]
    if not roi_radius.size:
        return np.zeros_like(roi_mask, dtype=bool)
    r_min = float(np.nanmin(roi_radius))
    r_max = float(np.nanmax(roi_radius))
    gap = max(float(settings.local_background_gap), 0.0)
    width = max(float(settings.local_background_width), 1.0e-9)
    inner_min = max(0.0, r_min - gap - width)
    inner_max = max(0.0, r_min - gap)
    outer_min = r_max + gap
    outer_max = r_max + gap + width
    chi_min, chi_max = sorted(
        (float(settings.chi_min_deg), float(settings.chi_max_deg))
    )
    radial_mask = ((radial_grid >= inner_min) & (radial_grid <= inner_max)) | (
        (radial_grid >= outer_min) & (radial_grid <= outer_max)
    )
    return radial_mask & (chi_grid >= chi_min) & (chi_grid <= chi_max)


def scaled_background_profile(
    image: np.ndarray,
    background_mask: np.ndarray,
    background_chi: np.ndarray,
    edges: np.ndarray,
    roi_counts: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Integrate background and scale it to the ROI pixel support."""

    background_sum, background_counts = integrate_by_chi(
        image,
        background_mask,
        background_chi,
        edges,
        mode=INTENSITY_SUM,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        background_mean = background_sum / background_counts
    if mode == INTENSITY_MEAN:
        return np.where(
            (background_counts > 0) & (roi_counts > 0),
            background_mean,
            np.nan,
        )
    return np.where(
        (background_counts > 0) & (roi_counts > 0),
        background_mean * roi_counts.astype(float),
        np.nan,
    )


def polynomial_baseline(
    chi_deg: np.ndarray,
    intensity: np.ndarray,
    *,
    degree: int,
    percentile: float,
) -> np.ndarray:
    """Fit a low-order smooth baseline to lower-intensity chi bins."""

    baseline = np.zeros_like(intensity, dtype=float)
    valid = np.isfinite(chi_deg) & np.isfinite(intensity)
    if np.count_nonzero(valid) < 2:
        return np.where(valid, baseline, np.nan)
    degree = max(0, min(int(degree), 5))
    degree = min(degree, max(0, int(np.count_nonzero(valid) - 1)))
    threshold = np.nanpercentile(intensity[valid], percentile)
    fit_mask = valid & (intensity <= threshold)
    if np.count_nonzero(fit_mask) <= degree:
        fit_mask = valid
    coeffs = np.polyfit(chi_deg[fit_mask], intensity[fit_mask], degree)
    fitted = np.polyval(coeffs, chi_deg)
    return np.where(valid, fitted, np.nan)


def normalize_profile(
    intensity: np.ndarray,
    normalization: str,
    *,
    bin_width_deg: float,
) -> np.ndarray:
    """Apply optional post-background normalization."""

    if normalization == NORMALIZE_NONE:
        return intensity
    finite = np.isfinite(intensity)
    if not np.any(finite):
        return intensity
    if normalization == NORMALIZE_MAX:
        scale = float(np.nanmax(np.abs(intensity[finite])))
    elif normalization == NORMALIZE_AREA:
        scale = float(np.nansum(np.abs(intensity[finite])) * bin_width_deg)
    else:
        scale = 1.0
    if scale <= 0.0 or not np.isfinite(scale):
        return intensity
    return intensity / scale


def pole_figure_record_from_result(
    roi: ROIRegion,
    result: PoleFigureResult,
    *,
    output_file_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build ROI metadata describing a generated pole figure."""

    label = result.settings.display_label or result.hkl_label
    return {
        "roi_id": roi.roi_id,
        "hkl_tag": roi_hkl_metadata(roi),
        "custom_label": label,
        "background_subtraction": result.background_record,
        "output_file_path": (
            str(output_file_path) if output_file_path is not None else None
        ),
        "generation_parameters": result.settings.as_dict(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "missing_fraction": result.missing_fraction,
        "bin_count": int(result.chi_deg.size),
        "current": True,
    }


def export_pole_figure_csv(
    result: PoleFigureResult,
    path: str | Path,
) -> Path:
    """Write generated pole-figure data to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "chi_deg",
            "intensity",
            "raw_intensity",
            "background",
            "valid_pixel_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.table_rows())
    return output_path


def _normalize_settings(settings: PoleFigureSettings) -> PoleFigureSettings:
    method = (
        settings.background_method
        if settings.background_method in BACKGROUND_METHODS
        else BACKGROUND_NONE
    )
    intensity_mode = (
        settings.intensity_mode
        if settings.intensity_mode in {INTENSITY_SUM, INTENSITY_MEAN}
        else INTENSITY_SUM
    )
    normalization = (
        settings.normalization
        if settings.normalization
        in {NORMALIZE_NONE, NORMALIZE_MAX, NORMALIZE_AREA}
        else NORMALIZE_NONE
    )
    return PoleFigureSettings(
        chi_min_deg=float(settings.chi_min_deg),
        chi_max_deg=float(settings.chi_max_deg),
        chi_bin_width_deg=max(float(settings.chi_bin_width_deg), 1.0e-6),
        intensity_mode=intensity_mode,
        background_method=method,
        background_constant=float(settings.background_constant),
        background_roi_id=settings.background_roi_id,
        local_background_gap=max(float(settings.local_background_gap), 0.0),
        local_background_width=max(
            float(settings.local_background_width), 1.0e-9
        ),
        polynomial_degree=max(0, min(int(settings.polynomial_degree), 5)),
        polynomial_percentile=float(
            np.clip(settings.polynomial_percentile, 0.0, 100.0)
        ),
        normalization=normalization,
        clip_negative=bool(settings.clip_negative),
        display_label=" ".join(settings.display_label.strip().split()),
    )


def _finite_float_or_none(value: Any) -> float | None:
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric
