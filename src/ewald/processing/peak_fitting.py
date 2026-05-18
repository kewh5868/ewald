"""ROI integration and Gaussian fitting helpers for peak refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

GAUSSIAN_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
PEAK_FIT_INTEGRATIONS = ("qxy", "qz", "azimuthal")
TWO_D_GAUSSIAN_EXPRESSION = (
    "offset + amplitude * exp(-0.5 * "
    "(((qxy - center_qxy) / sigma_qxy)^2 + "
    "((qz - center_qz) / sigma_qz)^2))"
)


@dataclass(slots=True)
class PeakRoiSlice:
    """Rectangular q-space image region prepared for peak fitting."""

    qxy: np.ndarray
    qz: np.ndarray
    intensity: np.ndarray

    @property
    def qxy_grid(self) -> np.ndarray:
        return np.meshgrid(self.qxy, self.qz)[0]

    @property
    def qz_grid(self) -> np.ndarray:
        return np.meshgrid(self.qxy, self.qz)[1]

    def as_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qxy_grid, qz_grid = np.meshgrid(self.qxy, self.qz)
        return qxy_grid, qz_grid, self.intensity


def image_axes(
    shape: tuple[int, int],
    axis_ranges: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return qxy and qz coordinate axes for a 2D q-space image."""

    height, width = shape
    qxy_min, qxy_max, qz_min, qz_max = axis_ranges
    return (
        np.linspace(qxy_min, qxy_max, max(width, 1)),
        np.linspace(qz_min, qz_max, max(height, 1)),
    )


def slice_peak_roi(
    image: Any,
    axis_ranges: tuple[float, float, float, float] | None,
    roi: dict[str, Any],
) -> PeakRoiSlice | None:
    """Extract the rectangular ROI described by a peak record."""

    if axis_ranges is None:
        return None
    array = np.asarray(image, dtype=float)
    if array.ndim != 2:
        return None
    values = (
        roi.get("qxy_min"),
        roi.get("qxy_max"),
        roi.get("qz_min"),
        roi.get("qz_max"),
    )
    if any(value is None for value in values):
        return None
    qxy_min, qxy_max = sorted((float(values[0]), float(values[1])))
    qz_min, qz_max = sorted((float(values[2]), float(values[3])))
    qxy_axis, qz_axis = image_axes(array.shape, axis_ranges)
    qxy_mask = (qxy_axis >= qxy_min) & (qxy_axis <= qxy_max)
    qz_mask = (qz_axis >= qz_min) & (qz_axis <= qz_max)
    if not np.any(qxy_mask) or not np.any(qz_mask):
        return None
    intensity = array[np.ix_(qz_mask, qxy_mask)]
    if intensity.size == 0 or not np.isfinite(intensity).any():
        return None
    return PeakRoiSlice(
        qxy=np.asarray(qxy_axis[qxy_mask], dtype=float),
        qz=np.asarray(qz_axis[qz_mask], dtype=float),
        intensity=np.asarray(intensity, dtype=float),
    )


def compute_peak_fit_integrations(
    image: Any,
    axis_ranges: tuple[float, float, float, float] | None,
    roi: dict[str, Any],
    *,
    azimuthal_roi: dict[str, Any] | None = None,
    azimuth_bins: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute qxy, qz, and azimuthal integrations for one peak ROI."""

    sliced = slice_peak_roi(image, axis_ranges, roi)
    if sliced is None:
        return {}
    fit_intensity, gap_mask = _gap_filtered_intensity(sliced.intensity)
    qxy_profile = _finite_sum_profile(fit_intensity, axis=0)
    qz_profile = _finite_sum_profile(fit_intensity, axis=1)
    integrations = {
        "qxy": _profile_dict(
            "qxy",
            "qxy",
            "Integrated intensity",
            sliced.qxy,
            qxy_profile,
        ),
        "qz": _profile_dict(
            "qz",
            "qz",
            "Integrated intensity",
            sliced.qz,
            qz_profile,
        ),
    }
    _annotate_gap_profile(integrations["qxy"], gap_mask, axis=0)
    _annotate_gap_profile(integrations["qz"], gap_mask, axis=1)
    azimuthal = None
    if azimuthal_roi is not None:
        azimuthal = _arch_azimuthal_profile(
            image,
            axis_ranges,
            azimuthal_roi,
            azimuth_bins=azimuth_bins,
        )
    if azimuthal is None:
        azimuthal = _azimuthal_profile(sliced, azimuth_bins=azimuth_bins)
    if azimuthal is not None:
        integrations["azimuthal"] = azimuthal
    return integrations


def fit_peak_integration(
    integration: dict[str, Any],
) -> dict[str, Any] | None:
    """Fit a single 1D integration profile with a Gaussian plus
    offset."""

    x_values = np.asarray(integration.get("x_values", []), dtype=float)
    y_values = np.asarray(integration.get("y_values", []), dtype=float)
    fit = _fit_gaussian_1d(x_values, y_values)
    if fit is None:
        return None
    fit["integration"] = integration.get("name", "")
    fit["x_label"] = integration.get("x_label", "")
    return fit


def fit_peak_integrations(
    integrations: dict[str, dict[str, Any]],
    *,
    names: tuple[str, ...] = PEAK_FIT_INTEGRATIONS,
) -> dict[str, dict[str, Any]]:
    """Fit each requested integration profile."""

    fits: dict[str, dict[str, Any]] = {}
    for name in names:
        integration = integrations.get(name)
        if integration is None:
            continue
        fit = fit_peak_integration(integration)
        if fit is not None:
            fits[name] = fit
    return fits


def fit_peak_roi_2d(
    image: Any,
    axis_ranges: tuple[float, float, float, float] | None,
    roi: dict[str, Any],
    integration_fits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Fit the full ROI with a 2D Gaussian seeded from 1D fits."""

    sliced = slice_peak_roi(image, axis_ranges, roi)
    if sliced is None:
        return None
    qxy_grid, qz_grid, intensity = sliced.as_mesh()
    fit_intensity, gap_mask = _gap_filtered_intensity(intensity)
    finite = (
        np.isfinite(qxy_grid)
        & np.isfinite(qz_grid)
        & np.isfinite(fit_intensity)
    )
    if np.count_nonzero(finite) < 6:
        return None

    seed = _seed_2d_from_profiles(sliced, integration_fits or {})
    x_flat = qxy_grid[finite].ravel()
    y_flat = qz_grid[finite].ravel()
    z_flat = fit_intensity[finite].ravel()
    qxy_step = _axis_step(sliced.qxy)
    qz_step = _axis_step(sliced.qz)
    qxy_range = max(
        float(np.nanmax(sliced.qxy) - np.nanmin(sliced.qxy)), qxy_step
    )
    qz_range = max(float(np.nanmax(sliced.qz) - np.nanmin(sliced.qz)), qz_step)
    z_min = float(np.nanmin(z_flat))
    z_max = float(np.nanmax(z_flat))
    z_span = max(z_max - z_min, 1.0e-12)
    p0 = [
        max(float(seed["amplitude"]), 1.0e-12),
        float(seed["center_qxy"]),
        float(seed["center_qz"]),
        max(float(seed["sigma_qxy"]), qxy_step),
        max(float(seed["sigma_qz"]), qz_step),
        float(seed["offset"]),
    ]
    lower = [
        0.0,
        float(np.nanmin(sliced.qxy)),
        float(np.nanmin(sliced.qz)),
        max(qxy_step / 10.0, 1.0e-12),
        max(qz_step / 10.0, 1.0e-12),
        z_min - 2.0 * z_span,
    ]
    upper = [
        np.inf,
        float(np.nanmax(sliced.qxy)),
        float(np.nanmax(sliced.qz)),
        max(qxy_range * 2.0, qxy_step),
        max(qz_range * 2.0, qz_step),
        z_max + 2.0 * z_span,
    ]
    status = "estimated"
    message = "Used seeded moment estimate."
    parameters = p0
    covariance_error: float | None = None
    try:
        from scipy.optimize import curve_fit

        parameters, covariance = curve_fit(
            _gaussian_2d_curve,
            (x_flat, y_flat),
            z_flat,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
        covariance_error = float(np.nanmean(np.diag(covariance)))
        status = "fit"
        message = "Converged."
    except Exception as exc:
        message = f"Estimated from 1D fits; optimizer did not converge: {exc}"

    fit_values = _gaussian_2d_curve((x_flat, y_flat), *parameters)
    statistics = _fit_statistics(z_flat, fit_values)
    if covariance_error is not None and np.isfinite(covariance_error):
        statistics["mean_parameter_variance"] = covariance_error
    amplitude, center_qxy, center_qz, sigma_qxy, sigma_qz, offset = (
        float(value) for value in parameters
    )
    return {
        "model_name": "seeded-convolved-2d-gaussian",
        "expression": TWO_D_GAUSSIAN_EXPRESSION,
        "status": status,
        "message": message,
        "center_qxy": center_qxy,
        "center_qz": center_qz,
        "amplitude": amplitude,
        "sigma_qxy": abs(sigma_qxy),
        "sigma_qz": abs(sigma_qz),
        "width_qxy_fwhm": abs(sigma_qxy) * GAUSSIAN_FWHM_FACTOR,
        "width_qz_fwhm": abs(sigma_qz) * GAUSSIAN_FWHM_FACTOR,
        "offset": offset,
        "statistics": statistics,
        "metadata": {
            "seed": seed,
            "masked_gap_pixels": int(np.count_nonzero(gap_mask)),
        },
    }


def evaluate_peak_fit_2d(
    image: Any,
    axis_ranges: tuple[float, float, float, float] | None,
    roi: dict[str, Any],
    fit: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return qxy, qz, observed intensity, and model grids for
    plotting."""

    sliced = slice_peak_roi(image, axis_ranges, roi)
    if sliced is None:
        return None
    qxy_grid, qz_grid, intensity = sliced.as_mesh()
    model = _gaussian_2d_curve(
        (qxy_grid, qz_grid),
        float(fit.get("amplitude", 0.0)),
        float(fit.get("center_qxy", 0.0)),
        float(fit.get("center_qz", 0.0)),
        max(abs(float(fit.get("sigma_qxy", 1.0))), 1.0e-12),
        max(abs(float(fit.get("sigma_qz", 1.0))), 1.0e-12),
        float(fit.get("offset", 0.0)),
    )
    return qxy_grid, qz_grid, intensity, np.asarray(model, dtype=float)


def _gap_filtered_intensity(
    intensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(intensity, dtype=float)
    gap_mask = _masked_gap_array_mask(array)
    if not np.any(gap_mask):
        return array, gap_mask
    filtered = np.array(array, dtype=float, copy=True)
    filtered[gap_mask] = np.nan
    return filtered, gap_mask


def _masked_gap_array_mask(intensity: np.ndarray) -> np.ndarray:
    array = np.asarray(intensity, dtype=float)
    finite = np.isfinite(array)
    if not np.any(finite):
        return ~finite
    floor = float(np.nanmin(array[finite]))
    ceiling = float(np.nanmax(array[finite]))
    dynamic_range = max(ceiling - floor, 0.0)
    tolerance = max(abs(floor) * 0.02, dynamic_range * 0.005, 1.0e-9)
    floor_mask = finite & (array <= floor + tolerance)
    mask = ~finite
    if array.ndim != 2 or not np.any(floor_mask):
        return mask
    row_floor_fraction = np.mean(floor_mask, axis=1)
    col_floor_fraction = np.mean(floor_mask, axis=0)
    row_gaps = row_floor_fraction >= 0.85
    col_gaps = col_floor_fraction >= 0.85
    if np.any(row_gaps):
        mask |= row_gaps[:, None]
    if np.any(col_gaps):
        mask |= col_gaps[None, :]
    return mask


def _finite_sum_profile(intensity: np.ndarray, *, axis: int) -> np.ndarray:
    finite = np.isfinite(intensity)
    values = np.where(finite, intensity, 0.0)
    counts = np.sum(finite, axis=axis)
    summed = np.sum(values, axis=axis)
    return np.where(counts > 0, summed, np.nan)


def _annotate_gap_profile(
    profile: dict[str, Any],
    gap_mask: np.ndarray,
    *,
    axis: int,
) -> None:
    if not np.any(gap_mask):
        return
    counts = np.sum(gap_mask, axis=axis)
    gap_bins = np.flatnonzero(counts > 0)
    if not gap_bins.size:
        return
    profile["metadata"] = {
        "masked_gap_bins": [int(value) for value in gap_bins],
        "masked_gap_pixel_count": int(np.count_nonzero(gap_mask)),
    }


def _profile_dict(
    name: str,
    x_label: str,
    y_label: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> dict[str, Any]:
    return {
        "name": name,
        "x_label": x_label,
        "y_label": y_label,
        "x_values": [float(value) for value in np.asarray(x_values)],
        "y_values": [float(value) for value in np.asarray(y_values)],
    }


def _azimuthal_profile(
    sliced: PeakRoiSlice,
    *,
    azimuth_bins: int | None,
) -> dict[str, Any] | None:
    qxy_grid, qz_grid, intensity = sliced.as_mesh()
    finite = (
        np.isfinite(qxy_grid) & np.isfinite(qz_grid) & np.isfinite(intensity)
    )
    if not np.any(finite):
        return None
    chi = np.degrees(np.arctan2(qxy_grid[finite], qz_grid[finite]))
    values = intensity[finite]
    if not chi.size:
        return None
    chi_min = float(np.nanmin(chi))
    chi_max = float(np.nanmax(chi))
    if not np.isfinite(chi_min) or not np.isfinite(chi_max):
        return None
    if chi_min == chi_max:
        chi_min -= 0.5
        chi_max += 0.5
    bin_count = azimuth_bins or max(
        12,
        min(180, int(np.ceil(np.sqrt(float(values.size)) * 2.0))),
    )
    edges = np.linspace(chi_min, chi_max, int(bin_count) + 1)
    bin_index = np.digitize(chi, edges) - 1
    valid = (bin_index >= 0) & (bin_index < int(bin_count))
    if not np.any(valid):
        return None
    integrated = np.bincount(
        bin_index[valid],
        weights=np.nan_to_num(values[valid], nan=0.0),
        minlength=int(bin_count),
    )
    counts = np.bincount(bin_index[valid], minlength=int(bin_count))
    integrated = np.where(counts > 0, integrated, np.nan)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return _profile_dict(
        "azimuthal",
        "chi (deg)",
        "Integrated intensity",
        centers,
        integrated,
    )


def _arch_azimuthal_profile(
    image: Any,
    axis_ranges: tuple[float, float, float, float] | None,
    roi: dict[str, Any],
    *,
    azimuth_bins: int | None,
) -> dict[str, Any] | None:
    if axis_ranges is None:
        return None
    array = np.asarray(image, dtype=float)
    if array.ndim != 2:
        return None
    values = (
        roi.get("qr_min"),
        roi.get("qr_max"),
        roi.get("chi_min"),
        roi.get("chi_max"),
    )
    if any(value is None for value in values):
        return None
    qr_min, qr_max = sorted((float(values[0]), float(values[1])))
    chi_min, chi_max = sorted((float(values[2]), float(values[3])))
    if qr_max <= qr_min or chi_max <= chi_min:
        return None
    qxy_axis, qz_axis = image_axes(array.shape, axis_ranges)
    qxy_grid, qz_grid = np.meshgrid(qxy_axis, qz_axis)
    center_qxy = float(roi.get("qxy_center", 0.0))
    center_qz = float(roi.get("qz_center", 0.0))
    qxy_relative = qxy_grid - center_qxy
    qz_relative = qz_grid - center_qz
    radius = np.hypot(qxy_relative, qz_relative)
    chi = np.degrees(np.arctan2(qxy_relative, qz_relative))
    mask = (
        (radius >= qr_min)
        & (radius <= qr_max)
        & (chi >= chi_min)
        & (chi <= chi_max)
        & np.isfinite(array)
    )
    if not np.any(mask):
        return None
    bin_count = azimuth_bins or max(
        12,
        min(180, int(np.ceil(abs(chi_max - chi_min)))),
    )
    edges = np.linspace(chi_min, chi_max, int(bin_count) + 1)
    bin_index = np.digitize(chi[mask], edges) - 1
    valid = (bin_index >= 0) & (bin_index < int(bin_count))
    if not np.any(valid):
        return None
    integrated = np.bincount(
        bin_index[valid],
        weights=np.nan_to_num(array[mask][valid], nan=0.0),
        minlength=int(bin_count),
    )
    counts = np.bincount(bin_index[valid], minlength=int(bin_count))
    integrated = np.where(counts > 0, integrated, np.nan)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return _profile_dict(
        "azimuthal",
        "chi (deg)",
        "Integrated intensity",
        centers,
        integrated,
    )


def _fit_gaussian_1d(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> dict[str, Any] | None:
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(finite) < 4:
        return None
    x = np.asarray(x_values[finite], dtype=float)
    y = np.asarray(y_values[finite], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_range = max(float(np.nanmax(x) - np.nanmin(x)), _axis_step(x))
    x_step = _axis_step(x)
    estimate = _moment_1d_seed(x, y)
    p0 = [
        max(float(estimate["amplitude"]), 1.0e-12),
        float(estimate["center"]),
        max(float(estimate["sigma"]), x_step),
        float(estimate["offset"]),
    ]
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_span = max(y_max - y_min, 1.0e-12)
    lower = [
        0.0,
        float(np.nanmin(x)),
        max(x_step / 10.0, 1.0e-12),
        y_min - 2.0 * y_span,
    ]
    upper = [
        np.inf,
        float(np.nanmax(x)),
        max(2.0 * x_range, x_step),
        y_max + 2.0 * y_span,
    ]
    status = "estimated"
    message = "Used moment estimate."
    parameters = p0
    try:
        from scipy.optimize import curve_fit

        parameters, _covariance = curve_fit(
            _gaussian_1d,
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=10000,
        )
        status = "fit"
        message = "Converged."
    except Exception as exc:
        message = f"Estimated; optimizer did not converge: {exc}"
    fit_values = _gaussian_1d(x, *parameters)
    statistics = _fit_statistics(y, fit_values)
    amplitude, center, sigma, offset = (float(value) for value in parameters)
    return {
        "model_name": "gaussian-1d",
        "status": status,
        "message": message,
        "amplitude": amplitude,
        "center": center,
        "sigma": abs(sigma),
        "width_fwhm": abs(sigma) * GAUSSIAN_FWHM_FACTOR,
        "offset": offset,
        "statistics": statistics,
    }


def _moment_1d_seed(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    baseline = float(np.nanpercentile(y, 10.0))
    weights = np.clip(y - baseline, 0.0, None)
    if np.nansum(weights) <= 0.0:
        center = float(x[int(np.nanargmax(y))])
        sigma = max(_axis_step(x), float(np.nanstd(x)))
    else:
        center = float(np.nansum(x * weights) / np.nansum(weights))
        variance = float(
            np.nansum(weights * (x - center) ** 2) / np.nansum(weights)
        )
        sigma = max(np.sqrt(max(variance, 0.0)), _axis_step(x))
    amplitude = max(float(np.nanmax(y) - baseline), 1.0e-12)
    return {
        "amplitude": amplitude,
        "center": center,
        "sigma": sigma,
        "offset": baseline,
    }


def _seed_2d_from_profiles(
    sliced: PeakRoiSlice,
    integration_fits: dict[str, dict[str, Any]],
) -> dict[str, float]:
    intensity = np.nan_to_num(sliced.intensity, nan=0.0)
    baseline = float(np.nanpercentile(intensity, 10.0))
    weights = np.clip(intensity - baseline, 0.0, None)
    qxy_grid, qz_grid, _ = sliced.as_mesh()
    if np.nansum(weights) <= 0.0:
        max_index = np.unravel_index(
            int(np.nanargmax(sliced.intensity)),
            sliced.intensity.shape,
        )
        center_qxy = float(qxy_grid[max_index])
        center_qz = float(qz_grid[max_index])
        sigma_qxy = max(_axis_step(sliced.qxy), float(np.nanstd(sliced.qxy)))
        sigma_qz = max(_axis_step(sliced.qz), float(np.nanstd(sliced.qz)))
    else:
        center_qxy = float(np.nansum(qxy_grid * weights) / np.nansum(weights))
        center_qz = float(np.nansum(qz_grid * weights) / np.nansum(weights))
        sigma_qxy = max(
            np.sqrt(
                float(
                    np.nansum(weights * (qxy_grid - center_qxy) ** 2)
                    / np.nansum(weights)
                )
            ),
            _axis_step(sliced.qxy),
        )
        sigma_qz = max(
            np.sqrt(
                float(
                    np.nansum(weights * (qz_grid - center_qz) ** 2)
                    / np.nansum(weights)
                )
            ),
            _axis_step(sliced.qz),
        )

    qxy_fit = integration_fits.get("qxy", {})
    qz_fit = integration_fits.get("qz", {})
    azimuthal_fit = integration_fits.get("azimuthal", {})
    if qxy_fit.get("center") is not None:
        center_qxy = float(qxy_fit["center"])
    if qz_fit.get("center") is not None:
        center_qz = float(qz_fit["center"])
    if qxy_fit.get("sigma") is not None:
        sigma_qxy = max(abs(float(qxy_fit["sigma"])), _axis_step(sliced.qxy))
    if qz_fit.get("sigma") is not None:
        sigma_qz = max(abs(float(qz_fit["sigma"])), _axis_step(sliced.qz))

    if azimuthal_fit.get("center") is not None:
        chi_rad = np.radians(float(azimuthal_fit["center"]))
        radius = max(np.hypot(center_qxy, center_qz), 1.0e-12)
        az_qxy = float(radius * np.sin(chi_rad))
        az_qz = float(radius * np.cos(chi_rad))
        center_qxy = (center_qxy + az_qxy) / 2.0
        center_qz = (center_qz + az_qz) / 2.0
        chi_sigma = abs(float(azimuthal_fit.get("sigma", 0.0)))
        chi_sigma_rad = np.radians(chi_sigma)
        sigma_qxy = np.sqrt(sigma_qxy**2 + (radius * chi_sigma_rad) ** 2)
        sigma_qz = np.sqrt(sigma_qz**2 + (radius * chi_sigma_rad) ** 2)

    return {
        "amplitude": max(float(np.nanmax(intensity) - baseline), 1.0e-12),
        "center_qxy": float(
            np.clip(center_qxy, np.nanmin(sliced.qxy), np.nanmax(sliced.qxy))
        ),
        "center_qz": float(
            np.clip(center_qz, np.nanmin(sliced.qz), np.nanmax(sliced.qz))
        ),
        "sigma_qxy": max(float(sigma_qxy), _axis_step(sliced.qxy)),
        "sigma_qz": max(float(sigma_qz), _axis_step(sliced.qz)),
        "offset": baseline,
    }


def _axis_step(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 1.0e-9
    diffs = np.diff(np.sort(values[np.isfinite(values)]))
    diffs = np.abs(diffs[diffs > 0.0])
    if diffs.size == 0:
        return 1.0e-9
    return max(float(np.nanmedian(diffs)), 1.0e-9)


def _gaussian_1d(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    offset: float,
) -> np.ndarray:
    sigma = max(abs(float(sigma)), 1.0e-12)
    return offset + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _gaussian_2d_curve(
    coordinates: tuple[np.ndarray, np.ndarray],
    amplitude: float,
    center_qxy: float,
    center_qz: float,
    sigma_qxy: float,
    sigma_qz: float,
    offset: float,
) -> np.ndarray:
    qxy, qz = coordinates
    sigma_qxy = max(abs(float(sigma_qxy)), 1.0e-12)
    sigma_qz = max(abs(float(sigma_qz)), 1.0e-12)
    exponent = ((qxy - center_qxy) / sigma_qxy) ** 2 + (
        (qz - center_qz) / sigma_qz
    ) ** 2
    return offset + amplitude * np.exp(-0.5 * exponent)


def _fit_statistics(
    observed: np.ndarray, fitted: np.ndarray
) -> dict[str, float]:
    residual = np.asarray(observed, dtype=float) - np.asarray(
        fitted, dtype=float
    )
    finite = np.isfinite(residual) & np.isfinite(observed)
    if not np.any(finite):
        return {}
    residual = residual[finite]
    observed = np.asarray(observed, dtype=float)[finite]
    ss_res = float(np.nansum(residual**2))
    ss_tot = float(np.nansum((observed - np.nanmean(observed)) ** 2))
    observed_norm = float(np.nansum(observed**2))
    return {
        "rmse": float(np.sqrt(np.nanmean(residual**2))),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0,
        "r_w": (
            float(np.sqrt(ss_res / observed_norm))
            if observed_norm > 0.0
            else 0.0
        ),
        "sum_squared_error": ss_res,
    }
