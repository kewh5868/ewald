"""Reusable GIWAXS simulation backend."""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

from ewald.data.models import ProjectState
from ewald.simulation.legacy import WAXSAFF

PEAK_TABLE_ATTR = "peak_table_json"
SIMULATION_MODE_ATTR = "simulation_mode"
SIMULATION_MODE_PATTERN = "giwaxs_pattern"
SIMULATION_MODE_EWALD_SWEEP = "ewald_sphere_sweep"


@dataclass(slots=True)
class GIWAXSSimulationRequest:
    """Inputs required to simulate a GIWAXS pattern."""

    cif_path: Path
    qxy_range: tuple[float, float] = (-3.0, 3.0)
    qz_range: tuple[float, float] = (0.0, 3.0)
    detector_shape: tuple[int, int] = (128, 256)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GIWAXSSimulationParameters:
    """Numerical controls for a GIWAXS simulation."""

    sigma_theta: float = 0.03
    sigma_phi: float = 0.25
    sigma_r: float = 0.035
    hkl_extent: int = 4
    theta_x_deg: float = 90.0
    theta_y_deg: float = 0.0
    qxy_min: float = -3.0
    qxy_max: float = 3.0
    qz_min: float = 0.0
    qz_max: float = 3.0
    resolution_x: int = 256
    resolution_z: int = 128

    def as_dict(self) -> dict[str, Any]:
        return {
            "sigma_theta": self.sigma_theta,
            "sigma_phi": self.sigma_phi,
            "sigma_r": self.sigma_r,
            "hkl_extent": self.hkl_extent,
            "theta_x_deg": self.theta_x_deg,
            "theta_y_deg": self.theta_y_deg,
            "qxy_min": self.qxy_min,
            "qxy_max": self.qxy_max,
            "qz_min": self.qz_min,
            "qz_max": self.qz_max,
            "resolution_x": self.resolution_x,
            "resolution_z": self.resolution_z,
        }

    @classmethod
    def from_mapping(
        cls, payload: dict[str, Any]
    ) -> "GIWAXSSimulationParameters":
        values = cls().as_dict()
        values.update({key: payload[key] for key in values if key in payload})
        values["hkl_extent"] = int(values["hkl_extent"])
        values["resolution_x"] = int(values["resolution_x"])
        values["resolution_z"] = int(values["resolution_z"])
        return cls(**values)


@dataclass(slots=True)
class EwaldSphereSweepParameters(GIWAXSSimulationParameters):
    """Numerical controls for a low-resolution theta sweep."""

    resolution_x: int = 96
    resolution_z: int = 64
    theta_x_min_deg: float = 0.0
    theta_x_max_deg: float = 180.0
    theta_x_step_deg: float = 15.0
    theta_y_min_deg: float = 0.0
    theta_y_max_deg: float = 345.0
    theta_y_step_deg: float = 15.0

    def as_dict(self) -> dict[str, Any]:
        payload = GIWAXSSimulationParameters.as_dict(self)
        payload.update(
            {
                "theta_x_min_deg": self.theta_x_min_deg,
                "theta_x_max_deg": self.theta_x_max_deg,
                "theta_x_step_deg": self.theta_x_step_deg,
                "theta_y_min_deg": self.theta_y_min_deg,
                "theta_y_max_deg": self.theta_y_max_deg,
                "theta_y_step_deg": self.theta_y_step_deg,
            }
        )
        return payload

    @classmethod
    def from_mapping(
        cls, payload: dict[str, Any]
    ) -> "EwaldSphereSweepParameters":
        values = cls().as_dict()
        values.update({key: payload[key] for key in values if key in payload})
        values["hkl_extent"] = int(values["hkl_extent"])
        values["resolution_x"] = int(values["resolution_x"])
        values["resolution_z"] = int(values["resolution_z"])
        return cls(**values)

    def frame_parameters(
        self,
        theta_x_deg: float,
        theta_y_deg: float,
    ) -> GIWAXSSimulationParameters:
        """Return single-pattern parameters for one sweep orientation."""

        return GIWAXSSimulationParameters(
            sigma_theta=self.sigma_theta,
            sigma_phi=self.sigma_phi,
            sigma_r=self.sigma_r,
            hkl_extent=self.hkl_extent,
            theta_x_deg=float(theta_x_deg),
            theta_y_deg=float(theta_y_deg),
            qxy_min=self.qxy_min,
            qxy_max=self.qxy_max,
            qz_min=self.qz_min,
            qz_max=self.qz_max,
            resolution_x=self.resolution_x,
            resolution_z=self.resolution_z,
        )


@dataclass(slots=True)
class StructureData:
    """Minimal structure representation used by the simulator."""

    path: Path
    lattice: np.ndarray
    species: list[str]
    frac_coords: np.ndarray

    @property
    def structure_id(self) -> str:
        digest = hashlib.sha1(str(self.path).encode("utf-8")).hexdigest()[:10]
        return f"{self.path.stem}_{digest}"

    def metadata(self) -> dict[str, Any]:
        return {
            "structure_id": self.structure_id,
            "structure_name": self.path.stem,
            "structure_path": str(self.path),
            "atom_count": len(self.species),
            "species": sorted(set(self.species)),
        }


@dataclass(slots=True)
class SimulationResult:
    """Simulation output record stored by an EWALD project."""

    data_id: str
    dataset_uri: str
    objective: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_id": self.data_id,
            "dataset_uri": self.dataset_uri,
            "objective": self.objective,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class GIWAXSImageComparison:
    """Aligned target/simulated images and fit-quality metrics."""

    target: xr.DataArray
    simulated: xr.DataArray
    fitted_simulated: xr.DataArray
    difference: xr.DataArray
    metrics: dict[str, Any]
    target_label: str = "Target"
    simulated_label: str = "Simulated"

    @property
    def experimental(self) -> xr.DataArray:
        """Backward-compatible alias for the comparison target image."""

        return self.target


@dataclass(slots=True)
class GIWAXSSimulationFitRecord:
    """One ranked structure/parameter comparison against an experiment."""

    structure_name: str
    structure_path: str
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "structure_name": self.structure_name,
            "structure_path": self.structure_path,
            "parameters": self.parameters,
            "metrics": self.metrics,
        }


def simulate_giwaxs(request: GIWAXSSimulationRequest) -> SimulationResult:
    """Run a GIWAXS simulation and return an in-memory result record."""

    parameters = GIWAXSSimulationParameters.from_mapping(
        {
            **request.parameters,
            "qxy_min": request.qxy_range[0],
            "qxy_max": request.qxy_range[1],
            "qz_min": request.qz_range[0],
            "qz_max": request.qz_range[1],
            "resolution_z": request.detector_shape[0],
            "resolution_x": request.detector_shape[1],
        }
    )
    data_array = simulate_giwaxs_image(request.cif_path, parameters)
    return SimulationResult(
        data_id=request.cif_path.stem,
        dataset_uri=f"simulations/{request.cif_path.stem}.nc",
        parameters={**parameters.as_dict(), **dict(data_array.attrs)},
    )


def compare_giwaxs_images(
    target: xr.DataArray | np.ndarray,
    simulated: xr.DataArray | np.ndarray,
    *,
    low_quantile: float = 1.0,
    high_quantile: float = 99.5,
    peak_weight: float = 4.0,
    target_label: str | None = None,
    simulated_label: str | None = None,
) -> GIWAXSImageComparison:
    """Fit and compare two q-space GIWAXS images on one detector grid.

    The simulated image is interpolated onto the target qxy/qz grid, both
    images are robustly normalized, and the simulated image is fit with a
    scale/offset term before residual metrics are calculated.
    """

    target_da = _standardize_qspace_image(target, "target")
    simulated_da = _standardize_qspace_image(simulated, "simulated")
    resolved_target_label = target_label or _comparison_image_label(
        target_da,
        fallback="Target",
    )
    resolved_simulated_label = simulated_label or _comparison_image_label(
        simulated_da,
        fallback="Simulated",
    )
    simulated_on_grid = _interpolate_to_grid(simulated_da, target_da)
    target_values = _robust_metric_image(
        np.asarray(target_da.values, dtype=float),
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )
    simulated_values = _robust_metric_image(
        np.asarray(simulated_on_grid.values, dtype=float),
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )
    scale, offset = _fit_image_scale_offset(
        target_values,
        simulated_values,
    )
    fitted_values = scale * simulated_values + offset
    difference_values = target_values - fitted_values
    metrics = _image_fit_metrics(
        target_values,
        fitted_values,
        difference_values,
        scale=scale,
        offset=offset,
        peak_weight=peak_weight,
    )
    fitted = xr.DataArray(
        fitted_values,
        dims=target_da.dims,
        coords=target_da.coords,
        name="fitted_simulated_intensity",
        attrs={"scale": scale, "offset": offset},
    )
    difference = xr.DataArray(
        difference_values,
        dims=target_da.dims,
        coords=target_da.coords,
        name="normalized_difference",
        attrs=metrics,
    )
    return GIWAXSImageComparison(
        target=xr.DataArray(
            target_values,
            dims=target_da.dims,
            coords=target_da.coords,
            name="normalized_target_intensity",
            attrs={
                **dict(target_da.attrs),
                "comparison_label": resolved_target_label,
            },
        ),
        simulated=simulated_on_grid,
        fitted_simulated=fitted,
        difference=difference,
        metrics=metrics,
        target_label=resolved_target_label,
        simulated_label=resolved_simulated_label,
    )


def rank_giwaxs_simulation_fits(
    target: xr.DataArray | np.ndarray,
    structures: Iterable[str | Path | tuple[str, str | Path]],
    parameter_grid: Iterable[GIWAXSSimulationParameters | dict[str, Any]],
) -> list[GIWAXSSimulationFitRecord]:
    """Rank structures and simulation parameters against a target image."""

    ranked: list[GIWAXSSimulationFitRecord] = []
    for structure_name, structure_path in _named_structure_paths(structures):
        for parameters in parameter_grid:
            params = (
                parameters
                if isinstance(parameters, GIWAXSSimulationParameters)
                else GIWAXSSimulationParameters.from_mapping(dict(parameters))
            )
            simulated = simulate_giwaxs_image(structure_path, params)
            comparison = compare_giwaxs_images(target, simulated)
            ranked.append(
                GIWAXSSimulationFitRecord(
                    structure_name=structure_name,
                    structure_path=str(structure_path),
                    parameters=params.as_dict(),
                    metrics=dict(comparison.metrics),
                )
            )
    ranked.sort(key=lambda item: _fit_record_sort_key(item.metrics))
    for index, record in enumerate(ranked, start=1):
        record.rank = index
    return ranked


def _named_structure_paths(
    structures: Iterable[str | Path | tuple[str, str | Path]],
) -> list[tuple[str, Path]]:
    named: list[tuple[str, Path]] = []
    for item in structures:
        if isinstance(item, tuple):
            name, path = item
            path_obj = Path(path)
            named.append((str(name), path_obj))
        else:
            path_obj = Path(item)
            named.append((path_obj.stem, path_obj))
    return named


def _fit_record_sort_key(
    metrics: dict[str, Any],
) -> tuple[float, float, float]:
    fit_score = float(
        metrics.get(
            "difference_rmse",
            metrics.get("fit_score", float("inf")),
        )
    )
    weighted_rmse = float(metrics.get("weighted_rmse", float("inf")))
    correlation = float(metrics.get("correlation", 0.0))
    return (fit_score, weighted_rmse, -correlation)


def save_giwaxs_comparison_plot(
    comparison: GIWAXSImageComparison,
    output_path: str | Path,
    *,
    title: str = "",
    target_label: str | None = None,
    simulated_label: str | None = None,
) -> Path:
    """Save target, fitted simulation, and residual maps."""

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - dependency guard.
        raise RuntimeError(
            "matplotlib is required to save comparison plots."
        ) from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = (
        comparison.target,
        comparison.fitted_simulated,
        comparison.difference,
    )
    target_title = target_label or comparison.target_label
    simulated_title = simulated_label or f"Fitted {comparison.simulated_label}"
    labels = (target_title, simulated_title, "Difference")
    cmaps = ("viridis", "viridis", "coolwarm")
    figure, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.2), constrained_layout=True
    )
    for axis, data_array, label, cmap in zip(axes, arrays, labels, cmaps):
        image = np.asarray(data_array.values, dtype=float)
        extent = _image_extent(data_array)
        if label == "Difference":
            vmax = float(np.nanquantile(np.abs(image), 0.99))
            vmin = -vmax if np.isfinite(vmax) and vmax > 0.0 else None
            vmax = vmax if np.isfinite(vmax) and vmax > 0.0 else None
        else:
            vmin = 0.0
            vmax = float(np.nanquantile(image[np.isfinite(image)], 0.995))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = None
        artist = axis.imshow(
            image,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(label)
        axis.set_xlabel(r"$q_{xy}$ ($\AA^{-1}$)")
        axis.set_ylabel(r"$q_z$ ($\AA^{-1}$)")
        figure.colorbar(artist, ax=axis, shrink=0.8)
    score = comparison.metrics.get("difference_rmse")
    if score is None:
        score = comparison.metrics.get("fit_score")
    subtitle = (
        f"difference RMSE {score:.4g}" if isinstance(score, float) else ""
    )
    figure.suptitle(" | ".join(item for item in (title, subtitle) if item))
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _standardize_qspace_image(
    image: xr.DataArray | np.ndarray,
    name: str,
) -> xr.DataArray:
    if isinstance(image, xr.DataArray):
        data = image
        rename: dict[str, str] = {}
        if "q_ip" in data.dims:
            rename["q_ip"] = "qxy"
        if "q_oop" in data.dims:
            rename["q_oop"] = "qz"
        if rename:
            data = data.rename(rename)
        if not {"qz", "qxy"}.issubset(data.dims):
            if data.ndim != 2:
                raise ValueError("GIWAXS comparison expects 2D images.")
            data = xr.DataArray(
                np.asarray(data.values, dtype=float),
                dims=("qz", "qxy"),
                coords={
                    "qz": np.arange(data.shape[0], dtype=float),
                    "qxy": np.arange(data.shape[1], dtype=float),
                },
                name=name,
            )
        return data.transpose("qz", "qxy")
    values = np.asarray(image, dtype=float)
    if values.ndim != 2:
        raise ValueError("GIWAXS comparison expects 2D images.")
    return xr.DataArray(
        values,
        dims=("qz", "qxy"),
        coords={
            "qz": np.arange(values.shape[0], dtype=float),
            "qxy": np.arange(values.shape[1], dtype=float),
        },
        name=name,
    )


def _comparison_image_label(data: xr.DataArray, *, fallback: str) -> str:
    label = data.attrs.get("comparison_label")
    if label:
        return str(label)
    if data.attrs.get(SIMULATION_MODE_ATTR):
        return "Simulated target" if fallback == "Target" else "Simulation"
    name = str(data.name or "").lower()
    if "experimental" in name:
        return "Experimental data"
    return fallback


def _interpolate_to_grid(
    source: xr.DataArray,
    target: xr.DataArray,
) -> xr.DataArray:
    try:
        interpolated = source.interp(
            qz=target.coords["qz"],
            qxy=target.coords["qxy"],
            kwargs={"fill_value": 0.0},
        )
    except Exception:
        if source.shape != target.shape:
            raise
        interpolated = xr.DataArray(
            np.asarray(source.values, dtype=float),
            dims=target.dims,
            coords=target.coords,
            name=source.name,
        )
    return interpolated.fillna(0.0).transpose("qz", "qxy")


def _robust_metric_image(
    values: np.ndarray,
    *,
    low_quantile: float,
    high_quantile: float,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return np.zeros_like(array, dtype=float)
    low = float(np.nanpercentile(finite, low_quantile))
    high = float(np.nanpercentile(finite, high_quantile))
    if not np.isfinite(high) or high <= low:
        high = float(np.nanmax(finite))
    scale = max(high - low, 1.0e-12)
    normalized = np.clip(
        (np.nan_to_num(array, nan=0.0) - low) / scale, 0.0, None
    )
    return np.sqrt(normalized)


def _fit_image_scale_offset(
    experimental: np.ndarray,
    simulated: np.ndarray,
) -> tuple[float, float]:
    mask = np.isfinite(experimental) & np.isfinite(simulated)
    if not np.any(mask):
        return 0.0, 0.0
    x = simulated[mask].ravel()
    y = experimental[mask].ravel()
    design = np.column_stack([x, np.ones_like(x)])
    try:
        scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        scale, offset = 1.0, 0.0
    if not np.isfinite(scale) or scale < 0.0:
        scale = 0.0
    if not np.isfinite(offset):
        offset = 0.0
    return float(scale), float(offset)


def _image_fit_metrics(
    experimental: np.ndarray,
    fitted: np.ndarray,
    difference: np.ndarray,
    *,
    scale: float,
    offset: float,
    peak_weight: float,
) -> dict[str, Any]:
    mask = (
        np.isfinite(experimental)
        & np.isfinite(fitted)
        & np.isfinite(difference)
    )
    if not np.any(mask):
        return {
            "fit_score": float("inf"),
            "difference_rmse": float("inf"),
            "difference_mae": float("inf"),
            "difference_max_abs": float("inf"),
            "rmse": float("inf"),
            "weighted_rmse": float("inf"),
            "mae": float("inf"),
            "correlation": 0.0,
            "scale": float(scale),
            "offset": float(offset),
            "pixel_count": 0,
        }
    exp = experimental[mask].ravel()
    fit = fitted[mask].ravel()
    diff = difference[mask].ravel()
    weights = 1.0 + max(float(peak_weight), 0.0) * np.clip(exp, 0.0, None)
    rmse = float(np.sqrt(np.mean(diff**2)))
    weighted_rmse = float(np.sqrt(np.average(diff**2, weights=weights)))
    mae = float(np.mean(np.abs(diff)))
    difference_max_abs = float(np.max(np.abs(diff)))
    exp_std = float(np.std(exp))
    fit_std = float(np.std(fit))
    if exp_std > 1.0e-12 and fit_std > 1.0e-12:
        correlation = float(np.corrcoef(exp, fit)[0, 1])
    else:
        correlation = 0.0
    high_threshold = float(np.nanpercentile(exp, 95.0))
    high_mask = exp >= high_threshold
    peak_rmse = (
        float(np.sqrt(np.mean(diff[high_mask] ** 2)))
        if np.any(high_mask)
        else rmse
    )
    peak_correlation = _masked_correlation(exp, fit, high_mask)
    model_threshold = float(np.nanpercentile(fit, 95.0))
    model_high_mask = fit >= model_threshold
    overlap = np.count_nonzero(high_mask & model_high_mask)
    union = np.count_nonzero(high_mask | model_high_mask)
    model_high_count = np.count_nonzero(model_high_mask)
    peak_high_count = np.count_nonzero(high_mask)
    peak_overlap_jaccard = float(overlap / union) if union else 0.0
    peak_precision = (
        float(overlap / model_high_count) if model_high_count else 0.0
    )
    peak_recall = float(overlap / peak_high_count) if peak_high_count else 0.0
    peak_focus_score = peak_rmse * (2.0 - max(peak_correlation, -1.0)) / 2.0
    peak_focus_score *= 1.0 + (1.0 - peak_overlap_jaccard)
    residual_correlation_score = (
        weighted_rmse * (2.0 - max(correlation, -1.0)) / 2.0
    )
    return {
        "fit_score": rmse,
        "difference_rmse": rmse,
        "difference_mae": mae,
        "difference_max_abs": difference_max_abs,
        "residual_correlation_score": float(residual_correlation_score),
        "peak_focus_score": float(peak_focus_score),
        "rmse": rmse,
        "weighted_rmse": weighted_rmse,
        "peak_rmse": peak_rmse,
        "mae": mae,
        "correlation": correlation,
        "peak_correlation": peak_correlation,
        "peak_overlap_jaccard": peak_overlap_jaccard,
        "peak_precision": peak_precision,
        "peak_recall": peak_recall,
        "scale": float(scale),
        "offset": float(offset),
        "pixel_count": int(exp.size),
        "peak_pixel_count": int(peak_high_count),
    }


def _masked_correlation(
    experimental: np.ndarray,
    fitted: np.ndarray,
    mask: np.ndarray,
) -> float:
    if not np.any(mask):
        return 0.0
    x = experimental[mask].ravel()
    y = fitted[mask].ravel()
    if x.size < 2 or float(np.std(x)) <= 1.0e-12:
        return 0.0
    if float(np.std(y)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _image_extent(data: xr.DataArray) -> tuple[float, float, float, float]:
    qxy = np.asarray(data.coords["qxy"].values, dtype=float)
    qz = np.asarray(data.coords["qz"].values, dtype=float)
    return (
        float(np.nanmin(qxy)),
        float(np.nanmax(qxy)),
        float(np.nanmin(qz)),
        float(np.nanmax(qz)),
    )


def simulate_giwaxs_image(
    structure_path: str | Path,
    parameters: GIWAXSSimulationParameters | None = None,
) -> xr.DataArray:
    """Compute a q_{xy}-q_{z} GIWAXS intensity image from a structure file.

    The implementation is adapted from the legacy pyWAXS simulation logic:
    reciprocal-lattice peaks are generated from a POSCAR/CIF structure, their
    intensities are weighted by the copied atomic form-factor table, and each
    reflection is smeared onto a q-space detector plane with Gaussian terms.
    """

    params = parameters or GIWAXSSimulationParameters()
    structure = load_structure(structure_path)
    return _simulate_giwaxs_image_from_structure(structure, params)


def simulate_ewald_sphere_sweep(
    structure_path: str | Path,
    parameters: EwaldSphereSweepParameters | None = None,
) -> xr.DataArray:
    """Compute a theta_x/theta_y sweep of low-resolution patterns."""

    params = parameters or EwaldSphereSweepParameters()
    structure = load_structure(structure_path)
    theta_x_axis = _angle_axis(
        params.theta_x_min_deg,
        params.theta_x_max_deg,
        params.theta_x_step_deg,
    )
    theta_y_axis = _angle_axis(
        params.theta_y_min_deg,
        params.theta_y_max_deg,
        params.theta_y_step_deg,
    )
    qxy_axis = np.linspace(params.qxy_min, params.qxy_max, params.resolution_x)
    qz_axis = np.linspace(params.qz_min, params.qz_max, params.resolution_z)
    frames = np.zeros(
        (
            theta_y_axis.size,
            theta_x_axis.size,
            params.resolution_z,
            params.resolution_x,
        ),
        dtype=float,
    )
    peak_counts: list[int] = []
    for theta_y_index, theta_y in enumerate(theta_y_axis):
        for theta_x_index, theta_x in enumerate(theta_x_axis):
            frame = _simulate_giwaxs_image_from_structure(
                structure,
                params.frame_parameters(theta_x, theta_y),
            )
            frames[theta_y_index, theta_x_index] = np.asarray(
                frame.values,
                dtype=float,
            )
            peak_counts.append(int(frame.attrs.get("peak_count", 0)))

    attrs = {
        **structure.metadata(),
        **params.as_dict(),
        SIMULATION_MODE_ATTR: SIMULATION_MODE_EWALD_SWEEP,
        "frame_count": int(theta_x_axis.size * theta_y_axis.size),
        "theta_x_count": int(theta_x_axis.size),
        "theta_y_count": int(theta_y_axis.size),
        "peak_count": int(sum(peak_counts)),
        "peak_count_max": int(max(peak_counts, default=0)),
        "legacy_source": "pyWAXS/pywaxs/simulation/WAXSSim.py",
    }
    return xr.DataArray(
        frames,
        dims=("theta_y", "theta_x", "qz", "qxy"),
        coords={
            "theta_y": theta_y_axis,
            "theta_x": theta_x_axis,
            "qz": qz_axis,
            "qxy": qxy_axis,
        },
        name="simulated_intensity",
        attrs=attrs,
    )


def _simulate_giwaxs_image_from_structure(
    structure: StructureData,
    params: GIWAXSSimulationParameters,
) -> xr.DataArray:
    peak_rows = _calculated_peak_rows(structure, params)
    qxy_axis = np.linspace(params.qxy_min, params.qxy_max, params.resolution_x)
    qz_axis = np.linspace(params.qz_min, params.qz_max, params.resolution_z)
    image = np.zeros((params.resolution_z, params.resolution_x), dtype=float)

    qxy_limits = sorted((params.qxy_min, params.qxy_max))
    sigma_qxy = max(float(params.sigma_r), 1.0e-6)
    sigma_qz = max(float(params.sigma_theta), 1.0e-6)
    gaussian_cutoff = 5.0

    for row in peak_rows:
        qxy = float(row["qxy"])
        qz = float(row["qz"])
        amplitude = float(row["amplitude"])
        if qxy < qxy_limits[0] or qxy > qxy_limits[1]:
            continue
        qxy_indices = np.flatnonzero(
            np.abs(qxy_axis - qxy) <= gaussian_cutoff * sigma_qxy
        )
        qz_indices = np.flatnonzero(
            np.abs(qz_axis - qz) <= gaussian_cutoff * sigma_qz
        )
        if qxy_indices.size == 0:
            qxy_indices = np.asarray(
                [int(np.nanargmin(np.abs(qxy_axis - qxy)))]
            )
        if qz_indices.size == 0:
            qz_indices = np.asarray([int(np.nanargmin(np.abs(qz_axis - qz)))])
        qxy_weights = np.exp(
            -0.5 * ((qxy_axis[qxy_indices] - qxy) / sigma_qxy) ** 2
        )
        qz_weights = np.exp(
            -0.5 * ((qz_axis[qz_indices] - qz) / sigma_qz) ** 2
        )
        image[np.ix_(qz_indices, qxy_indices)] += (
            amplitude * qz_weights[:, np.newaxis] * qxy_weights[np.newaxis, :]
        )

    attrs = {
        **structure.metadata(),
        **params.as_dict(),
        SIMULATION_MODE_ATTR: SIMULATION_MODE_PATTERN,
        "peak_count": len(peak_rows),
        PEAK_TABLE_ATTR: json.dumps(peak_rows, separators=(",", ":")),
        "legacy_source": "pyWAXS/pywaxs/simulation/WAXSSim.py",
    }
    return xr.DataArray(
        image,
        dims=("qz", "qxy"),
        coords={"qz": qz_axis, "qxy": qxy_axis},
        name="simulated_intensity",
        attrs=attrs,
    )


def run_and_store_simulation(
    project: ProjectState,
    structure_path: str | Path,
    output_directory: str | Path,
    *,
    parameters: GIWAXSSimulationParameters | None = None,
    target_data_id: str | None = None,
) -> dict[str, Any]:
    """Run a simulation, save it, and attach the record to a project."""

    if (
        target_data_id is not None
        and project.data_file_by_id(target_data_id) is None
    ):
        raise KeyError(f"No data file with id {target_data_id!r}")

    params = parameters or GIWAXSSimulationParameters()
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    data_array = simulate_giwaxs_image(structure_path, params)
    structure_id = str(data_array.attrs["structure_id"])
    simulation_id = _simulation_id(structure_id, params)
    dataset_path = output_path / f"{simulation_id}.nc"
    data_array.to_netcdf(dataset_path)
    record = {
        "simulation_id": simulation_id,
        "simulation_mode": SIMULATION_MODE_PATTERN,
        "data_id": target_data_id,
        "structure_id": structure_id,
        "structure_name": data_array.attrs.get("structure_name"),
        "structure_path": data_array.attrs.get("structure_path"),
        "dataset_uri": str(dataset_path),
        "parameters": params.as_dict(),
        "metadata": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "peak_count": data_array.attrs.get("peak_count", 0),
            "atom_count": data_array.attrs.get("atom_count", 0),
            "species": data_array.attrs.get("species", []),
            "legacy_source": data_array.attrs.get("legacy_source"),
        },
    }
    project.simulations[simulation_id] = record
    if target_data_id is not None:
        project.link_simulation_to_data_file(simulation_id, target_data_id)
    return record


def run_and_store_ewald_sphere_sweep(
    project: ProjectState,
    structure_path: str | Path,
    output_directory: str | Path,
    *,
    parameters: EwaldSphereSweepParameters | None = None,
    target_data_id: str | None = None,
) -> dict[str, Any]:
    """Run a theta sweep, save it, and attach the record to a project."""

    if (
        target_data_id is not None
        and project.data_file_by_id(target_data_id) is None
    ):
        raise KeyError(f"No data file with id {target_data_id!r}")

    params = parameters or EwaldSphereSweepParameters()
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    data_array = simulate_ewald_sphere_sweep(structure_path, params)
    structure_id = str(data_array.attrs["structure_id"])
    simulation_id = _simulation_id(structure_id, params, suffix="sphere")
    dataset_path = output_path / f"{simulation_id}.nc"
    data_array.to_netcdf(dataset_path)
    record = {
        "simulation_id": simulation_id,
        "simulation_mode": SIMULATION_MODE_EWALD_SWEEP,
        "data_id": target_data_id,
        "structure_id": structure_id,
        "structure_name": data_array.attrs.get("structure_name"),
        "structure_path": data_array.attrs.get("structure_path"),
        "dataset_uri": str(dataset_path),
        "parameters": params.as_dict(),
        "metadata": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame_count": data_array.attrs.get("frame_count", 0),
            "theta_x_count": data_array.attrs.get("theta_x_count", 0),
            "theta_y_count": data_array.attrs.get("theta_y_count", 0),
            "peak_count": data_array.attrs.get("peak_count", 0),
            "peak_count_max": data_array.attrs.get("peak_count_max", 0),
            "atom_count": data_array.attrs.get("atom_count", 0),
            "species": data_array.attrs.get("species", []),
            "legacy_source": data_array.attrs.get("legacy_source"),
        },
    }
    project.simulations[simulation_id] = record
    if target_data_id is not None:
        project.link_simulation_to_data_file(simulation_id, target_data_id)
    return record


def load_simulation_data(record: dict[str, Any]) -> xr.DataArray | None:
    """Load a stored simulation image from a project simulation record."""

    dataset_uri = record.get("dataset_uri")
    if not dataset_uri:
        return None
    path = Path(str(dataset_uri))
    if not path.exists():
        return None
    loaded = xr.load_dataarray(path)
    return loaded


def is_ewald_sphere_sweep_record(record: dict[str, Any]) -> bool:
    """Return True when a stored simulation record is a theta sweep."""

    return record.get("simulation_mode") == SIMULATION_MODE_EWALD_SWEEP


def is_ewald_sphere_sweep_data(data: xr.DataArray) -> bool:
    """Return True when a loaded simulation contains theta sweep frames."""

    return {"theta_x", "theta_y", "qz", "qxy"}.issubset(data.dims)


def reconstruct_ewald_sphere_points(
    data: xr.DataArray,
    *,
    intensity_quantile: float = 0.995,
    max_points: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a point-cloud reconstruction from a stored theta sweep.

    The sweep patterns are recorded in the lab q_{xy}/q_{z} plane. For each
    selected bright pixel, the q-vector is rotated back by the frame orientation so the
    accumulated point cloud approximates the single-crystal reciprocal sphere.
    """

    if not is_ewald_sphere_sweep_data(data):
        raise ValueError(
            "Ewald sphere reconstruction requires theta sweep data."
        )

    sweep = data.transpose("theta_y", "theta_x", "qz", "qxy")
    values = np.asarray(sweep.values, dtype=float)
    finite = values[np.isfinite(values)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

    quantile = min(max(float(intensity_quantile), 0.0), 1.0)
    threshold = float(np.nanquantile(finite, quantile))
    mask = np.isfinite(values) & (values >= threshold) & (values > 0.0)
    indices = np.argwhere(mask)
    if indices.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

    intensities = values[tuple(indices.T)]
    point_limit = max(0, int(max_points))
    if point_limit and intensities.size > point_limit:
        selection = np.argpartition(intensities, -point_limit)[-point_limit:]
        indices = indices[selection]
        intensities = intensities[selection]

    theta_y_axis = np.asarray(sweep.coords["theta_y"].values, dtype=float)
    theta_x_axis = np.asarray(sweep.coords["theta_x"].values, dtype=float)
    qz_axis = np.asarray(sweep.coords["qz"].values, dtype=float)
    qxy_axis = np.asarray(sweep.coords["qxy"].values, dtype=float)
    points = np.empty((indices.shape[0], 3), dtype=float)
    for point_index, (theta_y_i, theta_x_i, qz_i, qxy_i) in enumerate(indices):
        base_vector = np.asarray(
            [qxy_axis[qxy_i], 0.0, qz_axis[qz_i]],
            dtype=float,
        )
        rotation = _orientation_rotation(
            theta_x_axis[theta_x_i],
            theta_y_axis[theta_y_i],
        )
        points[point_index] = base_vector @ rotation.T
    return points, intensities


def calculate_giwaxs_peak_rows(
    structure_path: str | Path,
    parameters: GIWAXSSimulationParameters | None = None,
) -> list[dict[str, Any]]:
    """Return calculated (hkl) q-space positions for a simulation."""

    params = parameters or GIWAXSSimulationParameters()
    return _calculated_peak_rows(load_structure(structure_path), params)


def load_structure(path: str | Path) -> StructureData:
    """Load CIF/POSCAR structures through pymatgen or a POSCAR fallback."""

    structure_path = Path(path)
    try:
        from pymatgen.core import Structure

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"Issues encountered while parsing CIF: .*fractional "
                    r"coordinates rounded to ideal values.*"
                ),
                category=UserWarning,
            )
            structure = Structure.from_file(str(structure_path))
        lattice = np.asarray(structure.lattice.matrix, dtype=float)
        species = [site.specie.symbol for site in structure]
        frac_coords = np.asarray([site.frac_coords for site in structure])
        return StructureData(structure_path, lattice, species, frac_coords)
    except Exception:
        if structure_path.suffix.lower() in {".cif", ".mcif"}:
            raise
    return _read_poscar(structure_path)


def _read_poscar(path: Path) -> StructureData:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 8:
        raise ValueError(f"{path} is not a valid POSCAR file.")
    scale = float(lines[1].split()[0])
    lattice = np.asarray(
        [
            [float(value) for value in lines[index].split()[:3]]
            for index in (2, 3, 4)
        ],
        dtype=float,
    )
    lattice *= scale
    species = lines[5].split()
    counts = [int(value) for value in lines[6].split()]
    coord_line = 7
    if lines[coord_line].lower().startswith("s"):
        coord_line += 1
    coordinate_kind = lines[coord_line].lower()
    start = coord_line + 1
    total_atoms = sum(counts)
    coords = np.asarray(
        [
            [float(value) for value in lines[start + index].split()[:3]]
            for index in range(total_atoms)
        ],
        dtype=float,
    )
    if coordinate_kind.startswith("c") or coordinate_kind.startswith("k"):
        coords = np.linalg.solve(lattice.T, coords.T).T
    atom_species = [
        symbol
        for symbol, count in zip(species, counts)
        for _index in range(count)
    ]
    return StructureData(path, lattice, atom_species, coords)


def _bragg_peaks(
    structure: StructureData,
    params: GIWAXSSimulationParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lattice = _rotated_lattice(structure.lattice, params)
    volume = np.dot(lattice[2], np.cross(lattice[0], lattice[1]))
    b1 = 2.0 * np.pi * np.cross(lattice[1], lattice[2]) / volume
    b2 = 2.0 * np.pi * np.cross(lattice[2], lattice[0]) / volume
    b3 = 2.0 * np.pi * np.cross(lattice[0], lattice[1]) / volume
    extent = max(0, int(params.hkl_extent))
    miller_range = np.arange(-extent, extent + 1)
    h, k, l = np.meshgrid(
        miller_range, miller_range, miller_range, indexing="ij"
    )
    hkl = np.column_stack([h.ravel(), k.ravel(), l.ravel()])
    keep = np.any(hkl != 0, axis=1)
    hkl = hkl[keep]
    q_vectors = (
        hkl[:, [0]] * b1[np.newaxis, :]
        + hkl[:, [1]] * b2[np.newaxis, :]
        + hkl[:, [2]] * b3[np.newaxis, :]
    )
    intensities = _structure_factor_intensity(structure, lattice, q_vectors)
    return hkl, q_vectors, intensities


def _calculated_peak_rows(
    structure: StructureData,
    params: GIWAXSSimulationParameters,
) -> list[dict[str, Any]]:
    hkl, q_vectors, intensities = _bragg_peaks(structure, params)
    qxy_limits = sorted((params.qxy_min, params.qxy_max))
    qz_limits = sorted((params.qz_min, params.qz_max))
    max_abs_qxy = max(abs(qxy_limits[0]), abs(qxy_limits[1]))
    sigma_phi = max(float(params.sigma_phi), 1.0e-6)
    rows: list[dict[str, Any]] = []
    for miller, q_vector, intensity in zip(hkl, q_vectors, intensities):
        if not np.isfinite(intensity) or intensity <= 0.0:
            continue
        qxy = float(np.hypot(q_vector[0], q_vector[1]))
        qz = float(abs(q_vector[2]))
        if qz < qz_limits[0] or qz > qz_limits[1]:
            continue
        if qxy < 0.0 or qxy > max_abs_qxy:
            continue
        for signed_qxy in _signed_qxy_positions(qxy, qxy_limits):
            phi_weight = _azimuthal_weight(q_vector, signed_qxy, sigma_phi)
            rows.append(
                {
                    "h": int(miller[0]),
                    "k": int(miller[1]),
                    "l": int(miller[2]),
                    "qxy": float(signed_qxy),
                    "qz": qz,
                    "intensity": float(intensity),
                    "amplitude": float(intensity * phi_weight),
                }
            )
    return rows


def _rotated_lattice(
    lattice: np.ndarray,
    params: GIWAXSSimulationParameters,
) -> np.ndarray:
    return np.asarray(lattice, dtype=float) @ _orientation_rotation(
        params.theta_x_deg,
        params.theta_y_deg,
    )


def _orientation_rotation(
    theta_x_deg: float,
    theta_y_deg: float,
) -> np.ndarray:
    theta_x = math.radians(theta_x_deg)
    theta_y = math.radians(theta_y_deg)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta_x), -np.sin(theta_x)],
            [0.0, np.sin(theta_x), np.cos(theta_x)],
        ]
    )
    ry = np.array(
        [
            [np.cos(theta_y), 0.0, -np.sin(theta_y)],
            [0.0, 1.0, 0.0],
            [np.sin(theta_y), 0.0, np.cos(theta_y)],
        ]
    )
    return rx @ ry


def _angle_axis(
    start: float,
    stop: float,
    step: float,
) -> np.ndarray:
    step_value = abs(float(step))
    if step_value <= 0.0:
        raise ValueError("Theta sweep step must be greater than zero.")
    start_value = float(start)
    stop_value = float(stop)
    if stop_value < start_value:
        step_value *= -1.0
    values: list[float] = []
    current = start_value
    tolerance = abs(step_value) * 1.0e-9
    if step_value > 0.0:
        while current <= stop_value + tolerance:
            values.append(float(current))
            current += step_value
    else:
        while current >= stop_value - tolerance:
            values.append(float(current))
            current += step_value
    if not values:
        values.append(start_value)
    return np.asarray(values, dtype=float)


def _structure_factor_intensity(
    structure: StructureData,
    lattice: np.ndarray,
    q_vectors: np.ndarray,
) -> np.ndarray:
    aff = WAXSAFF.AFF()
    q2 = np.sum(q_vectors * q_vectors, axis=1)
    structure_factor = np.zeros(q_vectors.shape[0], dtype=complex)
    for symbol, frac_coord in zip(structure.species, structure.frac_coords):
        form_factor = _atomic_form_factor(aff, symbol, q2)
        real_position = frac_coord[0] * lattice[0]
        real_position += frac_coord[1] * lattice[1]
        real_position += frac_coord[2] * lattice[2]
        phase = np.exp(1j * (q_vectors @ real_position))
        structure_factor += form_factor * phase
    return np.abs(structure_factor) ** 2


def _atomic_form_factor(
    aff: np.ndarray,
    symbol: str,
    q2: np.ndarray,
) -> np.ndarray:
    aff_key = _aff_symbol(symbol)
    row_id = WAXSAFF.atom_dict.get(aff_key)
    if row_id is None:
        row_id = WAXSAFF.atom_dict.get(symbol)
    if row_id is None:
        raise ValueError(f"No atomic form factor for {symbol!r}.")
    row_index = np.searchsorted(aff[:, 0], row_id)
    row = aff[row_index]
    factor = np.zeros_like(q2, dtype=float)
    for a_col, b_col in ((1, 2), (3, 4), (5, 6), (7, 8)):
        factor += row[a_col] * np.exp(-row[b_col] * q2 / (16.0 * np.pi**2))
    factor += row[9]
    return factor


def _aff_symbol(symbol: str) -> str:
    aliases = {
        "Si": "Siv",
    }
    return aliases.get(symbol, symbol)


def _signed_qxy_positions(
    qxy: float,
    qxy_limits: list[float],
) -> list[float]:
    positions: list[float] = []
    for candidate in (-qxy, qxy):
        if qxy_limits[0] <= candidate <= qxy_limits[1]:
            positions.append(candidate)
    return list(dict.fromkeys(positions))


def _azimuthal_weight(
    q_vector: np.ndarray,
    signed_qxy: float,
    sigma_phi: float,
) -> float:
    qxy = float(np.hypot(q_vector[0], q_vector[1]))
    if qxy <= 1.0e-12:
        return 1.0
    phi = 0.0 if signed_qxy >= 0 else np.pi
    raw_phi = np.arctan2(q_vector[1], q_vector[0])
    distance = abs(phi - raw_phi)
    distance = abs(abs(distance - np.pi) - np.pi)
    return float(np.exp(-0.5 * (distance / sigma_phi) ** 2))


def _simulation_id(
    structure_id: str,
    params: GIWAXSSimulationParameters,
    *,
    suffix: str = "sim",
) -> str:
    payload = repr(sorted(params.as_dict().items())).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:10]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nonce = time.time_ns() % 1_000_000_000
    return f"{structure_id}_{suffix}_{timestamp}_{nonce:09d}_{digest}"
