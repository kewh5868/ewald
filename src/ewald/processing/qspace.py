"""PyFAI-backed reciprocal-space and caking transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xarray as xr

HC_KEV_ANGSTROM = 12.398419843320026


@dataclass(frozen=True, slots=True)
class CakingConfig:
    """Configuration for azimuthal caking with pyFAI."""

    npt_radial: int = 1000
    npt_azimuthal: int = 360
    radial_unit: str = "q_nm^-1"
    radial_range: tuple[float, float] | None = None
    azimuth_range: tuple[float, float] | None = None
    correct_solid_angle: bool = True
    polarization_factor: float | None = None
    method: tuple[str, str, str] = ("bbox", "csr", "cython")


@dataclass(frozen=True, slots=True)
class GrazingIncidenceConfig:
    """Configuration for GIWAXS/GISAXS qIP-qOOP mapping with pyFAI."""

    npt_ip: int = 1000
    npt_oop: int = 1000
    unit_ip: str = "qip_A^-1"
    unit_oop: str = "qoop_A^-1"
    ip_range: tuple[float, float] | None = None
    oop_range: tuple[float, float] | None = None
    xray_energy_kev: float | None = None
    incident_angle_deg: float = 0.0
    tilt_angle_deg: float = 0.0
    sample_orientation: int = 6
    correct_solid_angle: bool = True
    polarization_factor: float | None = 0.95
    normalization_factor: float = 1.0
    dummy: float | None = None
    delta_dummy: float | None = None
    method: tuple[str, str, str] = ("no", "histogram", "cython")


def load_azimuthal_integrator(poni_file: str | Path) -> Any:
    """Load a pyFAI AzimuthalIntegrator from a PONI file."""

    import pyFAI

    return pyFAI.load(str(poni_file))


def load_fiber_integrator(poni_file: str | Path) -> Any:
    """Load a pyFAI FiberIntegrator for grazing-incidence transforms."""

    from pyFAI.integrator.fiber import FiberIntegrator

    return load_azimuthal_integrator(poni_file).promote(type_=FiberIntegrator)


def xray_energy_kev_to_wavelength_m(energy_kev: float) -> float:
    """Convert X-ray photon energy in keV to wavelength in meters."""

    if energy_kev <= 0:
        raise ValueError("X-ray energy must be greater than zero.")
    return (HC_KEV_ANGSTROM / energy_kev) * 1.0e-10


def xray_energy_kev_from_wavelength_m(wavelength_m: float) -> float:
    """Convert X-ray wavelength in meters to photon energy in keV."""

    if wavelength_m <= 0:
        raise ValueError("X-ray wavelength must be greater than zero.")
    return HC_KEV_ANGSTROM / (wavelength_m * 1.0e10)


def cake_image(
    data: Any,
    poni_file: str | Path,
    *,
    config: CakingConfig | None = None,
    mask: Any | None = None,
    dark: Any | None = None,
    flat: Any | None = None,
) -> xr.DataArray:
    """Return a caked detector image with azimuth and radial
    coordinates."""

    settings = config or CakingConfig()
    ai = load_azimuthal_integrator(poni_file)
    result = ai.integrate2d(
        data,
        settings.npt_radial,
        settings.npt_azimuthal,
        unit=settings.radial_unit,
        radial_range=settings.radial_range,
        azimuth_range=settings.azimuth_range,
        correctSolidAngle=settings.correct_solid_angle,
        polarization_factor=settings.polarization_factor,
        mask=mask,
        dark=dark,
        flat=flat,
        method=settings.method,
    )
    return xr.DataArray(
        result.intensity,
        dims=("azimuth", "radial"),
        coords={
            "azimuth": result.azimuthal,
            "radial": result.radial,
        },
        name="caked_intensity",
        attrs={
            "transform": "pyfai.integrate2d",
            "radial_unit": str(result.radial_unit),
            "azimuthal_unit": str(result.azimuthal_unit),
            "poni_file": str(poni_file),
        },
    )


def map_grazing_incidence_qspace(
    data: Any,
    poni_file: str | Path,
    *,
    config: GrazingIncidenceConfig | None = None,
    mask: Any | None = None,
    dark: Any | None = None,
    flat: Any | None = None,
) -> xr.DataArray:
    """Return an Ewald-sphere-aware qIP-qOOP reciprocal-space map."""

    settings = config or GrazingIncidenceConfig()
    fi = load_fiber_integrator(poni_file)
    if settings.xray_energy_kev is not None:
        fi.wavelength = xray_energy_kev_to_wavelength_m(
            settings.xray_energy_kev
        )
    result = fi.integrate2d_grazing_incidence(
        data,
        npt_ip=settings.npt_ip,
        npt_oop=settings.npt_oop,
        unit_ip=settings.unit_ip,
        unit_oop=settings.unit_oop,
        ip_range=settings.ip_range,
        oop_range=settings.oop_range,
        incident_angle=settings.incident_angle_deg,
        tilt_angle=settings.tilt_angle_deg,
        angle_unit="deg",
        sample_orientation=settings.sample_orientation,
        correctSolidAngle=settings.correct_solid_angle,
        polarization_factor=settings.polarization_factor,
        mask=mask,
        dummy=settings.dummy,
        delta_dummy=settings.delta_dummy,
        dark=dark,
        flat=flat,
        method=settings.method,
        normalization_factor=settings.normalization_factor,
    )
    return xr.DataArray(
        result.intensity,
        dims=("q_oop", "q_ip"),
        coords={
            "q_oop": result.outofplane,
            "q_ip": result.inplane,
        },
        name="qspace_intensity",
        attrs={
            "transform": "pyfai.integrate2d_grazing_incidence",
            "q_ip_unit": str(result.ip_unit),
            "q_oop_unit": str(result.oop_unit),
            "xray_energy_kev": settings.xray_energy_kev,
            "wavelength_m": (
                float(fi.wavelength) if fi.wavelength is not None else None
            ),
            "incident_angle_deg": settings.incident_angle_deg,
            "tilt_angle_deg": settings.tilt_angle_deg,
            "sample_orientation": settings.sample_orientation,
            "correct_solid_angle": settings.correct_solid_angle,
            "polarization_factor": settings.polarization_factor,
            "normalization_factor": settings.normalization_factor,
            "dummy": settings.dummy,
            "delta_dummy": settings.delta_dummy,
            "poni_file": str(poni_file),
        },
    )
