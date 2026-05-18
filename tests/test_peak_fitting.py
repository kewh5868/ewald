"""Tests for peak ROI integration and Gaussian fitting helpers."""

import numpy as np
import pytest

from ewald.processing.peak_fitting import (
    compute_peak_fit_integrations,
    fit_peak_integrations,
    fit_peak_roi_2d,
)


def test_peak_fit_integrations_seed_2d_gaussian_center():
    qxy = np.linspace(-1.0, 1.0, 81)
    qz = np.linspace(-1.0, 1.0, 81)
    qxy_grid, qz_grid = np.meshgrid(qxy, qz)
    image = 3.0 + 100.0 * np.exp(
        -0.5
        * (((qxy_grid - 0.2) / 0.07) ** 2 + ((qz_grid + 0.15) / 0.12) ** 2)
    )
    roi = {
        "qxy_min": -0.2,
        "qxy_max": 0.55,
        "qz_min": -0.55,
        "qz_max": 0.2,
    }

    integrations = compute_peak_fit_integrations(
        image,
        (-1.0, 1.0, -1.0, 1.0),
        roi,
    )
    fits = fit_peak_integrations(integrations)
    fit_2d = fit_peak_roi_2d(image, (-1.0, 1.0, -1.0, 1.0), roi, fits)

    assert set(integrations) == {"qxy", "qz", "azimuthal"}
    assert set(fits) == {"qxy", "qz", "azimuthal"}
    assert fit_2d is not None
    assert fits["qxy"]["center"] == pytest.approx(0.2, abs=0.03)
    assert fits["qz"]["center"] == pytest.approx(-0.15, abs=0.03)
    assert fit_2d["center_qxy"] == pytest.approx(0.2, abs=0.03)
    assert fit_2d["center_qz"] == pytest.approx(-0.15, abs=0.03)
    assert fit_2d["expression"]


def test_peak_fit_azimuthal_integration_can_use_arch_roi():
    qxy = np.linspace(-1.0, 1.0, 101)
    qz = np.linspace(-1.0, 1.0, 101)
    qxy_grid, qz_grid = np.meshgrid(qxy, qz)
    radius = np.hypot(qxy_grid, qz_grid)
    chi = np.degrees(np.arctan2(qxy_grid, qz_grid))
    image = np.where(
        (radius >= 0.45) & (radius <= 0.55) & (chi >= -30.0) & (chi <= 30.0),
        10.0 + chi,
        0.0,
    )
    box_roi = {
        "qxy_min": -0.8,
        "qxy_max": 0.8,
        "qz_min": -0.8,
        "qz_max": 0.8,
    }
    arch_roi = {
        "kind": "arch",
        "qxy_center": 0.0,
        "qz_center": 0.0,
        "qr_min": 0.45,
        "qr_max": 0.55,
        "chi_min": -30.0,
        "chi_max": 30.0,
    }

    integrations = compute_peak_fit_integrations(
        image,
        (-1.0, 1.0, -1.0, 1.0),
        box_roi,
        azimuthal_roi=arch_roi,
    )

    azimuthal = integrations["azimuthal"]
    assert min(azimuthal["x_values"]) >= -30.0
    assert max(azimuthal["x_values"]) <= 30.0
    assert np.nanmax(azimuthal["y_values"]) > 0.0
    assert "r_w" in fit_peak_integrations(integrations)["qxy"]["statistics"]
