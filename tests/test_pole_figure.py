"""Pole-figure ROI metadata and reduction tests."""

from __future__ import annotations

import numpy as np
import pytest

from ewald.data.models import (
    ProjectState,
    ROIRegion,
    roi_hkl_metadata,
    roi_pole_figure_status,
    set_roi_hkl_metadata,
)
from ewald.processing.pole_figure import (
    BACKGROUND_CONSTANT,
    BACKGROUND_LOCAL_ANNULAR,
    PoleFigureSettings,
    generate_pole_figure,
    pole_figure_record_from_result,
)


def test_roi_hkl_metadata_round_trips_with_project(tmp_path):
    project = ProjectState(name="hkl project")
    roi = project.add_roi_region(
        ROIRegion(
            target_id="detector",
            kind="box",
            name="Peak ROI",
            qxy_min=-1.0,
            qxy_max=1.0,
            qz_min=0.0,
            qz_max=2.0,
        )
    )

    project.set_roi_hkl_tag(
        "detector",
        roi.roi_id or "",
        h=1,
        k=0,
        l=2,
        label="(102) shoulder",
    )

    from ewald.io.project import load_project, save_project

    path = save_project(project, tmp_path / "hkl_project")
    loaded = load_project(path)
    loaded_roi = loaded.rois_for_target("detector")[0]

    assert roi_hkl_metadata(loaded_roi) == {
        "h": 1,
        "k": 0,
        "l": 2,
        "label": "(102) shoulder",
    }


def test_roi_hkl_rejects_malformed_values():
    roi = ROIRegion(target_id="detector")

    with pytest.raises(ValueError):
        set_roi_hkl_metadata(roi, h="1.5", k=0, l=1)


def test_generate_pole_figure_with_constant_background():
    qxy = np.linspace(-1.0, 1.0, 41)
    qz = np.linspace(0.0, 2.0, 41)
    qxy_grid, _qz_grid = np.meshgrid(qxy, qz)
    image = 10.0 + np.abs(qxy_grid) * 5.0
    roi = ROIRegion(
        target_id="detector",
        kind="box",
        roi_id="box",
        name="Box",
        qxy_min=-1.0,
        qxy_max=1.0,
        qz_min=0.0,
        qz_max=2.0,
    )
    settings = PoleFigureSettings(
        chi_min_deg=-90.0,
        chi_max_deg=90.0,
        chi_bin_width_deg=10.0,
        background_method=BACKGROUND_CONSTANT,
        background_constant=10.0,
        intensity_mode="mean",
    )

    result = generate_pole_figure(
        roi,
        image,
        (-1.0, 1.0, 0.0, 2.0),
        settings=settings,
    )

    assert result is not None
    assert result.chi_deg.size == 18
    assert np.nanmin(result.intensity) >= -1.0e-9
    assert result.background_record["method"] == BACKGROUND_CONSTANT
    assert result.missing_fraction < 1.0


def test_local_annular_background_and_staleness_metadata():
    qxy = np.linspace(-1.0, 1.0, 81)
    qz = np.linspace(0.0, 2.0, 81)
    qxy_grid, qz_grid = np.meshgrid(qxy, qz)
    radius = np.hypot(qxy_grid, qz_grid)
    image = 4.0 + 20.0 * np.exp(-((radius - 1.0) ** 2) / 0.004)
    project = ProjectState()
    roi = project.add_roi_region(
        ROIRegion(
            target_id="detector",
            kind="arch",
            roi_id="arch",
            name="Arch",
            qr_min=0.93,
            qr_max=1.07,
            chi_min=-60.0,
            chi_max=60.0,
            integration_axis="chi",
            integration_direction="azimuthal",
        )
    )
    result = generate_pole_figure(
        roi,
        image,
        (-1.0, 1.0, 0.0, 2.0),
        settings=PoleFigureSettings(
            chi_min_deg=-60.0,
            chi_max_deg=60.0,
            background_method=BACKGROUND_LOCAL_ANNULAR,
            local_background_gap=0.03,
            local_background_width=0.08,
            intensity_mode="mean",
        ),
    )
    assert result is not None

    record = pole_figure_record_from_result(roi, result)
    project.set_roi_pole_figure_metadata("detector", roi.roi_id or "", record)

    assert roi_pole_figure_status(roi) == "Current"
    roi.qr_max = 1.2
    project.mark_roi_pole_figures_stale("detector", roi.roi_id or "")
    assert roi_pole_figure_status(roi) == "Stale"
