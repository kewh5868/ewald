"""Generate EWALD UI screenshots for documentation pages."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtCore, QtWidgets  # noqa: E402

from ewald.data.models import (  # noqa: E402
    ImageCorrectionState,
    ProjectState,
    ROIRegion,
    set_roi_hkl_metadata,
)
from ewald.io.importers import build_data_group_from_paths  # noqa: E402
from ewald.ui.data_viewer import DataViewerPane  # noqa: E402
from ewald.ui.giwaxs_simulation import GIWAXSSimulationWindow  # noqa: E402
from ewald.ui.main_window import MainWindow  # noqa: E402
from ewald.ui.notation import set_qspace_axis_labels  # noqa: E402
from ewald.ui.peak_identification import PeakIdentificationPane  # noqa: E402
from ewald.ui.pole_figure import PoleFigureGeneratorWindow  # noqa: E402
from ewald.ui.structure_analysis import StructureAnalysisPane  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "assets" / "screenshots" / "tutorials"
TMP_DIR = Path(tempfile.mkdtemp(prefix="ewald-doc-screenshots-"))


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    captures = [
        capture_main_window_overview(app),
        capture_apply_corrections(app),
        capture_data_viewer_rois(app),
        capture_peak_identification(app),
        capture_peak_fitting(app),
        capture_structure_analysis(app),
        capture_pole_figure(app),
        capture_simulation(app),
    ]
    for path in captures:
        print(path.relative_to(ROOT))
    return 0


def capture_main_window_overview(app: QtWidgets.QApplication) -> Path:
    project, data_id = _example_project(confirmed=True)
    window = MainWindow(project=project, settings=_settings("main"))
    _select_file(window)
    viewer = window.tabs.widget(0)
    if isinstance(viewer, DataViewerPane):
        _prepare_viewer_for_roi_demo(viewer)
        viewer.add_roi_from_bounds(-1.0, 0.1, 0.6, 1.5)
        viewer.add_coupled_roi_pair_from_bounds(0.25, 1.25, 0.35, 1.35)
        window.data_tree.set_project(project)
        _select_file(window)
    window.tabs.setCurrentIndex(0)
    return _save_widget(
        app,
        window,
        "ui-main-window.png",
        size=(1500, 950),
    )


def capture_apply_corrections(app: QtWidgets.QApplication) -> Path:
    project, data_id = _example_project(confirmed=False)
    state = ImageCorrectionState(
        target_id=data_id,
        mask_asset_id=project.masks[0].asset_id,
        calibrant_asset_id=project.calibrants[0].asset_id,
        xray_energy_kev=12.7,
        reflected_beam_x_px=486.0,
        reflected_beam_y_px=508.0,
        sample_stoichiometry="MAPbI3",
        sample_density_g_cm3=4.16,
        refractive_index_delta=3.2e-6,
        critical_angle_deg=0.145,
        confirmed=False,
    )
    project.set_image_corrections(state)
    window = MainWindow(project=project, settings=_settings("corrections"))
    _select_file(window)
    _set_tab(window, "Apply Image Corrections")
    return _save_widget(
        app,
        window,
        "apply-corrections.png",
        size=(1500, 950),
    )


def capture_data_viewer_rois(app: QtWidgets.QApplication) -> Path:
    project, data_id = _synthetic_project()
    viewer = DataViewerPane(project, data_id)
    _prepare_viewer_for_roi_demo(viewer)
    roi = viewer.add_roi_from_bounds(-1.25, -0.2, 0.45, 1.25)
    pair = viewer.add_coupled_roi_pair_from_bounds(0.25, 1.4, 0.35, 1.55)
    if roi is not None:
        viewer._toggle_roi_channel(roi.roi_id, 1, True)
        viewer._add_channel_peak_marker(1, roi.roi_id, -0.68, 250.0)
    if pair is not None:
        viewer._select_roi(pair[0].roi_id)
    return _save_widget(
        app,
        viewer,
        "data-viewer-rois.png",
        size=(1400, 900),
    )


def capture_peak_identification(app: QtWidgets.QApplication) -> Path:
    pane = _peak_identification_pane()
    pane.peak_table.selectRow(1)
    pane.peak_finder_status_label.setText(
        "Tutorial capture: manual and detected peak candidates are ready."
    )
    return _save_widget(
        app,
        pane,
        "peak-identification.png",
        size=(1400, 900),
    )


def capture_peak_fitting(app: QtWidgets.QApplication) -> Path:
    pane = _peak_identification_pane()
    pane.peak_table.selectRow(1)
    pane.side_tabs.setMinimumWidth(540)
    pane.roi_width.setValue(0.35)
    pane.roi_height.setValue(0.35)
    pane.apply_roi_to_selected_peak()
    pane.run_integrations_for_selected_roi()
    pane.side_tabs.setCurrentIndex(2)
    pane._set_fit_status(
        "Tutorial capture: qxy, qz, and azimuthal traces are ready for "
        "fitting."
    )
    return _save_widget(
        app,
        pane,
        "peak-fitting.png",
        size=(1650, 900),
    )


def _peak_identification_pane() -> PeakIdentificationPane:
    project, data_id = _synthetic_project()
    pane = PeakIdentificationPane(project, data_id)
    pane.image_data = _synthetic_qspace_image()
    pane.axis_ranges = (-2.2, 2.2, 0.0, 2.4)
    pane.coordinate_space = "qspace"
    if pane.plot_widget is not None:
        set_qspace_axis_labels(pane.plot_widget)
    pane._set_initial_image()
    for qxy, qz in [(-1.2, 0.7), (-0.35, 1.35), (0.42, 1.1), (1.15, 1.75)]:
        pane.add_peak_at(qxy, qz, source="tutorial", record_history=False)
    return pane


def capture_structure_analysis(app: QtWidgets.QApplication) -> Path:
    project, data_id = _synthetic_project()
    project.peak_sets[data_id] = [
        {
            "peak_id": "p1",
            "label": "Peak 01",
            "qxy": -1.2,
            "qz": 0.7,
            "source": "peak-fit",
        },
        {
            "peak_id": "p2",
            "label": "Peak 02",
            "qxy": -0.35,
            "qz": 1.35,
            "source": "peak-fit",
        },
        {
            "peak_id": "p3",
            "label": "Peak 03",
            "qxy": 0.42,
            "qz": 1.1,
            "source": "peak-fit",
        },
        {
            "peak_id": "p4",
            "label": "Peak 04",
            "qxy": 1.15,
            "qz": 1.75,
            "source": "peak-fit",
        },
    ]
    project.fits[data_id] = {
        "peak_fit": {
            "p1": {"fit_2d": {"center_qxy": -1.18, "center_qz": 0.72}},
            "p2": {"fit_2d": {"center_qxy": -0.36, "center_qz": 1.33}},
            "p3": {"fit_2d": {"center_qxy": 0.44, "center_qz": 1.12}},
            "p4": {"fit_2d": {"center_qxy": 1.13, "center_qz": 1.73}},
        }
    }
    pane = StructureAnalysisPane(
        project,
        data_id,
        image_data=_synthetic_qspace_image(),
        axis_ranges=(-2.2, 2.2, 0.0, 2.4),
        coordinate_space="qspace",
        generated_cif_directory=TMP_DIR / "generated_cifs",
    )
    pane.peak_table.selectRow(1)
    pane.status_label.setText(
        "Tutorial capture: fitted peaks are imported and ready for candidate "
        "ranking."
    )
    return _save_widget(
        app,
        pane,
        "structure-analysis.png",
        size=(1500, 950),
    )


def capture_pole_figure(app: QtWidgets.QApplication) -> Path:
    project, data_id = _synthetic_project()
    roi = project.add_roi_region(
        ROIRegion(
            target_id=data_id,
            kind="arch",
            roi_id="tutorial_pole_arc",
            name="(102) Azimuthal Arc",
            qr_min=1.0,
            qr_max=1.45,
            chi_min=-70.0,
            chi_max=75.0,
            integration_axis="chi",
            integration_direction="azimuthal",
        )
    )
    set_roi_hkl_metadata(roi, h=1, k=0, l=2, label="(102)")
    window = PoleFigureGeneratorWindow(
        project=project,
        project_path=TMP_DIR / "pole_figure_tutorial.ewld",
        data_id=data_id,
        roi=roi,
        image_data=_synthetic_qspace_image(),
        axis_ranges=(-2.2, 2.2, 0.0, 2.4),
    )
    window.chi_bin_width.setValue(5.0)
    window.normalization.setCurrentIndex(1)
    window.display_label_edit.setText("(102) azimuthal distribution")
    window.generate()
    return _save_widget(
        app,
        window,
        "pole-figure.png",
        size=(1300, 850),
    )


def capture_simulation(app: QtWidgets.QApplication) -> Path:
    project = ProjectState(name="Simulation Tutorial")
    window = GIWAXSSimulationWindow(
        project=project,
        output_directory=TMP_DIR / "simulations",
        settings=_settings("simulation"),
    )
    window.import_structure_path(
        ROOT / "example" / "structures" / "2hPbI2.cif"
    )
    window.hkl_extent.setValue(2)
    window.resolution_x.setValue(96)
    window.resolution_z.setValue(72)
    window.theta_x.setValue(90.0)
    window.theta_y.setValue(0.0)
    record = window.run_selected_simulation()
    if record is not None and hasattr(window, "right_tabs"):
        window.right_tabs.setCurrentWidget(window.result_pane)
    return _save_widget(
        app,
        window,
        "giwaxs-simulation.png",
        size=(1500, 950),
    )


def _example_project(*, confirmed: bool) -> tuple[ProjectState, str]:
    path = next((ROOT / "example").glob("*.tiff"))
    group, _report = build_data_group_from_paths([path], group_name="Example")
    data_id = str(group.data_files[0].data_id)
    project = ProjectState(name="EWALD Tutorial Project")
    project.add_data_group(group)
    mask = project.add_mask(
        ROOT / "example" / "mask.edf", target_ids=[data_id]
    )
    calibrant = project.add_calibrant(
        ROOT / "example" / "calib.poni",
        target_ids=[data_id],
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            xray_energy_kev=12.7,
            image_rotation_deg=0,
            pyfai_sample_orientation=1,
            correct_solid_angle=True,
            polarization_factor=0.95,
            confirmed=confirmed,
        )
    )
    return project, data_id


def _synthetic_project() -> tuple[ProjectState, str]:
    project = ProjectState(name="Synthetic Tutorial Project")
    data_file = project.add_data_file(TMP_DIR / "synthetic_tutorial.tiff")
    data_id = str(data_file.data_id)
    project.set_image_corrections(
        ImageCorrectionState(target_id=data_id, confirmed=True)
    )
    return project, data_id


def _prepare_viewer_for_roi_demo(viewer: DataViewerPane) -> None:
    viewer.image_data = _synthetic_qspace_image()
    viewer.axis_ranges = (-2.2, 2.2, 0.0, 2.4)
    viewer.coordinate_space = "qspace"
    viewer.roi_controls_enabled = True
    viewer._set_roi_controls_enabled(True)
    if viewer.plot_widget is not None:
        set_qspace_axis_labels(viewer.plot_widget)
    viewer._set_initial_image()


def _synthetic_qspace_image() -> np.ndarray:
    qxy = np.linspace(-2.2, 2.2, 360)
    qz = np.linspace(0.0, 2.4, 260)
    x_grid, z_grid = np.meshgrid(qxy, qz)
    image = 0.04 + 0.02 * np.sin(6.0 * x_grid) ** 2
    peaks = [
        (-1.2, 0.7, 0.10, 0.08, 1.2),
        (-0.35, 1.35, 0.08, 0.12, 0.9),
        (0.42, 1.1, 0.09, 0.10, 1.1),
        (1.15, 1.75, 0.12, 0.09, 0.8),
    ]
    for x0, z0, sx, sz, amp in peaks:
        image += amp * np.exp(
            -0.5 * ((x_grid - x0) / sx) ** 2 - 0.5 * ((z_grid - z0) / sz) ** 2
        )
    return image.astype(float)


def _select_file(window: MainWindow) -> None:
    root = window.data_tree.tree.topLevelItem(0)
    if root is not None and root.childCount():
        window.data_tree.tree.setCurrentItem(root.child(0))
        _process_events()


def _set_tab(window: MainWindow, title: str) -> None:
    for index in range(window.tabs.count()):
        if window.tabs.tabText(index) == title:
            window.tabs.setCurrentIndex(index)
            _process_events()
            return


def _settings(name: str) -> QtCore.QSettings:
    path = TMP_DIR / f"{name}.ini"
    return QtCore.QSettings(str(path), QtCore.QSettings.Format.IniFormat)


def _save_widget(
    app: QtWidgets.QApplication,
    widget: QtWidgets.QWidget,
    filename: str,
    *,
    size: tuple[int, int],
) -> Path:
    path = OUTPUT_DIR / filename
    widget.resize(*size)
    widget.show()
    widget.raise_()
    for _ in range(6):
        _process_events()
    pixmap = widget.grab()
    if pixmap.isNull():
        raise RuntimeError(f"Could not capture screenshot: {filename}")
    pixmap.save(str(path), "PNG")
    widget.close()
    widget.deleteLater()
    _process_events()
    app.processEvents()
    return path


def _process_events() -> None:
    app = QtWidgets.QApplication.instance()
    if app is not None:
        QtCore.QCoreApplication.sendPostedEvents(None, 0)
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
