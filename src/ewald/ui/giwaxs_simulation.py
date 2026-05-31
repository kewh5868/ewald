"""Standalone and embedded GIWAXS simulation UI."""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from ewald.crystallography.cif import (
    extract_cif_lattice_parameters,
    infer_crystal_system_from_lattice,
)
from ewald.data.models import ProjectState
from ewald.simulation.giwaxs import (
    PEAK_TABLE_ATTR,
    SIMULATION_MODE_EWALD_SWEEP,
    SIMULATION_MODE_PATTERN,
    EwaldSphereSweepParameters,
    GIWAXSImageComparison,
    GIWAXSSimulationParameters,
    calculate_giwaxs_peak_rows,
    compare_giwaxs_images,
    is_ewald_sphere_sweep_data,
    is_ewald_sphere_sweep_record,
    load_simulation_data,
    load_structure,
    reconstruct_ewald_sphere_points,
    run_and_store_ewald_sphere_sweep,
    run_and_store_simulation,
)
from ewald.ui.data_viewer import ImagePlotToolbar
from ewald.ui.notation import (
    QSPACE_UNITS_HTML,
    QXY_HTML,
    QZ_HTML,
    enable_rich_text_items,
    qt_tooltip,
    rich_label,
    set_qspace_axis_labels,
    set_rich_text_table_headers,
)

STRUCTURE_FILE_FILTER = (
    "Structure Files (*.cif *.mcif POSCAR* CONTCAR* *.vasp);;All Files (*)"
)
STRUCTURE_HISTORY_SETTING = "giwaxs/structure_history"
STRUCTURE_HISTORY_LIMIT = 12
SIMULATION_CACHE_STATUS_READY = "Cached result displayed."
SIMULATION_CACHE_STATUS_MISS = (
    "Displayed plot may not match current inputs; run to compute this view."
)
PATTERN_CACHE_SCALABLE_FIELDS = {
    "hkl_extent",
    "resolution_x",
    "resolution_z",
}
ORIENTATION_PRESETS = {
    "single_crystal": {
        "label": "Single crystal",
        "theta_x": 90.0,
        "theta_y": 0.0,
        "sigma_theta": 0.005,
        "sigma_phi": 0.005,
        "sigma_r": 0.02,
        "tooltip": (
            "Narrow tilt and azimuth spreads, approximating one oriented "
            "crystal domain."
        ),
    },
    "vertical_2d": {
        "label": "2D vertical",
        "theta_x": 90.0,
        "theta_y": 0.0,
        "sigma_theta": 0.03,
        "sigma_phi": 0.25,
        "sigma_r": 0.035,
        "tooltip": (
            "Default GIWAXS-like film orientation with the layer normal "
            "upright in the viewer."
        ),
    },
    "horizontal_2d": {
        "label": "2D horizontal",
        "theta_x": 0.0,
        "theta_y": 0.0,
        "sigma_theta": 0.03,
        "sigma_phi": 0.25,
        "sigma_r": 0.035,
        "tooltip": (
            "Layer-normal direction laid closer to the sample plane for a "
            "limiting in-plane structure view."
        ),
    },
    "isotropic": {
        "label": "Fully isotropic",
        "theta_x": 0.0,
        "theta_y": 0.0,
        "sigma_theta": 2.0,
        "sigma_phi": math.pi,
        "sigma_r": 0.06,
        "tooltip": (
            "Very broad tilt and azimuth spreads, approximating a powder-like "
            "orientation distribution."
        ),
    },
}
HKL_TABLE_HEADERS = [
    "(hkl)",
    f"{QXY_HTML} ({QSPACE_UNITS_HTML})",
    f"{QZ_HTML} ({QSPACE_UNITS_HTML})",
    "Intensity",
    "Status",
]
PEAK_INFO_SIGNIFICANT_DIGITS = 3
STRUCTURE_PREVIEW_MIN_SIDE = 160
STRUCTURE_PREVIEW_PREFERRED_SIDE = 200
STRUCTURE_PREVIEW_MAX_SIDE = 220

try:  # pragma: no cover - exercised through Qt tests when installed.
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None


class GIWAXSSimulationResultPane(QtWidgets.QWidget):
    """Display one stored GIWAXS simulation image."""

    sweepFrameChanged = QtCore.Signal(float, float)
    sweepPlaybackStarted = QtCore.Signal()
    sweepPlaybackPaused = QtCore.Signal()
    sweepPlaybackStopped = QtCore.Signal()

    def __init__(
        self,
        project: ProjectState,
        simulation_id: str | None = None,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.simulation_id = simulation_id
        self.image_data: np.ndarray | None = None
        self.data_array: Any | None = None
        self.axis_ranges: tuple[float, float, float, float] | None = None
        self.view_box: Any | None = None
        self.sweep_data: Any | None = None
        self.frame_index = 0
        self.peak_rows: list[dict[str, Any]] = []
        self.reconstruction_windows: list[QtWidgets.QMainWindow] = []
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance_sweep_frame)
        self._build_style_controls()
        self._build_plot()
        self._build_playback_controls()
        self._build_peak_controls()
        self._build_peak_table()
        self._build_metadata()
        self._build_layout()
        self.set_simulation(simulation_id)

    def set_simulation(self, simulation_id: str | None) -> None:
        self.simulation_id = simulation_id
        record = (
            self.project.simulations.get(simulation_id)
            if simulation_id is not None
            else None
        )
        if not record:
            self.metadata.setPlainText("No GIWAXS simulation selected.")
            self._clear_display()
            self._set_peak_rows([])
            return
        data = load_simulation_data(record)
        if data is None:
            self.metadata.setPlainText(
                "Simulation output could not be loaded."
            )
            self._clear_display()
            self._set_peak_rows([])
            return
        self._display_dataarray(data)
        self._set_peak_rows(_peak_rows_from_data(data, record))
        self.metadata.setHtml(_record_html(record))

    def _build_style_controls(self) -> None:
        self.colormap_combo = QtWidgets.QComboBox()
        for name in ("viridis", "magma", "turbo", "gray"):
            self.colormap_combo.addItem(name.title(), name)
        self.colormap_combo.currentIndexChanged.connect(
            self._apply_image_style
        )

        self.level_min = _level_spinbox()
        self.level_max = _level_spinbox()
        self.level_min.valueChanged.connect(self._apply_manual_levels)
        self.level_max.valueChanged.connect(self._apply_manual_levels)

        self.quantile_check = QtWidgets.QCheckBox("Quantile")
        self.quantile_check.setChecked(True)
        self.quantile_check.toggled.connect(self._apply_image_style)
        self.quantile_low = _quantile_spinbox(1.0)
        self.quantile_high = _quantile_spinbox(99.0)
        self.quantile_low.valueChanged.connect(
            self._handle_quantile_controls_changed
        )
        self.quantile_high.valueChanged.connect(
            self._handle_quantile_controls_changed
        )

        self.auto_contrast_button = QtWidgets.QToolButton()
        self.auto_contrast_button.setText("Auto")
        self.auto_contrast_button.clicked.connect(self._set_quantile_levels)

        self.zoom_in_button = QtWidgets.QToolButton()
        self.zoom_in_button.setText("Zoom In")
        self.zoom_in_button.clicked.connect(lambda: self._zoom_image(0.75))
        self.zoom_out_button = QtWidgets.QToolButton()
        self.zoom_out_button.setText("Zoom Out")
        self.zoom_out_button.clicked.connect(lambda: self._zoom_image(1.35))
        self.zoom_fit_button = QtWidgets.QToolButton()
        self.zoom_fit_button.setText("Autoscale")
        self.zoom_fit_button.clicked.connect(self._reset_image_zoom)
        self.pan_button = QtWidgets.QToolButton()
        self.pan_button.setText("Pan")
        self.pan_button.setCheckable(True)
        self.pan_button.toggled.connect(self._set_pan_mode)

    def _build_plot(self) -> None:
        if pg is None:
            self.plot_widget = QtWidgets.QLabel("GIWAXS Simulation")
            self.plot_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.image_item = None
            self.hkl_scatter = None
            self.view_box = None
            return
        self.plot_widget = pg.PlotWidget()
        self.view_box = self.plot_widget.getViewBox()
        set_qspace_axis_labels(self.plot_widget)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self._set_pan_mode(False)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot_widget.addItem(self.image_item)
        self.hkl_scatter = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkBrush(255, 255, 255, 45),
            pen=pg.mkPen("#f59e0b", width=1.3),
            hoverable=True,
            hoverBrush=pg.mkBrush("#f59e0b"),
            hoverPen=pg.mkPen("#111827", width=1.0),
            tip=_peak_tip,
        )
        self.hkl_scatter.setZValue(15)
        self.plot_widget.addItem(self.hkl_scatter)

    def _build_playback_controls(self) -> None:
        self.playback_controls = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(self.playback_controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.play_button = QtWidgets.QToolButton()
        self.play_button.setText("Play")
        self.play_button.setToolTip(
            "Animate frames in the Ewald sphere sweep. Pause freezes the "
            "current frame; Stop restores the selected crystal orientation."
        )
        self.play_button.clicked.connect(self._toggle_sweep_playback)
        layout.addWidget(self.play_button)

        self.stop_button = QtWidgets.QToolButton()
        self.stop_button.setText("Stop")
        self.stop_button.setToolTip(
            "Stop sweep playback and snap the structure preview back to the "
            "selected theta X/theta Y values."
        )
        self.stop_button.clicked.connect(self._stop_sweep_playback)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setToolTip(
            "Scrub through Ewald sphere sweep frames. Each frame uses one "
            "theta X/theta Y orientation."
        )
        self.frame_slider.valueChanged.connect(self._set_sweep_frame)
        layout.addWidget(self.frame_slider, stretch=1)

        self.frame_label = QtWidgets.QLabel("Frame 0/0")
        self.frame_label.setMinimumWidth(170)
        layout.addWidget(self.frame_label)

        self.frame_interval = _spinbox(150, 25, 5000)
        self.frame_interval.setSuffix(" ms")
        self.frame_interval.setToolTip("Milliseconds between sweep frames.")
        layout.addWidget(self.frame_interval)

        self.open_reconstruction_button = QtWidgets.QPushButton("Open 3D")
        self.open_reconstruction_button.clicked.connect(
            self.open_reconstruction_viewer
        )
        layout.addWidget(self.open_reconstruction_button)
        self.playback_controls.setVisible(False)
        self.open_reconstruction_button.setEnabled(False)

    def _build_peak_controls(self) -> None:
        self.show_hkl_points = QtWidgets.QCheckBox("Point grid")
        self.show_hkl_points.setChecked(True)
        self.show_hkl_points.setToolTip(
            qt_tooltip(
                f"Show calculated (hkl) {QXY_HTML}/{QZ_HTML} points on the "
                "simulation image."
            )
        )
        self.show_hkl_points.toggled.connect(self._sync_hkl_overlay)
        self.hkl_count_label = QtWidgets.QLabel("0 calculated points")
        self.hkl_count_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

    def _build_peak_table(self) -> None:
        self.hkl_table = QtWidgets.QTableWidget(0, len(HKL_TABLE_HEADERS))
        set_rich_text_table_headers(self.hkl_table, HKL_TABLE_HEADERS)
        enable_rich_text_items(self.hkl_table)
        self.hkl_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.hkl_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.hkl_table.setAlternatingRowColors(True)
        self.hkl_table.setMinimumHeight(150)
        self.hkl_table.setMaximumHeight(230)

    def _build_metadata(self) -> None:
        self.metadata = QtWidgets.QTextEdit()
        self.metadata.setReadOnly(True)
        self.metadata.setMaximumHeight(150)

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self.plot_toolbar = ImagePlotToolbar(
            colormap_combo=self.colormap_combo,
            level_min=self.level_min,
            level_max=self.level_max,
            quantile_check=self.quantile_check,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
            auto_contrast_button=self.auto_contrast_button,
            zoom_in_button=self.zoom_in_button,
            zoom_out_button=self.zoom_out_button,
            autoscale_button=self.zoom_fit_button,
            pan_button=self.pan_button,
        )
        layout.addWidget(self.plot_toolbar)
        layout.addWidget(self.plot_widget, stretch=1)
        layout.addWidget(self.playback_controls)
        overlay_layout = QtWidgets.QHBoxLayout()
        overlay_layout.addWidget(self.show_hkl_points)
        overlay_layout.addStretch(1)
        overlay_layout.addWidget(self.hkl_count_label)
        layout.addLayout(overlay_layout)
        layout.addWidget(self.hkl_table)
        layout.addWidget(self.metadata)

    def _display_dataarray(self, data: Any) -> None:
        self.data_array = data
        if is_ewald_sphere_sweep_data(data):
            self.sweep_data = data.transpose("theta_y", "theta_x", "qz", "qxy")
            frame_count = int(
                self.sweep_data.sizes["theta_y"]
                * self.sweep_data.sizes["theta_x"]
            )
            self.playback_controls.setVisible(True)
            self.open_reconstruction_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, max(0, frame_count - 1))
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            self._set_sweep_frame(0)
            return

        self.play_timer.stop()
        self.play_button.setText("Play")
        self.stop_button.setEnabled(False)
        self.sweep_data = None
        self.playback_controls.setVisible(False)
        self.open_reconstruction_button.setEnabled(False)
        self._display_2d_dataarray(data)

    def current_sweep_orientation(self) -> tuple[float, float] | None:
        if self.sweep_data is None:
            return None
        frame_count = int(
            self.sweep_data.sizes["theta_y"] * self.sweep_data.sizes["theta_x"]
        )
        if frame_count <= 0:
            return None
        theta_x_count = int(self.sweep_data.sizes["theta_x"])
        frame_index = max(0, min(int(self.frame_index), frame_count - 1))
        theta_y_index = frame_index // theta_x_count
        theta_x_index = frame_index % theta_x_count
        frame = self.sweep_data.isel(
            theta_y=theta_y_index,
            theta_x=theta_x_index,
        )
        return (
            float(frame.coords["theta_x"].values),
            float(frame.coords["theta_y"].values),
        )

    def _display_2d_dataarray(self, data: Any) -> None:
        array = np.asarray(data.values, dtype=float)
        self.image_data = array
        finite = array[np.isfinite(array)]
        if finite.size:
            self.level_min.blockSignals(True)
            self.level_max.blockSignals(True)
            self.level_min.setValue(float(np.nanmin(finite)))
            self.level_max.setValue(float(np.nanmax(finite)))
            self.level_min.blockSignals(False)
            self.level_max.blockSignals(False)
        if self.image_item is None or pg is None:
            self._set_quantile_levels()
            return
        self.image_item.setImage(array, autoLevels=False)
        qxy = np.asarray(data.coords["qxy"].values, dtype=float)
        qz = np.asarray(data.coords["qz"].values, dtype=float)
        qxy_min = float(np.nanmin(qxy))
        qxy_max = float(np.nanmax(qxy))
        qz_min = float(np.nanmin(qz))
        qz_max = float(np.nanmax(qz))
        self.axis_ranges = (qxy_min, qxy_max, qz_min, qz_max)
        self.image_item.setRect(
            QtCore.QRectF(
                qxy_min,
                qz_min,
                qxy_max - qxy_min,
                qz_max - qz_min,
            )
        )
        self.plot_widget.setRange(
            xRange=(qxy_min, qxy_max),
            yRange=(qz_min, qz_max),
        )
        self._set_quantile_levels()

    def _clear_display(self) -> None:
        self.play_timer.stop()
        self.play_button.setText("Play")
        self.stop_button.setEnabled(False)
        self.image_data = None
        self.data_array = None
        self.axis_ranges = None
        self.sweep_data = None
        self.frame_index = 0
        self.playback_controls.setVisible(False)
        self.open_reconstruction_button.setEnabled(False)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText("Frame 0/0")
        if self.image_item is not None:
            self.image_item.clear()

    def _set_sweep_frame(self, index: int) -> None:
        if self.sweep_data is None:
            return
        frame_count = int(
            self.sweep_data.sizes["theta_y"] * self.sweep_data.sizes["theta_x"]
        )
        if frame_count <= 0:
            return
        self.frame_index = max(0, min(int(index), frame_count - 1))
        theta_x_count = int(self.sweep_data.sizes["theta_x"])
        theta_y_index = self.frame_index // theta_x_count
        theta_x_index = self.frame_index % theta_x_count
        frame = self.sweep_data.isel(
            theta_y=theta_y_index,
            theta_x=theta_x_index,
        )
        theta_x = float(frame.coords["theta_x"].values)
        theta_y = float(frame.coords["theta_y"].values)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.frame_index)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(
            (
                f"Frame {self.frame_index + 1}/{frame_count}  "
                f"theta X {theta_x:g}, theta Y {theta_y:g}"
            )
        )
        self._display_2d_dataarray(frame)
        self.sweepFrameChanged.emit(theta_x, theta_y)

    def _toggle_sweep_playback(self) -> None:
        if self.sweep_data is None:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("Play")
            self.sweepPlaybackPaused.emit()
            return
        self.play_timer.start(self.frame_interval.value())
        self.play_button.setText("Pause")
        self.stop_button.setEnabled(True)
        self.sweepPlaybackStarted.emit()

    def _stop_sweep_playback(self) -> None:
        if self.sweep_data is None:
            return
        self.play_timer.stop()
        self.play_button.setText("Play")
        self.stop_button.setEnabled(False)
        self.sweepPlaybackStopped.emit()

    def _advance_sweep_frame(self) -> None:
        if self.sweep_data is None:
            self.play_timer.stop()
            return
        frame_count = int(
            self.sweep_data.sizes["theta_y"] * self.sweep_data.sizes["theta_x"]
        )
        if frame_count <= 0:
            return
        self._set_sweep_frame((self.frame_index + 1) % frame_count)

    def open_reconstruction_viewer(self) -> QtWidgets.QMainWindow | None:
        if self.sweep_data is None or self.simulation_id is None:
            QtWidgets.QMessageBox.information(
                self,
                "3D Reconstruction",
                "Select an Ewald sphere theta sweep before opening 3D.",
            )
            return None
        record = self.project.simulations.get(self.simulation_id, {})
        viewer = EwaldSphereReconstructionWindow(
            self.sweep_data,
            record if isinstance(record, dict) else {},
            parent=self.window(),
        )
        self.reconstruction_windows.append(viewer)
        viewer.destroyed.connect(
            lambda _obj=None, window=viewer: self._forget_reconstruction_window(
                window
            )
        )
        viewer.show()
        viewer.raise_()
        viewer.activateWindow()
        return viewer

    def _forget_reconstruction_window(
        self,
        window: QtWidgets.QMainWindow,
    ) -> None:
        if window in self.reconstruction_windows:
            self.reconstruction_windows.remove(window)

    def _apply_manual_levels(self) -> None:
        if not self.quantile_check.isChecked():
            self._apply_image_style()

    def _zoom_image(self, factor: float) -> None:
        if pg is None or self.view_box is None:
            return
        self.view_box.scaleBy((factor, factor))

    def _reset_image_zoom(self) -> None:
        if pg is None or self.plot_widget is None or self.axis_ranges is None:
            return
        x_min, x_max, y_min, y_max = self.axis_ranges
        self.plot_widget.setRange(
            xRange=(x_min, x_max),
            yRange=(y_min, y_max),
            padding=0.0,
        )

    def _set_pan_mode(self, enabled: bool) -> None:
        if self.view_box is not None:
            self.view_box.setMouseEnabled(x=enabled, y=enabled)
        if self.plot_widget is not None:
            self.plot_widget.setCursor(
                QtCore.Qt.CursorShape.OpenHandCursor
                if enabled
                else QtCore.Qt.CursorShape.ArrowCursor
            )

    def _handle_quantile_controls_changed(self) -> None:
        if self.quantile_check.isChecked():
            self._set_quantile_levels()
        else:
            self._apply_image_style()

    def _set_quantile_levels(self) -> None:
        if self.image_data is None:
            return
        low = min(self.quantile_low.value(), self.quantile_high.value())
        high = max(self.quantile_low.value(), self.quantile_high.value())
        finite = self.image_data[np.isfinite(self.image_data)]
        if not finite.size:
            return
        levels = np.nanquantile(finite, [low / 100.0, high / 100.0])
        self.level_min.blockSignals(True)
        self.level_max.blockSignals(True)
        self.level_min.setValue(float(levels[0]))
        self.level_max.setValue(float(levels[1]))
        self.level_min.blockSignals(False)
        self.level_max.blockSignals(False)
        self._apply_image_style()

    def _apply_image_style(self) -> None:
        if self.image_item is None or self.image_data is None:
            return
        if self.quantile_check.isChecked():
            low = min(self.quantile_low.value(), self.quantile_high.value())
            high = max(self.quantile_low.value(), self.quantile_high.value())
            finite = self.image_data[np.isfinite(self.image_data)]
            if finite.size:
                levels = np.nanquantile(finite, [low / 100.0, high / 100.0])
            else:
                levels = (self.level_min.value(), self.level_max.value())
        else:
            levels = (self.level_min.value(), self.level_max.value())
        if levels[0] == levels[1]:
            levels = (levels[0], levels[1] + 1.0)
        self.image_item.setLevels(levels)
        self._apply_colormap()

    def _apply_colormap(self) -> None:
        if pg is None or self.image_item is None:
            return
        name = str(self.colormap_combo.currentData())
        try:
            cmap = pg.colormap.get(name)
            self.image_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        except Exception:
            if name == "gray":
                self.image_item.setLookupTable(None)

    def _set_peak_rows(self, rows: list[dict[str, Any]]) -> None:
        self.peak_rows = [row for row in rows if _valid_peak_row(row)]
        self._sync_peak_table()
        self._sync_hkl_overlay()

    def _sync_peak_table(self) -> None:
        self.hkl_table.setRowCount(len(self.peak_rows))
        for row_index, row in enumerate(self.peak_rows):
            values = [
                _format_hkl(row),
                _format_float(row.get("qxy")),
                _format_float(row.get("qz")),
                _format_float(row.get("intensity")),
                _peak_status_text(row),
            ]
            tooltip = _peak_tooltip(row)
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(tooltip)
                if _is_forbidden_peak_row(row):
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#7f1d1d")))
                    item.setBackground(QtGui.QBrush(QtGui.QColor("#fef2f2")))
                self.hkl_table.setItem(row_index, column, item)
        self.hkl_table.resizeColumnsToContents()

    def _sync_hkl_overlay(self, *_args: Any) -> None:
        count = len(self.peak_rows)
        forbidden_count = sum(
            1 for row in self.peak_rows if _is_forbidden_peak_row(row)
        )
        indexable_count = count - forbidden_count
        noun = "point" if indexable_count == 1 else "points"
        suffix = (
            f", {forbidden_count} forbidden"
            if forbidden_count
            else ""
        )
        self.hkl_count_label.setText(
            f"{indexable_count} indexable {noun}{suffix}"
        )
        if self.hkl_scatter is None or pg is None:
            return
        if not self.show_hkl_points.isChecked():
            self.hkl_scatter.setData(spots=[])
            self.hkl_scatter.setVisible(False)
            return
        spots = [
            {
                "pos": (float(row["qxy"]), float(row["qz"])),
                "data": row,
            }
            for row in self.peak_rows
            if not _is_forbidden_peak_row(row)
        ]
        self.hkl_scatter.setData(spots=spots, hoverable=True, tip=_peak_tip)
        self.hkl_scatter.setVisible(bool(spots))


class GIWAXSComparisonPane(QtWidgets.QWidget):
    """Show target, simulated, and residual GIWAXS fit maps."""

    def __init__(
        self,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.comparison: GIWAXSImageComparison | None = None
        self.ranked_comparisons: list[dict[str, Any]] = []
        self._syncing_ranked_table = False
        self.image_items: list[Any] = []
        self._build_widgets()
        self.clear("Run a comparison to display fit maps.")

    def set_comparison(self, comparison: GIWAXSImageComparison) -> None:
        self.ranked_comparisons = []
        self._sync_ranked_table([])
        self._set_active_comparison(comparison)

    def set_ranked_comparisons(
        self,
        comparisons: list[dict[str, Any]],
    ) -> None:
        self.ranked_comparisons = list(comparisons)
        self._sync_ranked_table(self.ranked_comparisons)
        if self.ranked_comparisons:
            self._set_active_comparison(
                self.ranked_comparisons[0]["comparison"]
            )
            self.ranked_table.selectRow(0)
        else:
            self.clear("No ranked comparison maps are available.")

    def _set_active_comparison(
        self,
        comparison: GIWAXSImageComparison,
    ) -> None:
        self.comparison = comparison
        metrics = comparison.metrics
        self.summary_label.setText(
            (
                "Difference RMSE "
                f"{_format_float(metrics.get('difference_rmse'))} | "
                f"corr {_format_float(metrics.get('correlation'))} | "
                f"peak RMSE {_format_float(metrics.get('peak_rmse'))}"
            )
        )
        rows = [
            ("difference_rmse", "Difference RMSE"),
            ("difference_mae", "Difference MAE"),
            ("difference_max_abs", "Max |difference|"),
            ("fit_score", "Objective score"),
            ("weighted_rmse", "Weighted RMSE"),
            ("rmse", "RMSE"),
            ("peak_rmse", "Peak RMSE"),
            ("mae", "MAE"),
            ("correlation", "Correlation"),
            ("residual_correlation_score", "Residual-correlation score"),
            ("peak_focus_score", "Peak focus score"),
            ("peak_correlation", "Peak correlation"),
            ("peak_overlap_jaccard", "Peak overlap"),
            ("peak_precision", "Peak precision"),
            ("peak_recall", "Peak recall"),
            ("scale", "Scale"),
            ("offset", "Offset"),
        ]
        self.metrics_table.setRowCount(len(rows))
        for row_index, (key, label) in enumerate(rows):
            self.metrics_table.setItem(
                row_index,
                0,
                QtWidgets.QTableWidgetItem(label),
            )
            self.metrics_table.setItem(
                row_index,
                1,
                QtWidgets.QTableWidgetItem(_format_float(metrics.get(key))),
            )
        self.metrics_table.resizeColumnsToContents()
        self._set_plot_titles(
            comparison.target_label,
            f"Fitted {comparison.simulated_label}",
            "Difference",
        )
        self._display_maps(comparison)

    def clear(self, message: str = "") -> None:
        self.comparison = None
        self.ranked_comparisons = []
        self.summary_label.setText(message)
        self._sync_ranked_table([])
        self.metrics_table.setRowCount(0)
        self._set_plot_titles("Target", "Simulated", "Difference")
        for item in self.image_items:
            if item is not None and hasattr(item, "clear"):
                item.clear()

    def _build_widgets(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        if pg is None:
            self.plot_widgets = [
                QtWidgets.QLabel(label)
                for label in ("Target", "Simulated", "Difference")
            ]
            for widget in self.plot_widgets:
                widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.image_items = [None, None, None]
        else:
            self.plot_widgets = []
            self.image_items = []
            for title in ("Target", "Simulated", "Difference"):
                plot = pg.PlotWidget(title=title)
                set_qspace_axis_labels(plot)
                plot.showGrid(x=True, y=True, alpha=0.18)
                image_item = pg.ImageItem(axisOrder="row-major")
                plot.addItem(image_item)
                self.plot_widgets.append(plot)
                self.image_items.append(image_item)
        plot_layout = QtWidgets.QHBoxLayout()
        for widget in self.plot_widgets:
            plot_layout.addWidget(widget)
        layout.addLayout(plot_layout, stretch=1)
        self.ranked_table = QtWidgets.QTableWidget(0, 5)
        self.ranked_table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Structure",
                "Difference RMSE",
                "Weighted RMSE",
                "Simulation",
            ]
        )
        self.ranked_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.ranked_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.ranked_table.setMaximumHeight(125)
        self.ranked_table.itemSelectionChanged.connect(
            self._handle_ranked_selection_changed
        )
        self.ranked_table.setVisible(False)
        layout.addWidget(self.ranked_table)
        self.metrics_table = QtWidgets.QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.metrics_table.setMaximumHeight(170)
        layout.addWidget(self.metrics_table)

    def _set_plot_titles(self, *titles: str) -> None:
        for widget, title in zip(self.plot_widgets, titles):
            if hasattr(widget, "setTitle"):
                widget.setTitle(title)
            elif isinstance(widget, QtWidgets.QLabel):
                widget.setText(title)

    def _display_maps(self, comparison: GIWAXSImageComparison) -> None:
        if pg is None:
            return
        maps = (
            comparison.target,
            comparison.fitted_simulated,
            comparison.difference,
        )
        for plot, image_item, data_array, is_difference in zip(
            self.plot_widgets,
            self.image_items,
            maps,
            (False, False, True),
        ):
            image = np.asarray(data_array.values, dtype=float)
            image_item.setImage(image, autoLevels=False)
            x_min, x_max, y_min, y_max = _qspace_extent(data_array)
            image_item.setRect(
                QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
            )
            if is_difference:
                vmax = float(np.nanquantile(np.abs(image), 0.99))
                if not np.isfinite(vmax) or vmax <= 0.0:
                    vmax = 1.0
                image_item.setLevels((-vmax, vmax))
                cmap = pg.colormap.get("CET-D1")
            else:
                finite = image[np.isfinite(image)]
                vmax = (
                    float(np.nanquantile(finite, 0.995))
                    if finite.size
                    else 1.0
                )
                image_item.setLevels((0.0, max(vmax, 1.0e-9)))
                cmap = pg.colormap.get("viridis")
            image_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
            plot.setRange(xRange=(x_min, x_max), yRange=(y_min, y_max))

    def _sync_ranked_table(self, comparisons: list[dict[str, Any]]) -> None:
        self._syncing_ranked_table = True
        try:
            self.ranked_table.clearSelection()
            self.ranked_table.setRowCount(len(comparisons))
            self.ranked_table.setVisible(bool(comparisons))
            for row, item in enumerate(comparisons):
                comparison = item["comparison"]
                metrics = comparison.metrics
                record = item.get("record", {})
                rank_item = QtWidgets.QTableWidgetItem(str(row + 1))
                rank_item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
                values = [
                    rank_item,
                    QtWidgets.QTableWidgetItem(
                        _comparison_structure_label(item)
                    ),
                    QtWidgets.QTableWidgetItem(
                        _format_float(metrics.get("difference_rmse"))
                    ),
                    QtWidgets.QTableWidgetItem(
                        _format_float(metrics.get("weighted_rmse"))
                    ),
                    QtWidgets.QTableWidgetItem(
                        str(
                            item.get("simulation_id")
                            or record.get("simulation_id")
                            or ""
                        )
                    ),
                ]
                for column, table_item in enumerate(values):
                    table_item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
                    self.ranked_table.setItem(row, column, table_item)
            self.ranked_table.resizeColumnsToContents()
        finally:
            self._syncing_ranked_table = False

    def _handle_ranked_selection_changed(self) -> None:
        if self._syncing_ranked_table:
            return
        selected = self.ranked_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if row < 0 or row >= len(self.ranked_comparisons):
            return
        self._set_active_comparison(self.ranked_comparisons[row]["comparison"])


class EwaldSphereReconstructionWindow(QtWidgets.QMainWindow):
    """Separate 3D point-cloud viewer for a stored Ewald sphere
    sweep."""

    def __init__(
        self,
        data: Any,
        record: dict[str, Any],
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.data = data
        self.record = record
        self.setWindowTitle("Ewald Sphere Reconstruction")
        self.resize(900, 720)
        self._build_controls()
        self._build_plot()
        self._build_layout()
        self._plot_reconstruction()

    def _build_controls(self) -> None:
        self.quantile = _quantile_spinbox(99.5)
        self.quantile.setToolTip(
            "Intensity percentile used for point selection."
        )
        self.max_points = _spinbox(50000, 500, 500000)
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._plot_reconstruction)
        self.status_label = QtWidgets.QLabel()

    def _build_plot(self) -> None:
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            from matplotlib.backends.backend_qtagg import (
                NavigationToolbar2QT as NavigationToolbar,
            )
            from matplotlib.figure import Figure
        except Exception as exc:  # pragma: no cover - dependency fallback.
            self.figure = None
            self.canvas = QtWidgets.QLabel(f"Matplotlib is unavailable: {exc}")
            self.canvas.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.toolbar = None
            self.axes = None
            return

        self.figure = Figure(figsize=(7, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.axes = self.figure.add_subplot(111, projection="3d")

    def _build_layout(self) -> None:
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Percentile"))
        controls.addWidget(self.quantile)
        controls.addWidget(QtWidgets.QLabel("Max points"))
        controls.addWidget(self.max_points)
        controls.addWidget(self.refresh_button)
        controls.addStretch(1)
        controls.addWidget(self.status_label)
        layout.addLayout(controls)

        if self.toolbar is not None:
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        self.setCentralWidget(root)

    def _plot_reconstruction(self) -> None:
        if self.axes is None:
            return
        points, intensities = reconstruct_ewald_sphere_points(
            self.data,
            intensity_quantile=self.quantile.value() / 100.0,
            max_points=self.max_points.value(),
        )
        self.axes.clear()
        self.axes.set_xlabel("$q_{x}$ (Å$^{-1}$)")
        self.axes.set_ylabel("$q_{y}$ (Å$^{-1}$)")
        self.axes.set_zlabel("$q_{z}$ (Å$^{-1}$)")
        if points.size:
            self.axes.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=intensities,
                cmap="viridis",
                s=4,
                alpha=0.72,
                linewidths=0,
            )
            _set_equal_3d_limits(self.axes, points)
        title = self.record.get("structure_name") or self.record.get(
            "simulation_id",
            "Ewald sphere",
        )
        self.axes.set_title(str(title))
        self.status_label.setText(f"{len(points):,} points")
        self.canvas.draw_idle()


class SquarePlotContainer(QtWidgets.QWidget):
    """Keep the contained plotting widget square inside a flexible
    layout."""

    def __init__(
        self,
        child: QtWidgets.QWidget,
        *,
        minimum_side: int = STRUCTURE_PREVIEW_MIN_SIDE,
        preferred_side: int = STRUCTURE_PREVIEW_PREFERRED_SIDE,
        maximum_side: int = STRUCTURE_PREVIEW_MAX_SIDE,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.child = child
        self.minimum_side = int(minimum_side)
        self.preferred_side = int(preferred_side)
        self.maximum_side = int(maximum_side)
        self.child.setParent(self)
        self.setMinimumSize(self.minimum_side, self.minimum_side)
        self.setMaximumHeight(self.maximum_side)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self.preferred_side, self.preferred_side)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(
            self.minimum_side,
            min(self.maximum_side, int(width)),
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        side = min(self.width(), self.height())
        x_offset = (self.width() - side) // 2
        y_offset = (self.height() - side) // 2
        self.child.setGeometry(x_offset, y_offset, side, side)
        super().resizeEvent(event)


class UnitCellStructureView(QtWidgets.QWidget):
    """Compact unit-cell atom preview for a loaded structure."""

    def __init__(
        self,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.structure_id: str | None = None
        self.atom_count = 0
        self.species_text = ""
        self.theta_x_deg = 90.0
        self.theta_y_deg = 0.0
        self._structure: Any | None = None
        self._edge_items: list[Any] = []
        self._schematic_items: list[Any] = []
        self._atom_item: Any | None = None
        self._build_widgets()
        self.clear()

    def set_structure(
        self,
        structure: Any | None,
        *,
        structure_id: str | None = None,
    ) -> None:
        if structure is None:
            self.clear()
            return
        self._structure = structure
        self.structure_id = structure_id or structure.structure_id
        self.atom_count = len(structure.species)
        self.species_text = ", ".join(sorted(set(structure.species)))
        self.info_label.setText(
            "\n".join(
                [
                    f"Loaded: {structure.path.stem}",
                    (
                        f"Atoms: {self.atom_count} | "
                        f"Species: {self.species_text}"
                    ),
                    _lattice_summary(structure.lattice),
                ]
            )
        )
        self._draw_structure(structure)

    def set_orientation(
        self,
        theta_x_deg: float,
        theta_y_deg: float,
        *,
        source: str = "Selected orientation",
    ) -> None:
        self.theta_x_deg = float(theta_x_deg)
        self.theta_y_deg = float(theta_y_deg)
        self.orientation_label.setText(
            (
                f"{source}: theta X {self.theta_x_deg:g} deg, "
                f"theta Y {self.theta_y_deg:g} deg"
            )
        )
        if self._structure is not None:
            self._draw_structure(self._structure)

    def set_error(self, message: str) -> None:
        self.clear()
        self.info_label.setText(message)

    def clear(self) -> None:
        self.structure_id = None
        self.atom_count = 0
        self.species_text = ""
        self._structure = None
        self.info_label.setText("No structure loaded.")
        self.orientation_label.setText(
            "Selected orientation: theta X 90 deg, theta Y 0 deg"
        )
        self._clear_plot()

    def _build_widgets(self) -> None:
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.orientation_label = QtWidgets.QLabel()
        self.orientation_label.setWordWrap(True)
        self.orientation_label.setToolTip(
            "The unit cell rotates with theta X/theta Y. The X-ray source "
            "and detector plane stay fixed in the lab frame."
        )
        layout.addWidget(self.orientation_label)

        if pg is None:
            self.plot_widget = QtWidgets.QLabel("Unit cell preview")
            self.plot_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.plot_widget.setMinimumSize(
                STRUCTURE_PREVIEW_MIN_SIDE,
                STRUCTURE_PREVIEW_MIN_SIDE,
            )
            self.plot_container = SquarePlotContainer(self.plot_widget)
            layout.addWidget(self.plot_container)
            return

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumSize(
            STRUCTURE_PREVIEW_MIN_SIDE,
            STRUCTURE_PREVIEW_MIN_SIDE,
        )
        self.plot_widget.setLabel("bottom", "x", units="A")
        self.plot_widget.setLabel("left", "z", units="A")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.18)
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()
        self.plot_container = SquarePlotContainer(self.plot_widget)
        layout.addWidget(self.plot_container)

    def _draw_structure(self, structure: Any) -> None:
        if pg is None or not hasattr(self.plot_widget, "addItem"):
            return
        self._clear_plot()

        rotation = _orientation_rotation_matrix(
            self.theta_x_deg,
            self.theta_y_deg,
        )
        corners = _unit_cell_corners(structure.lattice) @ rotation
        projected_corners = _project_cell_points(corners)
        for start, end in _unit_cell_edges():
            item = pg.PlotDataItem(
                [projected_corners[start, 0], projected_corners[end, 0]],
                [projected_corners[start, 1], projected_corners[end, 1]],
                pen=pg.mkPen("#64748b", width=1.2),
            )
            self.plot_widget.addItem(item)
            self._edge_items.append(item)

        frac_coords = np.asarray(structure.frac_coords, dtype=float)
        if frac_coords.size == 0:
            self.plot_widget.enableAutoRange()
            return
        atom_positions = (
            frac_coords @ np.asarray(structure.lattice, dtype=float) @ rotation
        )
        order = np.argsort(atom_positions[:, 1])
        atom_positions = atom_positions[order]
        species = [structure.species[index] for index in order]
        projected_atoms = _project_cell_points(atom_positions)
        brushes = [pg.mkBrush(_cpk_color(symbol)) for symbol in species]
        sizes = [11 + 4 * _relative_atom_radius(symbol) for symbol in species]
        self._atom_item = pg.ScatterPlotItem(
            x=projected_atoms[:, 0],
            y=projected_atoms[:, 1],
            size=sizes,
            brush=brushes,
            pen=pg.mkPen("#111827", width=0.6),
        )
        self.plot_widget.addItem(self._atom_item)
        self._draw_lab_geometry_schematic(corners)
        self.plot_widget.enableAutoRange()

    def _draw_lab_geometry_schematic(
        self, rotated_corners: np.ndarray
    ) -> None:
        if pg is None or not hasattr(self.plot_widget, "addItem"):
            return
        center = np.nanmean(rotated_corners, axis=0)
        span = np.nanmax(np.ptp(rotated_corners, axis=0))
        if not np.isfinite(span) or span <= 0.0:
            span = 1.0

        ray_start = center + np.asarray(
            [-1.25 * span, -0.55 * span, 0.12 * span]
        )
        ray_end = center + np.asarray(
            [-0.08 * span, -0.08 * span, 0.02 * span]
        )
        ray_points = _project_cell_points(np.vstack([ray_start, ray_end]))
        ray = pg.PlotDataItem(
            ray_points[:, 0],
            ray_points[:, 1],
            pen=pg.mkPen("#2563eb", width=2.2),
        )
        self.plot_widget.addItem(ray)
        self._schematic_items.append(ray)
        for start, end in _arrowhead_segments(ray_points[0], ray_points[1]):
            head = pg.PlotDataItem(
                [start[0], end[0]],
                [start[1], end[1]],
                pen=pg.mkPen("#2563eb", width=2.2),
            )
            self.plot_widget.addItem(head)
            self._schematic_items.append(head)

        ray_label = pg.TextItem("incident X-ray", color="#1d4ed8")
        ray_label.setAnchor((0.0, 1.0))
        ray_label.setPos(float(ray_points[0, 0]), float(ray_points[0, 1]))
        self.plot_widget.addItem(ray_label)
        self._schematic_items.append(ray_label)

        plane_center = center + np.asarray(
            [0.9 * span, 0.22 * span, 0.15 * span]
        )
        plane_y = np.asarray([0.0, 0.52 * span, 0.0])
        plane_z = np.asarray([0.0, 0.0, 0.62 * span])
        plane_corners = np.vstack(
            [
                plane_center - plane_y - plane_z,
                plane_center + plane_y - plane_z,
                plane_center + plane_y + plane_z,
                plane_center - plane_y + plane_z,
                plane_center - plane_y - plane_z,
            ]
        )
        projected_plane = _project_cell_points(plane_corners)
        detector = pg.PlotDataItem(
            projected_plane[:, 0],
            projected_plane[:, 1],
            pen=pg.mkPen(
                "#16a34a", width=1.4, style=QtCore.Qt.PenStyle.DashLine
            ),
        )
        self.plot_widget.addItem(detector)
        self._schematic_items.append(detector)
        plane_label = pg.TextItem("detector plane", color="#15803d")
        plane_label.setAnchor((0.0, 0.0))
        plane_label.setPos(
            float(projected_plane[2, 0]),
            float(projected_plane[2, 1]),
        )
        self.plot_widget.addItem(plane_label)
        self._schematic_items.append(plane_label)

    def _clear_plot(self) -> None:
        if pg is None or not hasattr(self.plot_widget, "removeItem"):
            return
        for item in self._edge_items:
            self.plot_widget.removeItem(item)
        self._edge_items.clear()
        for item in self._schematic_items:
            self.plot_widget.removeItem(item)
        self._schematic_items.clear()
        if self._atom_item is not None:
            self.plot_widget.removeItem(self._atom_item)
            self._atom_item = None


class OrientationDistributionView(QtWidgets.QWidget):
    """3D octahedral crystallite orientation distribution preview."""

    def __init__(
        self,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.theta_x_deg = 90.0
        self.theta_y_deg = 0.0
        self.sigma_theta = 0.03
        self.sigma_phi = 0.25
        self.sigma_r = 0.035
        self._pending_parameters: (
            tuple[float, float, float, float, float] | None
        ) = None
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(180)
        self._refresh_timer.timeout.connect(self.refresh_distribution)
        self._build_widgets()
        self.refresh_distribution()

    def _build_widgets(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Crystallite orientations")
        title.setToolTip(
            "Small octahedra sample the crystallite orientation distribution. "
            "Similar orientations indicate narrow sigma values; scattered "
            "orientations indicate broad sigma values."
        )
        header.addWidget(title)
        header.addStretch(1)
        self.auto_update_check = QtWidgets.QCheckBox("Auto")
        self.auto_update_check.setChecked(True)
        self.auto_update_check.setToolTip(
            "Automatically redraw the distribution after theta or sigma "
            "changes."
        )
        header.addWidget(self.auto_update_check)
        self.refresh_button = QtWidgets.QToolButton()
        self.refresh_button.setText("Refresh")
        self.refresh_button.setToolTip(
            "Redraw the orientation distribution with the current theta and "
            "sigma values."
        )
        self.refresh_button.clicked.connect(self.refresh_distribution)
        header.addWidget(self.refresh_button)
        layout.addLayout(header)

        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            from matplotlib.figure import Figure
        except Exception as exc:  # pragma: no cover - dependency fallback.
            self.figure = None
            self.canvas = QtWidgets.QLabel(f"Matplotlib is unavailable: {exc}")
            self.canvas.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.axes = None
        else:
            self.figure = Figure(figsize=(3.0, 3.0), tight_layout=True)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumHeight(220)
            self.axes = self.figure.add_subplot(111, projection="3d")
        layout.addWidget(self.canvas, stretch=1)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def set_distribution(
        self,
        *,
        theta_x_deg: float,
        theta_y_deg: float,
        sigma_theta: float,
        sigma_phi: float,
        sigma_r: float,
    ) -> None:
        self._pending_parameters = (
            float(theta_x_deg),
            float(theta_y_deg),
            float(sigma_theta),
            float(sigma_phi),
            float(sigma_r),
        )
        if self.auto_update_check.isChecked():
            self._refresh_timer.start()

    def refresh_distribution(self) -> None:
        if self._pending_parameters is not None:
            (
                self.theta_x_deg,
                self.theta_y_deg,
                self.sigma_theta,
                self.sigma_phi,
                self.sigma_r,
            ) = self._pending_parameters
            self._pending_parameters = None
        if self.axes is None:
            self.status_label.setText("Orientation preview unavailable.")
            return
        self._plot_distribution()

    def _plot_distribution(self) -> None:
        assert self.axes is not None
        self.axes.clear()
        self.axes.set_axis_off()
        self.axes.view_init(elev=22, azim=-42)

        base_rotation = _orientation_rotation_matrix(
            self.theta_x_deg,
            self.theta_y_deg,
        )
        spread = _distribution_spread_degrees(
            self.sigma_theta,
            self.sigma_phi,
            self.sigma_r,
        )
        rotations = _sample_orientation_distribution(
            base_rotation,
            sigma_theta=self.sigma_theta,
            sigma_phi=self.sigma_phi,
            sigma_r=self.sigma_r,
        )
        centers = _orientation_distribution_centers(
            len(rotations),
            spread_degrees=spread,
        )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        vertices, faces = _octahedron_mesh()
        for index, (center, rotation) in enumerate(zip(centers, rotations)):
            scale = 0.09
            points = vertices @ rotation.T * scale + center
            color = "#2563eb" if index == 0 else "#38bdf8"
            alpha = 0.72 if index == 0 else 0.34
            collection = Poly3DCollection(
                [points[list(face)] for face in faces],
                facecolors=color,
                edgecolors="#0f172a",
                linewidths=0.25,
                alpha=alpha,
            )
            self.axes.add_collection3d(collection)

        self.axes.set_xlim(-1.2, 1.2)
        self.axes.set_ylim(-1.2, 1.2)
        self.axes.set_zlim(-1.2, 1.2)
        self.status_label.setText(
            f"{len(rotations)} crystallites, approx. spread {spread:g} deg"
        )
        self.canvas.draw_idle()


class GIWAXSSimulationWindow(QtWidgets.QMainWindow):
    """GIWAXS simulation workspace linked to an optional EWALD
    project."""

    simulationCreated = QtCore.Signal(str)
    simulationLinked = QtCore.Signal(str)
    projectChanged = QtCore.Signal()

    def __init__(
        self,
        *,
        project: ProjectState | None = None,
        project_path: Path | None = None,
        output_directory: Path | None = None,
        initial_data_id: str | None = None,
        settings: QtCore.QSettings | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project or ProjectState(
            name="GIWAXS Simulation Session"
        )
        self.project_path = project_path
        self.output_directory = output_directory or _default_output_directory(
            project_path
        )
        self.initial_data_id = initial_data_id
        self.target_data_id = initial_data_id
        self.structures: dict[str, dict[str, Any]] = {}
        self._simulation_cache: dict[str, list[str]] = {}
        self._displayed_cache_record_id: str | None = None
        self._sweep_visual_active = False
        self.settings = (
            settings
            if settings is not None
            else QtCore.QSettings("EWALD", "EWALD")
        )
        self.structure_history = _read_structure_history(self.settings)
        self.project_structure_paths = self._project_loaded_cif_paths()
        self.setWindowTitle("GIWAXS Simulation")
        self.resize(1200, 760)

        self._build_actions()
        self._build_tree()
        self._build_controls()
        self.structure_viewer = UnitCellStructureView()
        self.orientation_distribution_view = OrientationDistributionView()
        self.result_pane = GIWAXSSimulationResultPane(self.project)
        self.comparison_pane = GIWAXSComparisonPane()
        self._connect_result_pane_signals()
        self._build_layout()
        self._load_project_simulations()
        self._load_project_generated_cifs()
        self._refresh_simulation_cache()
        self._refresh_tree()
        if self.structures:
            self._set_active_structure(next(iter(self.structures)))
        self._sync_structure_orientation()
        self._sync_orientation_distribution()

    @property
    def simulation_id(self) -> str | None:
        """Currently displayed simulation id, for tab-level callers."""

        return self.result_pane.simulation_id

    def import_structure_path(self, path: str | Path) -> str:
        structure = load_structure(path)
        metadata = structure.metadata()
        loaded_cif = self._remember_imported_cif(structure)
        if loaded_cif is not None:
            metadata["source"] = "loaded_cif"
            metadata["loaded_cif_id"] = str(loaded_cif["cif_id"])
        self._register_structure(
            structure,
            metadata=metadata,
            name=structure.path.stem,
            remember_path=True,
            persist_history=True,
            select=True,
        )
        if loaded_cif is not None:
            self.projectChanged.emit()
        return structure.structure_id

    def _remember_imported_cif(
        self,
        structure: Any,
    ) -> dict[str, Any] | None:
        structure_path = Path(structure.path)
        if structure_path.suffix.lower() not in {".cif", ".mcif"}:
            return None
        try:
            cif_text = structure_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            cif_text = structure_path.read_text(encoding="latin-1")
        except OSError:
            return None
        try:
            lattice = extract_cif_lattice_parameters(structure_path)
        except Exception:
            lattice = _lattice_parameters_from_matrix(structure.lattice)
        record = self.project.remember_loaded_cif(
            structure_path,
            cif_text=cif_text,
            lattice=lattice,
            crystal_system=infer_crystal_system_from_lattice(lattice),
            label=structure_path.stem,
            target_id=self.target_data_id,
        )
        self.project_structure_paths = self._project_loaded_cif_paths()
        return record

    def refresh_project_context(
        self,
        *,
        project_path: Path | None = None,
        target_data_id: str | None = None,
    ) -> None:
        """Refresh project-derived structures, links, and cached results."""

        if project_path is not None:
            self.project_path = project_path
            self.output_directory = _default_output_directory(project_path)
        self.project_structure_paths = self._project_loaded_cif_paths()
        self._populate_data_file_links(
            target_data_id
            if target_data_id is not None
            else self.target_data_id
        )
        self._refresh_structure_history_field()
        self._load_project_simulations()
        self._load_project_generated_cifs()
        self._refresh_simulation_cache()
        self._refresh_tree()
        if (
            self.structures
            and self.structure_viewer.structure_id not in self.structures
        ):
            self._set_active_structure(next(iter(self.structures)))
        self._maybe_display_cached_result()

    def _register_structure(
        self,
        structure: Any,
        *,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        remember_path: bool = False,
        persist_history: bool = False,
        select: bool = False,
    ) -> str:
        structure_id = str(structure.structure_id)
        existing = self.structures.get(structure_id, {})
        simulation_ids = list(
            existing.get(
                "simulation_ids",
                _simulation_ids_for_structure(self.project, structure_id),
            )
        )
        self.structures[structure_id] = {
            "structure_id": structure_id,
            "path": str(structure.path),
            "name": name or structure.path.stem,
            "metadata": metadata or structure.metadata(),
            "structure_data": structure,
            "simulation_ids": simulation_ids,
        }
        if remember_path:
            self._remember_structure_path(
                structure.path,
                persist=persist_history,
            )
        if select:
            self._refresh_tree()
            self._set_active_structure(structure_id)
        return structure_id

    def _load_project_generated_cifs(self, limit: int | None = None) -> int:
        loaded = 0
        for record in self._top_generated_cif_records(limit):
            cif_path = self._ensure_generated_cif_path(record)
            if cif_path is None:
                continue
            try:
                structure = load_structure(cif_path)
            except Exception as exc:
                record["load_error"] = str(exc)
                continue
            metadata = {
                **structure.metadata(),
                **self._generated_cif_structure_metadata(record),
            }
            self._register_structure(
                structure,
                metadata=metadata,
                name=_generated_cif_label(record),
                remember_path=True,
                persist_history=False,
                select=False,
            )
            loaded += 1
        return loaded

    def _generated_cif_structure_metadata(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = {
            "source": "Structure Analysis generated CIF",
            "generated_cif_id": str(
                record.get("cif_id") or record.get("id") or ""
            ),
            "candidate_id": record.get("candidate_id"),
            "rank": record.get("rank"),
            "score": record.get("score"),
            "data_id": record.get("data_id"),
        }
        for key in ("space_group", "wyckoff_combination"):
            if record.get(key) is not None:
                metadata[key] = record.get(key)
        return metadata

    def _annotate_record_from_structure(
        self,
        record: dict[str, Any],
        structure: dict[str, Any],
    ) -> None:
        metadata = structure.get("metadata", {})
        if not isinstance(metadata, dict):
            return
        loaded_cif_id = metadata.get("loaded_cif_id")
        if loaded_cif_id:
            record["source"] = "loaded_cif"
            record["loaded_cif_id"] = str(loaded_cif_id)
            record_metadata = record.setdefault("metadata", {})
            if isinstance(record_metadata, dict):
                record_metadata["loaded_cif_id"] = record["loaded_cif_id"]
        generated_cif_id = metadata.get("generated_cif_id")
        if not generated_cif_id:
            return
        record["source"] = "generated_cif"
        record["generated_cif_id"] = str(generated_cif_id)
        record["generated_cif_rank"] = metadata.get("rank")
        record["generated_cif_score"] = metadata.get("score")
        record_metadata = record.setdefault("metadata", {})
        if isinstance(record_metadata, dict):
            record_metadata["generated_cif_id"] = record["generated_cif_id"]
            record_metadata["generated_cif_rank"] = record[
                "generated_cif_rank"
            ]
            record_metadata["generated_cif_score"] = record[
                "generated_cif_score"
            ]
            if metadata.get("candidate_id") is not None:
                record_metadata["candidate_id"] = metadata["candidate_id"]

    def run_selected_simulation(self) -> dict[str, Any] | None:
        structure_id = self._selected_structure_id()
        if structure_id is None and self.structures:
            structure_id = next(iter(self.structures))
        if structure_id is None:
            return None
        structure = self.structures[structure_id]
        params = self.parameters()
        cached_record = self._find_compatible_pattern_record(
            structure_id,
            params,
        )
        if cached_record is not None:
            if self.target_data_id is not None:
                self._apply_target_link_to_cached_record(cached_record)
            self._annotate_record_from_structure(cached_record, structure)
            self._display_cached_record(cached_record)
            self.projectChanged.emit()
            return cached_record
        record = run_and_store_simulation(
            self.project,
            structure["path"],
            self.output_directory,
            parameters=params,
            target_data_id=self.target_data_id,
        )
        self._annotate_record_from_structure(record, structure)
        if record["simulation_id"] not in structure["simulation_ids"]:
            structure["simulation_ids"].append(record["simulation_id"])
        self._refresh_simulation_cache()
        self._refresh_tree()
        self._set_active_structure(structure_id)
        self.result_pane.set_simulation(record["simulation_id"])
        self.open_reconstruction_action.setEnabled(False)
        self.open_3d_button.setEnabled(False)
        self._set_cache_status(
            f"New simulation computed: {record['simulation_id']}",
            record["simulation_id"],
        )
        self.simulationCreated.emit(record["simulation_id"])
        self.projectChanged.emit()
        return record

    def run_selected_mode(self) -> dict[str, Any] | None:
        if self.simulation_mode.currentData() == SIMULATION_MODE_EWALD_SWEEP:
            return self.run_ewald_sphere_sweep()
        return self.run_selected_simulation()

    def run_ewald_sphere_sweep(self) -> dict[str, Any] | None:
        structure_id = self._selected_structure_id()
        if structure_id is None and self.structures:
            structure_id = next(iter(self.structures))
        if structure_id is None:
            return None
        structure = self.structures[structure_id]
        params = self.sweep_parameters()
        cached_record = self._find_exact_cached_record(
            structure_id,
            SIMULATION_MODE_EWALD_SWEEP,
            params.as_dict(),
        )
        if cached_record is not None:
            if self.target_data_id is not None:
                self._apply_target_link_to_cached_record(cached_record)
            self._annotate_record_from_structure(cached_record, structure)
            self._display_cached_record(cached_record)
            self.projectChanged.emit()
            return cached_record
        record = run_and_store_ewald_sphere_sweep(
            self.project,
            structure["path"],
            self.output_directory,
            parameters=params,
            target_data_id=self.target_data_id,
        )
        self._annotate_record_from_structure(record, structure)
        if record["simulation_id"] not in structure["simulation_ids"]:
            structure["simulation_ids"].append(record["simulation_id"])
        self._refresh_simulation_cache()
        self._refresh_tree()
        self._set_active_structure(structure_id)
        self.result_pane.set_simulation(record["simulation_id"])
        self.open_reconstruction_action.setEnabled(True)
        self.open_3d_button.setEnabled(True)
        self._set_cache_status(
            f"New sweep computed: {record['simulation_id']}",
            record["simulation_id"],
        )
        self.simulationCreated.emit(record["simulation_id"])
        self.projectChanged.emit()
        return record

    def compare_displayed_simulation_to_target(
        self,
    ) -> GIWAXSImageComparison | None:
        simulation_id = self.result_pane.simulation_id
        if simulation_id is None or self.result_pane.data_array is None:
            self.comparison_pane.clear("Run or select a simulation first.")
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentWidget(self.comparison_pane)
            return None
        if is_ewald_sphere_sweep_data(self.result_pane.data_array):
            self.comparison_pane.clear(
                "Select a single-pattern simulation for 2D comparison."
            )
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentWidget(self.comparison_pane)
            return None
        if self.target_data_id is None:
            self.comparison_pane.clear("Link a target data file first.")
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentWidget(self.comparison_pane)
            return None
        target = _target_qspace_for_project(
            self.project,
            self.target_data_id,
        )
        if target is None:
            self.comparison_pane.clear(
                "No q-space target product is available for comparison."
            )
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentWidget(self.comparison_pane)
            return None
        comparison = compare_giwaxs_images(
            target,
            self.result_pane.data_array,
        )
        record = self.project.simulations.get(simulation_id)
        if isinstance(record, dict):
            record["fit_metrics"] = comparison.metrics
        self.comparison_pane.set_comparison(comparison)
        if hasattr(self, "right_tabs"):
            self.right_tabs.setCurrentWidget(self.comparison_pane)
        self._refresh_tree()
        self.projectChanged.emit()
        return comparison

    def rank_stored_simulations_against_target(self) -> list[dict[str, Any]]:
        if self.target_data_id is None:
            self.comparison_pane.clear("Link a target data file first.")
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentWidget(self.comparison_pane)
            return []
        target = _target_qspace_for_project(
            self.project,
            self.target_data_id,
        )
        if target is None:
            self.comparison_pane.clear(
                "No q-space target product is available for ranking."
            )
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentWidget(self.comparison_pane)
            return []
        ranked: list[dict[str, Any]] = []
        for simulation_id, record in self.project.simulations.items():
            if not isinstance(record, dict):
                continue
            if record.get("simulation_mode") != SIMULATION_MODE_PATTERN:
                continue
            data = load_simulation_data(record)
            if data is None or is_ewald_sphere_sweep_data(data):
                continue
            comparison = compare_giwaxs_images(target, data)
            record["fit_metrics"] = dict(comparison.metrics)
            record["fit_target_data_id"] = self.target_data_id
            ranked.append(
                {
                    "simulation_id": simulation_id,
                    "record": record,
                    "comparison": comparison,
                    "fit_score": float(comparison.metrics["difference_rmse"]),
                }
            )
        ranked.sort(
            key=lambda item: (
                float(
                    item["comparison"].metrics.get(
                        "difference_rmse",
                        item["comparison"].metrics.get(
                            "fit_score",
                            float("inf"),
                        ),
                    )
                ),
                float(
                    item["comparison"].metrics.get(
                        "weighted_rmse",
                        float("inf"),
                    )
                ),
                -float(item["comparison"].metrics.get("correlation", 0.0)),
            )
        )
        for index, item in enumerate(ranked, start=1):
            item["record"]["fit_rank"] = index
        self._refresh_tree()
        if ranked:
            best = ranked[0]
            self.result_pane.set_simulation(str(best["simulation_id"]))
            self.comparison_pane.set_ranked_comparisons(ranked)
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentWidget(self.comparison_pane)
        else:
            self.comparison_pane.clear(
                "No single-pattern simulations to rank."
            )
        self.projectChanged.emit()
        return ranked

    def run_top_generated_cif_fit_comparisons(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.target_data_id is None:
            self.comparison_pane.clear("Link a target data file first.")
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentWidget(self.comparison_pane)
            return []
        target = _target_qspace_for_project(
            self.project,
            self.target_data_id,
        )
        if target is None:
            self.comparison_pane.clear(
                "No q-space target product is available for generated CIF comparison."
            )
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentWidget(self.comparison_pane)
            return []
        records = self._top_generated_cif_records(limit)
        if not records:
            self.comparison_pane.clear(
                "No generated CIF records are available."
            )
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentWidget(self.comparison_pane)
            return []

        params = self.parameters()
        ranked: list[dict[str, Any]] = []
        for cif_record in records:
            cif_path = self._ensure_generated_cif_path(cif_record)
            if cif_path is None:
                continue
            structure_id = self.import_structure_path(cif_path)
            cached_record = self._find_compatible_pattern_record(
                structure_id,
                params,
            )
            if cached_record is not None:
                self._apply_target_link_to_cached_record(cached_record)
                record = cached_record
            else:
                record = run_and_store_simulation(
                    self.project,
                    cif_path,
                    self.output_directory,
                    parameters=params,
                    target_data_id=self.target_data_id,
                )
            record["source"] = "generated_cif"
            record["generated_cif_id"] = str(cif_record.get("cif_id", ""))
            record["generated_cif_rank"] = cif_record.get("rank")
            record["generated_cif_score"] = cif_record.get("score")
            record["fit_target_data_id"] = self.target_data_id
            metadata = record.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["generated_cif_id"] = record["generated_cif_id"]
                metadata["generated_cif_rank"] = record["generated_cif_rank"]
                metadata["generated_cif_score"] = record["generated_cif_score"]
            structure = self.structures.get(structure_id)
            if structure is not None:
                simulation_ids = structure.setdefault("simulation_ids", [])
                if record["simulation_id"] not in simulation_ids:
                    simulation_ids.append(record["simulation_id"])
            data = load_simulation_data(record)
            if data is None or is_ewald_sphere_sweep_data(data):
                continue
            comparison = compare_giwaxs_images(
                target,
                data,
                simulated_label=_generated_cif_label(cif_record),
            )
            record["fit_metrics"] = dict(comparison.metrics)
            ranked.append(
                {
                    "simulation_id": record["simulation_id"],
                    "record": record,
                    "comparison": comparison,
                    "generated_cif": cif_record,
                    "fit_score": float(comparison.metrics["difference_rmse"]),
                }
            )
        ranked.sort(key=_comparison_sort_key)
        for index, item in enumerate(ranked, start=1):
            item["record"]["fit_rank"] = index
        self._refresh_simulation_cache()
        self._refresh_tree()
        if ranked:
            best = ranked[0]
            self.result_pane.set_simulation(str(best["simulation_id"]))
            self.comparison_pane.set_ranked_comparisons(ranked)
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentWidget(self.comparison_pane)
        else:
            self.comparison_pane.clear(
                "No generated CIF simulations could be compared."
            )
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentWidget(self.comparison_pane)
        self.projectChanged.emit()
        return ranked

    def run_generated_cif_fit_comparisons(self) -> list[dict[str, Any]]:
        return self.run_top_generated_cif_fit_comparisons(
            self.generated_cif_limit.value()
        )

    def parameters(self) -> GIWAXSSimulationParameters:
        return GIWAXSSimulationParameters(
            sigma_theta=self.sigma_theta.value(),
            sigma_phi=self.sigma_phi.value(),
            sigma_r=self.sigma_r.value(),
            hkl_extent=self.hkl_extent.value(),
            theta_x_deg=self.theta_x.value(),
            theta_y_deg=self.theta_y.value(),
            qxy_min=self.qxy_min.value(),
            qxy_max=self.qxy_max.value(),
            qz_min=self.qz_min.value(),
            qz_max=self.qz_max.value(),
            resolution_x=self.resolution_x.value(),
            resolution_z=self.resolution_z.value(),
        )

    def sweep_parameters(self) -> EwaldSphereSweepParameters:
        return EwaldSphereSweepParameters(
            sigma_theta=self.sigma_theta.value(),
            sigma_phi=self.sigma_phi.value(),
            sigma_r=self.sigma_r.value(),
            hkl_extent=self.hkl_extent.value(),
            theta_x_deg=self.sweep_theta_x_min.value(),
            theta_y_deg=self.sweep_theta_y_min.value(),
            qxy_min=self.qxy_min.value(),
            qxy_max=self.qxy_max.value(),
            qz_min=self.qz_min.value(),
            qz_max=self.qz_max.value(),
            resolution_x=self.resolution_x.value(),
            resolution_z=self.resolution_z.value(),
            theta_x_min_deg=self.sweep_theta_x_min.value(),
            theta_x_max_deg=self.sweep_theta_x_max.value(),
            theta_x_step_deg=self.sweep_theta_x_step.value(),
            theta_y_min_deg=self.sweep_theta_y_min.value(),
            theta_y_max_deg=self.sweep_theta_y_max.value(),
            theta_y_step_deg=self.sweep_theta_y_step.value(),
        )

    def _build_actions(self) -> None:
        self.import_action = QtGui.QAction("Import Structure", self)
        self.run_action = QtGui.QAction("Run Simulation", self)
        self.run_sweep_action = QtGui.QAction("Run Sphere Sweep", self)
        self.open_reconstruction_action = QtGui.QAction("Open 3D", self)
        self.open_reconstruction_action.setEnabled(False)
        self.import_action.triggered.connect(self.import_structure)
        self.run_action.triggered.connect(self.run_selected_mode)
        self.run_sweep_action.triggered.connect(self.run_ewald_sphere_sweep)
        self.open_reconstruction_action.triggered.connect(
            self.open_selected_reconstruction
        )
        toolbar = self.addToolBar("Simulation")
        toolbar.setMovable(False)
        toolbar.addAction(self.import_action)
        toolbar.addAction(self.run_action)
        toolbar.addAction(self.run_sweep_action)
        toolbar.addAction(self.open_reconstruction_action)

    def open_selected_reconstruction(self) -> QtWidgets.QMainWindow | None:
        return self.result_pane.open_reconstruction_viewer()

    def _build_tree(self) -> None:
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Simulation Session", "Value"])
        enable_rich_text_items(self.tree)
        self.tree.setColumnWidth(0, 260)
        self.tree.itemSelectionChanged.connect(self._handle_tree_selection)

    def _build_controls(self) -> None:
        self.import_structure_combo = QtWidgets.QComboBox()
        self.import_structure_combo.setEditable(True)
        self.import_structure_combo.setInsertPolicy(
            QtWidgets.QComboBox.InsertPolicy.NoInsert
        )
        self.import_structure_combo.addItems(
            _dedupe_structure_history(
                [*self.project_structure_paths, *self.structure_history]
            )
        )
        self.import_structure_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.import_structure_combo.setToolTip(
            "Choose a previously imported structure or enter a path."
        )
        line_edit = self.import_structure_combo.lineEdit()
        if line_edit is not None:
            line_edit.returnPressed.connect(self.import_structure_from_field)
        self.import_structure_field = QtWidgets.QWidget()
        import_layout = QtWidgets.QHBoxLayout(self.import_structure_field)
        import_layout.setContentsMargins(0, 0, 0, 0)
        import_layout.setSpacing(4)
        import_layout.addWidget(self.import_structure_combo, stretch=1)
        self.import_structure_button = QtWidgets.QPushButton("Load")
        self.import_structure_button.setToolTip(
            "Load the structure path currently shown in the path field."
        )
        self.import_structure_button.clicked.connect(
            self.import_structure_from_field
        )
        import_layout.addWidget(self.import_structure_button)
        self.browse_structure_button = QtWidgets.QPushButton("Browse")
        self.browse_structure_button.setToolTip(
            "Browse for CIF, MCIF, POSCAR, CONTCAR, or VASP structure files."
        )
        self.browse_structure_button.clicked.connect(self.import_structure)
        import_layout.addWidget(self.browse_structure_button)
        self.remove_structure_button = QtWidgets.QPushButton("Remove")
        self.remove_structure_button.setToolTip(
            "Remove the selected structure from this simulation session. The "
            "source file is not deleted."
        )
        self.remove_structure_button.clicked.connect(
            self.remove_active_structure
        )
        import_layout.addWidget(self.remove_structure_button)

        self.simulation_mode = QtWidgets.QComboBox()
        self.simulation_mode.setToolTip(
            "Single pattern computes one GIWAXS image for the selected "
            "orientation. Theta sweep computes many frames by rotating the "
            "crystal through theta X/theta Y values."
        )
        self.simulation_mode.addItem("Single pattern", SIMULATION_MODE_PATTERN)
        self.simulation_mode.setItemData(
            0,
            "Compute one detector pattern at the current theta X/theta Y "
            "orientation.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.simulation_mode.addItem(
            "Ewald sphere theta sweep",
            SIMULATION_MODE_EWALD_SWEEP,
        )
        self.simulation_mode.setItemData(
            1,
            "Compute a grid of low-resolution patterns while sweeping theta "
            "X and theta Y, then replay the frames as an Ewald sphere video.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.simulation_mode.currentIndexChanged.connect(
            self._handle_mode_changed
        )

        self.sigma_theta = _double_spinbox(0.03, 1.0e-6, 10.0, 0.005)
        self.sigma_phi = _double_spinbox(0.25, 1.0e-6, 10.0, 0.05)
        self.sigma_r = _double_spinbox(0.035, 1.0e-6, 10.0, 0.005)
        self.hkl_extent = _spinbox(4, 0, 25)
        self.theta_x = _double_spinbox(90.0, -360.0, 360.0, 1.0)
        self.theta_y = _double_spinbox(0.0, -360.0, 360.0, 1.0)
        self.qxy_min = _double_spinbox(-3.0, -100.0, 100.0, 0.1)
        self.qxy_max = _double_spinbox(3.0, -100.0, 100.0, 0.1)
        self.qz_min = _double_spinbox(0.0, -100.0, 100.0, 0.1)
        self.qz_max = _double_spinbox(3.0, -100.0, 100.0, 0.1)
        self.resolution_x = _spinbox(256, 16, 2048)
        self.resolution_z = _spinbox(128, 16, 2048)

        self.rotation_increment = _double_spinbox(5.0, 0.1, 90.0, 1.0)
        self.rotation_increment.setSuffix(" deg")
        self.rotate_x_neg_button = QtWidgets.QToolButton()
        self.rotate_x_neg_button.setText("-X")
        self.rotate_x_pos_button = QtWidgets.QToolButton()
        self.rotate_x_pos_button.setText("+X")
        self.rotate_y_neg_button = QtWidgets.QToolButton()
        self.rotate_y_neg_button.setText("-Y")
        self.rotate_y_pos_button = QtWidgets.QToolButton()
        self.rotate_y_pos_button.setText("+Y")
        for button, axis, sign in (
            (self.rotate_x_neg_button, "theta X", -1),
            (self.rotate_x_pos_button, "theta X", 1),
            (self.rotate_y_neg_button, "theta Y", -1),
            (self.rotate_y_pos_button, "theta Y", 1),
        ):
            button.setToolTip(
                f"Rotate {axis} by {sign * self.rotation_increment.value():g} "
                "degrees. Cached results are displayed when available."
            )
        self.rotate_x_neg_button.clicked.connect(
            lambda: self._rotate_theta(self.theta_x, -1.0)
        )
        self.rotate_x_pos_button.clicked.connect(
            lambda: self._rotate_theta(self.theta_x, 1.0)
        )
        self.rotate_y_neg_button.clicked.connect(
            lambda: self._rotate_theta(self.theta_y, -1.0)
        )
        self.rotate_y_pos_button.clicked.connect(
            lambda: self._rotate_theta(self.theta_y, 1.0)
        )

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.setToolTip(
            "Choose a common limiting orientation/spread case. Applying a "
            "preset updates theta and sigma values without running a "
            "simulation."
        )
        for preset_id, preset in ORIENTATION_PRESETS.items():
            self.preset_combo.addItem(str(preset["label"]), preset_id)
            self.preset_combo.setItemData(
                self.preset_combo.count() - 1,
                preset["tooltip"],
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        self.apply_preset_button = QtWidgets.QPushButton("Apply")
        self.apply_preset_button.setToolTip(
            "Apply the selected orientation preset without starting a "
            "simulation."
        )
        self.apply_preset_button.clicked.connect(self.apply_orientation_preset)

        self.sweep_controls = QtWidgets.QGroupBox("Ewald sphere sweep")
        self.sweep_controls.setToolTip(
            "Theta sweep rotates the crystal through a grid of theta X and "
            "theta Y values and computes one low-resolution detector pattern "
            "per frame. This differs from changing sigma values, which "
            "broadens the orientation distribution within a single pattern."
        )
        sweep_layout = QtWidgets.QGridLayout(self.sweep_controls)
        sweep_layout.setContentsMargins(8, 8, 8, 8)
        self.sweep_theta_x_min = _double_spinbox(0.0, -360.0, 360.0, 5.0)
        self.sweep_theta_x_max = _double_spinbox(180.0, -360.0, 360.0, 5.0)
        self.sweep_theta_x_step = _double_spinbox(15.0, 0.1, 360.0, 1.0)
        self.sweep_theta_y_min = _double_spinbox(0.0, -360.0, 360.0, 5.0)
        self.sweep_theta_y_max = _double_spinbox(345.0, -360.0, 360.0, 5.0)
        self.sweep_theta_y_step = _double_spinbox(15.0, 0.1, 360.0, 1.0)
        for spinbox in (
            self.sweep_theta_x_min,
            self.sweep_theta_x_max,
            self.sweep_theta_x_step,
            self.sweep_theta_y_min,
            self.sweep_theta_y_max,
            self.sweep_theta_y_step,
        ):
            spinbox.setToolTip(
                "Sweep-frame orientation controls. The video replays these "
                "theta values without changing the selected single-pattern "
                "orientation."
            )
        sweep_layout.addWidget(QtWidgets.QLabel("Axis"), 0, 0)
        sweep_layout.addWidget(QtWidgets.QLabel("Min"), 0, 1)
        sweep_layout.addWidget(QtWidgets.QLabel("Max"), 0, 2)
        sweep_layout.addWidget(QtWidgets.QLabel("Step"), 0, 3)
        sweep_layout.addWidget(QtWidgets.QLabel("theta X"), 1, 0)
        sweep_layout.addWidget(self.sweep_theta_x_min, 1, 1)
        sweep_layout.addWidget(self.sweep_theta_x_max, 1, 2)
        sweep_layout.addWidget(self.sweep_theta_x_step, 1, 3)
        sweep_layout.addWidget(QtWidgets.QLabel("theta Y"), 2, 0)
        sweep_layout.addWidget(self.sweep_theta_y_min, 2, 1)
        sweep_layout.addWidget(self.sweep_theta_y_max, 2, 2)
        sweep_layout.addWidget(self.sweep_theta_y_step, 2, 3)
        self.link_data_file_combo = QtWidgets.QComboBox()
        self.link_data_file_combo.currentIndexChanged.connect(
            self._handle_target_data_file_changed
        )
        self._populate_data_file_links(self.initial_data_id)
        self.link_selected_button = QtWidgets.QPushButton("Link Selected")
        self.link_selected_button.setToolTip(
            "Link the displayed simulation record to the selected data file."
        )
        self.link_selected_button.clicked.connect(
            self.link_selected_simulation_to_data_file
        )
        self.link_data_file_field = QtWidgets.QWidget()
        link_layout = QtWidgets.QHBoxLayout(self.link_data_file_field)
        link_layout.setContentsMargins(0, 0, 0, 0)
        link_layout.setSpacing(4)
        link_layout.addWidget(self.link_data_file_combo, stretch=1)
        link_layout.addWidget(self.link_selected_button)

        self.generated_cif_limit = _spinbox(5, 1, 50)
        self.generated_cif_limit.setToolTip(
            "Number of top ranked generated CIF candidates to simulate and compare."
        )
        self.generated_cif_compare_button = QtWidgets.QPushButton(
            "Run + Compare"
        )
        self.generated_cif_compare_button.setToolTip(
            "Simulate top generated CIF candidates and open ranked difference maps."
        )
        self.generated_cif_compare_button.clicked.connect(
            self.run_generated_cif_fit_comparisons
        )
        self.generated_cif_field = QtWidgets.QWidget()
        generated_layout = QtWidgets.QHBoxLayout(self.generated_cif_field)
        generated_layout.setContentsMargins(0, 0, 0, 0)
        generated_layout.setSpacing(4)
        generated_layout.addWidget(self.generated_cif_limit)
        generated_layout.addWidget(
            self.generated_cif_compare_button, stretch=1
        )

        self.cache_status_label = QtWidgets.QLabel(
            SIMULATION_CACHE_STATUS_MISS
        )
        self.cache_status_label.setWordWrap(True)
        self.cache_status_label.setToolTip(
            "Shows whether the current detector view came from a reused "
            "simulation or from a newly computed run."
        )

        self.controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(8)

        input_group = QtWidgets.QGroupBox("Input files")
        input_layout = QtWidgets.QGridLayout(input_group)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setHorizontalSpacing(6)
        input_layout.setVerticalSpacing(5)
        input_layout.addWidget(QtWidgets.QLabel("Structure"), 0, 0)
        input_layout.addWidget(self.import_structure_field, 0, 1)
        input_layout.addWidget(QtWidgets.QLabel("Data file"), 1, 0)
        input_layout.addWidget(self.link_data_file_field, 1, 1)
        input_layout.addWidget(QtWidgets.QLabel("Generated CIFs"), 2, 0)
        input_layout.addWidget(self.generated_cif_field, 2, 1)
        input_layout.setColumnStretch(1, 1)
        controls_layout.addWidget(input_group)

        mode_group = QtWidgets.QGroupBox("Mode and presets")
        mode_layout = QtWidgets.QGridLayout(mode_group)
        mode_layout.setContentsMargins(8, 8, 8, 8)
        _add_grid_control(
            mode_layout,
            0,
            0,
            "Mode",
            self.simulation_mode,
            self.simulation_mode.toolTip(),
        )
        mode_layout.addWidget(QtWidgets.QLabel("Preset"), 1, 0)
        mode_layout.addWidget(self.preset_combo, 1, 1)
        mode_layout.addWidget(self.apply_preset_button, 1, 2)
        mode_layout.setColumnStretch(1, 1)
        controls_layout.addWidget(mode_group)
        controls_layout.addWidget(self.sweep_controls)

        parameters_group = QtWidgets.QGroupBox("Pattern parameters")
        parameter_layout = QtWidgets.QGridLayout(parameters_group)
        parameter_layout.setContentsMargins(8, 8, 8, 8)
        _add_grid_control(
            parameter_layout,
            0,
            0,
            "sigma theta",
            self.sigma_theta,
            f"Tilt spread used to smear reflections along {QZ_HTML}.",
        )
        _add_grid_control(
            parameter_layout,
            0,
            1,
            "sigma phi",
            self.sigma_phi,
            "Azimuthal orientation spread around the sample normal.",
        )
        _add_grid_control(
            parameter_layout,
            1,
            0,
            "sigma r",
            self.sigma_r,
            f"Radial broadening used for {QXY_HTML} peak width.",
        )
        _add_grid_control(
            parameter_layout,
            1,
            1,
            "hkl extent",
            self.hkl_extent,
            "Maximum absolute h, k, and l index included in the simulation.",
        )
        _add_grid_control(
            parameter_layout,
            2,
            0,
            "theta X",
            self.theta_x,
            "Crystal rotation around lab X for the selected orientation.",
        )
        _add_grid_control(
            parameter_layout,
            2,
            1,
            "theta Y",
            self.theta_y,
            "Crystal rotation around lab Y for the selected orientation.",
        )
        _add_grid_control(
            parameter_layout,
            3,
            0,
            f"{QXY_HTML} min",
            self.qxy_min,
            f"Detector lower bound in {QXY_HTML}.",
        )
        _add_grid_control(
            parameter_layout,
            3,
            1,
            f"{QXY_HTML} max",
            self.qxy_max,
            f"Detector upper bound in {QXY_HTML}.",
        )
        _add_grid_control(
            parameter_layout,
            4,
            0,
            f"{QZ_HTML} min",
            self.qz_min,
            f"Detector lower bound in {QZ_HTML}.",
        )
        _add_grid_control(
            parameter_layout,
            4,
            1,
            f"{QZ_HTML} max",
            self.qz_max,
            f"Detector upper bound in {QZ_HTML}.",
        )
        _add_grid_control(
            parameter_layout,
            5,
            0,
            "resolution x",
            self.resolution_x,
            f"Number of detector pixels along {QXY_HTML}.",
        )
        _add_grid_control(
            parameter_layout,
            5,
            1,
            "resolution z",
            self.resolution_z,
            f"Number of detector pixels along {QZ_HTML}.",
        )
        controls_layout.addWidget(parameters_group)

        rotation_group = QtWidgets.QGroupBox("Crystal rotation")
        rotation_layout = QtWidgets.QGridLayout(rotation_group)
        rotation_layout.setContentsMargins(8, 8, 8, 8)
        rotation_layout.addWidget(QtWidgets.QLabel("Step"), 0, 0)
        rotation_layout.addWidget(self.rotation_increment, 0, 1)
        rotation_layout.addWidget(self.rotate_x_neg_button, 1, 0)
        rotation_layout.addWidget(self.rotate_x_pos_button, 1, 1)
        rotation_layout.addWidget(self.rotate_y_neg_button, 1, 2)
        rotation_layout.addWidget(self.rotate_y_pos_button, 1, 3)
        controls_layout.addWidget(rotation_group)

        controls_layout.addWidget(self.cache_status_label)
        controls_layout.addStretch(1)

        self.run_bar = QtWidgets.QWidget()
        run_layout = QtWidgets.QHBoxLayout(self.run_bar)
        run_layout.setContentsMargins(8, 6, 8, 6)
        run_layout.setSpacing(6)
        self.run_mode_button = QtWidgets.QPushButton("Run Simulation")
        self.run_mode_button.setToolTip(
            "Run the selected simulation mode. If a compatible cached result "
            "already exists, it is displayed instead of recomputing."
        )
        self.run_mode_button.clicked.connect(self.run_selected_mode)
        run_layout.addWidget(self.run_mode_button, stretch=1)
        self.run_sweep_button = QtWidgets.QPushButton("Run Sphere Sweep")
        self.run_sweep_button.setToolTip(
            "Compute the theta X/theta Y Ewald sphere sweep and enable video "
            "playback."
        )
        self.run_sweep_button.clicked.connect(self.run_ewald_sphere_sweep)
        run_layout.addWidget(self.run_sweep_button)
        self.open_3d_button = QtWidgets.QPushButton("Open 3D")
        self.open_3d_button.setEnabled(False)
        self.open_3d_button.clicked.connect(self.open_selected_reconstruction)
        run_layout.addWidget(self.open_3d_button)
        self.compare_button = QtWidgets.QPushButton("Compare")
        self.compare_button.setToolTip(
            "Compare the displayed simulation against the linked q-space "
            "target image."
        )
        self.compare_button.clicked.connect(
            self.compare_displayed_simulation_to_target
        )
        run_layout.addWidget(self.compare_button)
        self.rank_fits_button = QtWidgets.QPushButton("Rank Fits")
        self.rank_fits_button.setToolTip(
            "Rank stored single-pattern simulations against the linked "
            "q-space target image."
        )
        self.rank_fits_button.clicked.connect(
            self.rank_stored_simulations_against_target
        )
        run_layout.addWidget(self.rank_fits_button)

        for control in self._parameter_controls():
            control.valueChanged.connect(self._handle_parameter_value_changed)
        self.theta_x.valueChanged.connect(
            self._handle_orientation_value_changed
        )
        self.theta_y.valueChanged.connect(
            self._handle_orientation_value_changed
        )
        self.sigma_theta.valueChanged.connect(
            self._handle_distribution_value_changed
        )
        self.sigma_phi.valueChanged.connect(
            self._handle_distribution_value_changed
        )
        self.sigma_r.valueChanged.connect(
            self._handle_distribution_value_changed
        )
        self.rotation_increment.valueChanged.connect(
            self._sync_rotation_button_tooltips
        )

    def _handle_mode_changed(self) -> None:
        sweep_mode = (
            self.simulation_mode.currentData() == SIMULATION_MODE_EWALD_SWEEP
        )
        self.run_action.setText(
            "Run Sweep" if sweep_mode else "Run Simulation"
        )
        if hasattr(self, "run_mode_button"):
            self.run_mode_button.setText(
                "Run Sweep" if sweep_mode else "Run Simulation"
            )
        if sweep_mode and self.resolution_x.value() == 256:
            self.resolution_x.setValue(
                EwaldSphereSweepParameters().resolution_x
            )
        if sweep_mode and self.resolution_z.value() == 128:
            self.resolution_z.setValue(
                EwaldSphereSweepParameters().resolution_z
            )
        self._handle_parameter_value_changed()

    def _parameter_controls(self) -> tuple[Any, ...]:
        return (
            self.sigma_theta,
            self.sigma_phi,
            self.sigma_r,
            self.hkl_extent,
            self.theta_x,
            self.theta_y,
            self.qxy_min,
            self.qxy_max,
            self.qz_min,
            self.qz_max,
            self.resolution_x,
            self.resolution_z,
            self.sweep_theta_x_min,
            self.sweep_theta_x_max,
            self.sweep_theta_x_step,
            self.sweep_theta_y_min,
            self.sweep_theta_y_max,
            self.sweep_theta_y_step,
        )

    def _connect_result_pane_signals(self) -> None:
        self.result_pane.sweepFrameChanged.connect(
            self._handle_sweep_frame_changed
        )
        self.result_pane.sweepPlaybackStarted.connect(
            self._handle_sweep_playback_started
        )
        self.result_pane.sweepPlaybackStopped.connect(
            self._handle_sweep_playback_stopped
        )

    def _handle_parameter_value_changed(self, *_args: Any) -> None:
        if not hasattr(self, "cache_status_label"):
            return
        self._maybe_display_cached_result()

    def _handle_orientation_value_changed(self, *_args: Any) -> None:
        self._sweep_visual_active = False
        self._sync_structure_orientation()
        self._sync_orientation_distribution()

    def _handle_distribution_value_changed(self, *_args: Any) -> None:
        self._sync_orientation_distribution()

    def _sync_structure_orientation(self) -> None:
        if not hasattr(self, "structure_viewer"):
            return
        self.structure_viewer.set_orientation(
            self.theta_x.value(),
            self.theta_y.value(),
            source="Selected orientation",
        )

    def _sync_orientation_distribution(self) -> None:
        if not hasattr(self, "orientation_distribution_view"):
            return
        self.orientation_distribution_view.set_distribution(
            theta_x_deg=self.theta_x.value(),
            theta_y_deg=self.theta_y.value(),
            sigma_theta=self.sigma_theta.value(),
            sigma_phi=self.sigma_phi.value(),
            sigma_r=self.sigma_r.value(),
        )

    def _handle_sweep_playback_started(self) -> None:
        self._sweep_visual_active = True
        orientation = self.result_pane.current_sweep_orientation()
        if orientation is not None:
            self._handle_sweep_frame_changed(*orientation)

    def _handle_sweep_frame_changed(
        self,
        theta_x_deg: float,
        theta_y_deg: float,
    ) -> None:
        if not hasattr(self, "structure_viewer"):
            return
        if not self._sweep_visual_active:
            return
        self._sweep_visual_active = True
        self.structure_viewer.set_orientation(
            theta_x_deg,
            theta_y_deg,
            source="Sweep frame",
        )

    def _handle_sweep_playback_stopped(self) -> None:
        self._sweep_visual_active = False
        self._sync_structure_orientation()

    def _rotate_theta(
        self,
        spinbox: QtWidgets.QDoubleSpinBox,
        direction: float,
    ) -> None:
        spinbox.setValue(
            spinbox.value()
            + float(direction) * self.rotation_increment.value()
        )

    def _sync_rotation_button_tooltips(self, *_args: Any) -> None:
        step = self.rotation_increment.value()
        self.rotate_x_neg_button.setToolTip(
            f"Rotate theta X by {-step:g} degrees."
        )
        self.rotate_x_pos_button.setToolTip(
            f"Rotate theta X by {step:g} degrees."
        )
        self.rotate_y_neg_button.setToolTip(
            f"Rotate theta Y by {-step:g} degrees."
        )
        self.rotate_y_pos_button.setToolTip(
            f"Rotate theta Y by {step:g} degrees."
        )

    def apply_orientation_preset(self) -> None:
        preset_id = str(self.preset_combo.currentData() or "")
        preset = ORIENTATION_PRESETS.get(preset_id)
        if preset is None:
            return
        for spinbox, key in (
            (self.theta_x, "theta_x"),
            (self.theta_y, "theta_y"),
            (self.sigma_theta, "sigma_theta"),
            (self.sigma_phi, "sigma_phi"),
            (self.sigma_r, "sigma_r"),
        ):
            spinbox.setValue(float(preset[key]))
        self._sync_structure_orientation()
        self._sync_orientation_distribution()
        self._maybe_display_cached_result()

    def remove_active_structure(self) -> None:
        structure_id = self._active_structure_id()
        if structure_id is None:
            self.import_structure_combo.setCurrentText("")
            return
        self.structures.pop(structure_id, None)
        self._refresh_tree()
        if self.structures:
            self._set_active_structure(next(iter(self.structures)))
        else:
            self.structure_viewer.clear()
        self._set_cache_status("Structure removed from this session.", None)

    def _active_structure_id(self) -> str | None:
        structure_id = self._selected_structure_id()
        if structure_id is not None:
            return structure_id
        if self.structure_viewer.structure_id in self.structures:
            return self.structure_viewer.structure_id
        if self.structures:
            return next(iter(self.structures))
        return None

    def _refresh_simulation_cache(self) -> None:
        cache: dict[str, list[str]] = {}
        for simulation_id, record in self.project.simulations.items():
            if not isinstance(record, dict):
                continue
            key = _record_cache_key(record)
            if key is None:
                continue
            cache.setdefault(key, []).append(simulation_id)
        self._simulation_cache = cache

    def _find_exact_cached_record(
        self,
        structure_id: str,
        mode: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any] | None:
        key = _simulation_cache_key(structure_id, mode, parameters)
        for simulation_id in self._simulation_cache.get(key, []):
            record = self.project.simulations.get(simulation_id)
            if _stored_record_is_loadable(record):
                return record
        return None

    def _find_compatible_pattern_record(
        self,
        structure_id: str,
        parameters: GIWAXSSimulationParameters,
    ) -> dict[str, Any] | None:
        exact_record = self._find_exact_cached_record(
            structure_id,
            SIMULATION_MODE_PATTERN,
            parameters.as_dict(),
        )
        if exact_record is not None:
            return exact_record

        requested = parameters.as_dict()
        candidates: list[dict[str, Any]] = []
        for record in self.project.simulations.values():
            if (
                not isinstance(record, dict)
                or record.get("simulation_mode") != SIMULATION_MODE_PATTERN
                or record.get("structure_id") != structure_id
                or not _stored_record_is_loadable(record)
            ):
                continue
            cached_parameters = record.get("parameters", {})
            if not isinstance(cached_parameters, dict):
                continue
            if _pattern_cache_parameters_cover(cached_parameters, requested):
                candidates.append(record)
        if not candidates:
            return None
        return max(candidates, key=_pattern_cache_rank)

    def _maybe_display_cached_result(self) -> None:
        structure_id = self._active_structure_id()
        if structure_id is None:
            self._set_cache_status(
                "Load a structure to use simulation cache.", None
            )
            return

        if self.simulation_mode.currentData() == SIMULATION_MODE_EWALD_SWEEP:
            cached = self._find_exact_cached_record(
                structure_id,
                SIMULATION_MODE_EWALD_SWEEP,
                self.sweep_parameters().as_dict(),
            )
        else:
            cached = self._find_compatible_pattern_record(
                structure_id,
                self.parameters(),
            )
        if cached is None:
            self._set_cache_status(SIMULATION_CACHE_STATUS_MISS, None)
            return
        if cached.get("simulation_id") != self.result_pane.simulation_id:
            self._display_cached_record(cached)
        else:
            self._set_cache_status(
                f"{SIMULATION_CACHE_STATUS_READY} {cached['simulation_id']}",
                str(cached["simulation_id"]),
            )

    def _display_cached_record(self, record: dict[str, Any]) -> None:
        simulation_id = record.get("simulation_id")
        if not simulation_id:
            return
        structure_id = record.get("structure_id")
        if structure_id in self.structures:
            self._set_active_structure(str(structure_id))
        self.result_pane.set_simulation(str(simulation_id))
        is_sweep = is_ewald_sphere_sweep_record(record)
        self.open_reconstruction_action.setEnabled(is_sweep)
        self.open_3d_button.setEnabled(is_sweep)
        self._set_cache_status(
            f"{SIMULATION_CACHE_STATUS_READY} {simulation_id}",
            str(simulation_id),
        )

    def _apply_target_link_to_cached_record(
        self,
        record: dict[str, Any],
    ) -> None:
        simulation_id = record.get("simulation_id")
        if not simulation_id:
            return
        if record.get("data_id") == self.target_data_id:
            return
        self.project.link_simulation_to_data_file(
            str(simulation_id),
            self.target_data_id,
        )
        self._refresh_tree()

    def _set_cache_status(
        self,
        message: str,
        simulation_id: str | None,
    ) -> None:
        if hasattr(self, "cache_status_label"):
            self.cache_status_label.setText(message)
        self._displayed_cache_record_id = simulation_id

    def _build_layout(self) -> None:
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)
        self.tree.setMinimumHeight(150)
        self.tree.setMaximumHeight(230)
        left_layout.addWidget(self.tree)

        self.left_scroll = QtWidgets.QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.left_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.left_scroll.setToolTip(
            "Program inputs scroll here so theta sweep controls stay readable."
        )
        self.left_scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(self.left_scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)
        scroll_layout.addWidget(self.controls)
        scroll_layout.addWidget(self.structure_viewer)
        scroll_layout.addWidget(self.orientation_distribution_view)
        self.left_scroll.setWidget(self.left_scroll_content)
        left_layout.addWidget(self.left_scroll, stretch=1)
        left_layout.addWidget(self.run_bar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        self.right_tabs = QtWidgets.QTabWidget()
        self.right_tabs.addTab(self.result_pane, "Simulation")
        self.right_tabs.addTab(self.comparison_pane, "Fit Compare")
        splitter.addWidget(self.right_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 770])
        self.setCentralWidget(splitter)

    def import_structure(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Structure",
            str(self._structure_dialog_directory()),
            STRUCTURE_FILE_FILTER,
        )
        if path:
            self.import_structure_path(path)

    def import_structure_from_field(self) -> str | None:
        path = self.import_structure_combo.currentText().strip()
        if not path:
            return None
        return self.import_structure_path(path)

    def _top_generated_cif_records(
        self,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        generated = self.project.reference_cifs.get("generated", {})
        if isinstance(generated, dict):
            records = [
                record
                for record in generated.values()
                if isinstance(record, dict)
            ]
        elif isinstance(generated, list):
            records = [
                record for record in generated if isinstance(record, dict)
            ]
        else:
            records = []
        if self.target_data_id is not None:
            target_records = [
                record
                for record in records
                if str(record.get("data_id", "")) == self.target_data_id
            ]
            if target_records:
                records = target_records
        records = sorted(records, key=_generated_cif_sort_key)
        return records[: max(1, int(limit or len(records) or 1))]

    def _ensure_generated_cif_path(
        self,
        record: dict[str, Any],
    ) -> Path | None:
        for key in ("path", "structure_path"):
            path_value = record.get(key)
            if path_value:
                path = Path(str(path_value))
                if path.exists():
                    return path
        cif_id = str(record.get("cif_id") or record.get("id") or "")
        cif_text = str(record.get("cif_text") or "")
        structure_record = self.project.structures.get(cif_id, {})
        if not cif_text and isinstance(structure_record, dict):
            cif_text = str(structure_record.get("cif_text") or "")
        if not cif_text.strip():
            return None
        directory = Path(self.output_directory) / "generated_cifs"
        directory.mkdir(parents=True, exist_ok=True)
        filename = _safe_generated_cif_filename(cif_id or "generated_cif")
        path = directory / filename
        if not path.exists() or path.read_text(encoding="utf-8") != cif_text:
            path.write_text(cif_text, encoding="utf-8")
        record["path"] = str(path)
        if cif_id:
            generated = self.project.reference_cifs.setdefault("generated", {})
            if isinstance(generated, dict):
                generated[cif_id] = record
            project_structure = self.project.structures.setdefault(cif_id, {})
            if isinstance(project_structure, dict):
                project_structure["path"] = str(path)
                project_structure["cif_text"] = cif_text
        return path

    def _load_project_simulations(self) -> None:
        for record in self.project.simulations.values():
            if not isinstance(record, dict):
                continue
            structure_id = record.get("structure_id")
            structure_path = record.get("structure_path")
            if not structure_id or not structure_path:
                continue
            entry = self.structures.setdefault(
                str(structure_id),
                {
                    "structure_id": str(structure_id),
                    "path": str(structure_path),
                    "name": Path(str(structure_path)).stem,
                    "metadata": {
                        "structure_id": structure_id,
                        "structure_path": structure_path,
                        "structure_name": record.get("structure_name"),
                    },
                    "structure_data": None,
                    "simulation_ids": [],
                },
            )
            simulation_id = record.get("simulation_id")
            if simulation_id and simulation_id not in entry["simulation_ids"]:
                entry["simulation_ids"].append(simulation_id)
            self._remember_structure_path(
                Path(str(structure_path)), persist=False
            )

    def _refresh_tree(self) -> None:
        self.tree.clear()
        root = QtWidgets.QTreeWidgetItem(
            [self.project.name, "Linked EWALD project"]
        )
        root.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "root"})
        self.tree.addTopLevelItem(root)
        for structure in self.structures.values():
            item = QtWidgets.QTreeWidgetItem([structure["name"], "Structure"])
            item.setData(
                0,
                QtCore.Qt.ItemDataRole.UserRole,
                {
                    "kind": "structure",
                    "structure_id": structure["structure_id"],
                },
            )
            root.addChild(item)
            metadata_item = QtWidgets.QTreeWidgetItem(["Metadata", ""])
            item.addChild(metadata_item)
            for key, value in sorted(structure["metadata"].items()):
                metadata_item.addChild(
                    QtWidgets.QTreeWidgetItem(
                        [_labelize(key), _format_value(value)]
                    )
                )
            for simulation_id in structure["simulation_ids"]:
                record = self.project.simulations.get(simulation_id, {})
                simulation_item = QtWidgets.QTreeWidgetItem(
                    [simulation_id, _simulation_type_label(record)]
                )
                simulation_item.setData(
                    0,
                    QtCore.Qt.ItemDataRole.UserRole,
                    {
                        "kind": "simulation",
                        "simulation_id": simulation_id,
                        "structure_id": structure["structure_id"],
                    },
                )
                item.addChild(simulation_item)
                self._add_record_details(simulation_item, record)
            item.setExpanded(True)
        root.setExpanded(True)

    def select_simulation(self, simulation_id: str) -> bool:
        """Select and display a stored simulation in this workspace."""

        target_id = str(simulation_id)
        item = self._simulation_tree_item(target_id)
        if item is None:
            self.result_pane.set_simulation(target_id)
            record = self.project.simulations.get(target_id)
            if isinstance(record, dict):
                data_id = record.get("data_id")
                self.set_target_data_id(str(data_id) if data_id else None)
            return False
        self.tree.setCurrentItem(item)
        return True

    def _simulation_tree_item(
        self,
        simulation_id: str,
    ) -> QtWidgets.QTreeWidgetItem | None:
        root = self.tree.invisibleRootItem()
        stack = [root.child(index) for index in range(root.childCount())]
        while stack:
            item = stack.pop()
            payload = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if (
                isinstance(payload, dict)
                and payload.get("kind") == "simulation"
                and str(payload.get("simulation_id")) == simulation_id
            ):
                return item
            stack.extend(
                item.child(index) for index in range(item.childCount())
            )
        return None

    def _add_record_details(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        record: dict[str, Any],
    ) -> None:
        metadata = record.get("metadata", {})
        parameters = record.get("parameters", {})
        for key in (
            "data_id",
            "fit_rank",
            "fit_target_data_id",
            "generated_cif_id",
            "generated_cif_rank",
            "cif_path",
            "dataset_uri",
            "structure_path",
        ):
            if record.get(key):
                parent.addChild(
                    QtWidgets.QTreeWidgetItem(
                        [_labelize(key), str(record[key])]
                    )
                )
        metadata_item = QtWidgets.QTreeWidgetItem(["Metadata", ""])
        parent.addChild(metadata_item)
        for key, value in sorted(metadata.items()):
            metadata_item.addChild(
                QtWidgets.QTreeWidgetItem(
                    [_labelize(key), _format_value(value)]
                )
            )
        fit_metrics = record.get("fit_metrics")
        if isinstance(fit_metrics, dict):
            fit_item = QtWidgets.QTreeWidgetItem(["Fit metrics", ""])
            parent.addChild(fit_item)
            for key, value in sorted(fit_metrics.items()):
                fit_item.addChild(
                    QtWidgets.QTreeWidgetItem(
                        [_labelize(key), _format_value(value)]
                    )
                )
        parameters_item = QtWidgets.QTreeWidgetItem(["Parameters", ""])
        parent.addChild(parameters_item)
        for key, value in sorted(parameters.items()):
            parameters_item.addChild(
                QtWidgets.QTreeWidgetItem(
                    [_labelize(key), _format_value(value)]
                )
            )

    def _handle_tree_selection(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        payload = items[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return
        if payload.get("kind") in {"structure", "simulation"}:
            self._set_active_structure(str(payload["structure_id"]))
        if payload.get("kind") == "simulation":
            self.result_pane.set_simulation(str(payload["simulation_id"]))
            record = self.project.simulations.get(
                str(payload["simulation_id"]), {}
            )
            if isinstance(record, dict):
                data_id = record.get("data_id")
                self.set_target_data_id(str(data_id) if data_id else None)
                self.open_reconstruction_action.setEnabled(
                    is_ewald_sphere_sweep_record(record)
                )
                if hasattr(self, "open_3d_button"):
                    self.open_3d_button.setEnabled(
                        is_ewald_sphere_sweep_record(record)
                    )
                self._set_cache_status(
                    f"Selected stored result: {payload['simulation_id']}",
                    str(payload["simulation_id"]),
                )
        elif hasattr(self, "open_reconstruction_action"):
            self.open_reconstruction_action.setEnabled(False)
            if hasattr(self, "open_3d_button"):
                self.open_3d_button.setEnabled(False)

    def _set_active_structure(self, structure_id: str) -> None:
        structure = self.structures.get(structure_id)
        if structure is None:
            self.structure_viewer.clear()
            return
        structure_data = structure.get("structure_data")
        if structure_data is None:
            try:
                structure_data = load_structure(structure["path"])
            except Exception as exc:
                self.structure_viewer.set_error(
                    f"Could not load structure: {exc}"
                )
                return
            structure["structure_data"] = structure_data
        self.structure_viewer.set_structure(
            structure_data,
            structure_id=structure_id,
        )
        if not self._sweep_visual_active:
            self._sync_structure_orientation()

    def _selected_structure_id(self) -> str | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        payload = items[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return None
        if payload.get("kind") == "structure":
            return str(payload["structure_id"])
        if payload.get("kind") == "simulation":
            return str(payload["structure_id"])
        return None

    def _selected_simulation_id(self) -> str | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        payload = items[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return None
        if payload.get("kind") == "simulation":
            return str(payload["simulation_id"])
        return None

    def set_target_data_id(self, data_id: str | None) -> None:
        normalized = (
            str(data_id)
            if data_id is not None
            and self.project.data_file_by_id(str(data_id)) is not None
            else None
        )
        self.target_data_id = normalized
        index = self.link_data_file_combo.findData(normalized)
        if index < 0:
            index = 0
        previous_blocked = self.link_data_file_combo.blockSignals(True)
        try:
            self.link_data_file_combo.setCurrentIndex(index)
        finally:
            self.link_data_file_combo.blockSignals(previous_blocked)
        if hasattr(self, "link_selected_button"):
            self.link_selected_button.setEnabled(
                self._selected_simulation_id() is not None
            )

    def selected_data_id(self) -> str | None:
        return self.target_data_id

    def link_selected_simulation_to_data_file(self) -> dict[str, Any] | None:
        simulation_id = self._selected_simulation_id()
        if simulation_id is None:
            simulation_id = self.result_pane.simulation_id
        if simulation_id is None:
            return None
        data_id = self.link_data_file_combo.currentData()
        target_data_id = str(data_id) if data_id else None
        record = self.project.link_simulation_to_data_file(
            simulation_id,
            target_data_id,
        )
        self.target_data_id = target_data_id
        self._refresh_tree()
        self.result_pane.set_simulation(simulation_id)
        self.simulationLinked.emit(simulation_id)
        self.projectChanged.emit()
        return record

    def _handle_target_data_file_changed(self) -> None:
        data_id = self.link_data_file_combo.currentData()
        self.target_data_id = str(data_id) if data_id else None

    def _populate_data_file_links(
        self,
        selected_data_id: str | None = None,
    ) -> None:
        self.link_data_file_combo.clear()
        self.link_data_file_combo.addItem("Unlinked", None)
        for data_file in _project_data_files(self.project):
            label = data_file.name or data_file.data_id or data_file.path.stem
            self.link_data_file_combo.addItem(
                f"{label} ({data_file.data_id})",
                data_file.data_id,
            )
        self.set_target_data_id(selected_data_id)

    def _remember_structure_path(
        self,
        path: Path,
        *,
        persist: bool = True,
    ) -> None:
        path_text = str(path)
        self.structure_history = _updated_structure_history(
            self.structure_history,
            path_text,
        )
        self._refresh_structure_history_field(path_text)
        if persist:
            self.settings.setValue(
                STRUCTURE_HISTORY_SETTING,
                self.structure_history,
            )
            self.settings.sync()

    def _refresh_structure_history_field(
        self,
        selected_path: str | None = None,
    ) -> None:
        previous_blocked = self.import_structure_combo.blockSignals(True)
        try:
            self.import_structure_combo.clear()
            entries = _dedupe_structure_history(
                [*self.project_structure_paths, *self.structure_history]
            )
            self.import_structure_combo.addItems(entries)
            if selected_path:
                self.import_structure_combo.setCurrentText(selected_path)
            elif entries:
                self.import_structure_combo.setCurrentIndex(0)
        finally:
            self.import_structure_combo.blockSignals(previous_blocked)

    def _structure_dialog_directory(self) -> Path:
        current_path = self.import_structure_combo.currentText().strip()
        for path_text in [
            current_path,
            *self.project_structure_paths,
            *self.structure_history,
        ]:
            if not path_text:
                continue
            candidate = Path(path_text).expanduser()
            directory = candidate if candidate.is_dir() else candidate.parent
            if str(directory) and directory.exists():
                return directory
        if self.project_path is not None:
            return self.project_path.parent
        return Path.home()

    def _project_loaded_cif_paths(self) -> list[str]:
        loaded = self.project.reference_cifs.get("loaded", {})
        if not isinstance(loaded, dict):
            return []
        paths: list[str] = []
        for record in loaded.values():
            if not isinstance(record, dict):
                continue
            path = self._ensure_loaded_cif_path(record)
            if path is not None:
                paths.append(str(path))
        return _dedupe_structure_history(paths)

    def _ensure_loaded_cif_path(
        self,
        record: dict[str, Any],
    ) -> Path | None:
        for key in ("local_path", "path", "structure_path"):
            path_value = record.get(key)
            if path_value:
                path = Path(str(path_value))
                if path.exists():
                    return path
        cif_text = str(record.get("cif_text") or "")
        if not cif_text.strip():
            return None
        cif_id = str(record.get("cif_id") or record.get("id") or "loaded_cif")
        directory = Path(self.output_directory) / "loaded_cifs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / _safe_generated_cif_filename(cif_id)
        if not path.exists() or path.read_text(encoding="utf-8") != cif_text:
            path.write_text(cif_text, encoding="utf-8")
        record["path"] = str(path)
        record["local_path"] = str(path)
        record["structure_path"] = str(path)
        return path


class GIWAXSSimulationPane(GIWAXSSimulationWindow):
    """Embedded main-workflow version of the GIWAXS simulation tool."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )


def _simulation_ids_for_structure(
    project: ProjectState,
    structure_id: str,
) -> list[str]:
    return [
        simulation_id
        for simulation_id, record in project.simulations.items()
        if isinstance(record, dict)
        and record.get("structure_id") == structure_id
    ]


def _project_data_files(project: ProjectState):
    data_files = []
    for group in project.data_groups:
        data_files.extend(group.data_files)
    data_files.extend(project.data_files)
    return data_files


def _target_qspace_for_project(
    project: ProjectState,
    data_id: str,
) -> Any | None:
    product_path = project.processed_products.get(data_id)
    if product_path:
        data = _load_qspace_product(Path(str(product_path)))
        if data is not None:
            return data
    data_file = project.data_file_by_id(data_id)
    if data_file is None or not data_file.usable_path.exists():
        return None
    return _load_qspace_product(data_file.usable_path)


def _experimental_qspace_for_project(
    project: ProjectState,
    data_id: str,
) -> Any | None:
    """Backward-compatible alias for loading a comparison target
    image."""

    return _target_qspace_for_project(project, data_id)


def _load_qspace_product(path: Path) -> Any | None:
    try:
        import xarray as xr
    except Exception:
        xr = None
    suffix = path.suffix.lower()
    if suffix == ".npz":
        try:
            with np.load(path) as data:
                intensity = np.asarray(
                    (
                        data["intensity"]
                        if "intensity" in data
                        else data["arr_0"]
                    ),
                    dtype=float,
                )
                qxy = np.asarray(
                    (
                        data["q_ip"]
                        if "q_ip" in data
                        else np.arange(intensity.shape[1])
                    ),
                    dtype=float,
                )
                qz = np.asarray(
                    (
                        data["q_oop"]
                        if "q_oop" in data
                        else np.arange(intensity.shape[0])
                    ),
                    dtype=float,
                )
                raw_label = (
                    data["target_label"] if "target_label" in data else None
                )
            if xr is None:
                return intensity
            attrs = {"comparison_label": "Experimental data"}
            if raw_label is not None:
                attrs["comparison_label"] = str(np.asarray(raw_label).item())
            return xr.DataArray(
                intensity,
                dims=("qz", "qxy"),
                coords={"qz": qz, "qxy": qxy},
                name="target_intensity",
                attrs=attrs,
            )
        except Exception:
            return None
    if xr is not None and suffix in {".nc", ".cdf", ".netcdf"}:
        try:
            return xr.load_dataarray(path)
        except Exception:
            return None
    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile

            image = np.asarray(tifffile.imread(path), dtype=float)
        except Exception:
            return None
        if xr is None:
            return image
        return xr.DataArray(
            image,
            dims=("qz", "qxy"),
            coords={
                "qz": np.arange(image.shape[0], dtype=float),
                "qxy": np.arange(image.shape[1], dtype=float),
            },
            name="target_intensity",
            attrs={"comparison_label": "Experimental data"},
        )
    return None


def _default_output_directory(project_path: Path | None) -> Path:
    if project_path is not None:
        return project_path.parent / "simulations"
    return Path.cwd() / "example" / "projects" / "simulations"


def _read_structure_history(settings: QtCore.QSettings) -> list[str]:
    value = settings.value(STRUCTURE_HISTORY_SETTING, [])
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = [value]
    else:
        raw_values = list(value)
    return _dedupe_structure_history(str(item) for item in raw_values)


def _updated_structure_history(history: list[str], path: str) -> list[str]:
    return _dedupe_structure_history([path, *history])


def _dedupe_structure_history(paths) -> list[str]:
    seen = set()
    history = []
    for path in paths:
        path = str(path).strip()
        if not path or path in seen:
            continue
        seen.add(path)
        history.append(path)
        if len(history) >= STRUCTURE_HISTORY_LIMIT:
            break
    return history


def _double_spinbox(
    value: float,
    minimum: float,
    maximum: float,
    step: float,
) -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(minimum, maximum)
    spinbox.setDecimals(5)
    spinbox.setSingleStep(step)
    spinbox.setValue(value)
    return spinbox


def _level_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(-1.0e12, 1.0e12)
    spinbox.setDecimals(4)
    spinbox.setKeyboardTracking(False)
    spinbox.setMaximumWidth(110)
    return spinbox


def _quantile_spinbox(value: float) -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(0.0, 100.0)
    spinbox.setDecimals(2)
    spinbox.setSingleStep(0.5)
    spinbox.setSuffix("%")
    spinbox.setValue(value)
    spinbox.setMaximumWidth(90)
    return spinbox


def _spinbox(value: int, minimum: int, maximum: int) -> QtWidgets.QSpinBox:
    spinbox = QtWidgets.QSpinBox()
    spinbox.setRange(minimum, maximum)
    spinbox.setValue(value)
    return spinbox


def _add_grid_control(
    layout: QtWidgets.QGridLayout,
    row: int,
    column: int,
    label_text: str,
    control: QtWidgets.QWidget,
    tooltip: str,
) -> None:
    label = (
        rich_label(label_text)
        if "<" in label_text
        else QtWidgets.QLabel(label_text)
    )
    label.setToolTip(qt_tooltip(tooltip))
    control.setToolTip(qt_tooltip(tooltip))
    layout.addWidget(label, row, column * 2)
    layout.addWidget(control, row, column * 2 + 1)


def _simulation_cache_key(
    structure_id: str,
    mode: str,
    parameters: dict[str, Any],
) -> str:
    payload = {
        "structure_id": str(structure_id),
        "mode": str(mode),
        "parameters": {
            key: _cache_value(value)
            for key, value in sorted(parameters.items())
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _record_cache_key(record: dict[str, Any]) -> str | None:
    structure_id = record.get("structure_id")
    mode = record.get("simulation_mode")
    parameters = record.get("parameters")
    if not structure_id or not mode or not isinstance(parameters, dict):
        return None
    return _simulation_cache_key(str(structure_id), str(mode), parameters)


def _stored_record_is_loadable(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    dataset_uri = record.get("dataset_uri")
    if not dataset_uri:
        return False
    return Path(str(dataset_uri)).exists()


def _pattern_cache_parameters_cover(
    cached: dict[str, Any],
    requested: dict[str, Any],
) -> bool:
    for key, requested_value in requested.items():
        if key not in cached:
            return False
        cached_value = cached[key]
        if key in PATTERN_CACHE_SCALABLE_FIELDS:
            try:
                if float(cached_value) < float(requested_value):
                    return False
            except (TypeError, ValueError):
                return False
            continue
        if _cache_value(cached_value) != _cache_value(requested_value):
            return False
    return True


def _pattern_cache_rank(record: dict[str, Any]) -> tuple[int, int, int]:
    parameters = record.get("parameters", {})
    if not isinstance(parameters, dict):
        return (0, 0, 0)
    hkl_extent = int(float(parameters.get("hkl_extent", 0)))
    resolution_x = int(float(parameters.get("resolution_x", 0)))
    resolution_z = int(float(parameters.get("resolution_z", 0)))
    return (
        hkl_extent,
        resolution_x * resolution_z,
        resolution_x + resolution_z,
    )


def _cache_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 9)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 9)


def _generated_cif_sort_key(
    record: dict[str, Any],
) -> tuple[float, float, str]:
    try:
        rank = float(record.get("rank", float("inf")))
    except (TypeError, ValueError):
        rank = float("inf")
    try:
        score = float(record.get("score", float("inf")))
    except (TypeError, ValueError):
        score = float("inf")
    return (rank, score, str(record.get("cif_id") or record.get("id") or ""))


def _generated_cif_label(record: dict[str, Any]) -> str:
    rank = record.get("rank")
    cif_id = str(record.get("cif_id") or record.get("id") or "generated CIF")
    if rank:
        return f"generated CIF rank {rank} ({cif_id})"
    return f"generated CIF ({cif_id})"


def _comparison_structure_label(item: dict[str, Any]) -> str:
    generated = item.get("generated_cif")
    if isinstance(generated, dict):
        return _generated_cif_label(generated)
    record = item.get("record", {})
    if isinstance(record, dict):
        return str(
            record.get("structure_name") or record.get("structure_id") or ""
        )
    return ""


def _comparison_sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
    comparison = item.get("comparison")
    metrics = getattr(comparison, "metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    difference_rmse = float(
        metrics.get("difference_rmse", metrics.get("fit_score", float("inf")))
    )
    weighted_rmse = float(metrics.get("weighted_rmse", float("inf")))
    correlation = float(metrics.get("correlation", 0.0))
    return (difference_rmse, weighted_rmse, -correlation)


def _safe_generated_cif_filename(cif_id: str) -> str:
    safe = "".join(
        (
            character
            if character.isalnum() or character in {"-", "_", "."}
            else "_"
        )
        for character in cif_id
    ).strip("._")
    return f"{safe or 'generated_cif'}.cif"


def _simulation_type_label(record: Any) -> str:
    if isinstance(record, dict) and is_ewald_sphere_sweep_record(record):
        return "Ewald sphere sweep"
    return "GIWAXS simulation"


def _record_html(record: dict[str, Any]) -> str:
    lines = [
        f"<b>{escape(str(record.get('simulation_id', 'GIWAXS simulation')))}</b>",
        "",
    ]
    lines.append(f"Mode: {escape(_simulation_type_label(record))}")
    lines.append("")
    for key in (
        "data_id",
        "structure_name",
        "cif_path",
        "structure_path",
        "dataset_uri",
    ):
        if record.get(key):
            lines.append(f"{_labelize(key)}: {escape(str(record[key]))}")
    lines.append("")
    lines.append("Metadata")
    for key, value in sorted(record.get("metadata", {}).items()):
        lines.append(f"{_labelize(key)}: {escape(_format_value(value))}")
    fit_metrics = record.get("fit_metrics")
    if isinstance(fit_metrics, dict):
        lines.append("")
        lines.append("Fit metrics")
        for key, value in sorted(fit_metrics.items()):
            lines.append(f"{_labelize(key)}: {escape(_format_value(value))}")
    lines.append("")
    lines.append("Parameters")
    for key, value in sorted(record.get("parameters", {}).items()):
        lines.append(f"{_labelize(key)}: {escape(_format_value(value))}")
    return "<br>".join(lines)


def _labelize(key: str) -> str:
    parts = [
        (
            QXY_HTML
            if part == "qxy"
            else QZ_HTML if part == "qz" else "CIF" if part == "cif" else part
        )
        for part in key.split("_")
    ]
    label = " ".join(parts)
    if label.startswith((QXY_HTML, QZ_HTML)):
        return label
    return label[:1].upper() + label[1:]


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_format_value(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {_format_value(val)}" for key, val in value.items()
        )
    return str(value)


def _peak_rows_from_data(
    data: Any,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    if is_ewald_sphere_sweep_data(data):
        return []
    raw_rows = data.attrs.get(PEAK_TABLE_ATTR)
    if isinstance(raw_rows, str) and raw_rows:
        try:
            rows = json.loads(raw_rows)
        except json.JSONDecodeError:
            rows = []
        if isinstance(rows, list):
            return [_normalize_peak_row(row) for row in rows if row]

    if is_ewald_sphere_sweep_data(data):
        return []

    structure_path = record.get("structure_path") or data.attrs.get(
        "structure_path"
    )
    if not structure_path:
        return []
    try:
        params = GIWAXSSimulationParameters.from_mapping(
            {
                **record.get("parameters", {}),
                **{
                    key: data.attrs[key]
                    for key in GIWAXSSimulationParameters().as_dict()
                    if key in data.attrs
                },
            }
        )
        return calculate_giwaxs_peak_rows(structure_path, params)
    except Exception:
        return []


def _normalize_peak_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key in ("h", "k", "l"):
        normalized[key] = int(row.get(key, 0))
    for key in ("qxy", "qz", "intensity", "amplitude", "relative_intensity"):
        value = row.get(key)
        normalized[key] = None if value is None else float(value)
    forbidden = bool(
        row.get("forbidden_reflection")
        or row.get("excluded_from_indexing")
        or str(row.get("reflection_status", "")).lower() == "forbidden"
    )
    normalized["forbidden_reflection"] = forbidden
    normalized["excluded_from_indexing"] = bool(
        row.get("excluded_from_indexing", forbidden)
    )
    normalized["reflection_status"] = str(
        row.get("reflection_status")
        or ("forbidden" if forbidden else "indexable")
    )
    return normalized


def _valid_peak_row(row: dict[str, Any]) -> bool:
    qxy = row.get("qxy")
    qz = row.get("qz")
    return qxy is not None and qz is not None and np.isfinite([qxy, qz]).all()


def _format_hkl(row: dict[str, Any]) -> str:
    return "({h}, {k}, {l})".format(
        h=int(row.get("h", 0)),
        k=int(row.get("k", 0)),
        l=int(row.get("l", 0)),
    )


def _format_float(
    value: Any,
    *,
    significant_digits: int = 4,
) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{significant_digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _qspace_extent(data_array: Any) -> tuple[float, float, float, float]:
    qxy = np.asarray(data_array.coords["qxy"].values, dtype=float)
    qz = np.asarray(data_array.coords["qz"].values, dtype=float)
    return (
        float(np.nanmin(qxy)),
        float(np.nanmax(qxy)),
        float(np.nanmin(qz)),
        float(np.nanmax(qz)),
    )


def _format_peak_info_float(value: Any) -> str:
    return _format_float(
        value,
        significant_digits=PEAK_INFO_SIGNIFICANT_DIGITS,
    )


def _is_forbidden_peak_row(row: dict[str, Any]) -> bool:
    return bool(
        row.get("forbidden_reflection")
        or row.get("excluded_from_indexing")
        or str(row.get("reflection_status", "")).lower() == "forbidden"
    )


def _peak_status_text(row: dict[str, Any]) -> str:
    if _is_forbidden_peak_row(row):
        return "Forbidden"
    return "Indexable"


def _peak_tip(*, x: float, y: float, data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    return _peak_tooltip(data, qxy=x, qz=y)


def _peak_tooltip(
    row: dict[str, Any],
    *,
    qxy: Any | None = None,
    qz: Any | None = None,
) -> str:
    qxy_value = row.get("qxy") if qxy is None else qxy
    qz_value = row.get("qz") if qz is None else qz
    lines = [
        f"(hkl): {_format_hkl(row)}",
        (
            f"{QXY_HTML}: {_format_peak_info_float(qxy_value)} "
            f"{QSPACE_UNITS_HTML}"
        ),
        (
            f"{QZ_HTML}: {_format_peak_info_float(qz_value)} "
            f"{QSPACE_UNITS_HTML}"
        ),
    ]
    intensity = row.get("intensity")
    if intensity is not None:
        lines.append(f"Intensity: {_format_peak_info_float(intensity)}")
    if _is_forbidden_peak_row(row):
        lines.append("Forbidden reflection; excluded from indexing/training")
    return qt_tooltip(
        "<br>".join(lines)
    )


def _set_equal_3d_limits(axes: Any, points: np.ndarray) -> None:
    lower = np.nanmin(points, axis=0)
    upper = np.nanmax(points, axis=0)
    center = (lower + upper) / 2.0
    radius = float(np.nanmax(upper - lower) / 2.0)
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 1.0
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)


def _unit_cell_corners(lattice: np.ndarray) -> np.ndarray:
    frac_corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    return frac_corners @ np.asarray(lattice, dtype=float)


def _unit_cell_edges() -> tuple[tuple[int, int], ...]:
    return (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )


def _project_cell_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return np.column_stack(
        [
            points[:, 0] + 0.36 * points[:, 1],
            points[:, 2] + 0.22 * points[:, 1],
        ]
    )


def _orientation_rotation_matrix(
    theta_x_deg: float,
    theta_y_deg: float,
) -> np.ndarray:
    theta_x = math.radians(float(theta_x_deg))
    theta_y = math.radians(float(theta_y_deg))
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta_x), -np.sin(theta_x)],
            [0.0, np.sin(theta_x), np.cos(theta_x)],
        ],
        dtype=float,
    )
    ry = np.array(
        [
            [np.cos(theta_y), 0.0, -np.sin(theta_y)],
            [0.0, 1.0, 0.0],
            [np.sin(theta_y), 0.0, np.cos(theta_y)],
        ],
        dtype=float,
    )
    return rx @ ry


def _axis_angle_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        return np.eye(3)
    x, y, z = axis / norm
    angle = math.radians(float(angle_deg))
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _arrowhead_segments(
    start: np.ndarray,
    end: np.ndarray,
    *,
    size: float = 0.18,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-12:
        return ((end, end), (end, end))
    unit = direction / length
    normal = np.asarray([-unit[1], unit[0]], dtype=float)
    head_length = min(size, length * 0.35)
    head_width = head_length * 0.55
    base = np.asarray(end, dtype=float) - unit * head_length
    return (
        (np.asarray(end, dtype=float), base + normal * head_width),
        (np.asarray(end, dtype=float), base - normal * head_width),
    )


def _distribution_spread_degrees(
    sigma_theta: float,
    sigma_phi: float,
    sigma_r: float,
) -> float:
    theta_deg = abs(float(sigma_theta)) * 180.0 / math.pi
    phi_deg = abs(float(sigma_phi)) * 180.0 / math.pi
    radial_deg = abs(float(sigma_r)) * 120.0
    return min(90.0, max(1.0, theta_deg * 1.5, phi_deg, radial_deg))


def _sample_orientation_distribution(
    base_rotation: np.ndarray,
    *,
    sigma_theta: float,
    sigma_phi: float,
    sigma_r: float,
    count: int = 28,
) -> list[np.ndarray]:
    rng = np.random.default_rng(1729)
    spread = _distribution_spread_degrees(sigma_theta, sigma_phi, sigma_r)
    rotations = [np.asarray(base_rotation, dtype=float)]
    for _index in range(max(0, int(count) - 1)):
        tilt_x = float(rng.normal(0.0, spread * 0.45))
        tilt_y = float(rng.normal(0.0, spread * 0.35))
        roll = float(rng.normal(0.0, spread * 0.55))
        perturbation = (
            _axis_angle_rotation(np.asarray([1.0, 0.0, 0.0]), tilt_x)
            @ _axis_angle_rotation(np.asarray([0.0, 1.0, 0.0]), tilt_y)
            @ _axis_angle_rotation(np.asarray([0.0, 0.0, 1.0]), roll)
        )
        rotations.append(np.asarray(base_rotation, dtype=float) @ perturbation)
    return rotations


def _orientation_distribution_centers(
    count: int,
    *,
    spread_degrees: float,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 3), dtype=float)
    centers = np.zeros((count, 3), dtype=float)
    if count == 1:
        return centers
    radius = 0.12 + 0.72 * min(max(spread_degrees, 0.0), 90.0) / 90.0
    indices = np.arange(1, count, dtype=float)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (indices - 0.5) / max(1.0, count - 1.0)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = indices * golden_angle
    centers[1:, 0] = radius * r * np.cos(theta)
    centers[1:, 1] = radius * r * np.sin(theta)
    centers[1:, 2] = radius * z
    return centers


def _octahedron_mesh() -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    vertices = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    faces = (
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    )
    return vertices, faces


def _lattice_parameters_from_matrix(
    lattice: Any,
) -> dict[str, float]:
    vectors = np.asarray(lattice, dtype=float)
    if vectors.shape != (3, 3):
        raise ValueError("Expected a 3x3 lattice matrix.")
    a_vec, b_vec, c_vec = vectors
    return {
        "a": _vector_length(a_vec),
        "b": _vector_length(b_vec),
        "c": _vector_length(c_vec),
        "alpha": _vector_angle_degrees(b_vec, c_vec),
        "beta": _vector_angle_degrees(a_vec, c_vec),
        "gamma": _vector_angle_degrees(a_vec, b_vec),
    }


def _vector_length(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def _vector_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first_length = _vector_length(first)
    second_length = _vector_length(second)
    if first_length <= 0.0 or second_length <= 0.0:
        raise ValueError("Lattice vectors must be non-zero.")
    cosine = float(np.dot(first, second) / (first_length * second_length))
    return float(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))


def _lattice_summary(lattice: np.ndarray) -> str:
    vectors = np.asarray(lattice, dtype=float)
    lengths = np.linalg.norm(vectors, axis=1)
    return "Cell: a={:.3g} A, b={:.3g} A, c={:.3g} A".format(*lengths)


def _cpk_color(symbol: str) -> str:
    colors = {
        "H": "#ffffff",
        "C": "#909090",
        "N": "#3050f8",
        "O": "#ff0d0d",
        "F": "#90e050",
        "Cl": "#90e050",
        "Br": "#a62929",
        "I": "#940094",
        "P": "#ff8000",
        "S": "#ffff30",
        "B": "#ffb5b5",
        "Li": "#cc80ff",
        "Be": "#c2ff00",
        "Na": "#ab5cf2",
        "Mg": "#8aff00",
        "Al": "#bfa6a6",
        "Si": "#f0c8a0",
        "K": "#8f40d4",
        "Ca": "#3dff00",
        "Sc": "#e6e6e6",
        "Ti": "#bfc2c7",
        "V": "#a6a6ab",
        "Cr": "#8a99c7",
        "Mn": "#9c7ac7",
        "Fe": "#e06633",
        "Co": "#f090a0",
        "Ni": "#50d050",
        "Cu": "#c88033",
        "Zn": "#7d80b0",
        "Ga": "#c28f8f",
        "Ge": "#668f8f",
        "As": "#bd80e3",
        "Se": "#ffa100",
        "Rb": "#702eb0",
        "Sr": "#00ff00",
        "Y": "#94ffff",
        "Zr": "#94e0e0",
        "Nb": "#73c2c9",
        "Mo": "#54b5b5",
        "Ag": "#c0c0c0",
        "Cd": "#ffd98f",
        "In": "#a67573",
        "Sn": "#668080",
        "Sb": "#9e63b5",
        "Te": "#d47a00",
        "Ba": "#00c900",
        "La": "#70d4ff",
        "W": "#2194d6",
        "Au": "#ffd123",
        "Hg": "#b8b8d0",
        "Pb": "#575961",
        "Bi": "#9e4fb5",
    }
    return colors.get(_element_symbol(symbol), "#ff1493")


def _relative_atom_radius(symbol: str) -> float:
    radii = {
        "H": 0.25,
        "C": 0.7,
        "N": 0.65,
        "O": 0.6,
        "F": 0.5,
        "P": 1.0,
        "S": 1.0,
        "Cl": 1.0,
        "Br": 1.15,
        "I": 1.3,
        "Li": 0.9,
        "Na": 1.2,
        "K": 1.4,
        "Si": 1.1,
        "Pb": 1.35,
    }
    return radii.get(_element_symbol(symbol), 0.85)


def _element_symbol(symbol: str) -> str:
    letters = "".join(char for char in str(symbol) if char.isalpha())
    if not letters:
        return str(symbol)
    if len(letters) == 1:
        return letters.upper()
    return f"{letters[0].upper()}{letters[1:].lower()}"
