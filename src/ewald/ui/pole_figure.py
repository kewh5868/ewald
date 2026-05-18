"""Deployable pole-figure generator tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from ewald.data.models import (
    ProjectState,
    ROIRegion,
    roi_hkl_metadata,
    roi_pole_figure_record,
    set_roi_hkl_metadata,
)
from ewald.processing.pole_figure import (
    BACKGROUND_CONSTANT,
    BACKGROUND_LOCAL_ANNULAR,
    BACKGROUND_NONE,
    BACKGROUND_POLYNOMIAL,
    BACKGROUND_ROI,
    INTENSITY_MEAN,
    INTENSITY_SUM,
    NORMALIZE_AREA,
    NORMALIZE_MAX,
    NORMALIZE_NONE,
    PoleFigureResult,
    PoleFigureSettings,
    export_pole_figure_csv,
    generate_pole_figure,
    pole_figure_record_from_result,
)

try:  # pragma: no cover - exercised through UI tests when installed.
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover
    Figure = None
    FigureCanvasQTAgg = None


class PoleFigureGeneratorWindow(QtWidgets.QMainWindow):
    """Separate pole-figure tool linked to an optional EWALD project."""

    poleFigureSaved = QtCore.Signal(str, str)

    def __init__(
        self,
        *,
        project: ProjectState | None = None,
        project_path: str | Path | None = None,
        data_id: str | None = None,
        roi: ROIRegion | None = None,
        image_data: np.ndarray | None = None,
        axis_ranges: tuple[float, float, float, float] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project or ProjectState(
            name="Pole Figure Generator Session"
        )
        self.project_path = Path(project_path) if project_path else None
        self.data_id = data_id
        self.active_roi: ROIRegion | None = None
        self.image_data = image_data
        self.axis_ranges = axis_ranges
        self.result: PoleFigureResult | None = None
        self.last_output_path: Path | None = None
        self.setWindowTitle("Pole Figure Generator")
        self.resize(1100, 760)

        self._build_actions()
        self._build_controls()
        self._build_plot()
        self._build_result_table()
        self._build_metadata()
        self._build_layout()
        self.set_context(
            data_id=data_id,
            roi=roi,
            image_data=image_data,
            axis_ranges=axis_ranges,
        )

    def set_context(
        self,
        *,
        data_id: str | None,
        roi: ROIRegion | None,
        image_data: np.ndarray | None,
        axis_ranges: tuple[float, float, float, float] | None,
    ) -> None:
        """Load selected ROI context from EWALD without duplicating the app."""

        self.data_id = data_id
        self.image_data = image_data
        self.axis_ranges = axis_ranges
        self.active_roi = roi
        self.result = None
        self.last_output_path = None
        self._populate_roi_combo(selected_roi_id=roi.roi_id if roi else None)
        self._populate_background_roi_combo()
        self._sync_fields_from_roi()
        self._apply_settings_from_roi()
        self._refresh_context_label()
        self._set_output_actions_enabled(False)
        if (
            roi is not None
            and image_data is not None
            and axis_ranges is not None
        ):
            self.generate()

    def _build_actions(self) -> None:
        self.generate_action = QtGui.QAction("Generate", self)
        self.save_metadata_action = QtGui.QAction("Save Metadata", self)
        self.export_csv_action = QtGui.QAction("Export CSV", self)
        self.export_png_action = QtGui.QAction("Export PNG", self)
        self.generate_action.triggered.connect(self.generate)
        self.save_metadata_action.triggered.connect(self.save_metadata)
        self.export_csv_action.triggered.connect(self.export_csv)
        self.export_png_action.triggered.connect(self.export_png)
        toolbar = self.addToolBar("Pole Figure")
        toolbar.setMovable(False)
        toolbar.addAction(self.generate_action)
        toolbar.addAction(self.save_metadata_action)
        toolbar.addAction(self.export_csv_action)
        toolbar.addAction(self.export_png_action)

    def _build_controls(self) -> None:
        self.context_label = QtWidgets.QLabel("No ROI selected")
        self.roi_combo = QtWidgets.QComboBox()
        self.roi_combo.currentIndexChanged.connect(self._handle_roi_changed)

        self.h_edit = QtWidgets.QLineEdit()
        self.k_edit = QtWidgets.QLineEdit()
        self.l_edit = QtWidgets.QLineEdit()
        for edit in (self.h_edit, self.k_edit, self.l_edit):
            edit.setMaximumWidth(70)
            edit.setPlaceholderText("int")
        self.hkl_label_edit = QtWidgets.QLineEdit()
        self.display_label_edit = QtWidgets.QLineEdit()

        self.chi_min = _double_spinbox(-90.0, -180.0, 180.0, suffix=" deg")
        self.chi_max = _double_spinbox(90.0, -180.0, 180.0, suffix=" deg")
        self.chi_bin_width = _double_spinbox(1.0, 0.05, 30.0, suffix=" deg")

        self.intensity_mode = QtWidgets.QComboBox()
        self.intensity_mode.addItem("Integrated sum", INTENSITY_SUM)
        self.intensity_mode.addItem("Mean per valid pixel", INTENSITY_MEAN)

        self.normalization = QtWidgets.QComboBox()
        self.normalization.addItem("None", NORMALIZE_NONE)
        self.normalization.addItem("Maximum = 1", NORMALIZE_MAX)
        self.normalization.addItem("Area = 1", NORMALIZE_AREA)

        self.clip_negative = QtWidgets.QCheckBox("Clip negative values")

        self.background_method = QtWidgets.QComboBox()
        self.background_method.addItem("No subtraction", BACKGROUND_NONE)
        self.background_method.addItem("Constant", BACKGROUND_CONSTANT)
        self.background_method.addItem(
            "Local annular/neighboring ROI",
            BACKGROUND_LOCAL_ANNULAR,
        )
        self.background_method.addItem(
            "User-selected background ROI",
            BACKGROUND_ROI,
        )
        self.background_method.addItem(
            "Polynomial smooth baseline",
            BACKGROUND_POLYNOMIAL,
        )

        self.background_constant = _double_spinbox(0.0, -1.0e9, 1.0e9)
        self.background_roi_combo = QtWidgets.QComboBox()
        self.local_gap = _double_spinbox(0.02, 0.0, 1.0e6)
        self.local_width = _double_spinbox(0.05, 1.0e-6, 1.0e6)
        self.polynomial_degree = QtWidgets.QSpinBox()
        self.polynomial_degree.setRange(0, 5)
        self.polynomial_degree.setValue(2)
        self.polynomial_percentile = _double_spinbox(
            60.0, 0.0, 100.0, suffix="%"
        )

        self.controls = QtWidgets.QGroupBox("Pole Figure")
        form = QtWidgets.QFormLayout(self.controls)
        form.addRow("Context", self.context_label)
        form.addRow("ROI", self.roi_combo)

        hkl_widget = QtWidgets.QWidget()
        hkl_layout = QtWidgets.QHBoxLayout(hkl_widget)
        hkl_layout.setContentsMargins(0, 0, 0, 0)
        hkl_layout.addWidget(QtWidgets.QLabel("h"))
        hkl_layout.addWidget(self.h_edit)
        hkl_layout.addWidget(QtWidgets.QLabel("k"))
        hkl_layout.addWidget(self.k_edit)
        hkl_layout.addWidget(QtWidgets.QLabel("l"))
        hkl_layout.addWidget(self.l_edit)
        hkl_layout.addStretch(1)
        form.addRow("hkl", hkl_widget)
        form.addRow("hkl label", self.hkl_label_edit)
        form.addRow("Display label", self.display_label_edit)
        form.addRow("chi min", self.chi_min)
        form.addRow("chi max", self.chi_max)
        form.addRow("Bin width", self.chi_bin_width)
        form.addRow("Intensity", self.intensity_mode)
        form.addRow("Normalization", self.normalization)
        form.addRow("", self.clip_negative)
        form.addRow("Background", self.background_method)
        form.addRow("Constant", self.background_constant)
        form.addRow("Background ROI", self.background_roi_combo)
        form.addRow("Local gap", self.local_gap)
        form.addRow("Local width", self.local_width)
        form.addRow("Polynomial degree", self.polynomial_degree)
        form.addRow("Baseline percentile", self.polynomial_percentile)

    def _build_plot(self) -> None:
        if Figure is None or FigureCanvasQTAgg is None:
            self.canvas = None
            self.figure = None
            self.axes = None
            self.plot_widget = QtWidgets.QLabel("Pole figure plot")
            self.plot_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            return
        self.figure = Figure(figsize=(6.0, 4.0), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.plot_widget = self.canvas

    def _build_result_table(self) -> None:
        self.result_table = QtWidgets.QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(
            ["chi (deg)", "Intensity", "Raw", "Background", "Valid px"]
        )
        self.result_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.result_table.setMinimumHeight(180)

    def _build_metadata(self) -> None:
        self.metadata_text = QtWidgets.QPlainTextEdit(readOnly=True)
        self.metadata_text.setMaximumHeight(150)

    def _build_layout(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.controls)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(self.plot_widget, stretch=1)
        right_layout.addWidget(self.result_table)
        right_layout.addWidget(self.metadata_text)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 760])
        self.setCentralWidget(splitter)

    def generate(self) -> PoleFigureResult | None:
        """Generate a pole figure for the active ROI."""

        if self.active_roi is None:
            self._show_message(
                "Select an ROI before generating a pole figure."
            )
            return None
        if self.image_data is None or self.axis_ranges is None:
            self._show_message(
                "Pole figure generation requires corrected q-space data."
            )
            return None
        if not self._store_hkl_fields():
            return None
        settings = self.settings()
        result = generate_pole_figure(
            self.active_roi,
            self.image_data,
            self.axis_ranges,
            settings=settings,
            background_roi=self._selected_background_roi(),
        )
        if result is None:
            self._show_message("No finite data were found inside the ROI.")
            return None
        self.result = result
        self._plot_result()
        self._populate_result_table()
        self._refresh_metadata_text()
        self._set_output_actions_enabled(True)
        return result

    def settings(self) -> PoleFigureSettings:
        """Return settings from the current controls."""

        return PoleFigureSettings(
            chi_min_deg=self.chi_min.value(),
            chi_max_deg=self.chi_max.value(),
            chi_bin_width_deg=self.chi_bin_width.value(),
            intensity_mode=str(self.intensity_mode.currentData()),
            background_method=str(self.background_method.currentData()),
            background_constant=self.background_constant.value(),
            background_roi_id=self.background_roi_combo.currentData(),
            local_background_gap=self.local_gap.value(),
            local_background_width=self.local_width.value(),
            polynomial_degree=self.polynomial_degree.value(),
            polynomial_percentile=self.polynomial_percentile.value(),
            normalization=str(self.normalization.currentData()),
            clip_negative=self.clip_negative.isChecked(),
            display_label=self.display_label_edit.text().strip(),
        )

    def save_metadata(self) -> dict[str, Any] | None:
        """Link current pole-figure metadata back to the ROI."""

        return self._save_metadata(output_file_path=self.last_output_path)

    def export_csv(self) -> Path | None:
        """Export generated pole-figure data and update ROI metadata."""

        if self.result is None and self.generate() is None:
            return None
        assert self.result is not None
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Pole Figure CSV",
            str(self._default_output_path(".csv")),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return None
        output_path = export_pole_figure_csv(self.result, path)
        self.last_output_path = output_path
        self._save_metadata(output_file_path=output_path)
        return output_path

    def export_png(self) -> Path | None:
        """Export the plotted pole figure and update ROI metadata."""

        if self.result is None and self.generate() is None:
            return None
        if self.figure is None:
            self._show_message("Plot export requires matplotlib.")
            return None
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Pole Figure PNG",
            str(self._default_output_path(".png")),
            "PNG Files (*.png);;All Files (*)",
        )
        if not path:
            return None
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(output_path, dpi=180)
        self.last_output_path = output_path
        self._save_metadata(output_file_path=output_path)
        return output_path

    def _save_metadata(
        self,
        *,
        output_file_path: str | Path | None = None,
    ) -> dict[str, Any] | None:
        if self.active_roi is None:
            return None
        if self.result is None and self.generate() is None:
            return None
        assert self.result is not None
        if not self._store_hkl_fields():
            return None
        record = pole_figure_record_from_result(
            self.active_roi,
            self.result,
            output_file_path=output_file_path,
        )
        if self.data_id and self.active_roi.roi_id:
            stored = self.project.set_roi_pole_figure_metadata(
                self.data_id,
                self.active_roi.roi_id,
                record,
            )
        else:
            from ewald.data.models import set_roi_pole_figure_record

            stored = set_roi_pole_figure_record(self.active_roi, record)
        self._refresh_metadata_text()
        if self.data_id and self.active_roi.roi_id:
            self.poleFigureSaved.emit(self.data_id, self.active_roi.roi_id)
        return stored

    def _plot_result(self) -> None:
        if self.result is None or self.axes is None or self.canvas is None:
            return
        result = self.result
        self.axes.clear()
        self.axes.plot(
            result.chi_deg,
            result.intensity,
            color="#2563eb",
            linewidth=1.8,
            label="Background corrected",
        )
        if np.isfinite(result.background).any():
            self.axes.plot(
                result.chi_deg,
                result.background,
                color="#f97316",
                linewidth=1.0,
                alpha=0.8,
                label="Background",
            )
        title = result.settings.display_label or result.hkl_label
        if not title:
            title = result.source_roi_name
        self.axes.set_title(title)
        self.axes.set_xlabel("chi (deg)")
        ylabel = "Intensity"
        if result.settings.normalization == NORMALIZE_MAX:
            ylabel = "Intensity (max-normalized)"
        elif result.settings.normalization == NORMALIZE_AREA:
            ylabel = "Intensity (area-normalized)"
        self.axes.set_ylabel(ylabel)
        self.axes.grid(True, alpha=0.25)
        self.axes.legend(fontsize=8, loc="best")
        self.canvas.draw_idle()

    def _populate_result_table(self) -> None:
        if self.result is None:
            self.result_table.setRowCount(0)
            return
        rows = self.result.table_rows()
        self.result_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row["chi_deg"],
                row["intensity"],
                row["raw_intensity"],
                row["background"],
                row["valid_pixel_count"],
            ]
            for column, value in enumerate(values):
                self.result_table.setItem(
                    row_index,
                    column,
                    QtWidgets.QTableWidgetItem(_format_table_value(value)),
                )
        self.result_table.resizeColumnsToContents()

    def _refresh_metadata_text(self) -> None:
        roi = self.active_roi
        if roi is None:
            self.metadata_text.setPlainText("No ROI selected.")
            return
        lines = [
            f"ROI: {roi.name or roi.roi_id}",
            f"ROI id: {roi.roi_id}",
            f"Data id: {self.data_id or ''}",
        ]
        hkl = roi_hkl_metadata(roi)
        if any(hkl.values()):
            lines.append(
                "hkl: "
                f"{hkl.get('h')}, {hkl.get('k')}, {hkl.get('l')} "
                f"{hkl.get('label') or ''}".strip()
            )
        if self.result is not None:
            lines.extend(
                [
                    f"Background: {self.result.background_record}",
                    f"Missing bins: {self.result.missing_fraction:.1%}",
                ]
            )
        record = roi_pole_figure_record(roi)
        if record is not None:
            lines.extend(
                [
                    f"Saved: {record.get('generated_at', '')}",
                    f"Output: {record.get('output_file_path') or ''}",
                    f"Current: {record.get('current')}",
                ]
            )
        self.metadata_text.setPlainText("\n".join(lines))

    def _populate_roi_combo(self, *, selected_roi_id: str | None) -> None:
        self.roi_combo.blockSignals(True)
        try:
            self.roi_combo.clear()
            if not self.data_id:
                self.roi_combo.addItem("No ROI", None)
                return
            for roi in self.project.rois_for_target(self.data_id):
                self.roi_combo.addItem(
                    roi.name or roi.roi_id or "ROI", roi.roi_id
                )
            if selected_roi_id is not None:
                index = self.roi_combo.findData(selected_roi_id)
                if index >= 0:
                    self.roi_combo.setCurrentIndex(index)
        finally:
            self.roi_combo.blockSignals(False)
        self.roi_combo.setEnabled(self.roi_combo.count() > 0)

    def _populate_background_roi_combo(self) -> None:
        self.background_roi_combo.blockSignals(True)
        try:
            self.background_roi_combo.clear()
            self.background_roi_combo.addItem("None", None)
            if self.data_id:
                active_id = self.active_roi.roi_id if self.active_roi else None
                for roi in self.project.rois_for_target(self.data_id):
                    if roi.roi_id == active_id:
                        continue
                    self.background_roi_combo.addItem(
                        roi.name or roi.roi_id or "ROI",
                        roi.roi_id,
                    )
        finally:
            self.background_roi_combo.blockSignals(False)

    def _handle_roi_changed(self) -> None:
        roi_id = self.roi_combo.currentData()
        if roi_id is None or self.data_id is None:
            return
        for roi in self.project.rois_for_target(self.data_id):
            if roi.roi_id == roi_id:
                self.active_roi = roi
                self.result = None
                self._populate_background_roi_combo()
                self._sync_fields_from_roi()
                self._apply_settings_from_roi()
                self._refresh_context_label()
                self._set_output_actions_enabled(False)
                return

    def _selected_background_roi(self) -> ROIRegion | None:
        background_roi_id = self.background_roi_combo.currentData()
        if background_roi_id is None or self.data_id is None:
            return None
        for roi in self.project.rois_for_target(self.data_id):
            if roi.roi_id == background_roi_id:
                return roi
        return None

    def _sync_fields_from_roi(self) -> None:
        roi = self.active_roi
        hkl = roi_hkl_metadata(roi) if roi is not None else {}
        self.h_edit.setText(_hkl_field_text(hkl.get("h")))
        self.k_edit.setText(_hkl_field_text(hkl.get("k")))
        self.l_edit.setText(_hkl_field_text(hkl.get("l")))
        self.hkl_label_edit.setText(str(hkl.get("label") or ""))
        self.display_label_edit.setText("")
        record = roi_pole_figure_record(roi) if roi is not None else None
        if record:
            label = record.get("custom_label")
            if label:
                self.display_label_edit.setText(str(label))

    def _apply_settings_from_roi(self) -> None:
        record = (
            roi_pole_figure_record(self.active_roi)
            if self.active_roi is not None
            else None
        )
        parameters = record.get("generation_parameters", {}) if record else {}
        if not isinstance(parameters, dict):
            parameters = {}
        settings = PoleFigureSettings(
            chi_min_deg=float(parameters.get("chi_min_deg", -90.0)),
            chi_max_deg=float(parameters.get("chi_max_deg", 90.0)),
            chi_bin_width_deg=float(parameters.get("chi_bin_width_deg", 1.0)),
            intensity_mode=str(
                parameters.get("intensity_mode", INTENSITY_SUM)
            ),
            background_method=str(
                parameters.get("background_method", BACKGROUND_NONE)
            ),
            background_constant=float(
                parameters.get("background_constant", 0.0)
            ),
            background_roi_id=parameters.get("background_roi_id"),
            local_background_gap=float(
                parameters.get("local_background_gap", 0.02)
            ),
            local_background_width=float(
                parameters.get("local_background_width", 0.05)
            ),
            polynomial_degree=int(parameters.get("polynomial_degree", 2)),
            polynomial_percentile=float(
                parameters.get("polynomial_percentile", 60.0)
            ),
            normalization=str(parameters.get("normalization", NORMALIZE_NONE)),
            clip_negative=bool(parameters.get("clip_negative", False)),
            display_label=str(parameters.get("display_label", "")),
        )
        self._apply_settings(settings)

    def _apply_settings(self, settings: PoleFigureSettings) -> None:
        self.chi_min.setValue(settings.chi_min_deg)
        self.chi_max.setValue(settings.chi_max_deg)
        self.chi_bin_width.setValue(settings.chi_bin_width_deg)
        _set_combo_data(self.intensity_mode, settings.intensity_mode)
        _set_combo_data(self.background_method, settings.background_method)
        _set_combo_data(self.normalization, settings.normalization)
        self.background_constant.setValue(settings.background_constant)
        if settings.background_roi_id:
            _set_combo_data(
                self.background_roi_combo,
                settings.background_roi_id,
            )
        self.local_gap.setValue(settings.local_background_gap)
        self.local_width.setValue(settings.local_background_width)
        self.polynomial_degree.setValue(settings.polynomial_degree)
        self.polynomial_percentile.setValue(settings.polynomial_percentile)
        self.clip_negative.setChecked(settings.clip_negative)
        if settings.display_label:
            self.display_label_edit.setText(settings.display_label)

    def _store_hkl_fields(self) -> bool:
        if self.active_roi is None:
            return True
        try:
            set_roi_hkl_metadata(
                self.active_roi,
                h=self.h_edit.text(),
                k=self.k_edit.text(),
                l=self.l_edit.text(),
                label=self.hkl_label_edit.text(),
            )
        except ValueError as exc:
            self._show_message(str(exc))
            return False
        return True

    def _refresh_context_label(self) -> None:
        if self.active_roi is None:
            self.context_label.setText("No ROI selected")
            return
        source = self.data_id or self.active_roi.target_id
        self.context_label.setText(
            f"{self.active_roi.name or self.active_roi.roi_id} from {source}"
        )

    def _set_output_actions_enabled(self, enabled: bool) -> None:
        self.save_metadata_action.setEnabled(enabled)
        self.export_csv_action.setEnabled(enabled)
        self.export_png_action.setEnabled(enabled)

    def _default_output_path(self, suffix: str) -> Path:
        directory = _default_output_directory(self.project_path)
        roi_label = "pole_figure"
        if self.active_roi is not None:
            roi_label = (
                self.active_roi.name or self.active_roi.roi_id or roi_label
            )
        stem = _safe_filename(roi_label)
        return directory / f"{stem}{suffix}"

    def _show_message(self, message: str) -> None:
        QtWidgets.QMessageBox.information(
            self, "Pole Figure Generator", message
        )


def _double_spinbox(
    value: float,
    minimum: float,
    maximum: float,
    *,
    suffix: str = "",
) -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(minimum, maximum)
    spinbox.setDecimals(4)
    spinbox.setSingleStep(1.0)
    spinbox.setKeyboardTracking(False)
    spinbox.setValue(value)
    spinbox.setSuffix(suffix)
    return spinbox


def _set_combo_data(combo: QtWidgets.QComboBox, value: Any) -> None:
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def _format_table_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.6g}"


def _hkl_field_text(value: Any) -> str:
    return "" if value is None else str(value)


def _default_output_directory(project_path: Path | None) -> Path:
    if project_path is not None:
        return project_path.parent / "pole_figures"
    return Path.cwd() / "example" / "projects" / "pole_figures"


def _safe_filename(value: str) -> str:
    safe = "".join(
        (
            character
            if character.isalnum() or character in {" ", ".", "_", "-"}
            else "_"
        )
        for character in value
    )
    return safe.strip(" ._") or "pole_figure"
