"""Image correction workflow for raw detector data."""

from __future__ import annotations

import json
import re
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from ewald.data.models import (
    CorrectionAssetRef,
    ImageCorrectionState,
    ProjectState,
)
from ewald.processing.low_q import (
    StructureOpticsEstimate,
    build_low_q_features,
    critical_angle_deg_from_delta,
    estimate_bright_spot_centroid,
    estimate_refractive_index_delta,
    estimate_refractive_index_from_structure,
)
from ewald.ui.notation import QXY_HTML, QZ_HTML, qt_tooltip
from ewald.ui.orientation import sample_orientation_for_image_transform

STRUCTURE_FILE_FILTER = (
    "Structure Files (*.cif *.mcif POSCAR* CONTCAR* *.vasp);;All Files (*)"
)
FILM_MATERIAL_MEMORY_SETTING = "film_material_memory"
_STOICHIOMETRY_TEXT_PATTERN = re.compile(r"^[A-Za-z0-9_ .()+\-\[\]]+$")

FIELD_TOOLTIPS = {
    "mask": (
        "Select the detector mask to apply to this image or stack. Masks can "
        "be shared by multiple loaded data items."
    ),
    "calibrant": (
        "Select the pyFAI PONI calibrant that defines detector distance, "
        "beam center, wavelength, and geometry for q-space mapping."
    ),
    "load_mask": (
        "Load a detector mask file and assign it to the selected data item."
    ),
    "load_calibrant": (
        "Load a pyFAI PONI calibrant and assign it to the selected data item."
    ),
    "pyfai_calibration": (
        "Open the pyFAI calibration and mask tool for creating or editing "
        "PONI and mask files."
    ),
    "energy": (
        "X-ray energy used for reciprocal-space conversion and film optics. "
        "If the selected PONI file includes a wavelength, this field is "
        "filled from that value."
    ),
    "solid_angle": (
        "Apply pyFAI's solid-angle correction so detector pixels are "
        "normalized for their angular coverage. This is enabled by default."
    ),
    "polarization": (
        "Apply pyFAI polarization correction to compensate for synchrotron "
        "beam polarization before mapping or integration."
    ),
    "polarization_factor": (
        "Polarization factor passed to pyFAI when polarization correction is "
        "enabled. 0.95 is the default for highly horizontally polarized "
        "synchrotron beams."
    ),
    "normalization": (
        "Scale factor passed through pyFAI normalization. Use this for "
        "incident-flux, exposure, or other external intensity normalization "
        "when available."
    ),
    "dummy": (
        "Enable pyFAI dummy-pixel handling. Use this only when invalid pixels "
        "are encoded as a known numeric value in the image."
    ),
    "dummy_value": (
        "Numeric dummy value passed to pyFAI. Pixels matching this value are "
        "treated as invalid during corrections."
    ),
    "delta_dummy": (
        "Tolerance around the dummy value passed to pyFAI as delta_dummy. "
        "Pixels within dummy +/- this value are treated as invalid."
    ),
    "rotation": (
        qt_tooltip(
            "Rotate the raw detector image before correction so the preview "
            f"orientation matches the {QXY_HTML}/{QZ_HTML} corrected image."
        )
    ),
    "mirror": (
        "Mirror the already-rotated detector image over its active y-axis. "
        "Together with rotation this covers the pyFAI orientation variants."
    ),
    "sample_orientation": (
        "pyFAI sample_orientation value derived from the chosen rotation and "
        "mirror state. It controls how pyFAI maps detector pixels into "
        "reciprocal space."
    ),
    "reflected_x": (
        "Pixel-space x coordinate for the reflected or specular beam spot. "
        "This anchors optional low-q identifiers."
    ),
    "reflected_y": (
        "Pixel-space y coordinate for the reflected or specular beam spot. "
        "This anchors optional low-q identifiers."
    ),
    "critical_angle": (
        "Film critical angle in degrees. Use a measured value or calculate "
        "it from the refractive-index delta estimate."
    ),
    "stoichiometry": (
        "Film stoichiometry used to estimate refractive-index delta, for "
        "example CH3NH3PbI3."
    ),
    "density": (
        "Film mass density used with stoichiometry and X-ray energy to "
        "estimate refractive-index delta."
    ),
    "film_memory": (
        "Saved film stoichiometry and density entries. Load one to populate "
        "the film optics inputs, or save the current values for reuse."
    ),
    "save_film_memory": (
        "Save the current film stoichiometry and positive density as a reusable "
        "memory item."
    ),
    "load_film_memory": (
        "Populate the film stoichiometry and density inputs from the selected "
        "memory item."
    ),
    "delete_film_memory": "Delete the selected film material memory item.",
    "clear_film_memory": "Delete all saved film material memory items.",
    "refractive_delta": (
        "Real refractive-index decrement delta for the film. This can be "
        "entered manually or estimated from chemistry and density."
    ),
    "estimate_delta": (
        "Estimate refractive-index delta and critical angle from the film "
        "stoichiometry, density, and X-ray energy."
    ),
    "estimate_structure": (
        "Load a reference CIF, POSCAR, CONTCAR, or VASP structure, estimate "
        "film density and refractive-index delta from the parsed composition, "
        "and review the values before applying them."
    ),
    "calculate_critical": (
        "Calculate the critical angle from the current refractive-index "
        "delta value."
    ),
    "estimate_reflected": (
        "Search the raw detector image for a bright low-q spot and use its "
        "centroid as the reflected beam position."
    ),
    "add_identifiers": (
        "Create optional low-q markers such as the direct beam, sample "
        "horizon, specular reflection, critical edge, and Yoneda band."
    ),
    "add_artifact": (
        "Add a pixel-space artifact box for a detector gap, beamstop shadow, "
        "streak, or other region to track during correction review."
    ),
    "apply": (
        "Assign the selected mask and PONI calibrant to this data item "
        "without locking the correction workflow."
    ),
    "confirm": (
        "Permanently confirm image corrections for this loaded data item. "
        "Reload the data if corrections need to be rebuilt later."
    ),
    "low_q_status": (
        "Summary of low-q identifiers and film-optics estimates currently "
        "stored for this data item."
    ),
}


class ApplyImageCorrectionsPane(QtWidgets.QWidget):
    """Collect and confirm mask/PONI corrections for one detector image."""

    correctionsApplied = QtCore.Signal(str)
    correctionsConfirmed = QtCore.Signal(str)

    def __init__(
        self,
        project: ProjectState,
        data_id: str,
        *,
        group_id: str | None = None,
        settings: QtCore.QSettings | None = None,
        load_mask_action: QtGui.QAction | None = None,
        load_calibrant_action: QtGui.QAction | None = None,
        pyfai_calibration_action: QtGui.QAction | None = None,
        pyfai_calibration_status: str = "",
        pyfai_calibration_status_tooltip: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.data_id = data_id
        self.group_id = group_id
        self.settings = settings or QtCore.QSettings("EWALD", "EWALD")
        self.load_mask_action = load_mask_action
        self.load_calibrant_action = load_calibrant_action
        self.pyfai_calibration_action = pyfai_calibration_action
        self._pyfai_calibration_status = pyfai_calibration_status
        self._pyfai_calibration_status_tooltip = (
            pyfai_calibration_status_tooltip
        )
        self.loaded_film_memory_id: str | None = None
        self._loading_film_memory = False

        self.load_mask_button: QtWidgets.QToolButton | None = None
        self.load_calibrant_button: QtWidgets.QToolButton | None = None
        self.pyfai_calibration_button: QtWidgets.QToolButton | None = None
        self.pyfai_calibration_status_label: QtWidgets.QLabel | None = None
        self.mask_combo = QtWidgets.QComboBox()
        self.calibrant_combo = QtWidgets.QComboBox()
        self.energy_kev = _energy_spinbox()
        self.solid_angle_check = QtWidgets.QCheckBox()
        self.solid_angle_check.setChecked(True)
        self.polarization_check = QtWidgets.QCheckBox()
        self.polarization_check.setChecked(True)
        self.polarization_factor = _polarization_spinbox()
        self.normalization_factor = _normalization_spinbox()
        self.dummy_check = QtWidgets.QCheckBox()
        self.dummy_value = _correction_value_spinbox()
        self.delta_dummy = _correction_value_spinbox()
        self.rotation_combo = QtWidgets.QComboBox()
        self.mirror_y_check = QtWidgets.QCheckBox()
        self.sample_orientation_combo = QtWidgets.QComboBox()
        self.reflected_x = _coordinate_spinbox()
        self.reflected_y = _coordinate_spinbox()
        self.critical_angle = _angle_spinbox()
        self.sample_stoichiometry = QtWidgets.QLineEdit()
        self.sample_stoichiometry.setPlaceholderText("e.g. CH3NH3PbI3")
        self.sample_density = _density_spinbox()
        self.film_memory_combo = QtWidgets.QComboBox()
        self.film_memory_combo.setToolTip(FIELD_TOOLTIPS["film_memory"])
        self.save_film_memory_button = QtWidgets.QPushButton("Save")
        self.save_film_memory_button.setToolTip(
            FIELD_TOOLTIPS["save_film_memory"]
        )
        self.load_film_memory_button = QtWidgets.QPushButton("Load")
        self.load_film_memory_button.setToolTip(
            FIELD_TOOLTIPS["load_film_memory"]
        )
        self.delete_film_memory_button = QtWidgets.QPushButton("Delete")
        self.delete_film_memory_button.setToolTip(
            FIELD_TOOLTIPS["delete_film_memory"]
        )
        self.clear_film_memory_button = QtWidgets.QPushButton("Clear All")
        self.clear_film_memory_button.setToolTip(
            FIELD_TOOLTIPS["clear_film_memory"]
        )
        self.film_memory_status = QtWidgets.QLabel("Manual film values.")
        self.film_memory_status.setWordWrap(True)
        self.refractive_delta = _refractive_delta_spinbox()
        self.low_q_status = QtWidgets.QLabel()
        self.low_q_status.setWordWrap(True)
        self.artifact_table = QtWidgets.QTableWidget(0, 5)
        self.artifact_table.setHorizontalHeaderLabels(
            ["Label", "x", "y", "Width", "Height"]
        )
        self.artifact_table.horizontalHeader().setStretchLastSection(True)
        self.artifact_table.setToolTip(
            "Optional pixel-space boxes for beamstop shadows, detector gaps, "
            "streaks, or other artifacts to track before confirming image "
            "corrections."
        )
        self._apply_field_tooltips()

        self._populate_asset_combos()
        self._populate_pyfai_correction_controls()
        self._populate_orientation_controls()
        self._populate_film_memory_controls()
        self._restore_existing_state()
        self._refresh_low_q_status()
        self._build_layout()

    def apply_selected_assets(self, *, emit_signal: bool = True) -> None:
        mask_id = self.mask_combo.currentData()
        calibrant_id = self.calibrant_combo.currentData()
        if mask_id:
            self.project.set_correction_asset_assignment(
                "mask",
                str(mask_id),
                self.data_id,
            )
        if calibrant_id:
            self.project.set_correction_asset_assignment(
                "calibrant",
                str(calibrant_id),
                self.data_id,
            )
        if emit_signal:
            self.correctionsApplied.emit(self.data_id)

    def confirm_corrections(self) -> None:
        mask_id = self.mask_combo.currentData()
        calibrant_id = self.calibrant_combo.currentData()
        if not mask_id or not calibrant_id:
            QtWidgets.QMessageBox.information(
                self,
                "Missing Corrections",
                "Select both a MASK and a PONI calibrant before confirming.",
            )
            return

        self.apply_selected_assets(emit_signal=False)
        existing = self.project.image_corrections.get(self.data_id)
        metadata = dict(existing.metadata) if existing is not None else {}
        metadata.update(
            {
                "workflow": "insight-style-preprocessing",
                "locked_after_confirmation": True,
                "sample_stoichiometry": self.sample_stoichiometry.text().strip(),
                "sample_density_g_cm3": self.sample_density.value(),
                "refractive_index_delta": (
                    self.refractive_delta.value()
                    if self.refractive_delta.value() > 0
                    else None
                ),
            }
        )
        state = ImageCorrectionState(
            target_id=self.data_id,
            mask_asset_id=str(mask_id),
            calibrant_asset_id=str(calibrant_id),
            xray_energy_kev=self.energy_kev.value(),
            image_rotation_deg=int(self.rotation_combo.currentData() or 0),
            image_mirrored_y=self.mirror_y_check.isChecked(),
            pyfai_sample_orientation=int(
                self.sample_orientation_combo.currentData() or 1
            ),
            correct_solid_angle=self.solid_angle_check.isChecked(),
            polarization_factor=(
                self.polarization_factor.value()
                if self.polarization_check.isChecked()
                else None
            ),
            normalization_factor=self.normalization_factor.value(),
            dummy=(
                self.dummy_value.value()
                if self.dummy_check.isChecked()
                else None
            ),
            delta_dummy=(
                self.delta_dummy.value()
                if self.dummy_check.isChecked()
                else None
            ),
            reflected_beam_x_px=self.reflected_x.value(),
            reflected_beam_y_px=self.reflected_y.value(),
            critical_angle_deg=self.critical_angle.value(),
            sample_stoichiometry=self.sample_stoichiometry.text().strip()
            or None,
            sample_density_g_cm3=self.sample_density.value(),
            refractive_index_delta=(
                self.refractive_delta.value()
                if self.refractive_delta.value() > 0
                else None
            ),
            artifact_regions=self.artifact_regions(),
            confirmed=True,
            metadata=metadata,
        )
        self.project.set_image_corrections(state)
        self.correctionsConfirmed.emit(self.data_id)

    def calculate_critical_angle_from_delta(self) -> None:
        delta = self.refractive_delta.value()
        if delta <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "Missing Refractive Delta",
                "Enter the real refractive-index decrement delta first.",
            )
            return
        self.critical_angle.setValue(critical_angle_deg_from_delta(delta))
        state = self._editable_state()
        state.critical_angle_deg = self.critical_angle.value()
        state.refractive_index_delta = delta
        self._store_chemistry_on_state(state)
        self._refresh_low_q_status(
            "Critical angle calculated from refractive-index delta."
        )

    def estimate_delta_from_chemistry(self) -> None:
        try:
            estimate = estimate_refractive_index_delta(
                self.sample_stoichiometry.text(),
                self.sample_density.value(),
                self.energy_kev.value(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.information(
                self,
                "Chemistry Estimate Failed",
                str(exc),
            )
            return
        self.refractive_delta.setValue(estimate.delta)
        self.critical_angle.setValue(estimate.critical_angle_deg)
        state = self._editable_state()
        state.critical_angle_deg = self.critical_angle.value()
        state.xray_energy_kev = self.energy_kev.value()
        state.refractive_index_delta = estimate.delta
        self._store_chemistry_on_state(state, estimate.as_dict())
        self._refresh_low_q_status(
            "Estimated refractive-index delta from stoichiometry and density."
        )

    def estimate_delta_from_structure(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Reference Structure",
            "",
            STRUCTURE_FILE_FILTER,
        )
        if not path:
            return
        try:
            estimate = estimate_refractive_index_from_structure(
                path,
                self.energy_kev.value(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.information(
                self,
                "Structure Estimate Failed",
                str(exc),
            )
            return
        dialog = StructureOpticsReviewDialog(estimate, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._apply_structure_optics_estimate(estimate)

    def estimate_reflected_beam_spot(self) -> None:
        data_file = self.project.data_file_by_id(self.data_id)
        if data_file is None:
            return
        try:
            import tifffile

            image = tifffile.imread(data_file.usable_path)
        except Exception:
            QtWidgets.QMessageBox.information(
                self,
                "Image Not Available",
                "The detector image could not be loaded for spot detection.",
            )
            return
        centroid = estimate_bright_spot_centroid(
            image,
            rotation_deg=int(self.rotation_combo.currentData() or 0),
            mirrored_y=self.mirror_y_check.isChecked(),
        )
        if centroid is None:
            QtWidgets.QMessageBox.information(
                self,
                "No Bright Spot",
                "No bright low-q spot could be identified in the image.",
            )
            return
        x_px, y_px = centroid
        self.reflected_x.setValue(x_px)
        self.reflected_y.setValue(y_px)
        self._refresh_low_q_status("Reflected beam spot estimate updated.")

    def add_low_q_identifiers(self) -> None:
        incident_angle = self._incident_angle_deg()
        critical_angle = self.critical_angle.value()
        critical_angle = critical_angle if critical_angle > 0 else None
        features = build_low_q_features(
            incident_angle_deg=incident_angle,
            critical_angle_deg=critical_angle,
            xray_energy_kev=self.energy_kev.value(),
            reflected_beam_x_px=self.reflected_x.value(),
            reflected_beam_y_px=self.reflected_y.value(),
        )
        state = self._editable_state()
        state.reflected_beam_x_px = self.reflected_x.value()
        state.reflected_beam_y_px = self.reflected_y.value()
        state.critical_angle_deg = critical_angle
        state.xray_energy_kev = self.energy_kev.value()
        self._store_chemistry_on_state(state)
        state.metadata["low_q_features"] = [
            feature.as_dict() for feature in features
        ]
        state.metadata["low_q_identifier_model"] = (
            "INSIGHT-inspired grazing-incidence geometry with explicit "
            "primary/direct beam, sample horizon (alpha_f = 0), specular "
            "reflection (alpha_f = alpha_i), critical edge q_c, Yoneda band "
            "(alpha_f = alpha_c), and low-q exclusion center."
        )
        self._refresh_low_q_status(f"{len(features)} low-q identifiers added.")
        self.correctionsApplied.emit(self.data_id)

    def add_artifact_region(self) -> None:
        row = self.artifact_table.rowCount()
        self.artifact_table.insertRow(row)
        defaults = [f"artifact-{row + 1}", "0", "0", "0", "0"]
        for column, value in enumerate(defaults):
            self.artifact_table.setItem(
                row,
                column,
                QtWidgets.QTableWidgetItem(value),
            )

    def artifact_regions(self) -> list[dict[str, Any]]:
        regions: list[dict[str, Any]] = []
        for row in range(self.artifact_table.rowCount()):
            label = _table_text(self.artifact_table, row, 0)
            regions.append(
                {
                    "label": label or f"artifact-{row + 1}",
                    "x": _table_float(self.artifact_table, row, 1),
                    "y": _table_float(self.artifact_table, row, 2),
                    "width": _table_float(self.artifact_table, row, 3),
                    "height": _table_float(self.artifact_table, row, 4),
                }
            )
        return regions

    def save_film_material_memory(self) -> None:
        """Save the current film stoichiometry and density for reuse."""

        try:
            stoichiometry = _validated_stoichiometry(
                self.sample_stoichiometry.text()
            )
            density = _positive_density(self.sample_density.value())
        except ValueError as exc:
            QtWidgets.QMessageBox.information(
                self,
                "Film Memory",
                str(exc),
            )
            return

        existing = _find_matching_film_memory(
            self.project.film_material_memory,
            stoichiometry,
            density,
        )
        if existing is not None:
            self.loaded_film_memory_id = str(existing["memory_id"])
            self._populate_film_memory_combo(self.loaded_film_memory_id)
            self._set_film_memory_loaded_status(existing)
            return

        item = self.project.remember_film_material(
            stoichiometry,
            density,
        )
        self.loaded_film_memory_id = str(item["memory_id"])
        self._persist_film_memory()
        self._populate_film_memory_combo(self.loaded_film_memory_id)
        self._set_film_memory_loaded_status(item)

    def load_selected_film_material_memory(self) -> None:
        """Populate film inputs from the selected memory item."""

        item = self._selected_film_memory_item()
        if item is None:
            return
        self._loading_film_memory = True
        try:
            self.sample_stoichiometry.setText(str(item["stoichiometry"]))
            self.sample_density.setValue(float(item["density_g_cm3"]))
        finally:
            self._loading_film_memory = False
        self.loaded_film_memory_id = str(item["memory_id"])
        self._set_film_memory_loaded_status(item)

    def delete_selected_film_material_memory(self) -> None:
        """Delete the selected film memory item."""

        memory_id = self.film_memory_combo.currentData()
        if not memory_id:
            return
        removed = self.project.remove_film_material_memory(str(memory_id))
        if not removed:
            return
        deleted_loaded = self.loaded_film_memory_id == str(memory_id)
        if deleted_loaded:
            self.loaded_film_memory_id = None
        self._persist_film_memory()
        self._populate_film_memory_combo()
        if deleted_loaded:
            self._set_film_memory_status(
                "Deleted the loaded memory item; current values remain in the "
                "fields."
            )
        else:
            self._set_film_memory_status("Deleted film memory item.")

    def clear_film_material_memory(self) -> None:
        """Delete all saved film memory items."""

        if not self.project.film_material_memory:
            return
        self.project.clear_film_material_memory()
        self.loaded_film_memory_id = None
        self._persist_film_memory()
        self._populate_film_memory_combo()
        self._set_film_memory_status("Cleared all film memory items.")

    def _build_layout(self) -> None:
        title = QtWidgets.QLabel("Apply Image Corrections")
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 4)
        title_font.setBold(True)
        title.setFont(title_font)

        intro = QtWidgets.QLabel(
            "Select the MASK and PONI calibrant, estimate the reflected beam "
            "position and critical angle, then box beamstop shadows, detector "
            "gaps, and other artifacts before confirming."
        )
        intro.setWordWrap(True)

        self.add_artifact_button = QtWidgets.QPushButton("Add Artifact Box")
        self.add_artifact_button.setToolTip(FIELD_TOOLTIPS["add_artifact"])
        self.add_artifact_button.clicked.connect(self.add_artifact_region)
        self.apply_button = QtWidgets.QPushButton("Apply Selected MASK/PONI")
        self.apply_button.setToolTip(FIELD_TOOLTIPS["apply"])
        self.apply_button.clicked.connect(self.apply_selected_assets)
        self.confirm_button = QtWidgets.QPushButton("Confirm Corrections")
        self.confirm_button.setToolTip(FIELD_TOOLTIPS["confirm"])
        self.confirm_button.clicked.connect(self.confirm_corrections)

        asset_box = QtWidgets.QGroupBox("Assets")
        asset_grid = QtWidgets.QGridLayout(asset_box)
        _configure_compact_grid(asset_grid)
        field_row = 0
        asset_tool_grid = self._build_asset_tool_grid()
        if asset_tool_grid is not None:
            asset_grid.addLayout(asset_tool_grid, field_row, 0, 1, 2)
            field_row += 1
        if self.pyfai_calibration_action is not None:
            self.pyfai_calibration_status_label = QtWidgets.QLabel()
            self.pyfai_calibration_status_label.setObjectName(
                "PyFAICalib2StatusLabel"
            )
            self.pyfai_calibration_status_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )
            self.set_pyfai_calibration_status(
                self._pyfai_calibration_status,
                self._pyfai_calibration_status_tooltip,
            )
            asset_grid.addWidget(
                self.pyfai_calibration_status_label,
                field_row,
                0,
                1,
                2,
            )
            field_row += 1
        _add_grid_field(
            asset_grid,
            field_row,
            0,
            "MASK",
            self.mask_combo,
            FIELD_TOOLTIPS["mask"],
        )
        field_row += 1
        _add_grid_field(
            asset_grid,
            field_row,
            0,
            "PONI calibrant",
            self.calibrant_combo,
            FIELD_TOOLTIPS["calibrant"],
        )
        field_row += 1
        asset_grid.addWidget(
            self.apply_button,
            field_row,
            1,
            1,
            1,
            QtCore.Qt.AlignmentFlag.AlignRight,
        )
        asset_grid.setColumnStretch(1, 1)

        pyfai_box = QtWidgets.QGroupBox("Correction Options")
        pyfai_grid = QtWidgets.QGridLayout(pyfai_box)
        _configure_compact_grid(pyfai_grid)
        beam_box = QtWidgets.QGroupBox("Beam and Intensity")
        beam_form = QtWidgets.QFormLayout(beam_box)
        _configure_compact_form(beam_form)
        _add_form_row(
            beam_form,
            "X-ray energy",
            self.energy_kev,
            FIELD_TOOLTIPS["energy"],
        )
        _add_form_row(
            beam_form,
            "Solid angle correction",
            self.solid_angle_check,
            FIELD_TOOLTIPS["solid_angle"],
        )
        _add_form_row(
            beam_form,
            "Polarization correction",
            self.polarization_check,
            FIELD_TOOLTIPS["polarization"],
        )
        _add_form_row(
            beam_form,
            "Polarization factor",
            self.polarization_factor,
            FIELD_TOOLTIPS["polarization_factor"],
        )
        _add_form_row(
            beam_form,
            "Normalization factor",
            self.normalization_factor,
            FIELD_TOOLTIPS["normalization"],
        )

        dummy_box = QtWidgets.QGroupBox("Dummy Pixels")
        dummy_form = QtWidgets.QFormLayout(dummy_box)
        _configure_compact_form(dummy_form)
        _add_form_row(
            dummy_form,
            "Use pyFAI dummy pixel value",
            self.dummy_check,
            FIELD_TOOLTIPS["dummy"],
        )
        _add_form_row(
            dummy_form,
            "pyFAI dummy pixel value",
            self.dummy_value,
            FIELD_TOOLTIPS["dummy_value"],
        )
        _add_form_row(
            dummy_form,
            "Dummy tolerance (delta_dummy)",
            self.delta_dummy,
            FIELD_TOOLTIPS["delta_dummy"],
        )
        pyfai_grid.addWidget(beam_box, 0, 0)
        pyfai_grid.addWidget(dummy_box, 0, 1)
        pyfai_grid.setColumnStretch(0, 2)
        pyfai_grid.setColumnStretch(1, 1)

        orientation_box = QtWidgets.QGroupBox("Detector Orientation")
        orientation_grid = QtWidgets.QGridLayout(orientation_box)
        _configure_compact_grid(orientation_grid)
        _add_grid_field(
            orientation_grid,
            0,
            0,
            "Image rotation",
            self.rotation_combo,
            FIELD_TOOLTIPS["rotation"],
        )
        _add_grid_field(
            orientation_grid,
            1,
            0,
            "pyFAI sample orientation",
            self.sample_orientation_combo,
            FIELD_TOOLTIPS["sample_orientation"],
        )
        _add_grid_field(
            orientation_grid,
            2,
            0,
            "Mirror over active y-axis",
            self.mirror_y_check,
            FIELD_TOOLTIPS["mirror"],
        )
        orientation_grid.setColumnStretch(1, 1)

        low_q_box = QtWidgets.QGroupBox("Low-q Identifiers")
        low_q_grid = QtWidgets.QGridLayout(low_q_box)
        _configure_compact_grid(low_q_grid)
        _add_grid_field(
            low_q_grid,
            0,
            0,
            "Reflected beam x",
            self.reflected_x,
            FIELD_TOOLTIPS["reflected_x"],
        )
        _add_grid_field(
            low_q_grid,
            0,
            2,
            "Reflected beam y",
            self.reflected_y,
            FIELD_TOOLTIPS["reflected_y"],
        )
        self.estimate_reflected_button = QtWidgets.QPushButton(
            "Find Reflected Beam Spot"
        )
        self.estimate_reflected_button.setToolTip(
            FIELD_TOOLTIPS["estimate_reflected"]
        )
        self.estimate_reflected_button.clicked.connect(
            self.estimate_reflected_beam_spot
        )
        self.add_identifiers_button = QtWidgets.QPushButton(
            "Add Low-q Identifiers"
        )
        self.add_identifiers_button.setToolTip(
            FIELD_TOOLTIPS["add_identifiers"]
        )
        self.add_identifiers_button.clicked.connect(self.add_low_q_identifiers)
        low_q_grid.addWidget(self.estimate_reflected_button, 1, 0, 1, 2)
        low_q_grid.addWidget(self.add_identifiers_button, 1, 2, 1, 2)
        low_q_grid.addWidget(self.low_q_status, 2, 0, 1, 4)
        low_q_grid.setColumnStretch(1, 1)
        low_q_grid.setColumnStretch(3, 1)

        film_box = QtWidgets.QGroupBox("Film Optics")
        film_grid = QtWidgets.QGridLayout(film_box)
        _configure_compact_grid(film_grid)
        _add_grid_field(
            film_grid,
            0,
            0,
            "Film stoichiometry",
            self.sample_stoichiometry,
            FIELD_TOOLTIPS["stoichiometry"],
            field_column_span=3,
        )
        _add_grid_field(
            film_grid,
            1,
            0,
            "Film density",
            self.sample_density,
            FIELD_TOOLTIPS["density"],
        )
        _add_grid_field(
            film_grid,
            1,
            2,
            "Critical angle",
            self.critical_angle,
            FIELD_TOOLTIPS["critical_angle"],
        )
        _add_grid_field(
            film_grid,
            2,
            0,
            "Refractive-index delta",
            self.refractive_delta,
            FIELD_TOOLTIPS["refractive_delta"],
            field_column_span=3,
        )
        memory_label = QtWidgets.QLabel("Memory")
        memory_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        memory_label.setToolTip(FIELD_TOOLTIPS["film_memory"])
        film_grid.addWidget(memory_label, 3, 0)
        film_grid.addWidget(self.film_memory_combo, 3, 1, 1, 3)
        memory_buttons = QtWidgets.QHBoxLayout()
        memory_buttons.setContentsMargins(0, 0, 0, 0)
        memory_buttons.setSpacing(6)
        memory_buttons.addWidget(self.save_film_memory_button)
        memory_buttons.addWidget(self.load_film_memory_button)
        memory_buttons.addWidget(self.delete_film_memory_button)
        memory_buttons.addWidget(self.clear_film_memory_button)
        film_grid.addLayout(memory_buttons, 4, 0, 1, 4)
        film_grid.addWidget(self.film_memory_status, 5, 0, 1, 4)
        self.estimate_delta_button = QtWidgets.QPushButton(
            "Estimate Delta from Film"
        )
        self.estimate_delta_button.setToolTip(FIELD_TOOLTIPS["estimate_delta"])
        self.estimate_delta_button.clicked.connect(
            self.estimate_delta_from_chemistry
        )
        self.estimate_structure_button = QtWidgets.QPushButton(
            "Load Structure Estimate"
        )
        self.estimate_structure_button.setToolTip(
            FIELD_TOOLTIPS["estimate_structure"]
        )
        self.estimate_structure_button.clicked.connect(
            self.estimate_delta_from_structure
        )
        self.calculate_critical_button = QtWidgets.QPushButton(
            "Calculate Critical Angle"
        )
        self.calculate_critical_button.setToolTip(
            FIELD_TOOLTIPS["calculate_critical"]
        )
        self.calculate_critical_button.clicked.connect(
            self.calculate_critical_angle_from_delta
        )
        film_grid.addWidget(self.estimate_delta_button, 6, 0, 1, 2)
        film_grid.addWidget(self.estimate_structure_button, 6, 2, 1, 2)
        film_grid.addWidget(self.calculate_critical_button, 7, 0, 1, 4)
        film_grid.setColumnStretch(1, 1)
        film_grid.setColumnStretch(3, 1)

        artifact_box = QtWidgets.QGroupBox("Artifacts")
        artifact_layout = QtWidgets.QVBoxLayout(artifact_box)
        _configure_compact_box_layout(artifact_layout)
        artifact_header = QtWidgets.QHBoxLayout()
        artifact_header.setContentsMargins(0, 0, 0, 0)
        artifact_header.setSpacing(6)
        artifact_header.addStretch(1)
        artifact_header.addWidget(self.add_artifact_button)
        artifact_layout.addLayout(artifact_header)
        self.artifact_table.setMinimumHeight(120)
        self.artifact_table.setMaximumHeight(220)
        artifact_layout.addWidget(self.artifact_table)

        content_grid = QtWidgets.QGridLayout()
        content_grid.setContentsMargins(0, 0, 0, 0)
        content_grid.setHorizontalSpacing(10)
        content_grid.setVerticalSpacing(8)
        content_grid.addWidget(asset_box, 0, 0)
        content_grid.addWidget(orientation_box, 0, 1)
        content_grid.addWidget(pyfai_box, 1, 0, 1, 2)
        content_grid.addWidget(low_q_box, 2, 0)
        content_grid.addWidget(film_box, 2, 1)
        content_grid.addWidget(artifact_box, 3, 0, 1, 2)
        content_grid.setColumnStretch(0, 1)
        content_grid.setColumnStretch(1, 1)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.confirm_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addLayout(content_grid)
        layout.addLayout(button_row)
        self._set_correction_tab_order()

    def _build_asset_tool_grid(self) -> QtWidgets.QGridLayout | None:
        if (
            self.load_mask_action is None
            and self.load_calibrant_action is None
            and self.pyfai_calibration_action is None
        ):
            return None
        asset_tool_grid = QtWidgets.QGridLayout()
        asset_tool_grid.setContentsMargins(0, 0, 0, 0)
        asset_tool_grid.setHorizontalSpacing(6)
        asset_tool_grid.setVerticalSpacing(4)
        self.asset_tool_grid = asset_tool_grid
        if self.load_mask_action is not None:
            self.load_mask_button = _labeled_action_button(
                self.load_mask_action,
                FIELD_TOOLTIPS["load_mask"],
            )
            asset_tool_grid.addWidget(self.load_mask_button, 0, 0)
        if self.load_calibrant_action is not None:
            self.load_calibrant_button = _labeled_action_button(
                self.load_calibrant_action,
                FIELD_TOOLTIPS["load_calibrant"],
            )
            asset_tool_grid.addWidget(self.load_calibrant_button, 0, 1)
        if self.pyfai_calibration_action is not None:
            self.pyfai_calibration_button = _labeled_action_button(
                self.pyfai_calibration_action,
                FIELD_TOOLTIPS["pyfai_calibration"],
                width=220,
            )
            asset_tool_grid.addWidget(
                self.pyfai_calibration_button,
                1,
                0,
                1,
                2,
            )
        asset_tool_grid.setColumnStretch(0, 1)
        asset_tool_grid.setColumnStretch(1, 1)
        return asset_tool_grid

    def set_pyfai_calibration_status(
        self,
        text: str,
        tooltip: str = "",
    ) -> None:
        self._pyfai_calibration_status = text
        self._pyfai_calibration_status_tooltip = tooltip
        if self.pyfai_calibration_status_label is None:
            return
        self.pyfai_calibration_status_label.setText(text)
        self.pyfai_calibration_status_label.setToolTip(tooltip)

    def _set_correction_tab_order(self) -> None:
        action_buttons = [
            button
            for button in (
                self.load_mask_button,
                self.load_calibrant_button,
                self.pyfai_calibration_button,
            )
            if button is not None
        ]
        widgets = [
            *action_buttons,
            self.mask_combo,
            self.calibrant_combo,
            self.apply_button,
            self.energy_kev,
            self.solid_angle_check,
            self.polarization_check,
            self.polarization_factor,
            self.normalization_factor,
            self.dummy_check,
            self.dummy_value,
            self.delta_dummy,
            self.rotation_combo,
            self.mirror_y_check,
            self.sample_orientation_combo,
            self.reflected_x,
            self.reflected_y,
            self.estimate_reflected_button,
            self.add_identifiers_button,
            self.sample_stoichiometry,
            self.sample_density,
            self.critical_angle,
            self.refractive_delta,
            self.film_memory_combo,
            self.save_film_memory_button,
            self.load_film_memory_button,
            self.delete_film_memory_button,
            self.clear_film_memory_button,
            self.estimate_delta_button,
            self.estimate_structure_button,
            self.calculate_critical_button,
            self.add_artifact_button,
            self.artifact_table,
            self.confirm_button,
        ]
        for current_widget, next_widget in zip(widgets, widgets[1:]):
            QtWidgets.QWidget.setTabOrder(current_widget, next_widget)

    def _populate_asset_combos(self) -> None:
        _populate_combo(
            self.mask_combo,
            self.project.masks,
            self._selected_asset_id("mask"),
        )
        _populate_combo(
            self.calibrant_combo,
            self.project.calibrants,
            self._selected_asset_id("calibrant"),
        )

    def _populate_pyfai_correction_controls(self) -> None:
        self._sync_energy_from_selected_calibrant()
        self.calibrant_combo.currentIndexChanged.connect(
            self._sync_energy_from_selected_calibrant
        )
        self.polarization_check.toggled.connect(
            self.polarization_factor.setEnabled
        )
        self.polarization_factor.setEnabled(
            self.polarization_check.isChecked()
        )
        self.dummy_check.toggled.connect(self._set_dummy_controls_enabled)
        self._set_dummy_controls_enabled(self.dummy_check.isChecked())

    def _populate_orientation_controls(self) -> None:
        for label, degrees in (
            ("0 deg", 0),
            ("90 deg clockwise", 90),
            ("180 deg", 180),
            ("270 deg clockwise", 270),
        ):
            self.rotation_combo.addItem(label, degrees)
        for orientation in range(1, 9):
            self.sample_orientation_combo.addItem(
                str(orientation), orientation
            )
        self.rotation_combo.currentIndexChanged.connect(
            self._sync_sample_orientation_from_transform
        )
        self.mirror_y_check.toggled.connect(
            self._sync_sample_orientation_from_transform
        )
        self._sync_sample_orientation_from_transform()

    def _populate_film_memory_controls(self) -> None:
        self.project.film_material_memory = _merged_film_memory_items(
            _read_film_material_memory(self.settings),
            self.project.film_material_memory,
        )
        self._persist_film_memory()
        self._populate_film_memory_combo()
        self.film_memory_combo.currentIndexChanged.connect(
            self._refresh_film_memory_actions
        )
        self.save_film_memory_button.clicked.connect(
            self.save_film_material_memory
        )
        self.load_film_memory_button.clicked.connect(
            self.load_selected_film_material_memory
        )
        self.delete_film_memory_button.clicked.connect(
            self.delete_selected_film_material_memory
        )
        self.clear_film_memory_button.clicked.connect(
            self.clear_film_material_memory
        )
        self.sample_stoichiometry.textEdited.connect(
            self._handle_film_inputs_changed
        )
        self.sample_density.valueChanged.connect(
            self._handle_film_inputs_changed
        )

    def _populate_film_memory_combo(
        self,
        selected_memory_id: str | None = None,
    ) -> None:
        current_id = selected_memory_id
        if current_id is None:
            current_id = str(self.film_memory_combo.currentData() or "")
        self.film_memory_combo.blockSignals(True)
        self.film_memory_combo.clear()
        if not self.project.film_material_memory:
            self.film_memory_combo.addItem("No saved film memories", None)
        else:
            for item in self.project.film_material_memory:
                self.film_memory_combo.addItem(
                    _film_memory_label(item),
                    str(item["memory_id"]),
                )
        if current_id:
            index = self.film_memory_combo.findData(current_id)
            if index >= 0:
                self.film_memory_combo.setCurrentIndex(index)
        self.film_memory_combo.blockSignals(False)
        self._refresh_film_memory_actions()

    def _refresh_film_memory_actions(self, *_args: Any) -> None:
        has_selection = self._selected_film_memory_item() is not None
        has_memory = bool(self.project.film_material_memory)
        self.load_film_memory_button.setEnabled(has_selection)
        self.delete_film_memory_button.setEnabled(has_selection)
        self.clear_film_memory_button.setEnabled(has_memory)

    def _selected_film_memory_item(self) -> dict[str, Any] | None:
        memory_id = self.film_memory_combo.currentData()
        if not memory_id:
            return None
        return _find_film_memory_item(
            self.project.film_material_memory,
            str(memory_id),
        )

    def _handle_film_inputs_changed(self, *_args: Any) -> None:
        if self._loading_film_memory:
            return
        if self.loaded_film_memory_id is None:
            self._set_film_memory_status("Manual film values.")
            return
        item = _find_film_memory_item(
            self.project.film_material_memory,
            self.loaded_film_memory_id,
        )
        if item is None:
            self.loaded_film_memory_id = None
            self._set_film_memory_status("Manual film values.")
            return
        if _current_film_inputs_match(
            item,
            self.sample_stoichiometry.text(),
            self.sample_density.value(),
        ):
            self._set_film_memory_loaded_status(item)
            return
        self._set_film_memory_status(
            f"Edited values from memory: {_film_memory_label(item)}."
        )

    def _set_film_memory_loaded_status(self, item: dict[str, Any]) -> None:
        self._set_film_memory_status(
            f"Loaded from memory: {_film_memory_label(item)}."
        )

    def _set_film_memory_status(self, message: str) -> None:
        self.film_memory_status.setText(message)

    def _persist_film_memory(self) -> None:
        _write_film_material_memory(
            self.settings,
            self.project.film_material_memory,
        )

    def _sync_sample_orientation_from_transform(self, *_args: Any) -> None:
        rotation = int(self.rotation_combo.currentData() or 0)
        orientation = sample_orientation_for_image_transform(
            rotation,
            mirrored_y=self.mirror_y_check.isChecked(),
        )
        _set_combo_value(self.sample_orientation_combo, orientation)

    def _sync_energy_from_selected_calibrant(self, *_args: Any) -> None:
        energy = self._selected_calibrant_energy_kev()
        if energy is not None:
            self.energy_kev.setValue(energy)

    def _selected_calibrant_energy_kev(self) -> float | None:
        calibrant_id = self.calibrant_combo.currentData()
        if not calibrant_id:
            return None
        try:
            asset = self.project.get_correction_asset(
                "calibrant",
                str(calibrant_id),
            )
        except KeyError:
            return None
        if asset.usable_path is None:
            return None
        try:
            from ewald.processing.qspace import (
                load_azimuthal_integrator,
                xray_energy_kev_from_wavelength_m,
            )

            integrator = load_azimuthal_integrator(asset.usable_path)
            wavelength = getattr(integrator, "wavelength", None)
            if wavelength is None:
                return None
            return xray_energy_kev_from_wavelength_m(float(wavelength))
        except Exception:
            return None

    def _set_dummy_controls_enabled(self, enabled: bool) -> None:
        self.dummy_value.setEnabled(enabled)
        self.delta_dummy.setEnabled(enabled)

    def _selected_asset_id(self, kind: str) -> str | None:
        direct = self.project.assigned_assets(kind, self.data_id)
        if direct:
            return direct[0].asset_id
        inherited = self.project.assigned_assets(kind, self.group_id)
        if inherited:
            return inherited[0].asset_id
        return None

    def _restore_existing_state(self) -> None:
        state = self.project.image_corrections.get(self.data_id)
        if state is None:
            return
        _set_combo_value(self.mask_combo, state.mask_asset_id)
        _set_combo_value(self.calibrant_combo, state.calibrant_asset_id)
        if state.xray_energy_kev is not None:
            self.energy_kev.setValue(state.xray_energy_kev)
        self.restore_orientation_from_state()
        self.solid_angle_check.setChecked(state.correct_solid_angle)
        self.polarization_check.setChecked(
            state.polarization_factor is not None
        )
        self.polarization_factor.setValue(state.polarization_factor or 0.95)
        self.polarization_factor.setEnabled(
            self.polarization_check.isChecked()
        )
        self.normalization_factor.setValue(state.normalization_factor)
        self.dummy_check.setChecked(state.dummy is not None)
        self.dummy_value.setValue(state.dummy or 0.0)
        self.delta_dummy.setValue(state.delta_dummy or 0.0)
        self._set_dummy_controls_enabled(self.dummy_check.isChecked())
        self.reflected_x.setValue(state.reflected_beam_x_px or 0.0)
        self.reflected_y.setValue(state.reflected_beam_y_px or 0.0)
        self.critical_angle.setValue(state.critical_angle_deg or 0.0)
        stoichiometry = state.sample_stoichiometry or state.metadata.get(
            "sample_stoichiometry"
        )
        if stoichiometry:
            self.sample_stoichiometry.setText(str(stoichiometry))
        density = state.sample_density_g_cm3 or state.metadata.get(
            "sample_density_g_cm3"
        )
        if density is not None:
            self.sample_density.setValue(float(density))
        delta = state.refractive_index_delta or state.metadata.get(
            "refractive_index_delta"
        )
        if delta is not None:
            self.refractive_delta.setValue(float(delta))
        for region in state.artifact_regions:
            row = self.artifact_table.rowCount()
            self.artifact_table.insertRow(row)
            values = [
                region.get("label", f"artifact-{row + 1}"),
                region.get("x", 0.0),
                region.get("y", 0.0),
                region.get("width", 0.0),
                region.get("height", 0.0),
            ]
            for column, value in enumerate(values):
                self.artifact_table.setItem(
                    row,
                    column,
                    QtWidgets.QTableWidgetItem(str(value)),
                )

    def restore_orientation_from_state(self) -> None:
        state = self.project.image_corrections.get(self.data_id)
        if state is None:
            return
        self.rotation_combo.blockSignals(True)
        self.mirror_y_check.blockSignals(True)
        _set_combo_value(self.rotation_combo, state.image_rotation_deg)
        self.mirror_y_check.setChecked(state.image_mirrored_y)
        self.rotation_combo.blockSignals(False)
        self.mirror_y_check.blockSignals(False)
        _set_combo_value(
            self.sample_orientation_combo,
            state.pyfai_sample_orientation,
        )

    def _editable_state(self) -> ImageCorrectionState:
        state = self.project.image_corrections.get(self.data_id)
        if state is None:
            state = ImageCorrectionState(target_id=self.data_id)
            self.project.image_corrections[self.data_id] = state
        return state

    def _store_chemistry_on_state(
        self,
        state: ImageCorrectionState,
        estimate: dict[str, Any] | None = None,
    ) -> None:
        stoichiometry = self.sample_stoichiometry.text().strip()
        state.sample_stoichiometry = stoichiometry or None
        state.sample_density_g_cm3 = self.sample_density.value()
        state.refractive_index_delta = (
            self.refractive_delta.value()
            if self.refractive_delta.value() > 0
            else None
        )
        state.metadata["sample_stoichiometry"] = state.sample_stoichiometry
        state.metadata["sample_density_g_cm3"] = state.sample_density_g_cm3
        state.metadata["refractive_index_delta"] = state.refractive_index_delta
        if estimate is not None:
            state.metadata["refractive_index_estimate"] = estimate

    def _apply_structure_optics_estimate(
        self,
        estimate: StructureOpticsEstimate,
    ) -> None:
        self.energy_kev.setValue(estimate.xray_energy_kev)
        self.sample_stoichiometry.setText(estimate.normalized_formula)
        self.sample_density.setValue(estimate.density_g_cm3)
        self.refractive_delta.setValue(estimate.delta)
        self.critical_angle.setValue(estimate.critical_angle_deg)
        state = self._editable_state()
        state.xray_energy_kev = estimate.xray_energy_kev
        state.critical_angle_deg = estimate.critical_angle_deg
        estimate_dict = estimate.as_dict()
        self._store_chemistry_on_state(state, estimate_dict)
        state.metadata["structure_optics_estimate"] = estimate_dict
        self._refresh_low_q_status(
            "Applied film optics estimate from reference structure."
        )

    def _incident_angle_deg(self) -> float:
        data_file = self.project.data_file_by_id(self.data_id)
        if data_file is None:
            return 0.0
        return float(data_file.metadata.get("incidence_angle_deg", 0.0))

    def _refresh_low_q_status(self, prefix: str | None = None) -> None:
        state = self.project.image_corrections.get(self.data_id)
        count = 0
        optics = ""
        if state is not None:
            count = len(state.metadata.get("low_q_features", []))
            if state.refractive_index_delta:
                optics = (
                    f" Film delta={state.refractive_index_delta:.3g}, "
                    f"critical angle={state.critical_angle_deg or 0.0:.4g} deg."
                )
        message = f"{count} low-q identifier(s) stored."
        if optics:
            message = f"{message}{optics}"
        if prefix:
            message = f"{prefix} {message}"
        self.low_q_status.setText(message)

    def _apply_field_tooltips(self) -> None:
        widgets = {
            self.mask_combo: FIELD_TOOLTIPS["mask"],
            self.calibrant_combo: FIELD_TOOLTIPS["calibrant"],
            self.energy_kev: FIELD_TOOLTIPS["energy"],
            self.solid_angle_check: FIELD_TOOLTIPS["solid_angle"],
            self.polarization_check: FIELD_TOOLTIPS["polarization"],
            self.polarization_factor: FIELD_TOOLTIPS["polarization_factor"],
            self.normalization_factor: FIELD_TOOLTIPS["normalization"],
            self.dummy_check: FIELD_TOOLTIPS["dummy"],
            self.dummy_value: FIELD_TOOLTIPS["dummy_value"],
            self.delta_dummy: FIELD_TOOLTIPS["delta_dummy"],
            self.rotation_combo: FIELD_TOOLTIPS["rotation"],
            self.mirror_y_check: FIELD_TOOLTIPS["mirror"],
            self.sample_orientation_combo: FIELD_TOOLTIPS[
                "sample_orientation"
            ],
            self.reflected_x: FIELD_TOOLTIPS["reflected_x"],
            self.reflected_y: FIELD_TOOLTIPS["reflected_y"],
            self.critical_angle: FIELD_TOOLTIPS["critical_angle"],
            self.sample_stoichiometry: FIELD_TOOLTIPS["stoichiometry"],
            self.sample_density: FIELD_TOOLTIPS["density"],
            self.film_memory_combo: FIELD_TOOLTIPS["film_memory"],
            self.save_film_memory_button: FIELD_TOOLTIPS["save_film_memory"],
            self.load_film_memory_button: FIELD_TOOLTIPS["load_film_memory"],
            self.delete_film_memory_button: FIELD_TOOLTIPS[
                "delete_film_memory"
            ],
            self.clear_film_memory_button: FIELD_TOOLTIPS["clear_film_memory"],
            self.film_memory_status: FIELD_TOOLTIPS["film_memory"],
            self.refractive_delta: FIELD_TOOLTIPS["refractive_delta"],
            self.low_q_status: FIELD_TOOLTIPS["low_q_status"],
        }
        for widget, tooltip in widgets.items():
            widget.setToolTip(tooltip)


class StructureOpticsReviewDialog(QtWidgets.QDialog):
    """Review a parsed structure before applying film-optics values."""

    def __init__(
        self,
        estimate: StructureOpticsEstimate,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review Structure Optics Estimate")
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Review the parsed structure composition and density before "
            "applying these values to the film optics inputs."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        form.addRow("File", _selectable_label(estimate.structure_path))
        form.addRow("Format", _selectable_label(estimate.file_format))
        form.addRow("Formula", _selectable_label(estimate.formula))
        form.addRow(
            "Reduced formula",
            _selectable_label(estimate.normalized_formula),
        )
        form.addRow(
            "Unit cell volume",
            _selectable_label(f"{estimate.unit_cell_volume_a3:.4g} A^3"),
        )
        form.addRow(
            "Density",
            _selectable_label(f"{estimate.density_g_cm3:.4g} g/cm3"),
        )
        form.addRow(
            "X-ray energy",
            _selectable_label(f"{estimate.xray_energy_kev:.4g} keV"),
        )
        form.addRow(
            "Refractive-index delta",
            _selectable_label(f"{estimate.delta:.6g}"),
        )
        form.addRow(
            "Real refractive index",
            _selectable_label(f"{estimate.refractive_index_real:.10g}"),
        )
        form.addRow(
            "Critical angle",
            _selectable_label(f"{estimate.critical_angle_deg:.6g} deg"),
        )
        layout.addLayout(form)

        composition_table = QtWidgets.QTableWidget(
            len(estimate.composition),
            2,
        )
        composition_table.setHorizontalHeaderLabels(["Element", "Atoms"])
        composition_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        for row, (element, amount) in enumerate(estimate.composition.items()):
            composition_table.setItem(
                row,
                0,
                QtWidgets.QTableWidgetItem(element),
            )
            composition_table.setItem(
                row,
                1,
                QtWidgets.QTableWidgetItem(f"{amount:g}"),
            )
        composition_table.horizontalHeader().setStretchLastSection(True)
        composition_table.setMinimumHeight(140)
        layout.addWidget(composition_table)

        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton(
            "Apply Values",
            QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole,
        )
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def _populate_combo(
    combo: QtWidgets.QComboBox,
    assets: list[CorrectionAssetRef],
    selected_id: str | None,
) -> None:
    combo.clear()
    combo.addItem("Select...", None)
    for asset in assets:
        combo.addItem(asset.name, asset.asset_id)
    _set_combo_value(combo, selected_id)


def _add_form_row(
    form: QtWidgets.QFormLayout,
    label: str,
    field: QtWidgets.QWidget,
    tooltip: str,
) -> None:
    label_widget = QtWidgets.QLabel(label)
    label_widget.setToolTip(tooltip)
    field.setToolTip(tooltip)
    form.addRow(label_widget, field)


def _add_grid_field(
    grid: QtWidgets.QGridLayout,
    row: int,
    column: int,
    label: str,
    field: QtWidgets.QWidget,
    tooltip: str,
    *,
    field_column_span: int = 1,
) -> None:
    label_widget = QtWidgets.QLabel(label)
    label_widget.setAlignment(
        QtCore.Qt.AlignmentFlag.AlignRight
        | QtCore.Qt.AlignmentFlag.AlignVCenter
    )
    label_widget.setToolTip(tooltip)
    field.setToolTip(tooltip)
    grid.addWidget(label_widget, row, column)
    grid.addWidget(field, row, column + 1, 1, field_column_span)


def _labeled_action_button(
    action: QtGui.QAction,
    tooltip: str,
    *,
    width: int = 138,
) -> QtWidgets.QToolButton:
    button = QtWidgets.QToolButton()
    button.setDefaultAction(action)
    button.setToolButtonStyle(
        QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
    )
    button.setFixedSize(width, 34)
    button.setIconSize(QtCore.QSize(20, 20))
    button.setAutoRaise(True)
    button.setToolTip(tooltip)
    return button


def _read_film_material_memory(
    settings: QtCore.QSettings,
) -> list[dict[str, Any]]:
    raw = settings.value(FILM_MATERIAL_MEMORY_SETTING, "[]")
    if raw in (None, ""):
        return []
    try:
        data = json.loads(str(raw)) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return _merged_film_memory_items(data)


def _write_film_material_memory(
    settings: QtCore.QSettings,
    items: list[dict[str, Any]],
) -> None:
    settings.setValue(
        FILM_MATERIAL_MEMORY_SETTING,
        json.dumps(_merged_film_memory_items(items), sort_keys=True),
    )
    settings.sync()


def _merged_film_memory_items(
    *sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_signatures: set[tuple[str, float]] = set()
    for source in sources:
        for item in source or []:
            normalized = _normalized_film_memory_item(item)
            if normalized is None:
                continue
            signature = _film_memory_signature(normalized)
            if signature in used_signatures:
                continue
            normalized["memory_id"] = _unique_film_memory_id(
                str(normalized["memory_id"]),
                used_ids,
            )
            used_ids.add(str(normalized["memory_id"]))
            used_signatures.add(signature)
            items.append(normalized)
    return items


def _normalized_film_memory_item(
    item: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    try:
        stoichiometry = _validated_stoichiometry(item.get("stoichiometry", ""))
        density = _positive_density(item.get("density_g_cm3"))
    except ValueError:
        return None
    memory_id = str(
        item.get("memory_id") or _film_memory_id_seed(stoichiometry, density)
    )
    return {
        "memory_id": memory_id,
        "label": str(
            item.get("label")
            or _film_memory_label_text(
                stoichiometry,
                density,
            )
        ),
        "stoichiometry": stoichiometry,
        "density_g_cm3": density,
        "refractive_index_delta": _optional_float(
            item.get("refractive_index_delta")
        ),
        "critical_angle_deg": _optional_float(item.get("critical_angle_deg")),
        "source": str(item.get("source") or "manual"),
        "metadata": (
            dict(item.get("metadata"))
            if isinstance(item.get("metadata"), dict)
            else {}
        ),
    }


def _validated_stoichiometry(value: object) -> str:
    formula = " ".join(str(value).strip().split())
    compact = formula.replace(" ", "")
    if not compact:
        raise ValueError("Enter a film stoichiometry before saving memory.")
    if not any(character.isalpha() for character in compact):
        raise ValueError(
            "Film stoichiometry must include at least one element."
        )
    if _STOICHIOMETRY_TEXT_PATTERN.fullmatch(formula) is None:
        raise ValueError(
            "Film stoichiometry can include letters, numbers, spaces, "
            "parentheses, brackets, dots, underscores, hyphens, and plus signs."
        )
    if not _formula_delimiters_are_balanced(compact):
        raise ValueError("Film stoichiometry has unbalanced parentheses.")
    return formula


def _positive_density(value: object) -> float:
    try:
        density = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Film density must be numeric.") from exc
    if density <= 0:
        raise ValueError("Film density must be positive.")
    return density


def _formula_delimiters_are_balanced(value: str) -> bool:
    stack: list[str] = []
    pairs = {")": "(", "]": "["}
    for character in value:
        if character in "([{":
            stack.append(character)
        elif character in pairs:
            if not stack or stack.pop() != pairs[character]:
                return False
    return not stack


def _film_memory_id_seed(stoichiometry: str, density: float) -> str:
    seed = f"film_{stoichiometry}_{density:g}"
    memory_id = re.sub(r"[^a-zA-Z0-9]+", "_", seed).strip("_").lower()
    return memory_id or "film_material"


def _unique_film_memory_id(seed: str, used: set[str]) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "_", seed).strip("_").lower()
    candidate = base or "film_material"
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}" if base else f"film_material_{index}"
        index += 1
    return candidate


def _film_memory_label(item: dict[str, Any]) -> str:
    return str(
        item.get("label")
        or _film_memory_label_text(
            str(item.get("stoichiometry", "")),
            float(item.get("density_g_cm3", 0.0)),
        )
    )


def _film_memory_label_text(stoichiometry: str, density: float) -> str:
    return f"{stoichiometry} ({density:g} g/cm3)"


def _find_film_memory_item(
    items: list[dict[str, Any]],
    memory_id: str,
) -> dict[str, Any] | None:
    for item in items:
        if str(item.get("memory_id")) == memory_id:
            return item
    return None


def _find_matching_film_memory(
    items: list[dict[str, Any]],
    stoichiometry: str,
    density: float,
) -> dict[str, Any] | None:
    signature = (stoichiometry.casefold(), round(float(density), 8))
    for item in items:
        if _film_memory_signature(item) == signature:
            return item
    return None


def _film_memory_signature(item: dict[str, Any]) -> tuple[str, float]:
    return (
        str(item.get("stoichiometry", "")).casefold(),
        round(float(item.get("density_g_cm3", 0.0)), 8),
    )


def _current_film_inputs_match(
    item: dict[str, Any],
    stoichiometry: str,
    density: float,
) -> bool:
    try:
        formula = _validated_stoichiometry(stoichiometry)
    except ValueError:
        return False
    return _film_memory_signature(item) == (
        formula.casefold(),
        round(float(density), 8),
    )


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _configure_compact_form(form: QtWidgets.QFormLayout) -> None:
    form.setContentsMargins(8, 8, 8, 8)
    form.setHorizontalSpacing(8)
    form.setVerticalSpacing(4)
    form.setLabelAlignment(
        QtCore.Qt.AlignmentFlag.AlignRight
        | QtCore.Qt.AlignmentFlag.AlignVCenter
    )
    form.setFieldGrowthPolicy(
        QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
    )


def _configure_compact_grid(grid: QtWidgets.QGridLayout) -> None:
    grid.setContentsMargins(8, 8, 8, 8)
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(6)


def _configure_compact_box_layout(layout: QtWidgets.QBoxLayout) -> None:
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)


def _set_combo_value(
    combo: QtWidgets.QComboBox,
    value: str | int | None,
) -> None:
    if value is None:
        return
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def _selectable_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setTextInteractionFlags(
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )
    return label


def _energy_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(0.1, 200.0)
    spinbox.setDecimals(4)
    spinbox.setSingleStep(0.1)
    spinbox.setSuffix(" keV")
    spinbox.setValue(12.7)
    return spinbox


def _polarization_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(-1.0, 1.0)
    spinbox.setDecimals(4)
    spinbox.setSingleStep(0.01)
    spinbox.setValue(0.95)
    return spinbox


def _normalization_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(1.0e-12, 1.0e12)
    spinbox.setDecimals(6)
    spinbox.setSingleStep(0.1)
    spinbox.setValue(1.0)
    return spinbox


def _correction_value_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(-1.0e12, 1.0e12)
    spinbox.setDecimals(6)
    spinbox.setSingleStep(1.0)
    return spinbox


def _coordinate_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(0.0, 1_000_000.0)
    spinbox.setDecimals(3)
    spinbox.setSuffix(" px")
    return spinbox


def _angle_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(0.0, 90.0)
    spinbox.setDecimals(4)
    spinbox.setSuffix(" deg")
    return spinbox


def _density_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(0.0001, 30.0)
    spinbox.setDecimals(4)
    spinbox.setSingleStep(0.1)
    spinbox.setSuffix(" g/cm3")
    spinbox.setValue(1.0)
    return spinbox


def _refractive_delta_spinbox() -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(0.0, 1.0e-2)
    spinbox.setDecimals(8)
    spinbox.setSingleStep(1.0e-7)
    return spinbox


def _table_text(
    table: QtWidgets.QTableWidget,
    row: int,
    column: int,
) -> str:
    item = table.item(row, column)
    if item is None:
        return ""
    return item.text().strip()


def _table_float(
    table: QtWidgets.QTableWidget,
    row: int,
    column: int,
) -> float:
    value = _table_text(table, row, column)
    try:
        return float(value)
    except ValueError:
        return 0.0
