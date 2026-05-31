"""Structure Analysis tab for fitted peaks and candidate lattices."""

from __future__ import annotations

import math
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from ewald.analysis.structure import (
    DEFAULT_FAMILY_DETECTION_KINDS,
    DEFAULT_PHASE_TAG,
    DEFAULT_PHASE_TAGS,
    FAMILY_KIND_QXY_MULTIPLES,
    FAMILY_KIND_QZ_MULTIPLES,
    FAMILY_KIND_SIMILAR_QXY,
    FAMILY_KIND_SIMILAR_QZ,
    PHASE_FORBIDDEN,
    PHASE_REJECTED,
    PHASE_UNASSIGNED,
    REFERENCE_MOLECULES,
    SIMPLE_CRYSTAL_SYSTEM_ORDER,
    WYCKOFF_COMBINATION_DISPLAY_LIMIT,
    WYCKOFF_CRYSTAL_SYSTEMS,
    CandidateSearchConfig,
    LatticeCandidate,
    StructurePeak,
    build_structure_peaks,
    format_hkl,
    generate_ranked_cif_records,
    group_peak_families,
    guess_lattice_candidates,
    refine_lattice_candidate,
    suggest_non_main_phase_peaks,
    wyckoff_combination_count,
    wyckoff_site_combinations,
    wyckoff_space_group_option,
    wyckoff_space_group_options,
)
from ewald.crystallography.overlay import (
    CRYSTAL_SYSTEMS,
    CrystalOverlayCalculator,
    CrystalOverlayParameters,
    normalize_quaternion,
)
from ewald.data.models import ProjectState
from ewald.ui.data_viewer import (
    IMAGE_COLORMAPS,
    ImageDisplayStyle,
    ImagePlotToolbar,
    _ImageAspectPlotFrame,
    _level_spinbox,
    _quantile_spinbox,
    apply_image_display_style,
)
from ewald.ui.notation import (
    QSPACE_UNITS_HTML,
    QXY_HTML,
    QZ_HTML,
    data_image_rect,
    enable_rich_text_items,
    qt_tooltip,
    rich_label,
    set_data_aspect_locked,
    set_data_image_plot_range,
    set_qspace_axis_labels,
    set_rich_text_table_headers,
)

try:  # pragma: no cover - exercised by Qt tests when installed.
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None

PEAK_COLUMNS = [
    "Peak ID",
    f"{QXY_HTML} center",
    f"{QZ_HTML} center",
    "Source",
    "Phase tag",
    "(hkl)",
    "Include",
    "Fit quality",
    "Notes/status",
]
COL_PEAK_ID = 0
COL_QXY = 1
COL_QZ = 2
COL_SOURCE = 3
COL_PHASE = 4
COL_HKL = 5
COL_INCLUDE = 6
COL_FIT_QUALITY = 7
COL_NOTES = 8

CANDIDATE_COLUMNS = [
    "Rank",
    "Crystal system",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "Score",
    "Matched",
    "Outliers",
    "Method",
]


class _CandidateGuessWorker(QtCore.QObject):
    """Run candidate guessing away from the GUI event loop."""

    finished = QtCore.Signal(object, float)
    failed = QtCore.Signal(str, float)

    def __init__(
        self,
        peaks: list[StructurePeak],
        config: CandidateSearchConfig,
    ) -> None:
        super().__init__()
        self.peaks = list(peaks)
        self.config = config

    @QtCore.Slot()
    def run(self) -> None:
        started_at = time.perf_counter()
        try:
            candidates = guess_lattice_candidates(self.peaks, self.config)
        except Exception as exc:
            self.failed.emit(str(exc), time.perf_counter() - started_at)
            return
        self.finished.emit(candidates, time.perf_counter() - started_at)


FAMILY_COLUMNS = [
    "Family",
    "Flag",
    "Confidence",
    "Type",
    "Phase",
    "Reference",
    "Peaks",
    "Reason",
    "Notes",
]
FAMILY_COL_ID = 0
FAMILY_COL_FLAG = 1
FAMILY_COL_CONFIDENCE = 2
FAMILY_COL_TYPE = 3
FAMILY_COL_PHASE = 4
FAMILY_COL_REFERENCE = 5
FAMILY_COL_PEAKS = 6
FAMILY_COL_REASON = 7
FAMILY_COL_NOTES = 8
CIF_COLUMNS = ["Rank", "Candidate", "Score", "Composition", "Status"]
STRUCTURE_OUTPUT_FULL = "full"
STRUCTURE_OUTPUT_SCAFFOLD_ONLY = "scaffold_only"
STRUCTURE_OUTPUT_FULL_PLUS_SCAFFOLD = "full_plus_scaffold"
WYCKOFF_SITE_COLUMNS = ["Site", "Multiplicity", "Free params", "Space group"]
WYCKOFF_COMBINATION_COLUMNS = [
    "Combination",
    "Total mult.",
    "Free params",
    "Sites",
]
ATOM_SPEC_COLUMNS = ["Element", "Stoich.", "Shared site", "Occupancy"]
ATOM_COL_ELEMENT = 0
ATOM_COL_STOICHIOMETRY = 1
ATOM_COL_SHARED_SITE = 2
ATOM_COL_OCCUPANCY = 3
STRUCTURE_PEAK_BRUSH = "#22c55e"
STRUCTURE_ACTIVE_PEAK_BRUSH = "#2f80ed"
FAMILY_FLAG_APPROPRIATE = "appropriate"
FAMILY_FLAG_INAPPROPRIATE = "inappropriate"
FAMILY_FLAG_CYCLE = ("", FAMILY_FLAG_APPROPRIATE, FAMILY_FLAG_INAPPROPRIATE)
FAMILY_SOURCE_MANUAL = "manual"
FAMILY_KIND_CONTROL_LABELS = {
    FAMILY_KIND_SIMILAR_QXY: "Similar qxy",
    FAMILY_KIND_SIMILAR_QZ: "Similar qz",
    FAMILY_KIND_QXY_MULTIPLES: "qxy multiples",
    FAMILY_KIND_QZ_MULTIPLES: "qz multiples",
}


class _FamilyTableWidget(QtWidgets.QTableWidget):
    """Family table with row drag reporting for persistent manual order."""

    familyDragStarted = QtCore.Signal()
    familyRowsReordered = QtCore.Signal(list)

    def startDrag(
        self,
        supported_actions: QtCore.Qt.DropActions,
    ) -> None:
        before = self._family_ids_in_row_order()
        self.familyDragStarted.emit()
        super().startDrag(supported_actions)
        after = self._family_ids_in_row_order()
        if after and after != before:
            self.familyRowsReordered.emit(after)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        before = self._family_ids_in_row_order()
        super().dropEvent(event)
        after = self._family_ids_in_row_order()
        if after and after != before:
            self.familyRowsReordered.emit(after)

    def supportedDropActions(self) -> QtCore.Qt.DropActions:
        return QtCore.Qt.DropAction.MoveAction

    def _family_ids_in_row_order(self) -> list[str]:
        family_ids: list[str] = []
        for row in range(self.rowCount()):
            item = self.item(row, FAMILY_COL_ID)
            if item is None:
                continue
            family_id = str(
                item.data(QtCore.Qt.ItemDataRole.UserRole + 1) or item.text()
            )
            if family_id:
                family_ids.append(family_id)
        return family_ids


@dataclass(frozen=True)
class _CifAtom:
    label: str
    symbol: str
    fract_x: float
    fract_y: float
    fract_z: float
    occupancy: float = 1.0


@dataclass(frozen=True)
class _ParsedCif:
    cif_id: str
    space_group: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    atoms: tuple[_CifAtom, ...]


class _SquarePreviewContainer(QtWidgets.QWidget):
    """Keep a child preview square inside flexible splitter space."""

    def __init__(self, child: QtWidgets.QWidget) -> None:
        super().__init__()
        self.child = child
        child.setParent(self)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(260, 260)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(220, int(width))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        side = min(self.width(), self.height())
        x_offset = (self.width() - side) // 2
        y_offset = (self.height() - side) // 2
        self.child.setGeometry(x_offset, y_offset, side, side)
        super().resizeEvent(event)


class _GeneratedCifPreview(QtWidgets.QWidget):
    """Compact atom/unit-cell visualizer for generated draft CIF
    records."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.cif_id: str | None = None
        self.atom_count = 0
        self.species_text = ""
        self._parsed: _ParsedCif | None = None
        self._edge_items: list[Any] = []
        self._atom_item: Any | None = None
        self._build_widgets()
        self.clear()

    def set_record(self, record: dict[str, Any] | None) -> None:
        if not record:
            self.clear()
            return
        parsed = _parse_generated_cif(
            str(record.get("cif_text", "")),
            cif_id=str(
                record.get("cif_id") or record.get("candidate_id") or ""
            ),
        )
        if parsed is None:
            self.clear()
            self.info_label.setText("Selected CIF could not be parsed.")
            return
        self._parsed = parsed
        self.cif_id = parsed.cif_id
        self.atom_count = len(parsed.atoms)
        self.species_text = ", ".join(
            sorted({_cif_element_symbol(atom.symbol) for atom in parsed.atoms})
        )
        self.info_label.setText(
            "\n".join(
                [
                    f"Preview: {parsed.cif_id}",
                    (
                        f"Atoms: {self.atom_count} | "
                        f"Species: {self.species_text or 'none'}"
                    ),
                    _cif_lattice_summary(parsed),
                ]
            )
        )
        self._draw_cif(parsed)

    def clear(self) -> None:
        self.cif_id = None
        self.atom_count = 0
        self.species_text = ""
        self._parsed = None
        self.info_label.setText("Generate or select a CIF to preview it.")
        self._clear_plot()

    def _build_widgets(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        header.addWidget(self.info_label, stretch=1)
        self.projection_combo = QtWidgets.QComboBox()
        self.projection_combo.addItem("Perspective", "perspective")
        self.projection_combo.addItem("a-b", "ab")
        self.projection_combo.addItem("a-c", "ac")
        self.projection_combo.addItem("b-c", "bc")
        self.projection_combo.currentIndexChanged.connect(self._redraw)
        header.addWidget(self.projection_combo)
        layout.addLayout(header)

        if pg is None:
            self.plot_widget = QtWidgets.QLabel("CIF visualizer")
            self.plot_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.plot_widget.setMinimumSize(220, 220)
            self.plot_container = _SquarePreviewContainer(self.plot_widget)
            layout.addWidget(self.plot_container, stretch=1)
            return

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumSize(220, 220)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.18)
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()
        self.plot_container = _SquarePreviewContainer(self.plot_widget)
        layout.addWidget(self.plot_container, stretch=1)

    def _redraw(self) -> None:
        if self._parsed is not None:
            self._draw_cif(self._parsed)

    def _draw_cif(self, parsed: _ParsedCif) -> None:
        if pg is None or not hasattr(self.plot_widget, "addItem"):
            return
        self._clear_plot()
        mode = str(self.projection_combo.currentData() or "perspective")
        lattice = _cif_lattice_matrix(parsed)
        corners = _cif_unit_cell_corners(lattice)
        projected_corners = _cif_project_points(corners, mode)
        for start, end in _cif_unit_cell_edges():
            item = pg.PlotDataItem(
                [projected_corners[start, 0], projected_corners[end, 0]],
                [projected_corners[start, 1], projected_corners[end, 1]],
                pen=pg.mkPen("#64748b", width=1.2),
            )
            self.plot_widget.addItem(item)
            self._edge_items.append(item)

        if not parsed.atoms:
            self.plot_widget.enableAutoRange()
            return
        frac_coords = np.asarray(
            [
                (atom.fract_x, atom.fract_y, atom.fract_z)
                for atom in parsed.atoms
            ],
            dtype=float,
        )
        cartesian = frac_coords @ lattice
        projected_atoms = _cif_project_points(cartesian, mode)
        order = np.argsort(projected_atoms[:, 1])
        spots = []
        for index in order:
            atom = parsed.atoms[int(index)]
            spots.append(
                {
                    "pos": (
                        float(projected_atoms[index, 0]),
                        float(projected_atoms[index, 1]),
                    ),
                    "data": atom,
                    "size": 11 + 4 * _cif_relative_atom_radius(atom.symbol),
                    "brush": pg.mkBrush(_cif_atom_color(atom.symbol)),
                    "pen": pg.mkPen("#111827", width=0.6),
                }
            )
        self._atom_item = pg.ScatterPlotItem(
            hoverable=True,
            tip=_cif_atom_tip,
        )
        self._atom_item.setData(spots=spots)
        self.plot_widget.addItem(self._atom_item)
        _set_cif_plot_labels(self.plot_widget, mode)
        self.plot_widget.enableAutoRange()

    def _clear_plot(self) -> None:
        if pg is None or not hasattr(self.plot_widget, "removeItem"):
            return
        for item in self._edge_items:
            self.plot_widget.removeItem(item)
        self._edge_items.clear()
        if self._atom_item is not None:
            self.plot_widget.removeItem(self._atom_item)
            self._atom_item = None


class StructureAnalysisPane(QtWidgets.QWidget):
    """Analyze fitted peak centers, phases, and candidate structures."""

    structureAnalysisChanged = QtCore.Signal(str)
    candidateOverlayRequested = QtCore.Signal(str)

    def __init__(
        self,
        project: ProjectState,
        data_id: str,
        *,
        image_data: np.ndarray | None = None,
        axis_ranges: tuple[float, float, float, float] | None = None,
        coordinate_space: str = "qspace",
        image_style: ImageDisplayStyle | None = None,
        project_path: Path | None = None,
        generated_cif_directory: Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.data_id = data_id
        self.image_data = image_data
        self.axis_ranges = axis_ranges
        self.coordinate_space = coordinate_space
        self.image_style = image_style or ImageDisplayStyle()
        self.project_path = Path(project_path) if project_path else None
        self.generated_cif_directory = (
            Path(generated_cif_directory)
            if generated_cif_directory is not None
            else _default_generated_cif_directory(self.project_path)
        )
        self.active_peak_id: str | None = None
        self.active_family_peak_id: str | None = None
        self._syncing_table = False
        self._syncing_peak_selection = False
        self._syncing_atom_table = False
        self._phase_controls: list[QtWidgets.QComboBox] = []
        self._family_shortcuts: list[QtGui.QShortcut] = []
        self._candidate_guess_thread: QtCore.QThread | None = None
        self._candidate_guess_worker: _CandidateGuessWorker | None = None
        self._candidate_guess_dialog: QtWidgets.QProgressDialog | None = None
        self._candidate_guess_started_at: float | None = None
        self._candidate_guess_estimate_seconds: float | None = None
        self._candidate_guess_timer = QtCore.QTimer(self)
        self._candidate_guess_timer.setInterval(1000)
        self._candidate_guess_timer.timeout.connect(
            self._update_candidate_guess_progress
        )
        self.view_box: Any | None = None
        self.roi_overlay_items: list[Any] = []
        self.candidate_grid_scatter: Any | None = None
        self.plot_frame: _ImageAspectPlotFrame | None = None
        self._analysis_state()
        self._refresh_imported_peaks(preserve_user_edits=True)

        self._build_plot()
        self._build_peak_table()
        self._build_controls()
        self._build_layout()
        self._set_initial_image()
        self._sync_all_views()

    def refresh_from_peak_fit(self) -> None:
        """Refresh unedited Structure Analysis rows from Peak Fit
        results."""

        self._refresh_imported_peaks(preserve_user_edits=True)
        self._sync_all_views()
        self._set_status(
            "Refreshed unedited peak centers from Peak Identification / "
            "Peak Fit."
        )
        self.structureAnalysisChanged.emit(self.data_id)

    def run_best_guess_refinement(self) -> LatticeCandidate | None:
        peaks = self._structure_peaks()
        if not peaks:
            self._set_status("No included peaks are available for fitting.")
            return None
        config = self._candidate_config()
        params = self._best_guess_parameters()
        candidate = refine_lattice_candidate(peaks, params, config)
        candidate.candidate_id = "best_guess_refined"
        candidate.method = "best-guess least-squares"
        self._store_candidates([candidate, *self._candidate_records()])
        self._sync_candidates()
        self._set_status(
            f"Refined {candidate.crystal_system}: score "
            f"{candidate.score:.4g}, matched {candidate.matched_count}."
        )
        self.structureAnalysisChanged.emit(self.data_id)
        return candidate

    def start_candidate_guessing(self) -> bool:
        if self._candidate_guess_thread is not None:
            self._set_status("Candidate guessing is already running.")
            return False
        peaks = self._structure_peaks()
        if not peaks:
            self._set_status("No included peaks are available for guessing.")
            return False
        config = self._candidate_config()
        estimate_seconds = _estimate_candidate_guess_seconds(peaks, config)
        dialog = self._candidate_guess_progress_dialog(estimate_seconds)
        thread = QtCore.QThread(self)
        worker = _CandidateGuessWorker(peaks, config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_candidate_guessing_finished)
        worker.failed.connect(self._handle_candidate_guessing_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_candidate_guess_thread)

        self._candidate_guess_thread = thread
        self._candidate_guess_worker = worker
        self._candidate_guess_dialog = dialog
        self._candidate_guess_started_at = time.perf_counter()
        self._candidate_guess_estimate_seconds = estimate_seconds
        self.guess_button.setEnabled(False)
        self._set_status(
            "Guessing candidate structures in the background. "
            f"Estimated time: {_format_duration(estimate_seconds)}."
        )
        dialog.show()
        self._candidate_guess_timer.start()
        thread.start()
        return True

    def run_candidate_guessing(self) -> list[LatticeCandidate]:
        """Run candidate guessing synchronously for scripts and tests.

        The GUI button uses :meth:`start_candidate_guessing` so the Qt event
        loop remains responsive during longer searches.
        """

        peaks = self._structure_peaks()
        if not peaks:
            self._set_status("No included peaks are available for guessing.")
            return []
        config = self._candidate_config()
        estimate_seconds = _estimate_candidate_guess_seconds(peaks, config)
        dialog = self._candidate_guess_progress_dialog(estimate_seconds)
        self.guess_button.setEnabled(False)
        self._set_status(
            "Guessing candidate structures... "
            f"estimated time {_format_duration(estimate_seconds)}."
        )
        try:
            dialog.show()
            QtWidgets.QApplication.processEvents()
            candidates = guess_lattice_candidates(peaks, config)
            dialog.setRange(0, 1)
            dialog.setValue(1)
            QtWidgets.QApplication.processEvents()
        finally:
            self.guess_button.setEnabled(True)
            dialog.close()
            dialog.deleteLater()
        self._store_candidates(candidates)
        self._sync_candidates()
        self._set_status(
            f"Generated {len(candidates)} ranked candidate structure(s)."
        )
        self.structureAnalysisChanged.emit(self.data_id)
        return candidates

    def _candidate_guess_progress_dialog(
        self,
        estimate_seconds: float,
    ) -> QtWidgets.QProgressDialog:
        dialog = QtWidgets.QProgressDialog(
            _candidate_guess_progress_text(0.0, estimate_seconds),
            "Cancel",
            0,
            0,
            self,
        )
        dialog.setWindowTitle("Guess Candidates")
        dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        dialog.setCancelButton(None)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(True)
        dialog.setAutoReset(True)
        return dialog

    def _update_candidate_guess_progress(self) -> None:
        dialog = self._candidate_guess_dialog
        if dialog is None or self._candidate_guess_started_at is None:
            return
        elapsed = time.perf_counter() - self._candidate_guess_started_at
        estimate = self._candidate_guess_estimate_seconds or 0.0
        dialog.setLabelText(_candidate_guess_progress_text(elapsed, estimate))

    @QtCore.Slot(object, float)
    def _handle_candidate_guessing_finished(
        self,
        candidates: object,
        elapsed_seconds: float,
    ) -> None:
        resolved = list(candidates) if isinstance(candidates, list) else []
        self._finish_candidate_guessing_dialog()
        self._store_candidates(resolved)
        self._sync_candidates()
        self._set_status(
            f"Generated {len(resolved)} ranked candidate structure(s) in "
            f"{_format_duration(elapsed_seconds)}."
        )
        self.structureAnalysisChanged.emit(self.data_id)

    @QtCore.Slot(str, float)
    def _handle_candidate_guessing_failed(
        self,
        message: str,
        elapsed_seconds: float,
    ) -> None:
        self._finish_candidate_guessing_dialog()
        self._set_status(
            "Candidate guessing failed after "
            f"{_format_duration(elapsed_seconds)}: {message}"
        )

    def _finish_candidate_guessing_dialog(self) -> None:
        self._candidate_guess_timer.stop()
        dialog = self._candidate_guess_dialog
        if dialog is not None:
            dialog.setRange(0, 1)
            dialog.setValue(1)
            dialog.close()
            dialog.deleteLater()
        self.guess_button.setEnabled(True)
        self._candidate_guess_dialog = None
        self._candidate_guess_started_at = None
        self._candidate_guess_estimate_seconds = None

    @QtCore.Slot()
    def _clear_candidate_guess_thread(self) -> None:
        self._candidate_guess_thread = None
        self._candidate_guess_worker = None

    def suggest_and_tag_outliers(self) -> list[dict[str, Any]]:
        candidate = self._selected_candidate()
        if candidate is None:
            self._set_status("Select or generate a candidate first.")
            return []
        suggestions = suggest_non_main_phase_peaks(
            self._structure_peaks(),
            candidate,
            self._candidate_config(),
        )
        suggested_ids = {item["peak_id"] for item in suggestions}
        if suggested_ids:
            peaks = []
            for peak in self._structure_peaks():
                if peak.peak_id in suggested_ids:
                    peak.phase_tag = PHASE_UNASSIGNED
                    peak.notes = _append_note(
                        peak.notes,
                        "Suggested non-main phase/unassigned",
                    )
                peaks.append(peak)
            self._store_peaks(peaks)
        self._sync_all_views()
        self._set_status(
            f"Tagged {len(suggestions)} possible secondary/unassigned peak(s)."
        )
        self.structureAnalysisChanged.emit(self.data_id)
        return suggestions

    def overlay_selected_candidate(self) -> LatticeCandidate | None:
        candidate = self._selected_candidate()
        if candidate is None:
            self._set_status("Select or generate a candidate first.")
            return None
        params = candidate.as_parameters()
        params.h_max = self.hkl_max.value()
        params.k_max = self.hkl_max.value()
        params.l_max = self.hkl_max.value()
        overlays = self.project.analysis_results.setdefault(
            "crystal_overlays",
            {},
        )
        overlays[self.data_id] = {
            "parameters": params.as_dict(),
            "show_overlay": True,
            "show_hkl_labels": True,
            "hkl_label_mode": "partial",
            "auto_update_overlay": True,
            "source": "structure_analysis",
            "candidate_id": candidate.candidate_id,
        }
        self._set_status(f"Sent {candidate.candidate_id} to Crystal Overlay.")
        self.candidateOverlayRequested.emit(self.data_id)
        self.structureAnalysisChanged.emit(self.data_id)
        return candidate

    def suggest_peak_families(self) -> list[dict[str, Any]]:
        enabled_kinds = self._enabled_family_detection_kinds()
        families = group_peak_families(
            self._structure_peaks(),
            tolerance=self.family_tolerance.value(),
            ratio_tolerance=self.family_ratio_tolerance.value(),
            phase_tag=str(self.family_phase_combo.currentText()),
            enabled_kinds=enabled_kinds,
        )
        state = self._analysis_state()
        manual_families = [
            family
            for family in self._family_records()
            if _is_manual_family(family)
        ]
        state["families"] = [*manual_families, *families]
        self._sync_families()
        if not enabled_kinds:
            self._set_status(
                "No auto family types selected; preserved custom families."
            )
            self.structureAnalysisChanged.emit(self.data_id)
            return families
        self._set_status(
            f"Suggested {len(families)} candidate family group(s) from "
            f"{len(enabled_kinds)} enabled auto type(s)."
        )
        self.structureAnalysisChanged.emit(self.data_id)
        return families

    def _enabled_family_detection_kinds(self) -> list[str]:
        return [
            kind
            for kind in DEFAULT_FAMILY_DETECTION_KINDS
            if self.family_kind_checks[kind].isChecked()
        ]

    def add_custom_peak_family(self) -> dict[str, Any] | None:
        state = self._analysis_state()
        requested_name = self.custom_family_name_edit.text().strip()
        family_id = _unique_family_id(
            requested_name or "Custom family",
            self._family_records(),
        )
        family = {
            "family_id": family_id,
            "kind": "custom",
            "phase_tag": str(self.family_phase_combo.currentText()),
            "reference": None,
            "peak_ids": [],
            "labels": [],
            "confidence": 1.0,
            "reason": (
                "User-defined family. Select this row, then click peaks in "
                "the plot to add members."
            ),
            "notes": "custom family added by user",
            "source": FAMILY_SOURCE_MANUAL,
            "manual_edited": True,
            "user_flag": FAMILY_FLAG_APPROPRIATE,
        }
        state["families"] = [*self._family_records(), family]
        self.custom_family_name_edit.clear()
        self._sync_families(select_family_id=family_id)
        self._set_status(
            f"Added empty custom family {family_id}. Click peaks in the plot "
            "to add members."
        )
        self.structureAnalysisChanged.emit(self.data_id)
        return family

    def _selected_peak_ids(self) -> list[str]:
        rows = sorted(
            {index.row() for index in self.peak_table.selectedIndexes()}
        )
        if not rows and self.peak_table.currentRow() >= 0:
            rows = [self.peak_table.currentRow()]
        peak_ids: list[str] = []
        seen: set[str] = set()
        for row in rows:
            item = self.peak_table.item(row, COL_PEAK_ID)
            if item is None:
                continue
            peak_id = str(
                item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text()
            )
            if peak_id and peak_id not in seen:
                peak_ids.append(peak_id)
                seen.add(peak_id)
        return peak_ids

    def add_reference_molecule(self, key: str) -> None:
        metadata = dict(REFERENCE_MOLECULES.get(key, {}))
        if not metadata:
            return
        metadata["label"] = key
        wyckoff = self._wyckoff_state()
        molecules = wyckoff.setdefault("molecules", [])
        molecules.append(metadata)
        self._sync_molecule_table()
        self.structureAnalysisChanged.emit(self.data_id)

    def generate_candidate_cifs(self) -> list[dict[str, Any]]:
        candidate = self._wyckoff_candidate()
        if candidate is None:
            self._set_status("Select a candidate before generating CIFs.")
            return []
        atom_specs = self._free_atom_specs()
        atoms = [spec["element"] for spec in atom_specs]
        density = self.density_spin.value()
        self._wyckoff_state()["free_atoms"] = atom_specs
        self._store_structure_output_mode()
        records = generate_ranked_cif_records(
            candidate,
            atoms=atoms,
            atom_specs=atom_specs,
            molecules=self._wyckoff_state().get("molecules", []),
            space_group_number=self._selected_space_group_number(),
            wyckoff_combinations=self._selected_wyckoff_combinations(),
            stoichiometry=self.stoichiometry_edit.text(),
            density_g_cm3=density if density > 0.0 else None,
            occupancy_constraints=_atom_occupancy_summary(atom_specs),
            limit=self.cif_count_spin.value(),
        )
        records = _structure_output_records(
            records,
            self._selected_structure_output_mode(),
            keep_elements=atoms,
        )
        records = self._write_generated_cif_files(records)
        self._wyckoff_state()["generated_cifs"] = records
        self._publish_generated_cifs(records)
        self._sync_cif_table()
        self._set_status(
            f"Generated {len(records)} ranked draft CIF file(s) in "
            f"{self.generated_cif_directory}."
        )
        self.structureAnalysisChanged.emit(self.data_id)
        return records

    def open_generated_cif_folder(self) -> Path | None:
        directory = self.generated_cif_directory
        if not directory.exists():
            self._set_status("Generate CIFs before opening the output folder.")
            return None
        opened = QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(directory))
        )
        if opened:
            self._set_status(f"Opened generated CIF folder: {directory}")
            return directory
        self._set_status(f"Could not open generated CIF folder: {directory}")
        return None

    def _selected_structure_output_mode(self) -> str:
        if not hasattr(self, "structure_output_combo"):
            return STRUCTURE_OUTPUT_FULL
        mode = str(
            self.structure_output_combo.currentData() or STRUCTURE_OUTPUT_FULL
        )
        if mode in {
            STRUCTURE_OUTPUT_FULL,
            STRUCTURE_OUTPUT_SCAFFOLD_ONLY,
            STRUCTURE_OUTPUT_FULL_PLUS_SCAFFOLD,
        }:
            return mode
        return STRUCTURE_OUTPUT_FULL

    def _store_structure_output_mode(self) -> None:
        self._wyckoff_state()[
            "structure_output_mode"
        ] = self._selected_structure_output_mode()

    def _sync_structure_output_combo_from_state(self) -> None:
        mode = str(
            self._wyckoff_state().get(
                "structure_output_mode",
                STRUCTURE_OUTPUT_FULL,
            )
        )
        if mode not in {
            STRUCTURE_OUTPUT_FULL,
            STRUCTURE_OUTPUT_SCAFFOLD_ONLY,
            STRUCTURE_OUTPUT_FULL_PLUS_SCAFFOLD,
        }:
            mode = STRUCTURE_OUTPUT_FULL
        for index in range(self.structure_output_combo.count()):
            if self.structure_output_combo.itemData(index) == mode:
                self.structure_output_combo.setCurrentIndex(index)
                return

    def _analysis_state(self) -> dict[str, Any]:
        analyses = self.project.analysis_results.setdefault(
            "structure_analysis",
            {},
        )
        return analyses.setdefault(
            self.data_id,
            {
                "peaks": [],
                "candidates": [],
                "families": [],
                "wyckoff": {},
            },
        )

    def _wyckoff_state(self) -> dict[str, Any]:
        return self._analysis_state().setdefault("wyckoff", {})

    def _peak_fit_records(self) -> dict[str, Any]:
        container = self.project.fits.get(self.data_id, {})
        if isinstance(container, dict):
            peak_fit = container.get("peak_fit", {})
            return peak_fit if isinstance(peak_fit, dict) else {}
        return {}

    def _refresh_imported_peaks(self, *, preserve_user_edits: bool) -> None:
        state = self._analysis_state()
        existing = state.get("peaks", []) if preserve_user_edits else []
        records = self.project.peak_sets.get(self.data_id, [])
        if records:
            peaks = build_structure_peaks(
                records,
                self._peak_fit_records(),
                existing=existing,
            )
            self._store_peaks(peaks)
        elif existing:
            self._store_peaks(
                [StructurePeak.from_dict(item) for item in existing]
            )

    def _structure_peaks(self) -> list[StructurePeak]:
        return [
            StructurePeak.from_dict(item)
            for item in self._analysis_state().get("peaks", [])
            if isinstance(item, dict)
        ]

    def _store_peaks(self, peaks: list[StructurePeak]) -> None:
        self._analysis_state()["peaks"] = [peak.as_dict() for peak in peaks]

    def _candidate_records(self) -> list[LatticeCandidate]:
        return [
            LatticeCandidate.from_dict(item)
            for item in self._analysis_state().get("candidates", [])
            if isinstance(item, dict)
        ]

    def _store_candidates(self, candidates: list[LatticeCandidate]) -> None:
        unique: list[LatticeCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.candidate_id in seen:
                continue
            seen.add(candidate.candidate_id)
            unique.append(candidate)
        self._analysis_state()["candidates"] = [
            candidate.as_dict() for candidate in unique
        ]

    def _family_records(self) -> list[dict[str, Any]]:
        return [
            family
            for family in self._analysis_state().get("families", [])
            if isinstance(family, dict)
        ]

    def _build_plot(self) -> None:
        if pg is None:
            self.plot_widget = QtWidgets.QLabel("Structure Analysis")
            self.plot_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.plot_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            self.image_item = None
            self.peak_scatter = None
            self.family_highlight_scatter = None
            self.view_box = None
            self.roi_overlay_items = []
            self.candidate_grid_scatter = None
            return
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.view_box = self.plot_widget.getViewBox()
        if self.coordinate_space == "qspace":
            set_qspace_axis_labels(self.plot_widget)
        else:
            self.plot_widget.setLabel("bottom", "x", units="px")
            self.plot_widget.setLabel("left", "y", units="px")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        set_data_aspect_locked(self.plot_widget)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot_widget.addItem(self.image_item)
        self.peak_scatter = pg.ScatterPlotItem(
            size=10,
            hoverable=True,
            tip=_peak_tip,
        )
        self.family_highlight_scatter = pg.ScatterPlotItem(
            size=18,
            symbol="o",
            brush=pg.mkBrush(255, 255, 255, 0),
            pen=pg.mkPen("#facc15", width=2.4),
            hoverable=True,
            tip=_peak_tip,
        )
        self.candidate_grid_scatter = pg.ScatterPlotItem(
            size=8,
            symbol="x",
            brush=pg.mkBrush(244, 63, 94, 110),
            pen=pg.mkPen("#f43f5e", width=1.4),
        )
        self.candidate_grid_scatter.setZValue(12)
        _make_plot_item_non_interactive(self.candidate_grid_scatter)
        self.family_highlight_scatter.setZValue(15)
        self.peak_scatter.setZValue(14)
        self.peak_scatter.sigClicked.connect(self._handle_peak_plot_clicked)
        self.family_highlight_scatter.sigClicked.connect(
            self._handle_family_plot_clicked
        )
        self.plot_widget.scene().sigMouseClicked.connect(
            self._handle_plot_scene_clicked
        )
        self.plot_widget.addItem(self.candidate_grid_scatter)
        self.plot_widget.addItem(self.peak_scatter)
        self.plot_widget.addItem(self.family_highlight_scatter)

    def _build_peak_table(self) -> None:
        self.peak_table = QtWidgets.QTableWidget(0, len(PEAK_COLUMNS))
        set_rich_text_table_headers(self.peak_table, PEAK_COLUMNS)
        enable_rich_text_items(self.peak_table)
        self.peak_table.horizontalHeader().setStretchLastSection(True)
        self.peak_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.peak_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.peak_table.itemChanged.connect(self._handle_peak_item_changed)
        self.peak_table.itemSelectionChanged.connect(
            self._handle_peak_selection
        )
        self.peak_table.setMinimumHeight(170)

    def _build_controls(self) -> None:
        self.colormap_combo = QtWidgets.QComboBox()
        for name in IMAGE_COLORMAPS:
            self.colormap_combo.addItem(name.title(), name)
        self.level_min = _level_spinbox()
        self.level_max = _level_spinbox()
        self.quantile_check = QtWidgets.QCheckBox("Quantile")
        self.quantile_low = _quantile_spinbox(self.image_style.quantile_low)
        self.quantile_high = _quantile_spinbox(self.image_style.quantile_high)
        self.auto_contrast_button = QtWidgets.QToolButton()
        self.auto_contrast_button.setText("Auto")
        self._set_image_style_controls(self.image_style)
        self.colormap_combo.currentIndexChanged.connect(
            self._apply_image_style_from_controls
        )
        self.level_min.valueChanged.connect(self._apply_manual_levels)
        self.level_max.valueChanged.connect(self._apply_manual_levels)
        self.quantile_check.toggled.connect(
            self._apply_image_style_from_controls
        )
        self.quantile_low.valueChanged.connect(
            self._apply_image_style_from_controls
        )
        self.quantile_high.valueChanged.connect(
            self._apply_image_style_from_controls
        )
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
        self._set_pan_mode(False)

        self.status_label = QtWidgets.QLabel(
            "Structure Analysis edits update this tab only; Peak Fit results "
            "remain the source data."
        )
        self.status_label.setWordWrap(True)
        self.refresh_button = QtWidgets.QToolButton()
        self.refresh_button.setText("Refresh From Peak Fit")
        self.refresh_button.clicked.connect(self.refresh_from_peak_fit)

        self.phase_filter_combo = QtWidgets.QComboBox()
        self.phase_filter_combo.addItems(DEFAULT_PHASE_TAGS)
        self.phase_filter_combo.currentTextChanged.connect(
            self._sync_candidate_phase_controls
        )
        self.family_phase_combo = QtWidgets.QComboBox()
        self.family_phase_combo.addItems(DEFAULT_PHASE_TAGS)

        self.guess_system_combo = QtWidgets.QComboBox()
        self.guess_system_combo.addItems(CRYSTAL_SYSTEMS.keys())
        self.lattice_a = _lattice_spinbox(6.3)
        self.lattice_b = _lattice_spinbox(6.3)
        self.lattice_c = _lattice_spinbox(6.3)
        self.lattice_alpha = _angle_spinbox(90.0)
        self.lattice_beta = _angle_spinbox(90.0)
        self.lattice_gamma = _angle_spinbox(90.0)
        self.hkl_max = _int_spinbox(4, 0, 12)
        self.q_tolerance = _double_spinbox(0.06, 0.001, 5.0, 0.01)
        self.relative_tolerance = _double_spinbox(0.035, 0.0, 1.0, 0.005)
        self.grid_points = _int_spinbox(16, 3, 40)
        self.candidate_grid_check = QtWidgets.QCheckBox("Show candidate grid")
        self.candidate_grid_check.setChecked(
            bool(self._analysis_state().get("show_candidate_grid", True))
        )
        self.candidate_grid_check.setToolTip(
            qt_tooltip(
                "Overlay the selected approximation candidate as predicted "
                "reciprocal-lattice points on the q-space image."
            )
        )
        self.candidate_grid_check.toggled.connect(
            self._handle_candidate_grid_toggled
        )
        self.hkl_max.valueChanged.connect(self._sync_candidate_grid_overlay)

        self.refine_button = QtWidgets.QToolButton()
        self.refine_button.setText("Refine Best Guess")
        self.refine_button.clicked.connect(self.run_best_guess_refinement)
        self.guess_button = QtWidgets.QToolButton()
        self.guess_button.setText("Guess Candidates")
        self.guess_button.clicked.connect(self.start_candidate_guessing)
        self.overlay_button = QtWidgets.QToolButton()
        self.overlay_button.setText("Overlay Selected")
        self.overlay_button.clicked.connect(self.overlay_selected_candidate)
        self.outliers_button = QtWidgets.QToolButton()
        self.outliers_button.setText("Tag Outliers")
        self.outliers_button.clicked.connect(self.suggest_and_tag_outliers)

        self.candidate_table = QtWidgets.QTableWidget(
            0,
            len(CANDIDATE_COLUMNS),
        )
        self.candidate_table.setHorizontalHeaderLabels(CANDIDATE_COLUMNS)
        enable_rich_text_items(self.candidate_table)
        self.candidate_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.candidate_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.candidate_table.itemSelectionChanged.connect(
            self._sync_wyckoff_candidate_combo
        )
        self.candidate_table.itemSelectionChanged.connect(
            self._sync_candidate_grid_overlay
        )

        self.family_tolerance = _double_spinbox(0.04, 0.001, 5.0, 0.01)
        self.family_ratio_tolerance = _double_spinbox(0.06, 0.001, 1.0, 0.01)
        self.family_confidence_filter = _double_spinbox(
            0.0,
            0.0,
            1.0,
            0.05,
            decimals=2,
        )
        self.family_confidence_filter.setToolTip(
            qt_tooltip(
                "Show only generated or edited families at or above this "
                "confidence score."
            )
        )
        self.family_confidence_filter.valueChanged.connect(
            lambda _value: self._sync_families()
        )
        self.family_kind_checks: dict[str, QtWidgets.QCheckBox] = {}
        for kind in DEFAULT_FAMILY_DETECTION_KINDS:
            checkbox = QtWidgets.QCheckBox(
                FAMILY_KIND_CONTROL_LABELS.get(kind, kind)
            )
            checkbox.setChecked(True)
            checkbox.setToolTip(
                qt_tooltip(
                    "Include this generated peak-family type when running "
                    "family autodetection."
                )
            )
            self.family_kind_checks[kind] = checkbox
        self.family_button = QtWidgets.QToolButton()
        self.family_button.setText("Suggest Peak Families")
        self.family_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.family_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.family_button.setMinimumWidth(
            self.family_button.sizeHint().width()
        )
        self.family_button.setToolTip(
            qt_tooltip(
                "Suggest peak-family rows from the included peaks and "
                "selected autodetection types."
            )
        )
        self.family_button.clicked.connect(self.suggest_peak_families)
        self.custom_family_name_edit = QtWidgets.QLineEdit()
        self.custom_family_name_edit.setPlaceholderText("Custom family name")
        self.custom_family_name_edit.setClearButtonEnabled(True)
        self.add_custom_family_button = QtWidgets.QToolButton()
        self.add_custom_family_button.setText("Add Custom Family")
        self.add_custom_family_button.setToolTip(
            qt_tooltip(
                "Create an empty user-defined family. Select its row, then "
                "click peaks in the plot to add members."
            )
        )
        self.add_custom_family_button.clicked.connect(
            self.add_custom_peak_family
        )
        self.family_flag_button = QtWidgets.QToolButton()
        self.family_flag_button.setText("Flag (F)")
        self.family_flag_button.setToolTip(
            qt_tooltip(
                "Cycle selected families through appropriate, "
                "inappropriate, and unreviewed flags."
            )
        )
        self.family_flag_button.clicked.connect(
            self.toggle_selected_family_flags
        )
        self.family_appropriate_button = QtWidgets.QToolButton()
        self.family_appropriate_button.setText("Appropriate (A)")
        self.family_appropriate_button.setToolTip(
            qt_tooltip(
                "Mark selected families as appropriate. Shortcuts: A from "
                "the family table or plot; Alt+A anywhere in Structure "
                "Analysis."
            )
        )
        self.family_appropriate_button.clicked.connect(
            lambda: self.set_selected_family_flag(FAMILY_FLAG_APPROPRIATE)
        )
        self.family_inappropriate_button = QtWidgets.QToolButton()
        self.family_inappropriate_button.setText("Inappropriate (I)")
        self.family_inappropriate_button.setToolTip(
            qt_tooltip(
                "Mark selected families as inappropriate. Shortcuts: I from "
                "the family table or plot; Alt+I anywhere in Structure "
                "Analysis."
            )
        )
        self.family_inappropriate_button.clicked.connect(
            lambda: self.set_selected_family_flag(FAMILY_FLAG_INAPPROPRIATE)
        )
        self.family_validate_button = QtWidgets.QToolButton()
        self.family_validate_button.setText("Validate Families")
        self.family_validate_button.setToolTip(
            qt_tooltip(
                "Remove families marked inappropriate and collapse duplicate "
                "families with the same phase and peak membership."
            )
        )
        self.family_validate_button.clicked.connect(
            self.validate_reviewed_families
        )
        self.family_highlight_appropriate_button = QtWidgets.QToolButton()
        self.family_highlight_appropriate_button.setText(
            "Highlight Appropriate"
        )
        self.family_highlight_appropriate_button.setCheckable(True)
        self.family_highlight_appropriate_button.setToolTip(
            qt_tooltip(
                "Highlight every peak that belongs to a family marked "
                "appropriate, so ungrouped peaks remain unringed."
            )
        )
        self.family_highlight_appropriate_button.toggled.connect(
            self._sync_peak_plot
        )
        self.family_delete_button = QtWidgets.QToolButton()
        self.family_delete_button.setText("Delete (D)")
        self.family_delete_button.setToolTip(
            qt_tooltip("Delete the selected peak families.")
        )
        self.family_delete_button.clicked.connect(
            self.delete_selected_families
        )
        self.family_remove_ring_button = QtWidgets.QToolButton()
        self.family_remove_ring_button.setText("Remove Ring (R)")
        self.family_remove_ring_button.setToolTip(
            qt_tooltip(
                "Remove the highlighted ring from the active peak family."
            )
        )
        self.family_remove_ring_button.clicked.connect(
            self.remove_active_family_ring
        )
        self.family_table = _FamilyTableWidget(0, len(FAMILY_COLUMNS))
        self.family_table.setHorizontalHeaderLabels(FAMILY_COLUMNS)
        enable_rich_text_items(self.family_table)
        self.family_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.family_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.family_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.family_table.setDragEnabled(True)
        self.family_table.setAcceptDrops(True)
        self.family_table.setDragDropOverwriteMode(False)
        self.family_table.setDropIndicatorShown(True)
        self.family_table.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )
        self.family_table.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.family_table.setToolTip(
            qt_tooltip(
                "Drag family rows to save a manual order; click headers to "
                "sort the view again."
            )
        )
        family_header = self.family_table.horizontalHeader()
        family_header.setSortIndicator(
            FAMILY_COL_ID,
            QtCore.Qt.SortOrder.AscendingOrder,
        )
        family_header.setSortIndicatorShown(True)
        family_header.sectionClicked.connect(
            self._handle_family_header_clicked
        )
        self.family_table.setSortingEnabled(True)
        self.family_table.itemSelectionChanged.connect(self._sync_peak_plot)
        self.family_table.familyDragStarted.connect(self._prepare_family_drag)
        self.family_table.familyRowsReordered.connect(
            self._handle_family_rows_reordered
        )
        self._install_family_shortcuts()

        self._build_wyckoff_controls()

    def _install_family_shortcuts(self) -> None:
        targets = [self.family_table]
        if isinstance(self.plot_widget, QtWidgets.QWidget):
            targets.append(self.plot_widget)
        shortcut_specs = [
            ("F", self.toggle_selected_family_flags),
            (
                "A",
                lambda: self.set_selected_family_flag(FAMILY_FLAG_APPROPRIATE),
            ),
            (
                "I",
                lambda: self.set_selected_family_flag(
                    FAMILY_FLAG_INAPPROPRIATE
                ),
            ),
            ("D", self.delete_selected_families),
            ("R", self.remove_active_family_ring),
        ]
        for target in targets:
            for key, callback in shortcut_specs:
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), target)
                shortcut.setContext(
                    QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
                )
                shortcut.activated.connect(callback)
                self._family_shortcuts.append(shortcut)
        pane_shortcut_specs = [
            (
                "Alt+A",
                lambda: self.set_selected_family_flag(FAMILY_FLAG_APPROPRIATE),
            ),
            (
                "Alt+I",
                lambda: self.set_selected_family_flag(
                    FAMILY_FLAG_INAPPROPRIATE
                ),
            ),
        ]
        for key, callback in pane_shortcut_specs:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(
                QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
            )
            shortcut.activated.connect(callback)
            self._family_shortcuts.append(shortcut)

    def _build_wyckoff_controls(self) -> None:
        self.wyckoff_candidate_combo = QtWidgets.QComboBox()
        self.wyckoff_candidate_combo.currentIndexChanged.connect(
            self._handle_wyckoff_candidate_changed
        )
        self.wyckoff_system_combo = QtWidgets.QComboBox()
        self.wyckoff_system_combo.addItems(WYCKOFF_CRYSTAL_SYSTEMS)
        self.wyckoff_system_combo.currentTextChanged.connect(
            self._sync_space_group_combo
        )
        self.space_group_combo = QtWidgets.QComboBox()
        self.space_group_combo.currentIndexChanged.connect(
            self._sync_wyckoff_registry_tables
        )
        self.wyckoff_site_count_spin = _int_spinbox(3, 1, 4)
        self.wyckoff_site_count_spin.valueChanged.connect(
            self._sync_wyckoff_registry_tables
        )
        self.wyckoff_registry_status = QtWidgets.QLabel()
        self.wyckoff_site_table = QtWidgets.QTableWidget(
            0, len(WYCKOFF_SITE_COLUMNS)
        )
        self.wyckoff_site_table.setHorizontalHeaderLabels(WYCKOFF_SITE_COLUMNS)
        self.wyckoff_site_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.wyckoff_combination_table = QtWidgets.QTableWidget(
            0, len(WYCKOFF_COMBINATION_COLUMNS)
        )
        self.wyckoff_combination_table.setHorizontalHeaderLabels(
            WYCKOFF_COMBINATION_COLUMNS
        )
        self.wyckoff_combination_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.atom_table = QtWidgets.QTableWidget(0, len(ATOM_SPEC_COLUMNS))
        self.atom_table.setHorizontalHeaderLabels(ATOM_SPEC_COLUMNS)
        self.atom_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.atom_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers
        )
        self.atom_table.setMinimumHeight(116)
        self.atom_table.horizontalHeader().setStretchLastSection(True)
        self.atom_table.itemChanged.connect(
            lambda _item: self._store_free_atom_specs()
        )
        self.add_atom_button = QtWidgets.QToolButton()
        self.add_atom_button.setText("Add Atoms")
        self.add_atom_button.setToolTip(
            qt_tooltip(
                "Add an optional free atom row with stoichiometry, shared-site, "
                "and occupancy controls."
            )
        )
        self.add_atom_button.clicked.connect(self.add_atom_spec_row)
        self.remove_atom_button = QtWidgets.QToolButton()
        self.remove_atom_button.setText("Remove Atoms")
        self.remove_atom_button.setToolTip(
            qt_tooltip("Remove the selected optional free atom row(s).")
        )
        self.remove_atom_button.clicked.connect(
            self.remove_selected_atom_specs
        )
        self._sync_atom_table_from_state()
        self.molecule_combo = QtWidgets.QComboBox()
        for key, metadata in REFERENCE_MOLECULES.items():
            self.molecule_combo.addItem(
                f"{key} - {metadata['name']}",
                key,
            )
        self.add_molecule_button = QtWidgets.QToolButton()
        self.add_molecule_button.setText("Add Molecule")
        self.add_molecule_button.clicked.connect(
            lambda: self.add_reference_molecule(
                str(self.molecule_combo.currentData())
            )
        )
        self.load_molecule_button = QtWidgets.QToolButton()
        self.load_molecule_button.setText("Load PDB")
        self.load_molecule_button.clicked.connect(self._load_custom_molecule)
        self.molecule_table = QtWidgets.QTableWidget(0, 4)
        self.molecule_table.setHorizontalHeaderLabels(
            ["Label", "Formula", "Source", "Path"]
        )
        enable_rich_text_items(self.molecule_table)
        self.molecule_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.stoichiometry_edit = QtWidgets.QLineEdit()
        self.density_spin = _double_spinbox(0.0, 0.0, 100.0, 0.1)
        self.density_spin.setSuffix(" g/cm3")
        self.structure_output_combo = QtWidgets.QComboBox()
        self.structure_output_combo.addItem(
            "Full organic fit",
            STRUCTURE_OUTPUT_FULL,
        )
        self.structure_output_combo.addItem(
            "Inorganic scaffold only",
            STRUCTURE_OUTPUT_SCAFFOLD_ONLY,
        )
        self.structure_output_combo.addItem(
            "Full fit + scaffold",
            STRUCTURE_OUTPUT_FULL_PLUS_SCAFFOLD,
        )
        self.structure_output_combo.setToolTip(
            qt_tooltip(
                "Choose whether generated CIFs include organic molecule "
                "atoms, only the inorganic scaffold, or both records side by "
                "side."
            )
        )
        self._sync_structure_output_combo_from_state()
        self.structure_output_combo.currentIndexChanged.connect(
            self._store_structure_output_mode
        )
        self.cif_count_spin = _int_spinbox(5, 1, 50)
        self.generate_cif_button = QtWidgets.QToolButton()
        self.generate_cif_button.setText("Generate Ranked CIFs")
        self.generate_cif_button.clicked.connect(self.generate_candidate_cifs)
        self.open_cif_folder_button = QtWidgets.QToolButton()
        self.open_cif_folder_button.setText("Show in Finder")
        self.open_cif_folder_button.setToolTip(
            qt_tooltip(
                "Open the folder containing generated CIF structures for "
                "inspection in external software."
            )
        )
        self.open_cif_folder_button.setEnabled(False)
        self.open_cif_folder_button.clicked.connect(
            self.open_generated_cif_folder
        )
        self.cif_table = QtWidgets.QTableWidget(0, len(CIF_COLUMNS))
        self.cif_table.setHorizontalHeaderLabels(CIF_COLUMNS)
        enable_rich_text_items(self.cif_table)
        self.cif_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.cif_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.cif_table.itemSelectionChanged.connect(self._sync_cif_visualizer)
        self.cif_visualizer = _GeneratedCifPreview()
        self._sync_space_group_combo()

    def _sync_atom_table_from_state(self) -> None:
        records = self._wyckoff_state().get("free_atoms")
        if not isinstance(records, list) or not records:
            records = [
                {"element": "Pb", "stoichiometry": 1.0},
                {"element": "I", "stoichiometry": 1.0},
            ]
        self._syncing_atom_table = True
        try:
            self.atom_table.setRowCount(0)
            for record in records:
                if not isinstance(record, dict):
                    continue
                self.add_atom_spec_row(
                    element=str(record.get("element") or ""),
                    stoichiometry=_float_or_default(
                        record.get("stoichiometry", record.get("count", 1.0)),
                        1.0,
                    ),
                    shared_site=str(record.get("shared_site") or ""),
                    occupancy=_float_or_default(
                        record.get("occupancy", 1.0),
                        1.0,
                    ),
                )
        finally:
            self._syncing_atom_table = False
        self._store_free_atom_specs()

    def add_atom_spec_row(
        self,
        *,
        element: str = "X",
        stoichiometry: float = 1.0,
        shared_site: str = "",
        occupancy: float = 1.0,
    ) -> None:
        row = self.atom_table.rowCount()
        self.atom_table.insertRow(row)
        element_item = QtWidgets.QTableWidgetItem(element)
        shared_item = QtWidgets.QTableWidgetItem(shared_site)
        self.atom_table.setItem(row, ATOM_COL_ELEMENT, element_item)
        self.atom_table.setCellWidget(
            row,
            ATOM_COL_STOICHIOMETRY,
            count_spinbox := _atom_count_spinbox(stoichiometry),
        )
        self.atom_table.setItem(row, ATOM_COL_SHARED_SITE, shared_item)
        self.atom_table.setCellWidget(
            row,
            ATOM_COL_OCCUPANCY,
            occupancy_spinbox := _occupancy_spinbox(occupancy),
        )
        count_spinbox.valueChanged.connect(
            lambda _value: self._store_free_atom_specs()
        )
        occupancy_spinbox.valueChanged.connect(
            lambda _value: self._store_free_atom_specs()
        )
        self.atom_table.resizeColumnsToContents()
        self._store_free_atom_specs()

    def remove_selected_atom_specs(self) -> None:
        rows = sorted(
            {index.row() for index in self.atom_table.selectedIndexes()},
            reverse=True,
        )
        if not rows and self.atom_table.currentRow() >= 0:
            rows = [self.atom_table.currentRow()]
        for row in rows:
            self.atom_table.removeRow(row)
        self._store_free_atom_specs()

    def _free_atom_specs(self) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for row in range(self.atom_table.rowCount()):
            element_item = self.atom_table.item(row, ATOM_COL_ELEMENT)
            element = str(element_item.text() if element_item else "").strip()
            if not element:
                continue
            shared_item = self.atom_table.item(row, ATOM_COL_SHARED_SITE)
            count_widget = self.atom_table.cellWidget(
                row,
                ATOM_COL_STOICHIOMETRY,
            )
            occupancy_widget = self.atom_table.cellWidget(
                row,
                ATOM_COL_OCCUPANCY,
            )
            stoichiometry = (
                count_widget.value()
                if isinstance(count_widget, QtWidgets.QDoubleSpinBox)
                else 1.0
            )
            occupancy = (
                occupancy_widget.value()
                if isinstance(occupancy_widget, QtWidgets.QDoubleSpinBox)
                else 1.0
            )
            specs.append(
                {
                    "element": element,
                    "stoichiometry": float(stoichiometry),
                    "shared_site": str(
                        shared_item.text() if shared_item else ""
                    ).strip(),
                    "occupancy": float(occupancy),
                }
            )
        return specs

    def _store_free_atom_specs(self) -> None:
        if self._syncing_atom_table:
            return
        self._wyckoff_state()["free_atoms"] = self._free_atom_specs()

    def _build_layout(self) -> None:
        self.analysis_tabs = QtWidgets.QTabWidget()
        self.analysis_tabs.setMinimumWidth(420)
        self.analysis_tabs.setMaximumWidth(640)
        self.approximation_tab = self._approximation_tab()
        self.family_tab = self._families_tab()
        self.wyckoff_tab = self._wyckoff_tab()
        self.analysis_tabs.addTab(
            self.approximation_tab, "Structure Approximation"
        )
        self.analysis_tabs.addTab(self.family_tab, "Peak Families")
        self.analysis_tabs.addTab(self.wyckoff_tab, "Wyckoff Mapping")

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

        plot_area = self.plot_widget
        if pg is not None and self.image_item is not None:
            self.plot_frame = _ImageAspectPlotFrame(self.plot_widget)
            plot_area = self.plot_frame
        plot_section = QtWidgets.QWidget()
        plot_section_layout = QtWidgets.QVBoxLayout(plot_section)
        plot_section_layout.setContentsMargins(0, 0, 0, 0)
        plot_section_layout.addWidget(self.plot_toolbar)
        plot_section_layout.addWidget(plot_area, stretch=1)

        plot_layout = QtWidgets.QHBoxLayout()
        plot_layout.addWidget(plot_section, stretch=1)
        plot_layout.addWidget(self.analysis_tabs)

        table_header = QtWidgets.QHBoxLayout()
        table_header.addWidget(self.refresh_button)
        table_header.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(plot_layout, stretch=1)
        layout.addLayout(table_header)
        layout.addWidget(self.peak_table)
        layout.addWidget(self.status_label)

    def _approximation_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.refine_button)
        buttons.addWidget(self.guess_button)
        buttons.addWidget(self.overlay_button)
        buttons.addWidget(self.outliers_button)
        buttons.addWidget(self.candidate_grid_check)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)

        def add_field(
            row: int,
            column: int,
            label: str | QtWidgets.QLabel,
            widget: QtWidgets.QWidget,
        ) -> None:
            if isinstance(label, QtWidgets.QLabel):
                label_widget = label
            else:
                label_widget = QtWidgets.QLabel(label)
            label_widget.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            form.addWidget(label_widget, row, column * 2)
            form.addWidget(widget, row, column * 2 + 1)

        add_field(0, 0, "Phase", self.phase_filter_combo)
        add_field(0, 1, "Crystal system", self.guess_system_combo)
        add_field(0, 2, "hkl max", self.hkl_max)
        add_field(1, 0, "a", self.lattice_a)
        add_field(1, 1, "b", self.lattice_b)
        add_field(1, 2, "c", self.lattice_c)
        add_field(2, 0, "alpha", self.lattice_alpha)
        add_field(2, 1, "beta", self.lattice_beta)
        add_field(2, 2, "gamma", self.lattice_gamma)
        add_field(
            3,
            0,
            rich_label(f"{QSPACE_UNITS_HTML} tolerance"),
            self.q_tolerance,
        )
        add_field(3, 1, "Relative tolerance", self.relative_tolerance)
        add_field(3, 2, "Grid points", self.grid_points)
        for column in (1, 3, 5):
            form.setColumnStretch(column, 1)
        layout.addLayout(form)
        layout.addWidget(self.candidate_table, stretch=1)
        return tab

    def _families_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Phase"))
        controls.addWidget(self.family_phase_combo)
        controls.addWidget(QtWidgets.QLabel("Coordinate tolerance"))
        controls.addWidget(self.family_tolerance)
        controls.addWidget(QtWidgets.QLabel("Ratio tolerance"))
        controls.addWidget(self.family_ratio_tolerance)
        controls.addWidget(QtWidgets.QLabel("Min confidence"))
        controls.addWidget(self.family_confidence_filter)
        controls.addStretch(1)
        suggest_controls = QtWidgets.QHBoxLayout()
        suggest_controls.addWidget(self.family_button)
        suggest_controls.addStretch(1)
        auto_type_controls = QtWidgets.QHBoxLayout()
        auto_type_controls.addWidget(QtWidgets.QLabel("Auto types"))
        for kind in DEFAULT_FAMILY_DETECTION_KINDS:
            auto_type_controls.addWidget(self.family_kind_checks[kind])
        auto_type_controls.addStretch(1)
        custom_controls = QtWidgets.QHBoxLayout()
        custom_controls.addWidget(QtWidgets.QLabel("Custom"))
        custom_controls.addWidget(self.custom_family_name_edit, stretch=1)
        custom_controls.addWidget(self.add_custom_family_button)
        review_controls = QtWidgets.QHBoxLayout()
        review_controls.addWidget(self.family_flag_button)
        review_controls.addWidget(self.family_appropriate_button)
        review_controls.addWidget(self.family_inappropriate_button)
        review_controls.addWidget(self.family_validate_button)
        review_controls.addWidget(self.family_highlight_appropriate_button)
        review_controls.addWidget(self.family_delete_button)
        review_controls.addWidget(self.family_remove_ring_button)
        review_controls.addStretch(1)
        layout.addLayout(suggest_controls)
        layout.addLayout(controls)
        layout.addLayout(auto_type_controls)
        layout.addLayout(custom_controls)
        layout.addLayout(review_controls)
        layout.addWidget(self.family_table, stretch=1)
        return tab

    def _wyckoff_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.wyckoff_subtabs = QtWidgets.QTabWidget()
        self.candidate_mapping_tab = QtWidgets.QWidget()
        candidate_layout = QtWidgets.QVBoxLayout(self.candidate_mapping_tab)

        setup_content = QtWidgets.QWidget()
        setup_layout = QtWidgets.QVBoxLayout(setup_content)
        setup_layout.setContentsMargins(0, 0, 0, 0)

        candidate_group = QtWidgets.QGroupBox("Candidate Mapping")
        candidate_form = QtWidgets.QFormLayout(candidate_group)
        candidate_form.addRow("Phase/candidate", self.wyckoff_candidate_combo)
        setup_layout.addWidget(candidate_group)

        composition_group = QtWidgets.QGroupBox("Composition & Molecules")
        composition_layout = QtWidgets.QVBoxLayout(composition_group)
        composition_form = QtWidgets.QFormLayout()
        atom_widget = QtWidgets.QWidget()
        atom_layout = QtWidgets.QVBoxLayout(atom_widget)
        atom_layout.setContentsMargins(0, 0, 0, 0)
        atom_button_row = QtWidgets.QHBoxLayout()
        atom_button_row.addWidget(self.add_atom_button)
        atom_button_row.addWidget(self.remove_atom_button)
        atom_button_row.addStretch(1)
        atom_layout.addLayout(atom_button_row)
        atom_layout.addWidget(self.atom_table)
        composition_form.addRow("Free atoms", atom_widget)
        molecule_row = QtWidgets.QWidget()
        molecule_layout = QtWidgets.QHBoxLayout(molecule_row)
        molecule_layout.setContentsMargins(0, 0, 0, 0)
        molecule_layout.addWidget(self.molecule_combo, stretch=1)
        molecule_layout.addWidget(self.add_molecule_button)
        molecule_layout.addWidget(self.load_molecule_button)
        composition_form.addRow("Reference molecule", molecule_row)
        composition_form.addRow("Stoichiometry", self.stoichiometry_edit)
        composition_form.addRow("Density", self.density_spin)
        composition_form.addRow(
            "Structure output", self.structure_output_combo
        )
        composition_layout.addLayout(composition_form)
        composition_layout.addWidget(self.molecule_table)
        setup_layout.addWidget(composition_group)

        generate_group = QtWidgets.QGroupBox("Generate")
        generate_layout = QtWidgets.QHBoxLayout(generate_group)
        generate_layout.addWidget(QtWidgets.QLabel("CIF count"))
        generate_layout.addWidget(self.cif_count_spin)
        generate_layout.addWidget(self.generate_cif_button)
        generate_layout.addWidget(self.open_cif_folder_button)
        generate_layout.addStretch(1)
        setup_layout.addWidget(generate_group)
        setup_layout.addStretch(1)

        setup_scroll = QtWidgets.QScrollArea()
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        setup_scroll.setWidget(setup_content)

        candidate_layout.addWidget(setup_scroll, stretch=1)

        self.wyckoff_registry_tab = QtWidgets.QWidget()
        registry_tab_layout = QtWidgets.QVBoxLayout(self.wyckoff_registry_tab)

        registry_filter_group = QtWidgets.QGroupBox("Registry Filters")
        registry_filter_form = QtWidgets.QFormLayout(registry_filter_group)
        registry_filter_form.addRow(
            "Crystal system", self.wyckoff_system_combo
        )
        registry_filter_form.addRow("Space group", self.space_group_combo)
        registry_filter_form.addRow(
            "Wyckoff site count", self.wyckoff_site_count_spin
        )
        registry_tab_layout.addWidget(registry_filter_group)

        registry_group = QtWidgets.QGroupBox("Wyckoff Registry")
        registry_layout = QtWidgets.QVBoxLayout(registry_group)
        self.wyckoff_registry_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical
        )
        self.wyckoff_registry_splitter.addWidget(self.wyckoff_site_table)
        self.wyckoff_registry_splitter.addWidget(
            self.wyckoff_combination_table
        )
        self.wyckoff_registry_splitter.setStretchFactor(0, 1)
        self.wyckoff_registry_splitter.setStretchFactor(1, 2)
        registry_layout.addWidget(self.wyckoff_registry_status)
        registry_layout.addWidget(self.wyckoff_registry_splitter, stretch=1)
        registry_tab_layout.addWidget(registry_group, stretch=1)

        self.generated_cifs_tab = QtWidgets.QWidget()
        generated_cifs_layout = QtWidgets.QVBoxLayout(self.generated_cifs_tab)
        cifs_group = QtWidgets.QGroupBox("Generated CIFs")
        cifs_layout = QtWidgets.QVBoxLayout(cifs_group)
        self.generated_cifs_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical
        )
        self.generated_cifs_splitter.addWidget(self.cif_table)
        self.generated_cifs_splitter.addWidget(self.cif_visualizer)
        self.generated_cifs_splitter.setStretchFactor(0, 1)
        self.generated_cifs_splitter.setStretchFactor(1, 2)
        cifs_layout.addWidget(self.generated_cifs_splitter, stretch=1)
        generated_cifs_layout.addWidget(cifs_group, stretch=1)

        self.wyckoff_subtabs.addTab(
            self.candidate_mapping_tab,
            "Candidate Mapping",
        )
        self.wyckoff_subtabs.addTab(
            self.wyckoff_registry_tab,
            "Wyckoff Registry",
        )
        self.wyckoff_subtabs.addTab(
            self.generated_cifs_tab,
            "Generated CIFs",
        )

        layout.addWidget(self.wyckoff_subtabs, stretch=1)
        return tab

    def _set_initial_image(self) -> None:
        if pg is None or self.image_item is None or self.image_data is None:
            return
        self.image_item.setImage(self.image_data, autoLevels=False)
        image_rect = data_image_rect(
            self.image_data.shape,
            self.axis_ranges,
        )
        self.image_item.setRect(image_rect)
        if self.plot_frame is not None:
            self.plot_frame.set_data_rect(image_rect)
        self.apply_image_style(self.image_style)
        set_data_image_plot_range(
            self.plot_widget,
            self.image_data.shape,
            self.axis_ranges,
        )

    def _apply_manual_levels(self) -> None:
        if not self.quantile_check.isChecked():
            self._apply_image_style_from_controls()

    def _zoom_image(self, factor: float) -> None:
        if pg is None or self.view_box is None:
            return
        self.view_box.scaleBy((factor, factor))

    def _reset_image_zoom(self) -> None:
        if pg is None or self.plot_widget is None or self.image_data is None:
            return
        set_data_image_plot_range(
            self.plot_widget,
            self.image_data.shape,
            self.axis_ranges,
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

    def image_display_style(self) -> ImageDisplayStyle:
        return ImageDisplayStyle(
            colormap=str(self.colormap_combo.currentData()),
            use_quantile=self.quantile_check.isChecked(),
            quantile_low=self.quantile_low.value(),
            quantile_high=self.quantile_high.value(),
            level_min=self.level_min.value(),
            level_max=self.level_max.value(),
        )

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
        self._apply_image_style_from_controls()

    def _apply_image_style_from_controls(self) -> None:
        self.apply_image_style(
            self.image_display_style(),
            sync_controls=False,
        )

    def apply_image_style(
        self,
        style: ImageDisplayStyle,
        *,
        sync_controls: bool = True,
    ) -> None:
        self.image_style = style
        if sync_controls:
            self._set_image_style_controls(style)
        apply_image_display_style(
            self.image_item,
            self.image_data,
            self.image_style,
        )

    def _set_image_style_controls(self, style: ImageDisplayStyle) -> None:
        controls = (
            self.colormap_combo,
            self.level_min,
            self.level_max,
            self.quantile_check,
            self.quantile_low,
            self.quantile_high,
        )
        for control in controls:
            control.blockSignals(True)
        try:
            index = self.colormap_combo.findData(style.colormap)
            if index >= 0:
                self.colormap_combo.setCurrentIndex(index)
            self.level_min.setValue(style.level_min)
            self.level_max.setValue(style.level_max)
            self.quantile_check.setChecked(style.use_quantile)
            self.quantile_low.setValue(style.quantile_low)
            self.quantile_high.setValue(style.quantile_high)
        finally:
            for control in controls:
                control.blockSignals(False)

    def _sync_all_views(self) -> None:
        self._sync_phase_options()
        self._sync_peak_table()
        self._sync_peak_plot()
        self._sync_candidates()
        self._sync_families()
        self._sync_molecule_table()
        self._sync_cif_table()
        self._sync_wyckoff_registry_tables()

    def refresh_roi_overlays(self, data_id: str | None = None) -> None:
        if data_id is not None and str(data_id) != self.data_id:
            return
        if self.coordinate_space != "qspace":
            self._clear_roi_overlays()
            return
        self._sync_roi_overlays()

    def _sync_peak_table(self) -> None:
        peaks = self._structure_peaks()
        self._phase_controls.clear()
        self._syncing_table = True
        try:
            self.peak_table.setRowCount(len(peaks))
            for row, peak in enumerate(peaks):
                self._set_peak_row(row, peak)
        finally:
            self._syncing_table = False
        self.peak_table.resizeColumnsToContents()

    def _set_peak_row(self, row: int, peak: StructurePeak) -> None:
        values = {
            COL_PEAK_ID: peak.peak_id,
            COL_QXY: _format_float(peak.qxy),
            COL_QZ: _format_float(peak.qz),
            COL_SOURCE: peak.source,
            COL_HKL: peak.hkl_label,
            COL_FIT_QUALITY: _format_float(peak.fit_quality),
            COL_NOTES: _status_text(peak),
        }
        for column, value in values.items():
            item = QtWidgets.QTableWidgetItem(str(value))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, peak.peak_id)
            if column in {COL_PEAK_ID, COL_SOURCE, COL_FIT_QUALITY}:
                item.setFlags(
                    item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable
                )
            self.peak_table.setItem(row, column, item)

        phase_combo = QtWidgets.QComboBox()
        phase_combo.setEditable(True)
        phase_combo.addItems(self._phase_labels())
        if phase_combo.findText(peak.phase_tag) < 0:
            phase_combo.addItem(peak.phase_tag)
        phase_combo.setCurrentText(peak.phase_tag)
        phase_combo.currentTextChanged.connect(
            lambda text, peak_id=peak.peak_id: self._set_peak_phase(
                peak_id,
                text,
            )
        )
        self.peak_table.setCellWidget(row, COL_PHASE, phase_combo)
        self._phase_controls.append(phase_combo)

        include_item = QtWidgets.QTableWidgetItem("")
        include_item.setData(QtCore.Qt.ItemDataRole.UserRole, peak.peak_id)
        include_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        include_item.setCheckState(
            QtCore.Qt.CheckState.Checked
            if peak.include
            else QtCore.Qt.CheckState.Unchecked
        )
        self.peak_table.setItem(row, COL_INCLUDE, include_item)

    def _sync_peak_plot(self) -> None:
        if pg is None or self.peak_scatter is None:
            return
        if self.coordinate_space != "qspace":
            self.peak_scatter.setData(spots=[])
            if self.family_highlight_scatter is not None:
                self.family_highlight_scatter.setData(spots=[])
            self._clear_roi_overlays()
            self._clear_candidate_grid_overlay()
            return
        highlighted_family_peak_ids = self._highlighted_family_peak_ids()
        if self.active_family_peak_id not in highlighted_family_peak_ids:
            self.active_family_peak_id = None
        spots = []
        family_spots = []
        for peak in self._structure_peaks():
            if peak.peak_id == self.active_peak_id:
                size = 14
                brush = pg.mkBrush(STRUCTURE_ACTIVE_PEAK_BRUSH)
                pen = pg.mkPen("#ffffff", width=1.0)
            else:
                size = 10
                brush = pg.mkBrush(STRUCTURE_PEAK_BRUSH)
                pen = pg.mkPen("#ffffff", width=0.8)
            spots.append(
                {
                    "pos": (peak.qxy, peak.qz),
                    "data": peak.as_dict(),
                    "symbol": "o",
                    "size": size,
                    "brush": brush,
                    "pen": pen,
                }
            )
            if peak.peak_id in highlighted_family_peak_ids:
                active_family_ring = peak.peak_id == self.active_family_peak_id
                family_spots.append(
                    {
                        "pos": (peak.qxy, peak.qz),
                        "data": peak.as_dict(),
                        "size": 22 if active_family_ring else 18,
                        "symbol": "o",
                        "brush": pg.mkBrush(255, 255, 255, 0),
                        "pen": pg.mkPen(
                            "#ef4444" if active_family_ring else "#facc15",
                            width=3.0 if active_family_ring else 2.4,
                        ),
                    }
                )
        self.peak_scatter.setData(spots=spots)
        if self.family_highlight_scatter is not None:
            self.family_highlight_scatter.setData(spots=family_spots)
        self._sync_roi_overlays()
        self._sync_candidate_grid_overlay()

    def _handle_candidate_grid_toggled(self, enabled: bool) -> None:
        self._analysis_state()["show_candidate_grid"] = bool(enabled)
        self._sync_candidate_grid_overlay()

    def _sync_candidate_grid_overlay(self, *_args: Any) -> None:
        if pg is None or self.candidate_grid_scatter is None:
            return
        if (
            self.coordinate_space != "qspace"
            or not self.candidate_grid_check.isChecked()
        ):
            self._clear_candidate_grid_overlay()
            return
        candidate = self._selected_candidate(fallback_first=False)
        if candidate is None:
            self._clear_candidate_grid_overlay()
            return
        self.candidate_grid_scatter.setData(
            spots=self._candidate_grid_spots(candidate)
        )

    def _clear_candidate_grid_overlay(self) -> None:
        if self.candidate_grid_scatter is not None:
            self.candidate_grid_scatter.setData(spots=[])

    def _candidate_grid_spots(
        self,
        candidate: LatticeCandidate,
    ) -> list[dict[str, Any]]:
        params = candidate.as_parameters()
        params.h_max = self.hkl_max.value()
        params.k_max = self.hkl_max.value()
        params.l_max = self.hkl_max.value()
        result = CrystalOverlayCalculator(params).project(params)
        if result.qxy.size == 0:
            return []
        keep = np.isfinite(result.qxy) & np.isfinite(result.qz)
        if self.axis_ranges is not None:
            x0, x1, y0, y1 = (float(value) for value in self.axis_ranges)
            x_min, x_max = sorted((x0, x1))
            y_min, y_max = sorted((y0, y1))
            x_pad = max((x_max - x_min) * 0.04, 1.0e-9)
            y_pad = max((y_max - y_min) * 0.04, 1.0e-9)
            keep &= (
                (result.qxy >= x_min - x_pad)
                & (result.qxy <= x_max + x_pad)
                & (result.qz >= y_min - y_pad)
                & (result.qz <= y_max + y_pad)
            )
        spots: list[dict[str, Any]] = []
        for qxy, qz, hkl in zip(
            result.qxy[keep],
            result.qz[keep],
            result.hkl[keep],
            strict=False,
        ):
            spots.append(
                {
                    "pos": (float(qxy), float(qz)),
                    "data": {
                        "candidate_id": candidate.candidate_id,
                        "hkl": format_hkl(hkl),
                        "qxy": float(qxy),
                        "qz": float(qz),
                    },
                }
            )
        return spots

    def _sync_roi_overlays(self) -> None:
        self._clear_roi_overlays()
        if pg is None or self.plot_widget is None:
            return
        for roi in self._structure_roi_records():
            points = _roi_overlay_points(roi)
            if points is None:
                continue
            x_values, y_values = points
            kind = str(roi.get("kind", "box")).lower()
            color = "#f59e0b" if kind == "arch" else "#38bdf8"
            item = pg.PlotDataItem(
                x_values,
                y_values,
                pen=pg.mkPen(color, width=1.5),
            )
            item.setZValue(13)
            _make_plot_item_non_interactive(item)
            self.plot_widget.addItem(item)
            self.roi_overlay_items.append(item)

    def _clear_roi_overlays(self) -> None:
        if pg is None or self.plot_widget is None:
            self.roi_overlay_items = []
            return
        for item in self.roi_overlay_items:
            try:
                self.plot_widget.removeItem(item)
            except Exception:
                pass
        self.roi_overlay_items = []

    def _structure_roi_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for roi in self.project.rois_for_target(self.data_id):
            payload = roi.as_dict() if hasattr(roi, "as_dict") else dict(roi)
            key = _roi_overlay_key(payload)
            if key not in seen:
                records.append(payload)
                seen.add(key)
        for peak in self._structure_peaks():
            metadata = peak.metadata if isinstance(peak.metadata, dict) else {}
            peak_record = metadata.get("peak_record", {})
            fit_record = metadata.get("fit_record", {})
            candidates = []
            if isinstance(peak_record, dict):
                candidates.extend(
                    [
                        peak_record.get("roi"),
                        peak_record.get("azimuthal_roi"),
                    ]
                )
            if isinstance(fit_record, dict):
                candidates.extend(
                    [
                        fit_record.get("roi"),
                        fit_record.get("azimuthal_roi"),
                    ]
                )
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                key = _roi_overlay_key(candidate)
                if key in seen:
                    continue
                records.append(dict(candidate))
                seen.add(key)
        return records

    def _sync_candidates(self) -> None:
        candidates = self._candidate_records()
        self.candidate_table.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            values = [
                row + 1,
                candidate.crystal_system,
                _format_float(candidate.a),
                _format_float(candidate.b),
                _format_float(candidate.c),
                _format_float(candidate.alpha),
                _format_float(candidate.beta),
                _format_float(candidate.gamma),
                _format_float(candidate.score),
                candidate.matched_count,
                candidate.outlier_count,
                candidate.method,
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setData(
                    QtCore.Qt.ItemDataRole.UserRole,
                    candidate.candidate_id,
                )
                self.candidate_table.setItem(row, column, item)
        self.candidate_table.resizeColumnsToContents()
        self._sync_wyckoff_candidate_combo()
        self._sync_candidate_grid_overlay()

    def _sync_families(self, select_family_id: str | None = None) -> None:
        if select_family_id is None:
            select_family_id = self._selected_family_id()
        min_confidence = self.family_confidence_filter.value()
        families = [
            family
            for family in self._family_records()
            if _family_confidence(family) >= min_confidence
        ]
        sorting_enabled = self.family_table.isSortingEnabled()
        header = self.family_table.horizontalHeader()
        sort_column = header.sortIndicatorSection()
        sort_order = header.sortIndicatorOrder()
        self.family_table.setSortingEnabled(False)
        try:
            self.family_table.setRowCount(len(families))
            for row, family in enumerate(families):
                peak_ids = [str(value) for value in family.get("peak_ids", [])]
                family_id = str(family.get("family_id", ""))
                confidence = _family_confidence(family)
                flag = _family_flag(family)
                reason = str(family.get("reason", ""))
                notes = str(family.get("notes", ""))
                values = [
                    family_id,
                    _family_flag_label(flag),
                    f"{confidence:.2f}",
                    family.get("kind", ""),
                    family.get("phase_tag", ""),
                    _format_float(family.get("reference")),
                    ", ".join(
                        family.get("labels", family.get("peak_ids", []))
                    ),
                    reason,
                    notes,
                ]
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    item.setFlags(
                        QtCore.Qt.ItemFlag.ItemIsEnabled
                        | QtCore.Qt.ItemFlag.ItemIsSelectable
                        | QtCore.Qt.ItemFlag.ItemIsDragEnabled
                        | QtCore.Qt.ItemFlag.ItemIsDropEnabled
                    )
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, peak_ids)
                    item.setData(
                        QtCore.Qt.ItemDataRole.UserRole + 1,
                        family_id,
                    )
                    item.setToolTip(reason or notes)
                    self._style_family_item(item, flag)
                    self.family_table.setItem(row, column, item)
        finally:
            self.family_table.setSortingEnabled(sorting_enabled)
        if sorting_enabled and 0 <= sort_column < len(FAMILY_COLUMNS):
            self.family_table.sortItems(sort_column, sort_order)
        self.family_table.resizeColumnsToContents()
        if select_family_id:
            self._select_family_by_id(select_family_id)
        self._sync_peak_plot()

    def _prepare_family_drag(self) -> None:
        if self.family_table.isSortingEnabled():
            self.family_table.setSortingEnabled(False)

    def _handle_family_header_clicked(self, column: int) -> None:
        if self.family_table.isSortingEnabled():
            return
        header = self.family_table.horizontalHeader()
        order = header.sortIndicatorOrder()
        self.family_table.setSortingEnabled(True)
        self.family_table.sortItems(column, order)

    def _handle_family_rows_reordered(self, family_ids: list[str]) -> None:
        if not family_ids:
            return
        select_family_id = self._selected_family_id() or family_ids[-1]
        self.family_table.setSortingEnabled(False)
        if not self._reorder_families(family_ids):
            return
        self._sync_families(select_family_id=select_family_id)
        self._set_status("Saved manual peak-family order.")
        self.structureAnalysisChanged.emit(self.data_id)

    def _reorder_families(self, ordered_family_ids: list[str]) -> bool:
        requested_ids: list[str] = []
        seen: set[str] = set()
        for family_id in ordered_family_ids:
            normalized = str(family_id)
            if not normalized or normalized in seen:
                continue
            requested_ids.append(normalized)
            seen.add(normalized)
        if len(requested_ids) < 2:
            return False

        families = self._family_records()
        family_by_id = {
            str(family.get("family_id", "")): family for family in families
        }
        reordered_visible = [
            family_by_id[family_id]
            for family_id in requested_ids
            if family_id in family_by_id
        ]
        if len(reordered_visible) < 2:
            return False

        visible_ids = {
            str(family.get("family_id", "")) for family in reordered_visible
        }
        reordered_iter = iter(reordered_visible)
        reordered: list[dict[str, Any]] = []
        for family in families:
            family_id = str(family.get("family_id", ""))
            if family_id in visible_ids:
                reordered.append(next(reordered_iter))
            else:
                reordered.append(family)

        before = [str(family.get("family_id", "")) for family in families]
        after = [str(family.get("family_id", "")) for family in reordered]
        if before == after:
            return False
        self._analysis_state()["families"] = reordered
        return True

    def _selected_family_peak_ids(self) -> set[str]:
        if not hasattr(self, "family_table"):
            return set()
        peak_ids: set[str] = set()
        for item in self.family_table.selectedItems():
            values = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if values:
                peak_ids.update(str(value) for value in values)
        return peak_ids

    def _highlighted_family_peak_ids(self) -> set[str]:
        peak_ids = set(self._selected_family_peak_ids())
        if (
            hasattr(self, "family_highlight_appropriate_button")
            and self.family_highlight_appropriate_button.isChecked()
        ):
            peak_ids.update(self._appropriate_family_peak_ids())
        return peak_ids

    def _appropriate_family_peak_ids(self) -> set[str]:
        peak_ids: set[str] = set()
        for family in self._family_records():
            if _family_flag(family) != FAMILY_FLAG_APPROPRIATE:
                continue
            peak_ids.update(
                str(value)
                for value in family.get("peak_ids", [])
                if str(value).strip()
            )
        return peak_ids

    def _selected_family_id(self) -> str | None:
        if not hasattr(self, "family_table"):
            return None
        row = self.family_table.currentRow()
        if row < 0:
            return None
        item = self.family_table.item(row, FAMILY_COL_ID)
        if item is None:
            return None
        return str(
            item.data(QtCore.Qt.ItemDataRole.UserRole + 1) or item.text()
        )

    def _select_family_by_id(self, family_id: str) -> bool:
        for row in range(self.family_table.rowCount()):
            item = self.family_table.item(row, FAMILY_COL_ID)
            if item is None:
                continue
            item_family_id = str(
                item.data(QtCore.Qt.ItemDataRole.UserRole + 1) or item.text()
            )
            if item_family_id != str(family_id):
                continue
            self.family_table.selectRow(row)
            self.family_table.scrollToItem(
                item,
                QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
            )
            return True
        return False

    def _selected_family_ids(self) -> list[str]:
        if not hasattr(self, "family_table"):
            return []
        rows = sorted(
            {index.row() for index in self.family_table.selectedIndexes()}
        )
        if not rows and self.family_table.currentRow() >= 0:
            rows = [self.family_table.currentRow()]
        family_ids: list[str] = []
        for row in rows:
            item = self.family_table.item(row, FAMILY_COL_ID)
            if item is None:
                continue
            family_id = str(
                item.data(QtCore.Qt.ItemDataRole.UserRole + 1) or item.text()
            )
            if family_id:
                family_ids.append(family_id)
        return family_ids

    def _family_by_id(self, family_id: str | None) -> dict[str, Any] | None:
        if not family_id:
            return None
        for family in self._family_records():
            if str(family.get("family_id", "")) == str(family_id):
                return family
        return None

    def _style_family_item(
        self,
        item: QtWidgets.QTableWidgetItem,
        flag: str,
    ) -> None:
        if flag == FAMILY_FLAG_APPROPRIATE:
            item.setBackground(QtGui.QColor("#dcfce7"))
        elif flag == FAMILY_FLAG_INAPPROPRIATE:
            item.setBackground(QtGui.QColor("#fee2e2"))

    def toggle_selected_family_flags(self) -> None:
        family_ids = self._selected_family_ids()
        if not family_ids:
            self._set_status("Select a peak family to flag.")
            return
        for family_id in family_ids:
            family = self._family_by_id(family_id)
            if family is None:
                continue
            flag = _family_flag(family)
            next_index = (FAMILY_FLAG_CYCLE.index(flag) + 1) % len(
                FAMILY_FLAG_CYCLE
            )
            family["user_flag"] = FAMILY_FLAG_CYCLE[next_index]
        self._sync_families(select_family_id=family_ids[-1])
        self._set_status(
            f"Updated review flag for {len(family_ids)} family(s)."
        )
        self.structureAnalysisChanged.emit(self.data_id)

    def set_selected_family_flag(self, flag: str) -> None:
        family_ids = self._selected_family_ids()
        if not family_ids:
            self._set_status("Select a peak family to flag.")
            return
        normalized = _family_flag({"user_flag": flag})
        for family_id in family_ids:
            family = self._family_by_id(family_id)
            if family is not None:
                family["user_flag"] = normalized
        self._sync_families(select_family_id=family_ids[-1])
        label = _family_flag_label(normalized).lower()
        self._set_status(f"Marked {len(family_ids)} family(s) as {label}.")
        self.structureAnalysisChanged.emit(self.data_id)

    def delete_selected_families(self) -> None:
        family_ids = set(self._selected_family_ids())
        if not family_ids:
            self._set_status("Select a peak family to delete.")
            return
        state = self._analysis_state()
        state["families"] = [
            family
            for family in self._family_records()
            if str(family.get("family_id", "")) not in family_ids
        ]
        if self.active_family_peak_id is not None:
            self.active_family_peak_id = None
        self._sync_families()
        self._set_status(f"Deleted {len(family_ids)} peak family record(s).")
        self.structureAnalysisChanged.emit(self.data_id)

    def validate_reviewed_families(self) -> None:
        families = self._family_records()
        if not families:
            self._set_status("No peak families are available to validate.")
            return
        reviewed_families = []
        removed_inappropriate = 0
        for family in families:
            if _family_flag(family) == FAMILY_FLAG_INAPPROPRIATE:
                removed_inappropriate += 1
                continue
            reviewed_families.append(family)
        cleaned_families, removed_duplicates = _deduplicate_families(
            reviewed_families
        )
        removed_total = removed_inappropriate + removed_duplicates
        if removed_total == 0:
            self._set_status(
                "Family validation found no inappropriate or duplicate "
                "family records."
            )
            return
        self._analysis_state()["families"] = cleaned_families
        self.active_family_peak_id = None
        self._sync_families()
        self._set_status(
            "Validated families: removed "
            f"{removed_inappropriate} inappropriate and "
            f"{removed_duplicates} duplicate family record(s)."
        )
        self.structureAnalysisChanged.emit(self.data_id)

    def add_peak_to_active_family(self, peak_id: str) -> bool:
        family_id = self._selected_family_id()
        family = self._family_by_id(family_id)
        if family is None:
            return False
        peak = next(
            (
                peak
                for peak in self._structure_peaks()
                if peak.peak_id == str(peak_id)
            ),
            None,
        )
        if peak is None:
            return False
        peak_ids = [str(value) for value in family.get("peak_ids", [])]
        labels = [str(value) for value in family.get("labels", [])]
        added = False
        if peak.peak_id not in peak_ids:
            peak_ids.append(peak.peak_id)
            labels.append(peak.label)
            family["peak_ids"] = peak_ids
            family["labels"] = labels
            _append_family_note(family, f"manually added {peak.label}")
            added = True
        if _is_manual_family(family):
            self._update_family_reference_from_members(family)
        if added:
            family["manual_edited"] = True
        self.active_family_peak_id = peak.peak_id
        self._sync_families(select_family_id=family_id)
        action = "Added" if added else "Selected"
        self._set_status(f"{action} {peak.label} in {family_id}.")
        self.structureAnalysisChanged.emit(self.data_id)
        return True

    def remove_active_family_ring(self) -> None:
        family_id = self._selected_family_id()
        family = self._family_by_id(family_id)
        if family is None:
            self._set_status("Select a peak family before removing a ring.")
            return
        if not self.active_family_peak_id:
            self._set_status(
                "Click a highlighted family ring before removing it."
            )
            return
        peak_ids = [str(value) for value in family.get("peak_ids", [])]
        if self.active_family_peak_id not in peak_ids:
            self._set_status(
                "The highlighted ring is not in the active family."
            )
            return
        peak_lookup = {
            peak.peak_id: peak.label for peak in self._structure_peaks()
        }
        removed_label = peak_lookup.get(
            self.active_family_peak_id,
            self.active_family_peak_id,
        )
        remaining_ids = [
            peak_id
            for peak_id in peak_ids
            if peak_id != self.active_family_peak_id
        ]
        if len(remaining_ids) < 2 and not _is_manual_family(family):
            state = self._analysis_state()
            state["families"] = [
                item
                for item in self._family_records()
                if str(item.get("family_id", "")) != str(family_id)
            ]
            self.active_family_peak_id = None
            self._sync_families()
            self._set_status(
                f"Removed {removed_label}; {family_id} was deleted because "
                "fewer than two rings remained."
            )
            self.structureAnalysisChanged.emit(self.data_id)
            return
        family["peak_ids"] = remaining_ids
        family["labels"] = [
            peak_lookup.get(peak_id, peak_id) for peak_id in remaining_ids
        ]
        family["manual_edited"] = True
        if _is_manual_family(family):
            self._update_family_reference_from_members(family)
        _append_family_note(family, f"manually removed {removed_label}")
        self.active_family_peak_id = None
        self._sync_families(select_family_id=family_id)
        self._set_status(f"Removed {removed_label} from {family_id}.")
        self.structureAnalysisChanged.emit(self.data_id)

    def _update_family_reference_from_members(
        self,
        family: dict[str, Any],
    ) -> None:
        peaks_by_id = {peak.peak_id: peak for peak in self._structure_peaks()}
        q_values = [
            peaks_by_id[peak_id].q_magnitude
            for peak_id in [str(value) for value in family.get("peak_ids", [])]
            if peak_id in peaks_by_id
        ]
        family["reference"] = float(np.mean(q_values)) if q_values else None

    def _sync_molecule_table(self) -> None:
        molecules = self._wyckoff_state().get("molecules", [])
        self.molecule_table.setRowCount(len(molecules))
        for row, molecule in enumerate(molecules):
            values = [
                molecule.get("label", ""),
                molecule.get("formula", ""),
                molecule.get("source", ""),
                molecule.get("path", ""),
            ]
            for column, value in enumerate(values):
                self.molecule_table.setItem(
                    row,
                    column,
                    QtWidgets.QTableWidgetItem(str(value)),
                )
        self.molecule_table.resizeColumnsToContents()

    def _sync_cif_table(self) -> None:
        records = self._wyckoff_state().get("generated_cifs", [])
        current_cif_id = self._selected_cif_id()
        selected_row = 0 if records else -1
        self.cif_table.blockSignals(True)
        try:
            self.cif_table.setRowCount(len(records))
            for row, record in enumerate(records):
                cif_id = record.get("cif_id")
                if current_cif_id is not None and cif_id == current_cif_id:
                    selected_row = row
                values = [
                    record.get("rank", row + 1),
                    record.get("candidate_id", ""),
                    _format_float(record.get("score")),
                    record.get("composition", ""),
                    record.get("status", ""),
                ]
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    item.setData(
                        QtCore.Qt.ItemDataRole.UserRole,
                        cif_id,
                    )
                    self.cif_table.setItem(row, column, item)
            self.cif_table.clearSelection()
            if selected_row >= 0:
                self.cif_table.selectRow(selected_row)
                self.cif_table.setCurrentCell(selected_row, 0)
        finally:
            self.cif_table.blockSignals(False)
        self.cif_table.resizeColumnsToContents()
        if hasattr(self, "open_cif_folder_button"):
            self.open_cif_folder_button.setEnabled(
                bool(records) and self.generated_cif_directory.exists()
            )
        self._sync_cif_visualizer()

    def _selected_cif_id(self) -> str | None:
        if not hasattr(self, "cif_table"):
            return None
        row = self.cif_table.currentRow()
        if row < 0:
            selected = self.cif_table.selectedItems()
            row = selected[0].row() if selected else -1
        if row < 0:
            return None
        item = self.cif_table.item(row, 0)
        if item is None:
            return None
        cif_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return str(cif_id) if cif_id is not None else None

    def _selected_cif_record(self) -> dict[str, Any] | None:
        records = [
            record
            for record in self._wyckoff_state().get("generated_cifs", [])
            if isinstance(record, dict)
        ]
        if not records:
            return None
        cif_id = self._selected_cif_id()
        if cif_id is not None:
            for record in records:
                if str(record.get("cif_id")) == cif_id:
                    return record
        return records[0]

    def _sync_cif_visualizer(self) -> None:
        if not hasattr(self, "cif_visualizer"):
            return
        self.cif_visualizer.set_record(self._selected_cif_record())

    def _sync_phase_options(self) -> None:
        labels = self._phase_labels()
        for combo in (self.phase_filter_combo, self.family_phase_combo):
            current = combo.currentText() or DEFAULT_PHASE_TAG
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems(labels)
                combo.setCurrentText(
                    current if current in labels else labels[0]
                )
            finally:
                combo.blockSignals(False)

    def _phase_labels(self) -> list[str]:
        labels = list(DEFAULT_PHASE_TAGS)
        for peak in self._structure_peaks():
            if peak.phase_tag and peak.phase_tag not in labels:
                labels.append(peak.phase_tag)
        return labels

    def _sync_candidate_phase_controls(self) -> None:
        current = self.phase_filter_combo.currentText()
        if current:
            self.family_phase_combo.setCurrentText(current)

    def _sync_wyckoff_candidate_combo(self) -> None:
        current = self.wyckoff_candidate_combo.currentData()
        self.wyckoff_candidate_combo.blockSignals(True)
        try:
            self.wyckoff_candidate_combo.clear()
            for candidate in self._candidate_records():
                label = (
                    f"{candidate.candidate_id} | {candidate.crystal_system} "
                    f"a={candidate.a:.3g}, b={candidate.b:.3g}, "
                    f"c={candidate.c:.3g}"
                )
                self.wyckoff_candidate_combo.addItem(
                    label, candidate.candidate_id
                )
            if current is not None:
                index = self.wyckoff_candidate_combo.findData(current)
                if index >= 0:
                    self.wyckoff_candidate_combo.setCurrentIndex(index)
        finally:
            self.wyckoff_candidate_combo.blockSignals(False)
        self._handle_wyckoff_candidate_changed()

    def _handle_wyckoff_candidate_changed(self) -> None:
        candidate = self._wyckoff_candidate()
        if candidate is None:
            return
        if candidate.crystal_system in WYCKOFF_CRYSTAL_SYSTEMS:
            self.wyckoff_system_combo.blockSignals(True)
            try:
                self.wyckoff_system_combo.setCurrentText(
                    candidate.crystal_system
                )
            finally:
                self.wyckoff_system_combo.blockSignals(False)
            self._sync_space_group_combo()

    def _sync_space_group_combo(self) -> None:
        current = self.space_group_combo.currentData()
        if isinstance(current, dict):
            current_number = current.get("number")
        else:
            current_number = current
        system = self.wyckoff_system_combo.currentText() or "Triclinic"
        options = wyckoff_space_group_options(system)
        self.space_group_combo.blockSignals(True)
        try:
            self.space_group_combo.clear()
            for option in options:
                self.space_group_combo.addItem(
                    f"{option.number} {option.symbol}",
                    option.as_dict(include_sites=False),
                )
            if current_number is not None:
                for index in range(self.space_group_combo.count()):
                    data = self.space_group_combo.itemData(index)
                    if isinstance(data, dict) and data.get("number") == int(
                        current_number
                    ):
                        self.space_group_combo.setCurrentIndex(index)
                        break
        finally:
            self.space_group_combo.blockSignals(False)
        self._sync_wyckoff_registry_tables()

    def _sync_wyckoff_registry_tables(self) -> None:
        space_group_number = self._selected_space_group_number()
        if space_group_number is None:
            return
        option = wyckoff_space_group_option(space_group_number)
        self._wyckoff_state()["space_group"] = option.as_dict(
            include_sites=False
        )
        self._wyckoff_state()["crystal_system"] = option.crystal_system
        self._wyckoff_state()["registered_sites"] = [
            site.as_dict() for site in option.sites
        ]

        self.wyckoff_site_table.setRowCount(len(option.sites))
        for row, site in enumerate(option.sites):
            values = [
                site.site_label,
                site.multiplicity,
                site.parameter_count,
                f"{option.number} {option.symbol}",
            ]
            for column, value in enumerate(values):
                self.wyckoff_site_table.setItem(
                    row,
                    column,
                    QtWidgets.QTableWidgetItem(str(value)),
                )

        site_count = self.wyckoff_site_count_spin.value()
        total = wyckoff_combination_count(
            space_group_number,
            site_count=site_count,
        )
        ordered_total = wyckoff_combination_count(
            space_group_number,
            site_count=site_count,
            ordered=True,
        )
        combinations = wyckoff_site_combinations(
            space_group_number,
            site_count=site_count,
            max_combinations=WYCKOFF_COMBINATION_DISPLAY_LIMIT,
        )
        self._wyckoff_state()["registered_combination_count"] = total
        self._wyckoff_state()["registered_assignment_count"] = ordered_total
        self._wyckoff_state()["displayed_combinations"] = combinations
        self.wyckoff_combination_table.setRowCount(len(combinations))
        for row, combination in enumerate(combinations):
            labels = combination.get("site_labels", [])
            values = [
                combination.get("combination_id", ""),
                combination.get("total_multiplicity", ""),
                combination.get("free_parameter_count", ""),
                ", ".join(labels),
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setData(
                    QtCore.Qt.ItemDataRole.UserRole,
                    combination.get("combination_id"),
                )
                self.wyckoff_combination_table.setItem(row, column, item)
        self.wyckoff_site_table.resizeColumnsToContents()
        self.wyckoff_combination_table.resizeColumnsToContents()
        shown = len(combinations)
        self.wyckoff_registry_status.setText(
            f"{option.crystal_system}: {len(option.sites)} sites, "
            f"{total} combinations, {ordered_total} assignments registered"
            + (f" ({shown} shown)" if shown < total else "")
        )

    def _selected_space_group_number(self) -> int | None:
        data = self.space_group_combo.currentData()
        if isinstance(data, dict):
            number = data.get("number")
            return int(number) if number is not None else None
        return int(data) if data is not None else None

    def _selected_wyckoff_combinations(self) -> list[dict[str, Any]]:
        space_group_number = self._selected_space_group_number()
        if space_group_number is None:
            return []
        selected_rows = {
            index.row()
            for index in self.wyckoff_combination_table.selectedIndexes()
        }
        displayed = self._wyckoff_state().get("displayed_combinations", [])
        if selected_rows:
            return [
                dict(displayed[row])
                for row in sorted(selected_rows)
                if row < len(displayed)
            ]
        return wyckoff_site_combinations(
            space_group_number,
            site_count=self.wyckoff_site_count_spin.value(),
            max_combinations=max(self.cif_count_spin.value() * 20, 20),
            ordered=True,
        )

    def _handle_peak_item_changed(
        self, item: QtWidgets.QTableWidgetItem
    ) -> None:
        if self._syncing_table:
            return
        peak_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if peak_id is None:
            return
        column = item.column()
        peaks = self._structure_peaks()
        for peak in peaks:
            if peak.peak_id != str(peak_id):
                continue
            if column == COL_QXY:
                peak.qxy = _safe_float(item.text(), peak.qxy)
                peak.source = "user edit"
                peak.metadata["user_edited_center"] = True
            elif column == COL_QZ:
                peak.qz = _safe_float(item.text(), peak.qz)
                peak.source = "user edit"
                peak.metadata["user_edited_center"] = True
            elif column == COL_HKL:
                peak.hkl_label = item.text().strip()
            elif column == COL_INCLUDE:
                peak.include = (
                    item.checkState() == QtCore.Qt.CheckState.Checked
                )
            elif column == COL_NOTES:
                peak.notes = item.text().strip()
            break
        self._store_peaks(peaks)
        self._sync_peak_plot()
        self.structureAnalysisChanged.emit(self.data_id)

    def _handle_peak_selection(self) -> None:
        if self._syncing_peak_selection:
            return
        row = self.peak_table.currentRow()
        if row < 0:
            return
        item = self.peak_table.item(row, COL_PEAK_ID)
        if item is None:
            return
        peak_id = str(item.data(QtCore.Qt.ItemDataRole.UserRole))
        self.active_peak_id = peak_id
        self._sync_peak_plot()
        self._ensure_peak_visible(peak_id)

    def _handle_peak_plot_clicked(
        self,
        _scatter: Any,
        points: list[Any],
        _event: Any,
    ) -> None:
        if not points:
            return
        payload = points[0].data()
        if isinstance(payload, dict):
            peak_id = str(payload.get("peak_id", ""))
        else:
            peak_id = str(payload or "")
        if peak_id:
            if isinstance(self.plot_widget, QtWidgets.QWidget):
                self.plot_widget.setFocus()
            if self._family_editing_active() and self._selected_family_id():
                if self.add_peak_to_active_family(peak_id):
                    return
            self._select_peak_by_id(peak_id, scroll_table=True)

    def _handle_family_plot_clicked(
        self,
        _scatter: Any,
        points: list[Any],
        _event: Any,
    ) -> None:
        if not points:
            return
        payload = points[0].data()
        if isinstance(payload, dict):
            peak_id = str(payload.get("peak_id", ""))
        else:
            peak_id = str(payload or "")
        if not peak_id:
            return
        if isinstance(self.plot_widget, QtWidgets.QWidget):
            self.plot_widget.setFocus()
        if self._family_editing_active():
            family_id = self._selected_family_id()
            family = self._family_by_id(family_id)
            if family is not None:
                family_peak_ids = {
                    str(value) for value in family.get("peak_ids", [])
                }
                if peak_id not in family_peak_ids:
                    if self.add_peak_to_active_family(peak_id):
                        return
        self.active_family_peak_id = peak_id
        self._select_peak_by_id(peak_id, scroll_table=True)
        self._set_status(
            "Selected family ring. Press R to remove it from the active family."
        )

    def _handle_plot_scene_clicked(self, event: Any) -> None:
        if not (
            self._family_editing_active()
            and self._selected_family_id()
            and self.view_box is not None
        ):
            return
        button = event.button() if hasattr(event, "button") else None
        if button is not None and button != QtCore.Qt.MouseButton.LeftButton:
            return
        peak_id = self._peak_id_near_scene_position(event.scenePos())
        if not peak_id:
            return
        if self.add_peak_to_active_family(peak_id):
            if hasattr(event, "accept"):
                event.accept()

    def _peak_id_near_scene_position(
        self,
        scene_pos: QtCore.QPointF,
        *,
        max_distance_px: float = 18.0,
    ) -> str | None:
        if self.view_box is None:
            return None
        best_peak_id: str | None = None
        best_distance = float(max_distance_px)
        for peak in self._structure_peaks():
            peak_scene_pos = self.view_box.mapViewToScene(
                QtCore.QPointF(float(peak.qxy), float(peak.qz))
            )
            distance = math.hypot(
                float(peak_scene_pos.x() - scene_pos.x()),
                float(peak_scene_pos.y() - scene_pos.y()),
            )
            if distance <= best_distance:
                best_distance = distance
                best_peak_id = peak.peak_id
        return best_peak_id

    def _family_editing_active(self) -> bool:
        return (
            hasattr(self, "analysis_tabs")
            and hasattr(self, "family_tab")
            and self.analysis_tabs.currentWidget() is self.family_tab
        )

    def _select_peak_by_id(
        self,
        peak_id: str,
        *,
        scroll_table: bool,
    ) -> bool:
        for row in range(self.peak_table.rowCount()):
            item = self.peak_table.item(row, COL_PEAK_ID)
            if item is None:
                continue
            if str(item.data(QtCore.Qt.ItemDataRole.UserRole)) != peak_id:
                continue
            self._syncing_peak_selection = True
            try:
                self.peak_table.selectRow(row)
                if scroll_table:
                    self.peak_table.scrollToItem(
                        item,
                        QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
                    )
            finally:
                self._syncing_peak_selection = False
            self.active_peak_id = peak_id
            self._sync_peak_plot()
            self._ensure_peak_visible(peak_id)
            return True
        return False

    def _ensure_peak_visible(self, peak_id: str) -> None:
        if pg is None or self.plot_widget is None:
            return
        peak = next(
            (
                item
                for item in self._structure_peaks()
                if item.peak_id == peak_id
            ),
            None,
        )
        if peak is None:
            return
        try:
            (x_min, x_max), (y_min, y_max) = self.plot_widget.viewRange()
        except Exception:
            return
        if x_min <= peak.qxy <= x_max and y_min <= peak.qz <= y_max:
            return
        x_span = max(float(x_max - x_min), 1.0e-9)
        y_span = max(float(y_max - y_min), 1.0e-9)
        self.plot_widget.setRange(
            xRange=(peak.qxy - x_span / 2.0, peak.qxy + x_span / 2.0),
            yRange=(peak.qz - y_span / 2.0, peak.qz + y_span / 2.0),
            padding=0.0,
        )

    def _set_peak_phase(self, peak_id: str, phase_tag: str) -> None:
        if self._syncing_table:
            return
        peaks = self._structure_peaks()
        for peak in peaks:
            if peak.peak_id == peak_id:
                peak.phase_tag = phase_tag.strip() or DEFAULT_PHASE_TAG
                if peak.phase_tag in {PHASE_REJECTED, PHASE_FORBIDDEN}:
                    peak.include = False
                break
        self._store_peaks(peaks)
        self._sync_phase_options()
        self._sync_peak_table()
        self._sync_peak_plot()
        self.structureAnalysisChanged.emit(self.data_id)

    def _candidate_config(self) -> CandidateSearchConfig:
        selected = self.guess_system_combo.currentText()
        if selected in SIMPLE_CRYSTAL_SYSTEM_ORDER:
            selected_index = SIMPLE_CRYSTAL_SYSTEM_ORDER.index(selected)
            systems = SIMPLE_CRYSTAL_SYSTEM_ORDER[: selected_index + 1]
        else:
            systems = SIMPLE_CRYSTAL_SYSTEM_ORDER
        return CandidateSearchConfig(
            crystal_systems=systems,
            hkl_max=self.hkl_max.value(),
            q_tolerance=self.q_tolerance.value(),
            relative_tolerance=self.relative_tolerance.value(),
            grid_points=self.grid_points.value(),
            phase_tag=str(self.phase_filter_combo.currentText()),
            orientation_quaternion=self._current_crystal_orientation_quaternion(),
        )

    def _best_guess_parameters(self) -> CrystalOverlayParameters:
        return CrystalOverlayParameters(
            crystal_system=self.guess_system_combo.currentText(),
            a=self.lattice_a.value(),
            b=self.lattice_b.value(),
            c=self.lattice_c.value(),
            alpha=self.lattice_alpha.value(),
            beta=self.lattice_beta.value(),
            gamma=self.lattice_gamma.value(),
            h_max=self.hkl_max.value(),
            k_max=self.hkl_max.value(),
            l_max=self.hkl_max.value(),
            orientation_quaternion=(
                self._current_crystal_orientation_quaternion()
                or (0.0, 0.0, 0.0, 1.0)
            ),
        ).constrained()

    def _current_crystal_orientation_quaternion(
        self,
    ) -> tuple[float, float, float, float] | None:
        overlays = self.project.analysis_results.get("crystal_overlays", {})
        state = overlays.get(self.data_id, {})
        parameters = state.get("parameters", {})
        orientation = parameters.get("orientation_quaternion")
        if orientation is None:
            return None
        return tuple(
            float(value) for value in normalize_quaternion(orientation)
        )

    def _selected_candidate(
        self,
        *,
        fallback_first: bool = True,
    ) -> LatticeCandidate | None:
        candidates = {
            item.candidate_id: item for item in self._candidate_records()
        }
        candidate_id = self._selected_candidate_id()
        if candidate_id in candidates:
            return candidates[str(candidate_id)]
        if fallback_first:
            return next(iter(candidates.values()), None)
        return None

    def _selected_candidate_id(self) -> str | None:
        row = self.candidate_table.currentRow()
        if row < 0:
            selected = self.candidate_table.selectedItems()
            row = selected[0].row() if selected else -1
        if row < 0:
            return None
        item = self.candidate_table.item(row, 0)
        if item is None:
            return None
        candidate_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return str(candidate_id) if candidate_id is not None else None

    def _wyckoff_candidate(self) -> LatticeCandidate | None:
        candidate_id = self.wyckoff_candidate_combo.currentData()
        candidates = {
            item.candidate_id: item for item in self._candidate_records()
        }
        if candidate_id in candidates:
            return candidates[str(candidate_id)]
        return self._selected_candidate()

    def _load_custom_molecule(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Reference Molecule PDB",
            "",
            "PDB Files (*.pdb);;All Files (*)",
        )
        if not path:
            return
        label = QtCore.QFileInfo(path).baseName()
        self._wyckoff_state().setdefault("molecules", []).append(
            {
                "label": label,
                "formula": "",
                "name": label,
                "source": "custom PDB",
                "path": path,
            }
        )
        self._sync_molecule_table()
        self.structureAnalysisChanged.emit(self.data_id)

    def _publish_generated_cifs(self, records: list[dict[str, Any]]) -> None:
        generated = self.project.reference_cifs.setdefault("generated", {})
        for record in records:
            cif_id = str(record["cif_id"])
            payload = {
                **record,
                "data_id": self.data_id,
                "source": "Structure Analysis Wyckoff Mapping",
            }
            generated[cif_id] = payload
            self.project.structures[cif_id] = {
                "structure_id": cif_id,
                "data_id": self.data_id,
                "source": "structure_analysis_generated_cif",
                "cif_text": record.get("cif_text", ""),
                "path": record.get("path"),
                "candidate_id": record.get("candidate_id"),
                "score": record.get("score"),
                "space_group": record.get("space_group"),
                "wyckoff_combination": record.get("wyckoff_combination"),
                "wyckoff_assignments": record.get("wyckoff_assignments"),
                "structure_variant": record.get(
                    "structure_variant",
                    STRUCTURE_OUTPUT_FULL,
                ),
                "parent_cif_id": record.get("parent_cif_id"),
            }

    def _write_generated_cif_files(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        directory = self.generated_cif_directory
        directory.mkdir(parents=True, exist_ok=True)
        self._wyckoff_state()["generated_cif_directory"] = str(directory)
        written_records: list[dict[str, Any]] = []
        for record in records:
            payload = dict(record)
            cif_text = str(payload.get("cif_text") or "")
            cif_id = str(payload.get("cif_id") or payload.get("id") or "")
            filename = _safe_generated_cif_filename(cif_id or "generated_cif")
            path = directory / filename
            if cif_text.strip():
                if (
                    not path.exists()
                    or path.read_text(encoding="utf-8") != cif_text
                ):
                    path.write_text(cif_text, encoding="utf-8")
                payload["path"] = str(path)
                payload["structure_path"] = str(path)
            written_records.append(payload)
        return written_records

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)


def _estimate_candidate_guess_seconds(
    peaks: list[StructurePeak],
    config: CandidateSearchConfig,
) -> float:
    peak_count = max(len(peaks), 1)
    system_count = max(len(tuple(config.crystal_systems)), 1)
    grid_points = max(int(config.grid_points), 1)
    hkl_extent = max(int(config.hkl_max), 1)
    projected_factor = 1.35 if config.enable_projected_axis_search else 1.0
    orientation_factor = (
        1.2 if config.orientation_quaternion is not None else 1.0
    )
    work_units = (
        peak_count
        * system_count
        * grid_points
        * hkl_extent
        * projected_factor
        * orientation_factor
    )
    candidate_refinement = max(int(config.max_candidates), 1) * 0.2
    return min(
        max(0.5 + work_units / 2200.0 + candidate_refinement, 1.0), 180.0
    )


def _candidate_guess_progress_text(
    elapsed_seconds: float,
    estimate_seconds: float,
) -> str:
    return (
        "Guessing candidate structures...\n"
        f"Elapsed: {_format_duration(elapsed_seconds)}\n"
        f"Estimated time: {_format_duration(estimate_seconds)}"
    )


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 10.0:
        return f"{seconds:.1f}s"
    if seconds < 60.0:
        return f"{round(seconds):.0f}s"
    minutes = int(seconds // 60)
    remainder = int(round(seconds % 60))
    if remainder == 60:
        minutes += 1
        remainder = 0
    return f"{minutes}m {remainder:02d}s"


def _lattice_spinbox(value: float) -> QtWidgets.QDoubleSpinBox:
    return _double_spinbox(value, 0.0001, 1000.0, 0.1, decimals=4)


def _angle_spinbox(value: float) -> QtWidgets.QDoubleSpinBox:
    return _double_spinbox(value, 0.01, 179.99, 1.0, decimals=3)


def _atom_count_spinbox(value: float) -> QtWidgets.QDoubleSpinBox:
    spinbox = _double_spinbox(value, 0.0, 999.0, 1.0, decimals=4)
    spinbox.setToolTip(
        qt_tooltip(
            "Optional stoichiometric amount for this free atom. Leave as 1 "
            "for a simple one-site atom entry."
        )
    )
    return spinbox


def _occupancy_spinbox(value: float) -> QtWidgets.QDoubleSpinBox:
    spinbox = _double_spinbox(value, 0.0, 1.0, 0.05, decimals=4)
    spinbox.setToolTip(
        qt_tooltip(
            "Optional site occupancy fraction written to the CIF atom-site "
            "occupancy column."
        )
    )
    return spinbox


def _family_confidence(family: dict[str, Any]) -> float:
    try:
        value = float(family.get("confidence", 0.0))
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(1.0, value))


def _family_flag(family: dict[str, Any]) -> str:
    flag = str(family.get("user_flag", "")).strip().lower()
    if flag in {FAMILY_FLAG_APPROPRIATE, FAMILY_FLAG_INAPPROPRIATE}:
        return flag
    return ""


def _family_flag_label(flag: str) -> str:
    if flag == FAMILY_FLAG_APPROPRIATE:
        return "Appropriate"
    if flag == FAMILY_FLAG_INAPPROPRIATE:
        return "Inappropriate"
    return "Unreviewed"


def _is_manual_family(family: dict[str, Any]) -> bool:
    return (
        str(family.get("source", "")).strip().lower() == FAMILY_SOURCE_MANUAL
        or str(family.get("kind", "")).strip().lower() == "custom"
    )


def _unique_family_id(
    requested: str,
    families: list[dict[str, Any]],
) -> str:
    base = " ".join(str(requested).strip().split()) or "Custom family"
    existing = {str(family.get("family_id", "")) for family in families}
    if base not in existing:
        return base
    index = 2
    while f"{base} {index}" in existing:
        index += 1
    return f"{base} {index}"


def _deduplicate_families(
    families: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    order: list[tuple[str, Any]] = []
    best_by_key: dict[tuple[str, Any], dict[str, Any]] = {}
    removed = 0
    for index, family in enumerate(families):
        key = _family_duplicate_key(family)
        if key is None:
            key = ("family", str(family.get("family_id", index)))
        else:
            key = ("peaks", key)
        if key not in best_by_key:
            order.append(key)
            best_by_key[key] = family
            continue
        removed += 1
        if _family_validation_rank(family) > _family_validation_rank(
            best_by_key[key]
        ):
            best_by_key[key] = family
    return [best_by_key[key] for key in order], removed


def _family_duplicate_key(
    family: dict[str, Any],
) -> tuple[str, tuple[str, ...]] | None:
    peak_ids = tuple(
        sorted(
            {
                str(peak_id).strip()
                for peak_id in family.get("peak_ids", [])
                if str(peak_id).strip()
            }
        )
    )
    if not peak_ids:
        return None
    phase = str(family.get("phase_tag", "")).strip().lower()
    return phase, peak_ids


def _family_validation_rank(family: dict[str, Any]) -> tuple[int, int, float]:
    flag = _family_flag(family)
    flag_score = 2 if flag == FAMILY_FLAG_APPROPRIATE else 1
    manual_score = 1 if _is_manual_family(family) else 0
    return flag_score, manual_score, _family_confidence(family)


def _append_family_note(family: dict[str, Any], note: str) -> None:
    existing = str(family.get("notes", "")).strip()
    if note in existing:
        return
    family["notes"] = f"{existing}; {note}" if existing else note


def _structure_output_records(
    records: list[dict[str, Any]],
    mode: str,
    *,
    keep_elements: list[str],
) -> list[dict[str, Any]]:
    if mode == STRUCTURE_OUTPUT_FULL:
        return records
    keep = {
        element
        for element in (_canonical_cif_element(item) for item in keep_elements)
        if element
    }
    if not keep:
        return records
    scaffold_records = [
        _inorganic_scaffold_record(record, keep_elements=keep)
        for record in records
    ]
    if mode == STRUCTURE_OUTPUT_SCAFFOLD_ONLY:
        return scaffold_records
    if mode == STRUCTURE_OUTPUT_FULL_PLUS_SCAFFOLD:
        paired: list[dict[str, Any]] = []
        for full_record, scaffold_record in zip(records, scaffold_records):
            paired.append(full_record)
            paired.append(scaffold_record)
        return paired
    return records


def _inorganic_scaffold_record(
    record: dict[str, Any],
    *,
    keep_elements: set[str],
) -> dict[str, Any]:
    source_cif_text = str(record.get("cif_text") or "")
    cif_text, composition = _filter_cif_text_to_elements(
        source_cif_text,
        keep_elements,
    )
    parent_cif_id = str(record.get("cif_id") or record.get("id") or "cif")
    organic_note = _organic_scaffold_note(record, source_cif_text)
    cif_text = _annotate_inorganic_scaffold_cif(
        cif_text,
        parent_cif_id,
        organic_note=organic_note,
    )
    payload = dict(record)
    payload.update(
        {
            "cif_id": f"{parent_cif_id}_inorganic_scaffold",
            "parent_cif_id": parent_cif_id,
            "structure_variant": STRUCTURE_OUTPUT_SCAFFOLD_ONLY,
            "status": "inorganic scaffold",
            "composition": _cif_formula_sum(composition),
            "composition_elements": dict(composition),
            "molecules": [],
            "coordinate_model": (
                f"{record.get('coordinate_model', 'generated')}"
                "_inorganic_scaffold"
            ),
            "cif_text": cif_text,
        }
    )
    payload.pop("path", None)
    payload.pop("structure_path", None)
    return payload


def _annotate_inorganic_scaffold_cif(
    cif_text: str,
    parent_cif_id: str,
    *,
    organic_note: str,
) -> str:
    lines = cif_text.splitlines()
    annotated: list[str] = []
    inserted_parent = False
    for line in lines:
        annotated.append(line)
        if line.startswith("data_") and not inserted_parent:
            annotated.append("# structure variant: inorganic scaffold only")
            annotated.append(f"# parent generated CIF: {parent_cif_id}")
            annotated.append(f"# organic molecules to add: {organic_note}")
            inserted_parent = True
    for index, line in enumerate(annotated):
        if line.startswith("# molecular species:"):
            annotated[index] = (
                "# molecular species: inorganic scaffold only; planned "
                f"organics: {organic_note}"
            )
            break
    return "\n".join(annotated).rstrip() + "\n"


def _organic_scaffold_note(record: dict[str, Any], cif_text: str) -> str:
    pieces: list[str] = []
    seen: set[str] = set()
    molecules = record.get("molecules", [])
    if isinstance(molecules, list):
        for molecule in molecules:
            if not isinstance(molecule, dict):
                continue
            label = str(
                molecule.get("label") or molecule.get("name") or ""
            ).strip()
            formula = str(molecule.get("formula") or "").strip()
            if not label:
                continue
            piece = f"{label} ({formula})" if formula else label
            if piece not in seen:
                pieces.append(piece)
                seen.add(piece)
    if pieces:
        return ", ".join(pieces)
    for line in cif_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("# molecular species:"):
            continue
        note = stripped.split(":", 1)[1].strip()
        if note and note.lower() not in {"none", "unspecified"}:
            return note
    return "none recorded"


def _filter_cif_text_to_elements(
    cif_text: str,
    keep_elements: set[str],
) -> tuple[str, dict[str, float]]:
    output: list[str] = []
    composition: dict[str, float] = {}
    in_loop = False
    atom_loop = False
    headers: list[str] = []
    type_index: int | None = None
    occupancy_index: int | None = None

    for line in cif_text.splitlines():
        stripped = line.strip()
        if stripped == "loop_":
            in_loop = True
            atom_loop = False
            headers = []
            type_index = None
            occupancy_index = None
            output.append(line)
            continue
        if in_loop and stripped.startswith("_"):
            header = stripped.split()[0].lower()
            headers.append(header)
            atom_loop = atom_loop or header.startswith("_atom_site_")
            output.append(line)
            continue
        if in_loop and atom_loop and stripped and not stripped.startswith("#"):
            parts = _split_cif_row(stripped)
            if type_index is None:
                try:
                    type_index = headers.index("_atom_site_type_symbol")
                except ValueError:
                    type_index = 1
                try:
                    occupancy_index = headers.index("_atom_site_occupancy")
                except ValueError:
                    occupancy_index = None
            if len(parts) > type_index:
                element = _canonical_cif_element(parts[type_index])
                if element in keep_elements:
                    output.append(line)
                    occupancy = (
                        _float_or_default(parts[occupancy_index], 1.0)
                        if occupancy_index is not None
                        and len(parts) > occupancy_index
                        else 1.0
                    )
                    composition[element] = (
                        composition.get(element, 0.0) + occupancy
                    )
            continue
        output.append(line)

    formula = _cif_formula_sum(composition)
    if formula:
        for index, line in enumerate(output):
            if line.strip().lower().startswith("_chemical_formula_sum"):
                output[index] = f"_chemical_formula_sum '{formula}'"
                break
    return "\n".join(output).rstrip() + "\n", composition


def _split_cif_row(line: str) -> list[str]:
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def _canonical_cif_element(value: Any) -> str:
    token = str(value).strip().strip("'\"")
    letters = "".join(character for character in token if character.isalpha())
    if not letters:
        return ""
    return letters[0].upper() + letters[1:].lower()


def _cif_formula_sum(composition: dict[str, float]) -> str:
    pieces = []
    for element in sorted(composition):
        count = composition[element]
        if count <= 1.0e-9:
            continue
        if abs(count - round(count)) < 1.0e-9:
            suffix = "" if int(round(count)) == 1 else str(int(round(count)))
        else:
            suffix = f"{count:g}"
        pieces.append(f"{element}{suffix}")
    return " ".join(pieces)


def _default_generated_cif_directory(project_path: Path | None) -> Path:
    if project_path is not None:
        return project_path.parent / project_path.stem / "generated_cifs"
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "example" / "projects"
        if candidate.exists():
            return candidate / "generated_cifs"
    return Path.cwd() / "generated_cifs"


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


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _atom_occupancy_summary(atom_specs: list[dict[str, Any]]) -> str:
    pieces = []
    for spec in atom_specs:
        element = str(spec.get("element") or "").strip()
        shared_site = str(spec.get("shared_site") or "").strip()
        occupancy = _float_or_default(spec.get("occupancy", 1.0), 1.0)
        if not element:
            continue
        details = []
        if shared_site:
            details.append(f"shared site {shared_site}")
        if abs(occupancy - 1.0) > 1.0e-9:
            details.append(f"occupancy {occupancy:g}")
        if details:
            pieces.append(f"{element}: " + ", ".join(details))
    return "; ".join(pieces)


def _double_spinbox(
    value: float,
    minimum: float,
    maximum: float,
    step: float,
    *,
    decimals: int = 4,
) -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox()
    spinbox.setRange(minimum, maximum)
    spinbox.setDecimals(decimals)
    spinbox.setSingleStep(step)
    spinbox.setValue(value)
    spinbox.setMaximumWidth(120)
    return spinbox


def _int_spinbox(value: int, minimum: int, maximum: int) -> QtWidgets.QSpinBox:
    spinbox = QtWidgets.QSpinBox()
    spinbox.setRange(minimum, maximum)
    spinbox.setValue(value)
    spinbox.setMaximumWidth(96)
    return spinbox


def _phase_color(phase_tag: str) -> QtGui.QColor:
    if phase_tag == DEFAULT_PHASE_TAG:
        return QtGui.QColor("#2563eb")
    if phase_tag == PHASE_UNASSIGNED:
        return QtGui.QColor("#9ca3af")
    if phase_tag == PHASE_REJECTED:
        return QtGui.QColor("#111827")
    if phase_tag == PHASE_FORBIDDEN:
        return QtGui.QColor("#7f1d1d")
    if "secondary" in phase_tag.lower() or phase_tag.endswith("2"):
        return QtGui.QColor("#dc2626")
    if "gap" in phase_tag.lower():
        return QtGui.QColor("#f59e0b")
    return QtGui.QColor("#7c3aed")


def _peak_tip(*, x: float, y: float, data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    hkl = data.get("hkl_label") or ""
    phase = data.get("phase_tag") or ""
    label = data.get("label") or data.get("peak_id") or "Peak"
    suffix = f" | {hkl}" if hkl else ""
    return qt_tooltip(
        "<br>".join(
            [
                f"{label}{suffix}",
                f"{QXY_HTML}={float(x):.4g}, {QZ_HTML}={float(y):.4g}",
                f"{phase}",
            ]
        )
    )


def _parse_generated_cif(
    cif_text: str,
    *,
    cif_id: str = "",
) -> _ParsedCif | None:
    cell = {
        "a": 1.0,
        "b": 1.0,
        "c": 1.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    }
    space_group = ""
    atoms: list[_CifAtom] = []
    lines = [line.strip() for line in cif_text.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("data_") and not cif_id:
            cif_id = line.removeprefix("data_").strip()
        elif line.startswith("_symmetry_space_group_name_H-M"):
            space_group = _cif_value(line)
        elif line.startswith("_cell_length_a"):
            cell["a"] = _cif_float(_cif_value(line), 1.0)
        elif line.startswith("_cell_length_b"):
            cell["b"] = _cif_float(_cif_value(line), 1.0)
        elif line.startswith("_cell_length_c"):
            cell["c"] = _cif_float(_cif_value(line), 1.0)
        elif line.startswith("_cell_angle_alpha"):
            cell["alpha"] = _cif_float(_cif_value(line), 90.0)
        elif line.startswith("_cell_angle_beta"):
            cell["beta"] = _cif_float(_cif_value(line), 90.0)
        elif line.startswith("_cell_angle_gamma"):
            cell["gamma"] = _cif_float(_cif_value(line), 90.0)

    index = 0
    while index < len(lines):
        if lines[index] != "loop_":
            index += 1
            continue
        index += 1
        headers: list[str] = []
        while index < len(lines) and lines[index].startswith("_"):
            headers.append(lines[index])
            index += 1
        if not any(header.startswith("_atom_site_") for header in headers):
            continue
        header_map = {header: column for column, header in enumerate(headers)}
        while index < len(lines):
            line = lines[index]
            if (
                line == "loop_"
                or line.startswith("_")
                or line.startswith("data_")
            ):
                break
            if not line.startswith("#"):
                atom = _cif_atom_from_tokens(_cif_tokens(line), header_map)
                if atom is not None:
                    atoms.append(atom)
            index += 1
    if not atoms and not cif_text.strip():
        return None
    return _ParsedCif(
        cif_id=cif_id or "generated_cif",
        space_group=space_group,
        a=cell["a"],
        b=cell["b"],
        c=cell["c"],
        alpha=cell["alpha"],
        beta=cell["beta"],
        gamma=cell["gamma"],
        atoms=tuple(atoms),
    )


def _cif_atom_from_tokens(
    tokens: list[str],
    header_map: dict[str, int],
) -> _CifAtom | None:
    def value(header: str, fallback: str = "") -> str:
        column = header_map.get(header)
        if column is None or column >= len(tokens):
            return fallback
        return tokens[column]

    label = value("_atom_site_label")
    symbol = value("_atom_site_type_symbol", label)
    if not label and not symbol:
        return None
    return _CifAtom(
        label=label or symbol,
        symbol=symbol or label,
        fract_x=_cif_float(value("_atom_site_fract_x"), 0.0),
        fract_y=_cif_float(value("_atom_site_fract_y"), 0.0),
        fract_z=_cif_float(value("_atom_site_fract_z"), 0.0),
        occupancy=_cif_float(value("_atom_site_occupancy"), 1.0),
    )


def _cif_value(line: str) -> str:
    tokens = _cif_tokens(line)
    if len(tokens) <= 1:
        return ""
    return " ".join(tokens[1:]).strip("'\"")


def _cif_tokens(line: str) -> list[str]:
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        return line.split()


def _cif_float(value: Any, fallback: float) -> float:
    text = str(value).strip().strip("'\"")
    if "(" in text:
        text = text.split("(", 1)[0]
    try:
        number = float(text)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) else fallback


def _cif_lattice_summary(parsed: _ParsedCif) -> str:
    lengths = (
        f"a={_format_float(parsed.a)}, "
        f"b={_format_float(parsed.b)}, "
        f"c={_format_float(parsed.c)} A"
    )
    angles = (
        f"alpha={_format_float(parsed.alpha)}, "
        f"beta={_format_float(parsed.beta)}, "
        f"gamma={_format_float(parsed.gamma)} deg"
    )
    group = f" | SG {parsed.space_group}" if parsed.space_group else ""
    return f"{lengths} | {angles}{group}"


def _cif_lattice_matrix(parsed: _ParsedCif) -> np.ndarray:
    a = max(float(parsed.a), 1e-9)
    b = max(float(parsed.b), 1e-9)
    c = max(float(parsed.c), 1e-9)
    alpha = math.radians(float(parsed.alpha))
    beta = math.radians(float(parsed.beta))
    gamma = math.radians(float(parsed.gamma))
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1e-9:
        sin_gamma = 1e-9
    c_x = c * math.cos(beta)
    c_y = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma))
    c_y /= sin_gamma
    c_z_sq = max(c * c - c_x * c_x - c_y * c_y, 1e-12)
    return np.asarray(
        [
            [a, 0.0, 0.0],
            [b * math.cos(gamma), b * sin_gamma, 0.0],
            [c_x, c_y, math.sqrt(c_z_sq)],
        ],
        dtype=float,
    )


def _cif_unit_cell_corners(lattice: np.ndarray) -> np.ndarray:
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


def _cif_unit_cell_edges() -> tuple[tuple[int, int], ...]:
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


def _cif_project_points(points: np.ndarray, mode: str) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if mode == "ab":
        return points[:, [0, 1]]
    if mode == "ac":
        return points[:, [0, 2]]
    if mode == "bc":
        return points[:, [1, 2]]
    return np.column_stack(
        [
            points[:, 0] + 0.36 * points[:, 1],
            points[:, 2] + 0.22 * points[:, 1],
        ]
    )


def _set_cif_plot_labels(plot_widget: Any, mode: str) -> None:
    labels = {
        "ab": ("a", "b"),
        "ac": ("a", "c"),
        "bc": ("b", "c"),
    }
    bottom, left = labels.get(mode, ("x", "z"))
    plot_widget.setLabel("bottom", bottom, units="A")
    plot_widget.setLabel("left", left, units="A")


def _cif_atom_tip(*, x: float, y: float, data: Any) -> str:
    if not isinstance(data, _CifAtom):
        return ""
    return qt_tooltip(
        "<br>".join(
            [
                f"{data.label} ({_cif_element_symbol(data.symbol)})",
                f"plot x={float(x):.4g}, y={float(y):.4g}",
                (
                    "fract "
                    f"({data.fract_x:.4g}, {data.fract_y:.4g}, "
                    f"{data.fract_z:.4g})"
                ),
            ]
        )
    )


def _cif_atom_color(symbol: str) -> str:
    colors = {
        "H": "#ffffff",
        "C": "#909090",
        "N": "#3050f8",
        "O": "#ff0d0d",
        "F": "#90e050",
        "P": "#ff8000",
        "S": "#ffff30",
        "Cl": "#1ff01f",
        "Br": "#a62929",
        "I": "#940094",
        "Li": "#cc80ff",
        "Na": "#ab5cf2",
        "K": "#8f40d4",
        "Si": "#f0c8a0",
        "Pb": "#575961",
        "Bi": "#9e4fb5",
    }
    return colors.get(_cif_element_symbol(symbol), "#ff1493")


def _cif_relative_atom_radius(symbol: str) -> float:
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
    return radii.get(_cif_element_symbol(symbol), 0.85)


def _cif_element_symbol(symbol: str) -> str:
    letters = "".join(char for char in str(symbol) if char.isalpha())
    if not letters:
        return str(symbol)
    if len(letters) == 1:
        return letters.upper()
    return f"{letters[0].upper()}{letters[1:].lower()}"


def sync_structure_peak_from_fit(
    project: ProjectState,
    data_id: str,
    peak_record: dict[str, Any],
    fit_store: dict[str, Any],
) -> StructurePeak | None:
    """Mirror a Peak Fit result into stored Structure Analysis peaks."""

    analyses = project.analysis_results.setdefault("structure_analysis", {})
    state = analyses.setdefault(
        data_id,
        {
            "peaks": [],
            "candidates": [],
            "families": [],
            "wyckoff": {},
        },
    )
    peak_id = _peak_record_id(peak_record)
    records = [
        record
        for record in project.peak_sets.get(data_id, [])
        if isinstance(record, dict)
    ]
    if peak_id and all(
        _peak_record_id(record) != peak_id for record in records
    ):
        records.append(peak_record)
    fit_records = dict(_peak_fit_records_for_project(project, data_id))
    if peak_id:
        fit_records[peak_id] = fit_store
    peaks = build_structure_peaks(
        records,
        fit_records,
        existing=state.get("peaks", []),
    )
    state["peaks"] = [peak.as_dict() for peak in peaks]
    for peak in peaks:
        if peak.peak_id == peak_id:
            return peak
    return None


def _peak_fit_records_for_project(
    project: ProjectState,
    data_id: str,
) -> dict[str, Any]:
    container = project.fits.get(data_id, {})
    if not isinstance(container, dict):
        return {}
    peak_fit = container.get("peak_fit", {})
    return peak_fit if isinstance(peak_fit, dict) else {}


def _peak_record_id(record: dict[str, Any]) -> str:
    return str(record.get("peak_id") or record.get("id") or "")


def _format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.5g}"
    except (TypeError, ValueError):
        return str(value)


def _roi_overlay_key(roi: dict[str, Any]) -> tuple[Any, ...]:
    roi_id = roi.get("roi_id")
    if roi_id:
        return ("id", str(roi_id))
    return (
        str(roi.get("kind", "box")).lower(),
        _rounded_roi_value(roi.get("qxy_min")),
        _rounded_roi_value(roi.get("qxy_max")),
        _rounded_roi_value(roi.get("qz_min")),
        _rounded_roi_value(roi.get("qz_max")),
        _rounded_roi_value(roi.get("qxy_center")),
        _rounded_roi_value(roi.get("qz_center")),
        _rounded_roi_value(roi.get("qr_min")),
        _rounded_roi_value(roi.get("qr_max")),
        _rounded_roi_value(roi.get("chi_min")),
        _rounded_roi_value(roi.get("chi_max")),
    )


def _rounded_roi_value(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return round(parsed, 9)


def _roi_overlay_points(
    roi: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    kind = str(roi.get("kind", "box")).lower()
    if kind == "arch":
        return _arch_roi_overlay_points(roi)
    return _box_roi_overlay_points(roi)


def _make_plot_item_non_interactive(item: Any) -> None:
    targets = [item]
    for attribute in ("curve", "scatter"):
        child = getattr(item, attribute, None)
        if child is not None:
            targets.append(child)
    targets.extend(item.childItems() if hasattr(item, "childItems") else [])
    seen: set[int] = set()
    for target in targets:
        target_id = id(target)
        if target_id in seen:
            continue
        seen.add(target_id)
        if hasattr(target, "setAcceptedMouseButtons"):
            target.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        if hasattr(target, "setAcceptTouchEvents"):
            target.setAcceptTouchEvents(False)
        if hasattr(target, "setAcceptHoverEvents"):
            target.setAcceptHoverEvents(False)
        if hasattr(target, "setClickable"):
            target.setClickable(False)


def _box_roi_overlay_points(
    roi: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    qxy_min = _rounded_roi_value(roi.get("qxy_min"))
    qxy_max = _rounded_roi_value(roi.get("qxy_max"))
    qz_min = _rounded_roi_value(roi.get("qz_min"))
    qz_max = _rounded_roi_value(roi.get("qz_max"))
    if None in {qxy_min, qxy_max, qz_min, qz_max}:
        return None
    x_min, x_max = sorted((float(qxy_min), float(qxy_max)))
    y_min, y_max = sorted((float(qz_min), float(qz_max)))
    return (
        np.asarray([x_min, x_max, x_max, x_min, x_min], dtype=float),
        np.asarray([y_min, y_min, y_max, y_max, y_min], dtype=float),
    )


def _arch_roi_overlay_points(
    roi: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    center_qxy = _rounded_roi_value(roi.get("qxy_center")) or 0.0
    center_qz = _rounded_roi_value(roi.get("qz_center")) or 0.0
    qr_min = _rounded_roi_value(roi.get("qr_min"))
    qr_max = _rounded_roi_value(roi.get("qr_max"))
    chi_min = _rounded_roi_value(roi.get("chi_min"))
    chi_max = _rounded_roi_value(roi.get("chi_max"))
    if None in {qr_min, qr_max, chi_min, chi_max}:
        return None
    inner, outer = sorted((max(float(qr_min), 0.0), max(float(qr_max), 0.0)))
    if outer <= inner:
        return None
    start, stop = sorted((float(chi_min), float(chi_max)))
    theta = np.radians(np.linspace(start, stop, 96))
    outer_x = center_qxy + outer * np.sin(theta)
    outer_y = center_qz + outer * np.cos(theta)
    inner_x = center_qxy + inner * np.sin(theta[::-1])
    inner_y = center_qz + inner * np.cos(theta[::-1])
    return (
        np.concatenate([outer_x, inner_x, outer_x[:1]]),
        np.concatenate([outer_y, inner_y, outer_y[:1]]),
    )


def _safe_float(text: str, fallback: float) -> float:
    try:
        value = float(text)
    except ValueError:
        return fallback
    return value if np.isfinite(value) else fallback


def _status_text(peak: StructurePeak) -> str:
    parts = []
    if peak.status:
        parts.append(peak.status)
    if peak.notes:
        parts.append(peak.notes)
    return "; ".join(parts)


def _append_note(current: str, note: str) -> str:
    if not current:
        return note
    if note in current:
        return current
    return f"{current}; {note}"
