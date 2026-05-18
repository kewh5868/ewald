"""Left-pane tree for experimental data, metadata, and fits."""

from __future__ import annotations

from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from ewald.data.models import (
    CorrectionAssetRef,
    DataFileRef,
    DataGroupRef,
    ProjectState,
    ROIRegion,
    roi_hkl_label,
    roi_pole_figure_status,
)
from ewald.simulation.giwaxs import SIMULATION_MODE_EWALD_SWEEP
from ewald.ui.notation import QXY_HTML, QZ_HTML, enable_rich_text_items


class DataTreePane(QtWidgets.QWidget):
    """Tree view of loaded experimental detector data."""

    dataFileSelected = QtCore.Signal(str)
    selectionChanged = QtCore.Signal(dict)
    newProjectRequested = QtCore.Signal()
    saveProjectRequested = QtCore.Signal()
    saveProjectAsRequested = QtCore.Signal()
    loadProjectRequested = QtCore.Signal()
    importFileRequested = QtCore.Signal()
    importFolderRequested = QtCore.Signal()

    def __init__(
        self,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.toolbar = QtWidgets.QWidget()
        self.toolbar_layout = QtWidgets.QVBoxLayout(self.toolbar)
        self.toolbar_layout.setContentsMargins(4, 4, 4, 4)
        self.toolbar_layout.setSpacing(3)
        self.project_setup_layout = self._button_row("Project Setup")
        self.data_import_layout = self._button_row("Data Import")
        self._build_buttons()

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Experimental Data", "Value"])
        enable_rich_text_items(self.tree)
        self.tree.setColumnWidth(0, 240)
        self.tree.itemSelectionChanged.connect(self._emit_selected_item)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.tree)

    def _build_buttons(self) -> None:
        style = self.style()
        self.new_button = self._tool_button(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon),
            "New Project",
            self.newProjectRequested,
            self.project_setup_layout,
        )
        self.save_button = self._tool_button(
            style.standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            ),
            "Save Project",
            self.saveProjectRequested,
            self.project_setup_layout,
        )
        self.save_as_button = self._tool_button(
            style.standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton
            ),
            "Save Project As...",
            self.saveProjectAsRequested,
            self.project_setup_layout,
        )
        self.load_button = self._tool_button(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon),
            "Load Project",
            self.loadProjectRequested,
            self.project_setup_layout,
        )
        self.import_file_button = self._tool_button(
            style.standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_FileDialogNewFolder
            ),
            "Import Data File",
            self.importFileRequested,
            self.data_import_layout,
        )
        self.import_folder_button = self._tool_button(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirIcon),
            "Import Data Folder",
            self.importFolderRequested,
            self.data_import_layout,
        )
        self.project_setup_layout.addStretch(1)
        self.data_import_layout.addStretch(1)

    def _button_row(self, label: str) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        row.setContentsMargins(0, 0, 0, 0)
        text = QtWidgets.QLabel(label)
        text.setMinimumWidth(84)
        row.addWidget(text)
        self.toolbar_layout.addLayout(row)
        return row

    def _tool_button(
        self,
        icon: QtGui.QIcon,
        tooltip: str,
        signal: QtCore.Signal,
        layout: QtWidgets.QHBoxLayout,
    ) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setIcon(icon)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setFixedSize(26, 26)
        button.setIconSize(QtCore.QSize(16, 16))
        button.clicked.connect(signal.emit)
        layout.addWidget(button)
        return button

    def set_project(self, project: ProjectState | None) -> None:
        self.tree.clear()
        root_label = (
            project.name if project is not None else "Experimental Data"
        )
        root_value = "Project" if project is not None else "No active project"
        root = QtWidgets.QTreeWidgetItem([root_label, root_value])
        root.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "root"})
        self.tree.addTopLevelItem(root)
        if project is None:
            root.setExpanded(True)
            return

        if project.data_groups:
            for group in project.data_groups:
                self._add_group(root, group, project)
        if project.data_files:
            ungrouped = DataGroupRef(
                name="Ungrouped", data_files=project.data_files
            )
            self._add_group(root, ungrouped, project)
        self._add_correction_assets(
            root, "Masks", "mask", project.masks, project
        )
        self._add_correction_assets(
            root,
            "Calibrants",
            "calibrant",
            project.calibrants,
            project,
        )
        self._add_simulations(root, project)

        root.setExpanded(True)

    def set_project_active(self, active: bool) -> None:
        """Enable controls that require an active project."""

        for button in (
            self.save_button,
            self.save_as_button,
            self.import_file_button,
            self.import_folder_button,
        ):
            button.setEnabled(active)

    def _add_group(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        group: DataGroupRef,
        project: ProjectState,
    ) -> None:
        if group.import_kind == "file" and len(group.data_files) == 1:
            self._add_file(parent, group.data_files[0], project, group)
            return

        value = _group_value(group)
        if len(group.data_files) != 1:
            value = value.replace(" file", " files")
        group_item = QtWidgets.QTreeWidgetItem([group.name, value])
        group_item.setData(
            0,
            QtCore.Qt.ItemDataRole.UserRole,
            {
                "kind": "group",
                "group_id": group.group_id,
                "name": group.name,
                "import_kind": group.import_kind,
            },
        )
        parent.addChild(group_item)

        self._add_group_metadata(group_item, group, project)
        self._add_group_analysis_scope(group_item, group)
        for data_file in group.data_files:
            self._add_file(group_item, data_file, project, group)
        group_item.setExpanded(True)

    def _add_group_metadata(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        group: DataGroupRef,
        project: ProjectState,
    ) -> None:
        metadata_item = QtWidgets.QTreeWidgetItem(
            ["Metadata", _group_value(group)]
        )
        parent.addChild(metadata_item)
        self._add_leaf(metadata_item, "Import type", group.import_kind)
        self._add_leaf(metadata_item, "Metadata type", group.metadata_type)
        self._add_leaf(metadata_item, "Delimiter", group.delimiter)
        self._add_leaf(metadata_item, "Path", group.path)
        self._add_leaf(
            metadata_item,
            "Mask",
            _asset_assignment_label(project, "mask", group.group_id),
        )
        self._add_leaf(
            metadata_item,
            "Calibrant",
            _asset_assignment_label(project, "calibrant", group.group_id),
        )
        self._add_parse_report(metadata_item, group)

    def _add_group_analysis_scope(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        group: DataGroupRef,
    ) -> None:
        scope_item = QtWidgets.QTreeWidgetItem(["Available Processing", ""])
        parent.addChild(scope_item)
        if group.import_kind == "folder":
            self._add_leaf(scope_item, "Allowed", "time/temperature workflows")
            self._add_leaf(scope_item, "Allowed", "transient analysis")
            self._add_leaf(scope_item, "Disabled", "single-image fits")
            self._add_leaf(scope_item, "Disabled", "lattice determination")
        else:
            self._add_leaf(scope_item, "Allowed", "single-image fits")
            self._add_leaf(scope_item, "Allowed", "lattice determination")
            self._add_leaf(scope_item, "Disabled", "transient folder analysis")

    def _add_parse_report(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        group: DataGroupRef,
    ) -> None:
        if not group.parse_report:
            return
        report = group.parse_report
        fields = {
            "Parse delimiter": report.get("delimiter"),
            "Parse file count": report.get("file_count"),
            "Parse token counts": report.get("token_counts"),
            "Consistent token count": report.get("consistent_token_count"),
            "Recurrent exposure time": report.get("recurrent_exposure_time_s"),
        }
        for key, value in fields.items():
            self._add_leaf(parent, key, value)
        files_requiring_input = report.get(
            "files_requiring_metadata_input", []
        )
        if files_requiring_input:
            self._add_leaf(
                parent,
                "Needs metadata review",
                str(len(files_requiring_input)),
            )

    def _add_file(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        data_file: DataFileRef,
        project: ProjectState,
        group: DataGroupRef | None = None,
    ) -> None:
        file_item = QtWidgets.QTreeWidgetItem(
            [_data_file_label(data_file), data_file.kind]
        )
        file_item.setData(
            0,
            QtCore.Qt.ItemDataRole.UserRole,
            {
                "kind": "file",
                "data_id": data_file.data_id,
                "name": _data_file_label(data_file),
                "group_id": group.group_id if group else None,
            },
        )
        parent.addChild(file_item)
        self._add_file_metadata(file_item, data_file, project, group)
        self._add_file_analysis_scope(file_item)
        self._add_rois(file_item, project.rois_for_target(data_file.data_id))
        self._add_linked_simulations(
            file_item,
            project,
            project.simulations_for_data_file(data_file.data_id),
        )
        self._add_fits(file_item, project.fits.get(data_file.data_id or ""))
        file_item.setExpanded(True)

    def _add_file_metadata(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        data_file: DataFileRef,
        project: ProjectState,
        group: DataGroupRef | None = None,
    ) -> None:
        metadata = data_file.metadata
        metadata_item = QtWidgets.QTreeWidgetItem(
            ["Metadata", _file_summary(data_file)]
        )
        parent.addChild(metadata_item)
        inherited_target_id = group.group_id if group else None
        self._add_leaf(metadata_item, "Name", _data_file_label(data_file))
        self._add_leaf(metadata_item, "Data type", data_file.kind)
        if group is not None:
            self._add_leaf(metadata_item, "Import type", group.import_kind)
            self._add_leaf(metadata_item, "Metadata type", group.metadata_type)
            self._add_leaf(metadata_item, "Delimiter", group.delimiter)
        self._add_leaf(
            metadata_item,
            "Mask",
            _asset_assignment_label(
                project,
                "mask",
                data_file.data_id,
                inherited_target_id,
            ),
        )
        self._add_leaf(
            metadata_item,
            "Calibrant",
            _asset_assignment_label(
                project,
                "calibrant",
                data_file.data_id,
                inherited_target_id,
            ),
        )
        correction_state = project.image_corrections.get(
            data_file.data_id or ""
        )
        correction_label = (
            "Confirmed"
            if correction_state and correction_state.confirmed
            else "Not applied"
        )
        self._add_leaf(metadata_item, "Image corrections", correction_label)
        self._add_leaf(metadata_item, "Path", data_file.path)
        if group is not None:
            self._add_parse_report(metadata_item, group)
        for key in _metadata_detail_keys(metadata):
            self._add_leaf(
                metadata_item,
                _labelize(key),
                _format_metadata_value(key, metadata[key]),
            )

    def _add_correction_assets(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        title: str,
        asset_kind: str,
        assets: list[CorrectionAssetRef],
        project: ProjectState,
    ) -> None:
        value = f"{len(assets)} loaded"
        collection_item = QtWidgets.QTreeWidgetItem([title, value])
        collection_item.setData(
            0,
            QtCore.Qt.ItemDataRole.UserRole,
            {"kind": f"{asset_kind}-collection"},
        )
        parent.addChild(collection_item)
        if len(assets) == 1:
            self._add_correction_asset_details(
                collection_item,
                assets[0],
                project,
                include_name=True,
            )
            return
        for asset in assets:
            self._add_correction_asset(collection_item, asset, project)

    def _add_correction_asset(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        asset: CorrectionAssetRef,
        project: ProjectState,
    ) -> None:
        asset_item = QtWidgets.QTreeWidgetItem(
            [asset.name, _correction_asset_label(asset.kind)]
        )
        asset_item.setData(
            0,
            QtCore.Qt.ItemDataRole.UserRole,
            {
                "kind": "correction-asset",
                "asset_kind": asset.kind,
                "asset_id": asset.asset_id,
                "name": asset.name,
            },
        )
        parent.addChild(asset_item)
        self._add_correction_asset_details(asset_item, asset, project)

    def _add_correction_asset_details(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        asset: CorrectionAssetRef,
        project: ProjectState,
        *,
        include_name: bool = False,
    ) -> None:
        if include_name:
            self._add_leaf(parent, "Name", asset.name)
        self._add_leaf(parent, "Path", asset.path)
        self._add_leaf(parent, "Source", asset.source)

        applied_item = QtWidgets.QTreeWidgetItem(
            ["Applied To", f"{len(asset.target_ids)} target(s)"]
        )
        parent.addChild(applied_item)
        if not asset.target_ids:
            self._add_leaf(applied_item, "Status", "Not applied")
            return
        for target_id in asset.target_ids:
            self._add_leaf(
                applied_item,
                _target_kind_label(project, target_id),
                _target_label(project, target_id),
            )

    def _add_simulations(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        project: ProjectState,
    ) -> None:
        simulations_item = QtWidgets.QTreeWidgetItem(
            ["GIWAXS Simulations", str(len(project.simulations))]
        )
        simulations_item.setData(
            0,
            QtCore.Qt.ItemDataRole.UserRole,
            {"kind": "simulation-collection"},
        )
        parent.addChild(simulations_item)
        if not project.simulations:
            self._add_leaf(simulations_item, "Status", "No simulations")
            return
        for simulation_id, record in project.simulations.items():
            self._add_simulation(
                simulations_item, simulation_id, record, project
            )

    def _add_linked_simulations(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        project: ProjectState,
        simulations: list[tuple[str, dict[str, Any]]],
    ) -> None:
        simulations_item = QtWidgets.QTreeWidgetItem(
            ["Linked Simulations", str(len(simulations))]
        )
        parent.addChild(simulations_item)
        if not simulations:
            self._add_leaf(simulations_item, "Status", "No simulations")
            return
        for simulation_id, record in simulations:
            self._add_simulation(
                simulations_item,
                simulation_id,
                record,
                project,
            )

    def _add_simulation(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        simulation_id: str,
        record: Any,
        project: ProjectState | None = None,
    ) -> None:
        if not isinstance(record, dict):
            self._add_leaf(parent, simulation_id, record)
            return
        label = record.get("structure_name") or simulation_id
        item = QtWidgets.QTreeWidgetItem(
            [str(label), _simulation_type_label(record)]
        )
        item.setData(
            0,
            QtCore.Qt.ItemDataRole.UserRole,
            {"kind": "simulation", "simulation_id": simulation_id},
        )
        parent.addChild(item)
        self._add_leaf(item, "Simulation id", simulation_id)
        for key in ("structure_path", "dataset_uri"):
            if record.get(key):
                self._add_leaf(item, _labelize(key), record[key])
        data_id = record.get("data_id")
        if data_id:
            self._add_leaf(item, "Data id", data_id)
            if project is not None:
                self._add_leaf(
                    item,
                    "Linked data file",
                    _target_label(project, str(data_id)),
                )
        metadata_item = QtWidgets.QTreeWidgetItem(["Metadata", ""])
        item.addChild(metadata_item)
        for key, value in sorted(record.get("metadata", {}).items()):
            self._add_leaf(metadata_item, _labelize(key), value)
        parameters_item = QtWidgets.QTreeWidgetItem(["Parameters", ""])
        item.addChild(parameters_item)
        for key, value in sorted(record.get("parameters", {}).items()):
            self._add_leaf(parameters_item, _labelize(key), value)

    def _add_file_analysis_scope(
        self,
        parent: QtWidgets.QTreeWidgetItem,
    ) -> None:
        scope_item = QtWidgets.QTreeWidgetItem(["Available Processing", ""])
        parent.addChild(scope_item)
        self._add_leaf(scope_item, "Allowed", "single-image fits")
        self._add_leaf(scope_item, "Allowed", "lattice determination")
        self._add_leaf(scope_item, "Disabled", "transient folder analysis")

    def _add_fits(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        fits: Any,
    ) -> None:
        fits_item = QtWidgets.QTreeWidgetItem(["Fits", ""])
        parent.addChild(fits_item)
        if not fits:
            self._add_leaf(fits_item, "Status", "No fits")
            return
        if isinstance(fits, list):
            for index, fit in enumerate(fits, start=1):
                fit_item = QtWidgets.QTreeWidgetItem([f"Fit {index}", ""])
                fits_item.addChild(fit_item)
                self._add_mapping(fit_item, fit)
            return
        if isinstance(fits, dict):
            self._add_mapping(fits_item, fits)
            return
        self._add_leaf(fits_item, "Value", fits)

    def _add_rois(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        rois: list[ROIRegion],
    ) -> None:
        roi_item = QtWidgets.QTreeWidgetItem(["ROIs", str(len(rois))])
        parent.addChild(roi_item)
        if not rois:
            self._add_leaf(roi_item, "Status", "No ROIs")
            return
        for roi in rois:
            child = QtWidgets.QTreeWidgetItem(
                [roi.name or roi.roi_id or "ROI", _roi_tree_value(roi)]
            )
            roi_item.addChild(child)
            hkl = roi_hkl_label(roi)
            if hkl:
                self._add_leaf(child, "hkl", hkl)
            pole_status = roi_pole_figure_status(roi)
            if pole_status:
                self._add_leaf(child, "Pole figure", pole_status)

    def _add_mapping(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        payload: dict[str, Any],
    ) -> None:
        for key, value in sorted(payload.items()):
            self._add_leaf(parent, _labelize(key), value)

    def _add_leaf(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        key: str,
        value: Any,
    ) -> None:
        parent.addChild(QtWidgets.QTreeWidgetItem([key, _format_value(value)]))

    def _emit_selected_item(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        payload = items[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            payload = {"kind": "detail"}
        self.selectionChanged.emit(payload)
        if payload.get("kind") == "file":
            data_id = payload.get("data_id")
            if data_id:
                self.dataFileSelected.emit(str(data_id))


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


def _simulation_type_label(record: dict[str, Any]) -> str:
    if record.get("simulation_mode") == SIMULATION_MODE_EWALD_SWEEP:
        return "Ewald sphere sweep"
    return "GIWAXS simulation"


def _group_value(group: DataGroupRef) -> str:
    prefix = "Folder" if group.import_kind == "folder" else "File"
    return f"{prefix}, {len(group.data_files)} file"


def _file_summary(data_file: DataFileRef) -> str:
    metadata = data_file.metadata
    pieces: list[str] = []
    if data_file.kind:
        pieces.append(data_file.kind)
    sample = metadata.get("sample_label")
    composition = metadata.get("sample_composition")
    if sample:
        pieces.append(str(sample))
    if composition:
        pieces.append(str(composition))
    detector = metadata.get("detector_type")
    if detector:
        pieces.append(str(detector).upper())
    return " | ".join(pieces)


def _data_file_label(data_file: DataFileRef) -> str:
    return data_file.name or data_file.data_id or data_file.path.stem


def _metadata_detail_keys(metadata: dict[str, Any]) -> list[str]:
    preferred = [
        "original_file_name",
        "sample_label",
        "sample_number",
        "sample_composition",
        "filtration_status",
        "concentration_molar",
        "substrate",
        "solution_volume_uL",
        "flow_rate_scfh",
        "x_position",
        "y_position",
        "z_position",
        "incidence_angle_deg",
        "frame_timestamp_s",
        "exposure_time_s",
        "detector_type",
        "frame_number",
        "run_id",
    ]
    ordered = [key for key in preferred if key in metadata]
    ordered.extend(key for key in sorted(metadata) if key not in ordered)
    return ordered


def _labelize(key: str) -> str:
    if key.startswith("_"):
        key = key[1:]
    parts = [
        QXY_HTML if part == "qxy" else QZ_HTML if part == "qz" else part
        for part in key.split("_")
    ]
    label = " ".join(parts)
    if label.startswith((QXY_HTML, QZ_HTML)):
        return label
    return label[:1].upper() + label[1:]


def _format_metadata_value(key: str, value: Any) -> str:
    if key == "_metadata_fields" and isinstance(value, list):
        fields = []
        for item in value:
            if not isinstance(item, dict):
                continue
            field_key = item.get("key")
            field_value = item.get("value")
            if field_key:
                fields.append(f"{field_key}={_format_value(field_value)}")
        return ", ".join(fields)
    units = {
        "concentration_molar": "M",
        "solution_volume_uL": "uL",
        "flow_rate_scfh": "scfh",
        "incidence_angle_deg": "deg",
        "frame_timestamp_s": "s",
        "exposure_time_s": "s",
    }
    formatted = _format_value(value)
    unit = units.get(key)
    if unit and formatted:
        return f"{formatted} {unit}"
    return formatted


def _asset_assignment_label(
    project: ProjectState,
    asset_kind: str,
    target_id: str | None,
    inherited_target_id: str | None = None,
) -> str:
    direct = project.assigned_assets(asset_kind, target_id)
    if direct:
        return ", ".join(asset.name for asset in direct)
    inherited = project.assigned_assets(asset_kind, inherited_target_id)
    if inherited:
        return ", ".join(f"{asset.name} (folder)" for asset in inherited)
    return "None"


def _correction_asset_label(asset_kind: str) -> str:
    if asset_kind == "mask":
        return "MASK"
    if asset_kind == "calibrant":
        return "PONI"
    return asset_kind


def _roi_tree_value(roi: ROIRegion) -> str:
    if roi.kind == "arch":
        return (
            f"arch qr {_format_value(roi.qr_min)}-"
            f"{_format_value(roi.qr_max)}, chi "
            f"{_format_value(roi.chi_min)}-{_format_value(roi.chi_max)}"
        )
    return (
        f"box {QXY_HTML} {_format_value(roi.qxy_min)}-"
        f"{_format_value(roi.qxy_max)}, {QZ_HTML} "
        f"{_format_value(roi.qz_min)}-{_format_value(roi.qz_max)}"
    )


def _target_label(project: ProjectState, target_id: str) -> str:
    for group in project.data_groups:
        if group.group_id == target_id:
            return group.name
        for data_file in group.data_files:
            if data_file.data_id == target_id:
                return _data_file_label(data_file)
    for data_file in project.data_files:
        if data_file.data_id == target_id:
            return _data_file_label(data_file)
    return target_id


def _target_kind_label(project: ProjectState, target_id: str) -> str:
    for group in project.data_groups:
        if group.group_id == target_id:
            return "Folder" if group.import_kind == "folder" else "Data set"
        for data_file in group.data_files:
            if data_file.data_id == target_id:
                return "Data file"
    for data_file in project.data_files:
        if data_file.data_id == target_id:
            return "Data file"
    return "Target"
