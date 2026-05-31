"""Main Qt6 window for the EWALD application shell."""

from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets

from ewald.app.metadata import APPLICATION_DISPLAY_NAME
from ewald.data.models import ProjectState
from ewald.io.importers import (
    build_data_group_from_folder,
    build_data_group_from_paths,
)
from ewald.io.project import (
    PROJECT_FILE_FILTER,
    load_project,
    normalize_project_path,
    save_project,
)
from ewald.ui.corrections import ApplyImageCorrectionsPane
from ewald.ui.data_tree import DataTreePane
from ewald.ui.data_viewer import DataViewerPane
from ewald.ui.giwaxs_simulation import (
    GIWAXSSimulationPane,
)
from ewald.ui.metadata_dialog import ManualMetadataDialog
from ewald.ui.peak_identification import PeakIdentificationPane
from ewald.ui.pole_figure import PoleFigureGeneratorWindow
from ewald.ui.pyfai_calib2 import (
    PYFAI_CALIB2_COMMAND,
    PyFAICalib2Launcher,
    PyFAICalib2Status,
)
from ewald.ui.structure_analysis import StructureAnalysisPane
from ewald.ui.theme import apply_application_theme
from ewald.version import __version__

GITHUB_URL = "https://github.com/kewh5868/ewald/"
DEVELOPER_NAME = "Keith White"
DEVELOPER_EMAIL = "keith.white@colorado.edu"
PROJECT_DIRECTORY_SETTING = "project_directory"
RECENT_PROJECTS_SETTING = "recent_projects"
MAX_RECENT_PROJECTS = 8
APP_TITLE = APPLICATION_DISPLAY_NAME


class MainWindow(QtWidgets.QMainWindow):
    """Application shell with tab boundaries for the EWALD workflow."""

    def __init__(
        self,
        *,
        project: ProjectState | None = None,
        project_path: str | Path | None = None,
        settings: QtCore.QSettings | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        apply_application_theme(QtWidgets.QApplication.instance())
        self.settings = settings or QtCore.QSettings("EWALD", "EWALD")
        self.project = project or ProjectState()
        self.project_active = project is not None
        self.project_path: Path | None = (
            normalize_project_path(project_path) if project_path else None
        )
        self.pyfai_calib2_launcher = PyFAICalib2Launcher(self)
        self.current_tree_selection: dict[str, str | None] = {"kind": "root"}
        self.setWindowTitle(APP_TITLE)
        self.resize(1600, 1000)
        self._build_status_bar()

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self._create_actions()
        self._build_menu()
        self._build_workflow_toolbar()
        self.pyfai_calib2_launcher.statusChanged.connect(
            self._handle_pyfai_calibration_status_changed
        )
        self.pyfai_calib2_launcher.launchSkipped.connect(
            self._handle_pyfai_calibration_launch_skipped
        )
        self._build_left_pane()
        self._build_tabs()
        self._refresh_project_views()
        self._refresh_workflow_context()
        self._refresh_save_status()
        if self.project_active and self.project_path is not None:
            self._remember_recent_project(self.project_path)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.pyfai_calib2_launcher.terminate()
        super().closeEvent(event)

    def _create_actions(self) -> None:
        self.new_project_action = QtGui.QAction("New Project", self)
        self.new_project_action.setShortcut(QtGui.QKeySequence.StandardKey.New)
        self.open_project_action = QtGui.QAction("Open Project", self)
        self.open_project_action.setShortcut(
            QtGui.QKeySequence.StandardKey.Open
        )
        self.load_files_action = QtGui.QAction("Import Data File", self)
        self.load_folder_action = QtGui.QAction("Import Data Folder", self)
        self.save_project_action = QtGui.QAction("Save Project", self)
        self.save_project_action.setShortcut(
            QtGui.QKeySequence.StandardKey.Save
        )
        self.save_project_as_action = QtGui.QAction("Save Project As", self)
        self.create_mask_action = QtGui.QAction("Create Mask", self)
        self.load_mask_action = QtGui.QAction("Load Mask", self)
        self.create_calibrant_action = QtGui.QAction("Create Calibrant", self)
        self.load_calibrant_action = QtGui.QAction("Load Calibrant", self)
        self.pyfai_calibration_action = QtGui.QAction(
            "PyFAI Calibration/Mask Tool",
            self,
        )
        self.giwaxs_simulation_action = QtGui.QAction(
            "GIWAXS Simulation", self
        )
        self.pole_figure_action = QtGui.QAction("Pole Figure Generator", self)
        self.toggle_file_manager_action = QtGui.QAction("File Manager", self)
        self.toggle_file_manager_action.setCheckable(True)
        self.toggle_file_manager_action.setChecked(True)
        self.github_action = QtGui.QAction("GitHub Repository", self)
        self.developer_info_action = QtGui.QAction(
            "Developer Information", self
        )
        self.version_info_action = QtGui.QAction("Version Information", self)
        self.exit_action = QtGui.QAction("Exit", self)
        self._set_action_icons()
        self._set_action_status_tips()

        self.new_project_action.triggered.connect(self.new_project)
        self.open_project_action.triggered.connect(self.open_project)
        self.load_files_action.triggered.connect(self.load_files)
        self.load_folder_action.triggered.connect(self.load_folder)
        self.save_project_action.triggered.connect(self.save_project)
        self.save_project_as_action.triggered.connect(self.save_project_as)
        self.create_mask_action.triggered.connect(self.create_mask)
        self.load_mask_action.triggered.connect(self.load_mask)
        self.create_calibrant_action.triggered.connect(self.create_calibrant)
        self.load_calibrant_action.triggered.connect(self.load_calibrant)
        self.pyfai_calibration_action.triggered.connect(
            self.open_pyfai_calibration_tool
        )
        self.giwaxs_simulation_action.triggered.connect(
            self.open_giwaxs_simulation_tool
        )
        self.pole_figure_action.triggered.connect(
            lambda _checked=False: self.open_pole_figure_tool()
        )
        self.toggle_file_manager_action.triggered.connect(
            self.set_file_manager_visible
        )
        self.github_action.triggered.connect(self.open_github_repository)
        self.developer_info_action.triggered.connect(
            self.show_developer_information
        )
        self.version_info_action.triggered.connect(
            self.show_version_information
        )
        self.exit_action.triggered.connect(self.close)

    def _set_action_icons(self) -> None:
        style = self.style()
        self.new_project_action.setIcon(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon)
        )
        self.open_project_action.setIcon(
            style.standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton
            )
        )
        self.load_files_action.setIcon(
            style.standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_FileDialogNewFolder
            )
        )
        self.load_folder_action.setIcon(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirIcon)
        )
        self.save_project_action.setIcon(
            style.standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            )
        )
        self.save_project_as_action.setIcon(
            style.standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton
            )
        )
        mask_icon = style.standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView
        )
        calibrant_icon = style.standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_FileIcon
        )
        self.create_mask_action.setIcon(mask_icon)
        self.load_mask_action.setIcon(mask_icon)
        self.create_calibrant_action.setIcon(calibrant_icon)
        self.load_calibrant_action.setIcon(calibrant_icon)
        self.pyfai_calibration_action.setIcon(calibrant_icon)
        self.giwaxs_simulation_action.setIcon(
            style.standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_FileDialogInfoView
            )
        )
        self.pole_figure_action.setIcon(
            style.standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_FileDialogContentsView
            )
        )
        self.github_action.setIcon(
            style.standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogHelpButton
            )
        )

    def _set_action_status_tips(self) -> None:
        tips = {
            self.new_project_action: "Create a new EWALD project.",
            self.open_project_action: "Open an existing EWALD project.",
            self.load_files_action: "Import one detector image into the project.",
            self.load_folder_action: "Import a folder of detector images.",
            self.save_project_action: "Save the active EWALD project.",
            self.save_project_as_action: "Save the project to a new file.",
            self.load_mask_action: "Assign a mask to the selected data.",
            self.load_calibrant_action: "Assign a PONI calibration file.",
            self.pyfai_calibration_action: "Open pyFAI calibration tools.",
            self.giwaxs_simulation_action: "Open the GIWAXS simulation workflow.",
            self.pole_figure_action: "Generate a pole figure from the selected ROI.",
        }
        for action, tip in tips.items():
            action.setStatusTip(tip)
            action.setToolTip(tip)

    def _build_menu(self) -> None:
        self.file_menu = self.menuBar().addMenu("File")
        self.file_menu.addAction(self.new_project_action)
        self.file_menu.addAction(self.open_project_action)
        self.recent_projects_menu = self.file_menu.addMenu("Open Recent")
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.load_files_action)
        self.file_menu.addAction(self.load_folder_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_project_action)
        self.file_menu.addAction(self.save_project_as_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)

        self.view_menu = self.menuBar().addMenu("View")
        self.view_menu.addAction(self.toggle_file_manager_action)

        self.tools_menu = self.menuBar().addMenu("Tools")
        self.tools_menu.addAction(self.load_mask_action)
        self.tools_menu.addAction(self.load_calibrant_action)
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.pyfai_calibration_action)
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.giwaxs_simulation_action)
        self.tools_menu.addAction(self.pole_figure_action)

        self.help_menu = self.menuBar().addMenu("Help")
        self.help_menu.addAction(self.github_action)
        self.help_menu.addAction(self.developer_info_action)
        self.help_menu.addAction(self.version_info_action)
        self._refresh_recent_projects_menu()

    def _build_workflow_toolbar(self) -> None:
        self.workflow_toolbar = QtWidgets.QToolBar("Workflow", self)
        self.workflow_toolbar.setObjectName("WorkflowToolbar")
        self.workflow_toolbar.setMovable(False)
        self.workflow_toolbar.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.workflow_context_label = QtWidgets.QLabel("Project")
        self.workflow_context_label.setMinimumWidth(220)
        self.workflow_toolbar.addWidget(self.workflow_context_label)
        self.workflow_toolbar.addSeparator()
        for action in (
            self.new_project_action,
            self.open_project_action,
            self.save_project_action,
            self.load_files_action,
            self.load_folder_action,
        ):
            self.workflow_toolbar.addAction(action)
        self.workflow_toolbar.addSeparator()
        for action in (
            self.pyfai_calibration_action,
            self.giwaxs_simulation_action,
            self.pole_figure_action,
        ):
            self.workflow_toolbar.addAction(action)
        self.addToolBar(
            QtCore.Qt.ToolBarArea.TopToolBarArea,
            self.workflow_toolbar,
        )

    def _build_status_bar(self) -> None:
        self.save_status_label = QtWidgets.QLabel()
        self.save_status_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.save_status_label.setMinimumWidth(260)
        self.statusBar().addPermanentWidget(self.save_status_label)

    def open_github_repository(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(GITHUB_URL))

    def open_giwaxs_simulation_tool(self) -> None:
        if not self.project_active:
            self._activate_project_from_simulation()
            self.current_tree_selection = {"kind": "root"}
            self._refresh_project_views()
            self._refresh_workflow_context()
        target_data_id = (
            self._selected_file_data_id()
            or self._selected_simulation_data_id()
        )
        pane = self._embedded_simulation_tab()
        if pane is None:
            self._refresh_right_tabs()
            pane = self._embedded_simulation_tab()
        if pane is None:
            return
        pane.refresh_project_context(
            project_path=self.project_path,
            target_data_id=target_data_id,
        )
        if target_data_id is not None:
            pane.set_target_data_id(target_data_id)
        self.tabs.setCurrentWidget(pane)

    def open_pole_figure_tool(
        self,
        data_id: str | None = None,
        roi: object | None = None,
        image_data: object | None = None,
        axis_ranges: object | None = None,
    ) -> None:
        if not hasattr(self, "pole_figure_window"):
            self.pole_figure_window = None
        if data_id is None or roi is None:
            viewer = self._active_data_viewer()
            if viewer is None:
                QtWidgets.QMessageBox.information(
                    self,
                    "Pole Figure Generator",
                    "Select an ROI in the Data Viewer first.",
                )
                return
            selected_roi = viewer._selected_roi_region()
            if selected_roi is None:
                QtWidgets.QMessageBox.information(
                    self,
                    "Pole Figure Generator",
                    "Select an ROI in the Data Viewer table first.",
                )
                return
            data_id = viewer.data_id
            roi = selected_roi
            image_data = viewer.image_data
            axis_ranges = viewer.axis_ranges
        if self.pole_figure_window is None:
            self.pole_figure_window = PoleFigureGeneratorWindow(
                project=self.project,
                project_path=self.project_path,
                parent=self,
            )
            self.pole_figure_window.poleFigureSaved.connect(
                self._handle_pole_figure_saved
            )
        self.pole_figure_window.set_context(
            data_id=str(data_id) if data_id is not None else None,
            roi=roi,
            image_data=image_data,
            axis_ranges=axis_ranges,
        )
        self.pole_figure_window.show()
        self.pole_figure_window.raise_()
        self.pole_figure_window.activateWindow()

    def open_pyfai_calibration_tool(self) -> None:
        self.pyfai_calib2_launcher.launch()

    def _handle_pyfai_calibration_status_changed(
        self,
        _status_text: str,
    ) -> None:
        self._refresh_pyfai_calibration_status()
        status = self.pyfai_calib2_launcher.status
        message = f"{PYFAI_CALIB2_COMMAND}: {status.value}"
        if status == PyFAICalib2Status.FAILED:
            error = self.pyfai_calib2_launcher.last_error
            if error:
                message = f"{message} ({error})"
            timeout_ms = 8000
        else:
            timeout_ms = 4000
        self.statusBar().showMessage(message, timeout_ms)

    def _handle_pyfai_calibration_launch_skipped(
        self,
        message: str,
    ) -> None:
        self.statusBar().showMessage(message, 4000)

    def _refresh_pyfai_calibration_status(self) -> None:
        text = self._pyfai_calibration_status_text()
        tooltip = self._pyfai_calibration_status_tooltip()
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            if isinstance(widget, ApplyImageCorrectionsPane):
                widget.set_pyfai_calibration_status(text, tooltip)

    def _pyfai_calibration_status_text(self) -> str:
        return (
            f"{PYFAI_CALIB2_COMMAND}: "
            f"{self.pyfai_calib2_launcher.status.value}"
        )

    def _pyfai_calibration_status_tooltip(self) -> str:
        if self.pyfai_calib2_launcher.status == PyFAICalib2Status.FAILED:
            return self.pyfai_calib2_launcher.last_error
        return ""

    def show_developer_information(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Developer Information",
            _developer_information_text(),
        )

    def show_version_information(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Version Information",
            _version_information_text(),
        )

    def _build_tabs(self) -> None:
        self.project_summary = QtWidgets.QTextEdit(readOnly=True)
        self._refresh_right_tabs()

    def _placeholder(self, title: str) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        label = QtWidgets.QLabel(title)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return widget

    def _refresh_project_summary(self) -> None:
        if not self.project_active:
            self.project_summary.setPlainText(
                "No active EWALD project\n\n"
                "Create a new project, load an existing project, or open the "
                "GIWAXS Simulation workflow to start from a structure file."
            )
            return
        lines = [
            self.project.name,
            "",
            f"Schema: {self.project.schema_version}",
            f"Data groups: {len(self.project.data_groups)}",
            f"Data files: {_project_data_file_count(self.project)}",
            f"Masks: {len(self.project.masks)}",
            f"Calibrants: {len(self.project.calibrants)}",
            f"ROIs: {_project_roi_count(self.project)}",
            f"Peak sets: {len(self.project.peak_sets)}",
            f"Fits: {len(self.project.fits)}",
            f"Analysis results: {len(self.project.analysis_results)}",
            f"Structures: {len(self.project.structures)}",
            f"Reference CIFs: {len(self.project.reference_cifs)}",
            f"Simulations: {len(self.project.simulations)}",
        ]
        lines.extend(self._selection_summary_lines())
        self.project_summary.setPlainText("\n".join(lines))

    def _build_left_pane(self) -> None:
        self.data_tree = DataTreePane()
        self.data_tree.newProjectRequested.connect(self.new_project)
        self.data_tree.saveProjectRequested.connect(self.save_project)
        self.data_tree.saveProjectAsRequested.connect(self.save_project_as)
        self.data_tree.loadProjectRequested.connect(self.open_project)
        self.data_tree.importFileRequested.connect(self.load_files)
        self.data_tree.importFolderRequested.connect(self.load_folder)
        self.data_tree.selectionChanged.connect(self._handle_tree_selection)
        self.data_dock = QtWidgets.QDockWidget("Experimental Data", self)
        self.data_dock.setWidget(self.data_tree)
        self.data_dock.setMinimumWidth(320)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            self.data_dock,
        )
        self.data_dock.visibilityChanged.connect(
            self._handle_file_manager_visibility_changed
        )

    def set_file_manager_visible(self, visible: bool) -> None:
        if not hasattr(self, "data_dock"):
            return
        self.data_dock.setVisible(visible)
        if visible:
            self.data_dock.raise_()

    def _handle_file_manager_visibility_changed(self, visible: bool) -> None:
        self.toggle_file_manager_action.setChecked(visible)

    def _refresh_project_views(self) -> None:
        self._refresh_project_summary()
        self.data_tree.set_project(
            self.project if self.project_active else None
        )
        self._refresh_right_tabs()

    def new_project(self) -> None:
        if not self._confirm_replace_current_project("creating a new project"):
            return
        name = self._request_new_project_name()
        if name is None:
            return
        previous_project = self.project
        previous_project_active = self.project_active
        previous_project_path = self.project_path
        previous_selection = dict(self.current_tree_selection)
        previous_pole_window = getattr(self, "pole_figure_window", None)
        self.project = ProjectState(name=name)
        self.project_active = True
        self.project_path = None
        self.pole_figure_window = None
        self.current_tree_selection = {"kind": "root"}
        if not self.save_project_as():
            self.project = previous_project
            self.project_active = previous_project_active
            self.project_path = previous_project_path
            self.current_tree_selection = previous_selection
            self.pole_figure_window = previous_pole_window
            self._refresh_project_views()
            self._refresh_workflow_context()
            self._refresh_save_status()
            return
        self._refresh_project_views()
        self._refresh_workflow_context()

    def open_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open EWALD Project",
            str(self._project_directory()),
            PROJECT_FILE_FILTER,
        )
        if not path:
            return
        if not self._confirm_replace_current_project(
            "opening a different project"
        ):
            return
        self._open_project_path(Path(path))

    def _open_recent_project(self, path_text: str) -> None:
        path = Path(path_text)
        if not path.exists():
            self._forget_recent_project(path)
            QtWidgets.QMessageBox.information(
                self,
                "Recent Project Not Found",
                f"The recent project could not be found:\n{path}",
            )
            return
        if not self._confirm_replace_current_project(
            "opening a recent project"
        ):
            return
        self._open_project_path(path)

    def _open_project_path(self, path: Path) -> bool:
        try:
            project = load_project(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Open Project Failed",
                f"EWALD could not open this project:\n{path}\n\n{exc}",
            )
            return False
        self.project = project
        self.project_active = True
        self.project_path = normalize_project_path(path)
        self.pole_figure_window = None
        self._remember_project_directory(self.project_path.parent)
        self._remember_recent_project(self.project_path)
        self._refresh_save_status()
        self.current_tree_selection = {"kind": "root"}
        self._refresh_project_views()
        self._refresh_workflow_context()
        return True

    def load_files(self) -> None:
        if not self._require_project():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Detector Image File",
            "",
            "Detector Images (*.tif *.tiff)",
        )
        if not path:
            return
        data_name = self._request_data_file_name(Path(path))
        context = self._request_metadata_context("file")
        if context is None:
            return
        group, report = build_data_group_from_paths(
            [path],
            group_name=data_name,
            import_kind="file",
            metadata_type=context["metadata_type"],
            delimiter=context["delimiter"],
            metadata_yml=context["metadata_yml"],
        )
        if group.data_files:
            group.data_files[0].name = data_name
        self.project.add_data_group(group)
        self.current_tree_selection = _selection_for_group(group)
        self._review_group_metadata(
            group, report.files_requiring_metadata_input
        )
        self._refresh_project_views()
        self._refresh_workflow_context()

    def load_folder(self) -> None:
        if not self._require_project():
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Load Detector Image Folder",
        )
        if not folder:
            return
        context = self._request_metadata_context("folder")
        if context is None:
            return
        group, report = build_data_group_from_folder(
            folder,
            metadata_type=context["metadata_type"],
            delimiter=context["delimiter"],
            metadata_yml=context["metadata_yml"],
        )
        if not group.data_files:
            QtWidgets.QMessageBox.information(
                self,
                "No Detector Images",
                "No TIFF detector images were found in the selected folder.",
            )
            return
        self.project.add_data_group(group)
        self.current_tree_selection = {
            "kind": "group",
            "group_id": group.group_id,
            "name": group.name,
            "import_kind": group.import_kind,
        }
        self._review_group_metadata(
            group, report.files_requiring_metadata_input
        )
        self._refresh_project_views()
        self._refresh_workflow_context()

    def create_mask(self) -> None:
        self.open_pyfai_calibration_tool()

    def load_mask(self) -> None:
        if not self._require_project():
            return
        target_ids = self._selected_target_ids()
        if not self._require_data_target(target_ids):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Mask",
            "",
            "Mask Files (*.edf *.tif *.tiff *.npy *.npz);;All Files (*)",
        )
        if not path:
            return
        self.project.add_mask(path, target_ids=target_ids)
        self._refresh_after_correction_change()

    def create_calibrant(self) -> None:
        self.open_pyfai_calibration_tool()

    def load_calibrant(self) -> None:
        if not self._require_project():
            return
        target_ids = self._selected_target_ids()
        if not self._require_data_target(target_ids):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Calibrant PONI",
            "",
            "PONI Files (*.poni);;All Files (*)",
        )
        if not path:
            return
        self.project.add_calibrant(path, target_ids=target_ids)
        self._refresh_after_correction_change()

    def save_project(self) -> bool:
        if not self._require_project():
            return False
        if self.project_path is None:
            return self.save_project_as()
        self.project_path = save_project(self.project, self.project_path)
        self._remember_recent_project(self.project_path)
        self._mark_project_saved()
        return True

    def save_project_as(self) -> bool:
        if not self._require_project():
            return False
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save EWALD Project",
            str(self._suggested_project_save_path()),
            PROJECT_FILE_FILTER,
        )
        if not path:
            return False
        self.project_path = save_project(self.project, path)
        self._remember_project_directory(self.project_path.parent)
        self._remember_recent_project(self.project_path)
        self._mark_project_saved()
        return True

    def _mark_project_saved(self) -> None:
        self._refresh_save_status(saved_now=True)
        if self.project_path is not None:
            self.statusBar().showMessage(
                f"Project saved: {self.project_path}",
                4000,
            )

    def _refresh_save_status(self, *, saved_now: bool = False) -> None:
        if not hasattr(self, "save_status_label"):
            return
        if not self.project_active:
            text = "No project saved"
            tooltip = "Create or open a project to enable saving."
        elif self.project_path is None:
            text = "Not saved"
            tooltip = "Save the project to choose a project file."
        elif saved_now:
            timestamp = QtCore.QTime.currentTime().toString("h:mm:ss AP")
            text = f"Last saved: {timestamp}"
            tooltip = f"Project file: {self.project_path}"
        else:
            text = f"Saved: {self.project_path.name}"
            tooltip = f"Project file: {self.project_path}"
        self.save_status_label.setText(text)
        self.save_status_label.setToolTip(tooltip)

    def _request_new_project_name(self) -> str | None:
        while True:
            name, accepted = QtWidgets.QInputDialog.getText(
                self,
                "New EWALD Project",
                "Project name:",
                text="Untitled EWALD Project",
            )
            if not accepted:
                return None
            name = name.strip()
            if name:
                return name
            QtWidgets.QMessageBox.information(
                self,
                "Project Name Required",
                "Enter a project name before creating a new EWALD project.",
            )

    def _project_directory(self) -> Path:
        saved = self.settings.value(PROJECT_DIRECTORY_SETTING, "")
        if saved:
            return Path(str(saved)).expanduser()
        return _default_project_directory()

    def _suggested_project_save_path(self) -> Path:
        directory = self._project_directory()
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{_project_filename_stem(self.project.name)}.ewld"

    def _remember_project_directory(self, directory: Path) -> None:
        self.settings.setValue(PROJECT_DIRECTORY_SETTING, str(directory))
        self.settings.sync()

    def _recent_project_paths(self) -> list[Path]:
        raw_value = self.settings.value(RECENT_PROJECTS_SETTING, [])
        if isinstance(raw_value, str):
            values = [raw_value]
        else:
            values = list(raw_value or [])
        paths: list[Path] = []
        seen: set[str] = set()
        for value in values:
            path = Path(str(value)).expanduser()
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
        return paths[:MAX_RECENT_PROJECTS]

    def _remember_recent_project(self, path: Path) -> None:
        normalized = normalize_project_path(path).expanduser()
        recent = [
            candidate
            for candidate in self._recent_project_paths()
            if candidate != normalized
        ]
        recent.insert(0, normalized)
        self.settings.setValue(
            RECENT_PROJECTS_SETTING,
            [str(candidate) for candidate in recent[:MAX_RECENT_PROJECTS]],
        )
        self.settings.sync()
        if hasattr(self, "recent_projects_menu"):
            self._refresh_recent_projects_menu()

    def _forget_recent_project(self, path: Path) -> None:
        normalized = normalize_project_path(path).expanduser()
        recent = [
            candidate
            for candidate in self._recent_project_paths()
            if candidate != normalized
        ]
        self.settings.setValue(
            RECENT_PROJECTS_SETTING,
            [str(candidate) for candidate in recent],
        )
        self.settings.sync()
        if hasattr(self, "recent_projects_menu"):
            self._refresh_recent_projects_menu()

    def _clear_recent_projects(self) -> None:
        self.settings.setValue(RECENT_PROJECTS_SETTING, [])
        self.settings.sync()
        self._refresh_recent_projects_menu()

    def _refresh_recent_projects_menu(self) -> None:
        self.recent_projects_menu.clear()
        recent = self._recent_project_paths()
        if not recent:
            empty_action = self.recent_projects_menu.addAction(
                "No Recent Projects"
            )
            empty_action.setEnabled(False)
            self.recent_projects_menu.setEnabled(False)
            return
        self.recent_projects_menu.setEnabled(True)
        for path in recent:
            action = self.recent_projects_menu.addAction(path.name)
            action.setToolTip(str(path))
            action.setStatusTip(f"Open {path}")
            action.triggered.connect(
                lambda _checked=False, recent_path=str(path): (
                    self._open_recent_project(recent_path)
                )
            )
        self.recent_projects_menu.addSeparator()
        clear_action = self.recent_projects_menu.addAction("Clear Recent")
        clear_action.triggered.connect(self._clear_recent_projects)

    def _confirm_replace_current_project(self, action: str) -> bool:
        if not self.project_active:
            return True
        response = QtWidgets.QMessageBox.question(
            self,
            "Save Current Project?",
            f"Do you want to save the current project before {action}?",
            (
                QtWidgets.QMessageBox.StandardButton.Save
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel
            ),
            QtWidgets.QMessageBox.StandardButton.Save,
        )
        if response == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save_project()
        return response == QtWidgets.QMessageBox.StandardButton.Discard

    def _handle_tree_selection(self, payload: dict) -> None:
        self.current_tree_selection = payload
        self._refresh_workflow_context()
        self._refresh_project_summary()
        self._refresh_right_tabs()

    def _refresh_workflow_context(self) -> None:
        label = self._selection_label()
        self.workflow_context_label.setText(label)
        self.load_files_action.setEnabled(self.project_active)
        self.load_folder_action.setEnabled(self.project_active)
        self.save_project_action.setEnabled(self.project_active)
        self.save_project_as_action.setEnabled(self.project_active)
        self.data_tree.set_project_active(self.project_active)
        has_target = self.project_active and bool(self._selected_target_ids())
        locked = self._selected_file_corrections_confirmed()
        self.giwaxs_simulation_action.setEnabled(True)
        self.pole_figure_action.setEnabled(self.project_active)
        self.pyfai_calibration_action.setEnabled(True)
        self.create_mask_action.setEnabled(True)
        self.create_calibrant_action.setEnabled(True)
        for action in (self.load_mask_action, self.load_calibrant_action):
            action.setEnabled(has_target and not locked)

    def _selection_label(self) -> str:
        if not self.project_active:
            return "No Project"
        payload = self.current_tree_selection
        kind = payload.get("kind")
        if kind == "file":
            return (
                f"Data file: {payload.get('name') or payload.get('data_id')}"
            )
        if kind == "group":
            import_kind = payload.get("import_kind")
            group_label = (
                "Data folder" if import_kind == "folder" else "Data set"
            )
            return f"{group_label}: {payload.get('name')}"
        if kind == "correction-asset":
            return f"{payload.get('name')} ({payload.get('asset_kind')})"
        return "Project"

    def _selection_summary_lines(self) -> list[str]:
        if not self.project_active:
            return []
        payload = self.current_tree_selection
        target_ids = self._selected_target_ids()
        if not target_ids:
            return ["", f"Selected: {self._selection_label()}"]
        target_id = target_ids[0]
        inherited_id = (
            payload.get("group_id") if payload.get("kind") == "file" else None
        )
        return [
            "",
            f"Selected: {self._selection_label()}",
            f"Mask: {self._asset_label('mask', target_id, inherited_id)}",
            f"Calibrant: {self._asset_label('calibrant', target_id, inherited_id)}",
        ]

    def _asset_label(
        self,
        asset_kind: str,
        target_id: str | None,
        inherited_target_id: str | None = None,
    ) -> str:
        direct = self.project.assigned_assets(asset_kind, target_id)
        if direct:
            return ", ".join(asset.name for asset in direct)
        inherited = self.project.assigned_assets(
            asset_kind, inherited_target_id
        )
        if inherited:
            return ", ".join(f"{asset.name} (folder)" for asset in inherited)
        return "None"

    def _selected_target_ids(self) -> list[str]:
        payload = self.current_tree_selection
        if payload.get("kind") == "group" and payload.get("group_id"):
            return [str(payload["group_id"])]
        if payload.get("kind") == "file" and payload.get("data_id"):
            return [str(payload["data_id"])]
        return []

    def _require_data_target(self, target_ids: list[str]) -> bool:
        if target_ids:
            return True
        QtWidgets.QMessageBox.information(
            self,
            "Select Data",
            "Select a loaded data file or data folder first.",
        )
        return False

    def _selected_file_corrections_confirmed(self) -> bool:
        if self.current_tree_selection.get("kind") != "file":
            return False
        return self.project.image_corrections_confirmed(
            self.current_tree_selection.get("data_id")
        )

    def _require_project(self) -> bool:
        if self.project_active:
            return True
        QtWidgets.QMessageBox.information(
            self,
            "No Project",
            "Create a new project or load an existing .ewld project first.",
        )
        return False

    def _created_asset_name(self, prefix: str) -> str:
        if prefix == "Mask":
            index = len(self.project.masks) + 1
        else:
            index = len(self.project.calibrants) + 1
        return f"{prefix} {index}"

    def _refresh_after_correction_change(self) -> None:
        tab_title = self._current_tab_title()
        self._refresh_project_views()
        self._refresh_workflow_context()
        self._restore_tab(tab_title)

    def _current_tab_title(self) -> str | None:
        index = self.tabs.currentIndex()
        if index < 0:
            return None
        return self.tabs.tabText(index)

    def _restore_tab(self, title: str | None) -> None:
        if title is None:
            return
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == title:
                self.tabs.setCurrentIndex(index)
                return

    def _refresh_right_tabs(self) -> None:
        self.tabs.clear()
        if not self.project_active:
            self.tabs.addTab(self.project_summary, "Project")
            return

        simulation_id = self._selected_simulation_id()
        if simulation_id is not None:
            pane = self._create_embedded_simulation_pane(
                self._selected_simulation_data_id()
            )
            pane.select_simulation(simulation_id)
            self.tabs.addTab(pane, "GIWAXS Simulation")
            return

        data_id = self._selected_viewer_data_id()
        if data_id is None:
            self.tabs.addTab(self.project_summary, "Project")
            self.tabs.addTab(
                self._create_embedded_simulation_pane(None),
                "GIWAXS Simulation",
            )
            return

        viewer = DataViewerPane(self.project, str(data_id))
        viewer.roiRegionsChanged.connect(self._handle_roi_regions_changed)
        viewer.previewOrientationChanged.connect(
            self._handle_preview_orientation_changed
        )
        viewer.integrationPeakMarkersPushed.connect(
            self._handle_integration_peak_markers_pushed
        )
        viewer.poleFigureRequested.connect(self.open_pole_figure_tool)
        self.tabs.addTab(viewer, "Data Viewer")

        if not self.project.image_corrections_confirmed(data_id):
            pane = ApplyImageCorrectionsPane(
                self.project,
                str(data_id),
                group_id=self.current_tree_selection.get("group_id"),
                settings=self.settings,
                load_mask_action=self.load_mask_action,
                load_calibrant_action=self.load_calibrant_action,
                pyfai_calibration_action=self.pyfai_calibration_action,
                pyfai_calibration_status=(
                    self._pyfai_calibration_status_text()
                ),
                pyfai_calibration_status_tooltip=(
                    self._pyfai_calibration_status_tooltip()
                ),
            )
            pane.correctionsApplied.connect(
                self._handle_image_corrections_applied
            )
            pane.correctionsConfirmed.connect(
                self._handle_image_corrections_confirmed
            )
            self.tabs.addTab(pane, "Apply Image Corrections")
            self.tabs.addTab(
                self._create_embedded_simulation_pane(str(data_id)),
                "GIWAXS Simulation",
            )
            return

        peak_pane = PeakIdentificationPane(
            self.project,
            str(data_id),
            image_style=viewer.image_display_style(),
        )
        viewer.imageStyleChanged.connect(peak_pane.apply_image_style)
        peak_pane.peakSetChanged.connect(self._handle_peak_set_changed)
        self.tabs.addTab(peak_pane, "Peak Identification")

        structure_pane = StructureAnalysisPane(
            self.project,
            str(data_id),
            image_data=viewer.image_data,
            axis_ranges=viewer.axis_ranges,
            coordinate_space=viewer.coordinate_space,
            image_style=viewer.image_display_style(),
            project_path=self.project_path,
        )
        viewer.imageStyleChanged.connect(structure_pane.apply_image_style)
        viewer.roiRegionsChanged.connect(structure_pane.refresh_roi_overlays)
        peak_pane.peakSetChanged.connect(structure_pane.refresh_from_peak_fit)
        structure_pane.structureAnalysisChanged.connect(
            self._handle_structure_analysis_changed
        )
        structure_pane.candidateOverlayRequested.connect(
            self._handle_structure_candidate_overlay_requested
        )
        self.tabs.addTab(structure_pane, "Structure Analysis")

        simulation_pane = self._create_embedded_simulation_pane(str(data_id))
        structure_pane.structureAnalysisChanged.connect(
            lambda changed_data_id, pane=simulation_pane: (
                self._refresh_embedded_simulation_from_structure_analysis(
                    pane,
                    changed_data_id,
                )
            )
        )
        self.tabs.addTab(
            simulation_pane,
            "GIWAXS Simulation",
        )

    def _create_embedded_simulation_pane(
        self,
        target_data_id: str | None,
    ) -> GIWAXSSimulationPane:
        pane = GIWAXSSimulationPane(
            project=self.project,
            project_path=self.project_path,
            initial_data_id=target_data_id,
            settings=self.settings,
            parent=self,
        )
        pane.projectChanged.connect(
            lambda data_id=target_data_id: (
                self._handle_embedded_simulation_project_changed(data_id)
            )
        )
        self.giwaxs_simulation_window = pane
        return pane

    def _embedded_simulation_tab(self) -> GIWAXSSimulationPane | None:
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            if isinstance(widget, GIWAXSSimulationPane):
                return widget
        return None

    def _handle_image_corrections_applied(self, data_id: str) -> None:
        self._refresh_project_summary()
        self.data_tree.set_project(self.project)
        self._refresh_workflow_context()

    def _handle_image_corrections_confirmed(self, data_id: str) -> None:
        self.current_tree_selection = {
            **self.current_tree_selection,
            "kind": "file",
            "data_id": data_id,
        }
        self._refresh_project_views()
        self._refresh_workflow_context()

    def _handle_roi_regions_changed(self, data_id: str) -> None:
        self._refresh_project_summary()
        self.data_tree.set_project(self.project)

    def _handle_pole_figure_saved(self, data_id: str, roi_id: str) -> None:
        self._refresh_project_summary()
        self.data_tree.set_project(self.project)
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            if (
                isinstance(widget, DataViewerPane)
                and widget.data_id == data_id
            ):
                widget.refresh_roi_table()
                widget._select_roi(roi_id)
                return

    def _handle_simulation_created(self, simulation_id: str) -> None:
        self._activate_project_from_simulation()
        self.current_tree_selection = {
            "kind": "simulation",
            "simulation_id": simulation_id,
        }
        self._refresh_project_views()
        self._refresh_workflow_context()

    def _handle_simulation_linked(self, simulation_id: str) -> None:
        self._activate_project_from_simulation()
        self.current_tree_selection = {
            "kind": "simulation",
            "simulation_id": simulation_id,
        }
        self._refresh_project_views()
        self._refresh_workflow_context()

    def _handle_simulation_project_changed(self) -> None:
        self._activate_project_from_simulation()
        self._refresh_project_summary()
        self.data_tree.set_project(self.project)
        self._refresh_workflow_context()

    def _activate_project_from_simulation(self) -> None:
        if self.project_active:
            return
        self.project_active = True
        self._refresh_save_status()

    def _handle_embedded_simulation_project_changed(
        self,
        data_id: str | None,
    ) -> None:
        self._activate_project_from_simulation()
        self._refresh_project_summary()
        self.data_tree.set_project(self.project)
        self._refresh_workflow_context()

    def _refresh_embedded_simulation_from_structure_analysis(
        self,
        pane: GIWAXSSimulationPane,
        data_id: str,
    ) -> None:
        if pane.selected_data_id() != data_id:
            return
        pane.refresh_project_context(
            project_path=self.project_path,
            target_data_id=data_id,
        )

    def _handle_peak_set_changed(self, data_id: str) -> None:
        self._refresh_project_summary()
        self.data_tree.set_project(self.project)

    def _handle_structure_analysis_changed(self, data_id: str) -> None:
        self._refresh_project_summary()
        self.data_tree.set_project(self.project)

    def _handle_structure_candidate_overlay_requested(
        self,
        data_id: str,
    ) -> None:
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            if (
                isinstance(widget, PeakIdentificationPane)
                and widget.data_id == data_id
            ):
                widget.restore_crystal_overlay_from_project()
                self.tabs.setCurrentIndex(index)
                return

    def _handle_integration_peak_markers_pushed(
        self,
        data_id: str,
        markers: object,
    ) -> None:
        marker_list = list(markers) if isinstance(markers, list) else []
        if not marker_list:
            return
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            if (
                isinstance(widget, PeakIdentificationPane)
                and widget.data_id == data_id
            ):
                widget.add_integration_markers(marker_list)
                self._handle_peak_set_changed(data_id)
                return

    def _handle_preview_orientation_changed(self, data_id: str) -> None:
        self._refresh_project_summary()
        self.data_tree.set_project(self.project)
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            if (
                isinstance(widget, ApplyImageCorrectionsPane)
                and widget.data_id == data_id
            ):
                widget.restore_orientation_from_state()

    def _selected_viewer_data_id(self) -> str | None:
        if self.current_tree_selection.get("kind") == "file":
            data_id = self.current_tree_selection.get("data_id")
            return str(data_id) if data_id else None
        if self.current_tree_selection.get("kind") == "root":
            data_files = _project_data_files(self.project)
            if len(data_files) == 1:
                return data_files[0].data_id
        return None

    def _active_data_viewer(self) -> DataViewerPane | None:
        widget = self.tabs.currentWidget()
        if isinstance(widget, DataViewerPane):
            return widget
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            if isinstance(widget, DataViewerPane):
                return widget
        return None

    def _selected_simulation_id(self) -> str | None:
        if self.current_tree_selection.get("kind") != "simulation":
            return None
        simulation_id = self.current_tree_selection.get("simulation_id")
        return str(simulation_id) if simulation_id else None

    def _selected_simulation_data_id(self) -> str | None:
        simulation_id = self._selected_simulation_id()
        if simulation_id is None:
            return None
        record = self.project.simulations.get(simulation_id)
        if not isinstance(record, dict):
            return None
        data_id = record.get("data_id")
        return str(data_id) if data_id else None

    def _selected_file_data_id(self) -> str | None:
        if self.current_tree_selection.get("kind") != "file":
            return None
        data_id = self.current_tree_selection.get("data_id")
        return str(data_id) if data_id else None

    def _request_metadata_context(
        self, import_kind: str
    ) -> dict[str, str | None] | None:
        dialog = MetadataImportContextDialog(import_kind, parent=self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.context()

    def _request_data_file_name(self, path: Path) -> str:
        default_name = path.stem
        name, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Data File Name",
            "Name for this data file (leave blank to use the file name):",
            text=default_name,
        )
        if not accepted:
            return default_name
        return name.strip() or default_name

    def _review_group_metadata(
        self,
        group,
        files_requiring_metadata_input: list[str],
    ) -> None:
        has_reviewable_metadata = any(
            data_file.metadata.get("_metadata_fields")
            or data_file.metadata.get("_unresolved_tokens")
            for data_file in group.data_files
        )
        if not has_reviewable_metadata and not files_requiring_metadata_input:
            return
        dialog = ManualMetadataDialog(group, parent=self)
        if dialog.has_review_rows():
            dialog.exec()


class MetadataImportContextDialog(QtWidgets.QDialog):
    """Collect user context for filename and sidecar metadata
    parsing."""

    def __init__(
        self,
        import_kind: str,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.import_kind = import_kind
        self.setWindowTitle("Metadata Import Context")
        self.metadata_type_combo = QtWidgets.QComboBox()
        self._populate_metadata_types(import_kind)
        self.delimiter_edit = QtWidgets.QLineEdit("_")
        self.sidecar_edit = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self._browse_sidecar)

        sidecar_layout = QtWidgets.QHBoxLayout()
        sidecar_layout.addWidget(self.sidecar_edit)
        sidecar_layout.addWidget(browse_button)

        form = QtWidgets.QFormLayout()
        form.addRow("Metadata type", self.metadata_type_combo)
        form.addRow("Filename delimiter", self.delimiter_edit)
        form.addRow("Metadata YAML", sidecar_layout)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def context(self) -> dict[str, str | None]:
        sidecar = self.sidecar_edit.text().strip()
        return {
            "metadata_type": self.metadata_type_combo.currentData(),
            "delimiter": self.delimiter_edit.text() or "_",
            "metadata_yml": sidecar or None,
        }

    def _populate_metadata_types(self, import_kind: str) -> None:
        if import_kind == "folder":
            options = [
                ("Filename tokens per file", "filename"),
                ("Filename tokens + folder YAML", "filename+yml"),
                ("TIFF headers per file", "header"),
                ("TIFF headers + folder YAML", "header+yml"),
                ("Folder YAML only", "yml"),
            ]
        else:
            options = [
                ("Filename tokens", "filename"),
                ("Filename tokens + sidecar YAML", "filename+yml"),
                ("TIFF header", "header"),
                ("TIFF header + sidecar YAML", "header+yml"),
                ("Sidecar YAML only", "yml"),
            ]
        for label, value in options:
            self.metadata_type_combo.addItem(label, value)

    def _browse_sidecar(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Metadata YAML",
            "",
            "YAML Files (*.yml *.yaml)",
        )
        if path:
            self.sidecar_edit.setText(path)


def _project_data_file_count(project: ProjectState) -> int:
    return len(_project_data_files(project))


def _project_roi_count(project: ProjectState) -> int:
    return sum(len(regions) for regions in project.roi_regions.values())


def _project_data_files(project: ProjectState):
    data_files = []
    for group in project.data_groups:
        data_files.extend(group.data_files)
    data_files.extend(project.data_files)
    return data_files


def _default_project_directory() -> Path:
    source_path = Path(__file__).resolve()
    for parent in source_path.parents:
        candidate = parent / "example" / "projects"
        if candidate.exists():
            return candidate
    return Path.home() / "EWALD" / "projects"


def _project_filename_stem(name: str) -> str:
    safe = "".join(
        (
            character
            if character.isalnum() or character in {" ", ".", "_", "-"}
            else "_"
        )
        for character in name
    )
    return safe.strip(" ._") or "ewald_project"


def _selection_for_group(group) -> dict[str, str | None]:
    if group.import_kind == "file" and len(group.data_files) == 1:
        data_file = group.data_files[0]
        name = data_file.name or data_file.data_id
        return {
            "kind": "file",
            "data_id": data_file.data_id,
            "group_id": group.group_id,
            "name": name,
        }
    return {
        "kind": "group",
        "group_id": group.group_id,
        "name": group.name,
        "import_kind": group.import_kind,
    }


def _developer_information_text() -> str:
    return (
        f"{APP_TITLE}\n\n"
        f"Developer: {DEVELOPER_NAME}\n"
        f"Email: {DEVELOPER_EMAIL}\n"
        f"GitHub: {GITHUB_URL}"
    )


def _version_information_text() -> str:
    return (
        f"{APP_TITLE}\n\n"
        f"Version: {__version__}\n"
        f"Project schema: {ProjectState().schema_version}"
    )
