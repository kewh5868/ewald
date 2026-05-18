"""Tests for the fresh experimental data tree and data viewer panes."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from ewald.data.models import (
    PEAK_POINT_KIND_COMMITTED,
    PEAK_POINT_KIND_GAP_ESTIMATED,
    ImageCorrectionState,
    ProjectState,
    ROIRegion,
)
from ewald.io.importers import (
    build_data_group_from_folder,
    build_data_group_from_paths,
)
from ewald.io.project import load_project
from ewald.io.project import save_project as write_project
from ewald.ui.data_tree import DataTreePane
from ewald.ui.data_viewer import (
    ROI_COL_COUPLED,
    ROI_COL_H,
    ROI_COL_HKL_LABEL,
    ROI_COL_K,
    ROI_COL_L,
    ROI_COL_POLE_FIGURE,
    ROI_COL_QXY_MAX,
    DataViewerPane,
    _apply_image_orientation,
    _apply_image_rotation,
    _integration_peak_qspace_coordinate,
)
from ewald.ui.main_window import (
    APP_TITLE,
    GITHUB_URL,
    MainWindow,
    MetadataImportContextDialog,
    _developer_information_text,
    _version_information_text,
)
from ewald.ui.metadata_dialog import ManualMetadataDialog
from ewald.ui.orientation import sample_orientation_for_image_transform
from ewald.version import __version__


def _child_with_text(parent, text):
    for index in range(parent.childCount()):
        child = parent.child(index)
        if child.text(0) == text:
            return child
    raise AssertionError(f"No child named {text!r}")


def _write_si_poscar(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "Si",
                "1.0",
                "5.43 0.0 0.0",
                "0.0 5.43 0.0",
                "0.0 0.0 5.43",
                "Si",
                "2",
                "Direct",
                "0.0 0.0 0.0",
                "0.25 0.25 0.25",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_data_tree_renders_groups_files_metadata_and_fits(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    project = ProjectState()
    project.add_data_group(group)
    project.fits[group.data_files[0].data_id] = [
        {"model_name": "gaussian-2d", "center_qx": 1.0, "center_qz": 0.5}
    ]

    pane = DataTreePane()
    qtbot.addWidget(pane)
    pane.set_project(project)

    root = pane.tree.topLevelItem(0)
    assert root.text(0) == "Untitled EWALD Project"
    file_item = root.child(0)
    assert file_item.text(0).startswith("sam22_")
    assert file_item.text(1) == "detector-image"
    assert [file_item.child(index).text(0) for index in range(5)] == [
        "Metadata",
        "Available Processing",
        "ROIs",
        "Linked Simulations",
        "Fits",
    ]
    metadata_item = file_item.child(0)
    assert (
        _child_with_text(metadata_item, "Data type").text(1)
        == "detector-image"
    )
    assert _child_with_text(metadata_item, "Sample number").text(1) == "22"
    assert _child_with_text(metadata_item, "Parse delimiter").text(1) == "_"


def test_main_window_has_left_data_dock(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert APP_TITLE == "EWALD"
    assert window.windowTitle() == APP_TITLE
    assert window.data_dock.windowTitle() == "Experimental Data"
    assert window.data_tree.tree.headerItem().text(0) == "Experimental Data"
    assert window.data_tree.new_button.toolTip() == "New Project"
    assert window.data_tree.import_file_button.toolTip() == "Import Data File"
    assert (
        window.data_tree.import_folder_button.toolTip() == "Import Data Folder"
    )
    assert (
        window.data_tree.project_setup_layout.indexOf(
            window.data_tree.new_button
        )
        >= 0
    )
    assert (
        window.data_tree.project_setup_layout.indexOf(
            window.data_tree.load_button
        )
        >= 0
    )
    assert (
        window.data_tree.data_import_layout.indexOf(
            window.data_tree.import_file_button
        )
        >= 0
    )
    assert (
        window.data_tree.data_import_layout.indexOf(
            window.data_tree.import_folder_button
        )
        >= 0
    )
    assert [action.text() for action in window.menuBar().actions()] == [
        "File",
        "View",
        "Tools",
        "Help",
    ]
    assert window.view_menu.title() == "View"
    assert [action.text() for action in window.view_menu.actions()] == [
        "File Manager"
    ]
    assert window.toggle_file_manager_action.isCheckable()
    assert window.toggle_file_manager_action.isChecked()
    assert not window.data_dock.isHidden()
    window.toggle_file_manager_action.trigger()
    assert window.data_dock.isHidden()
    assert not window.toggle_file_manager_action.isChecked()
    window.toggle_file_manager_action.trigger()
    assert not window.data_dock.isHidden()
    assert window.toggle_file_manager_action.isChecked()
    window.data_dock.hide()
    assert window.data_dock.isHidden()
    assert not window.toggle_file_manager_action.isChecked()
    assert window.tools_menu.title() == "Tools"
    assert [action.text() for action in window.tools_menu.actions()] == [
        "Load Mask",
        "Load Calibrant",
        "",
        "PyFAI Calibration/Mask Tool",
        "",
        "GIWAXS Simulation",
        "Pole Figure Generator",
    ]
    assert window.help_menu.title() == "Help"
    assert [action.text() for action in window.help_menu.actions()] == [
        "GitHub Repository",
        "Developer Information",
        "Version Information",
    ]
    assert GITHUB_URL in _developer_information_text()
    assert "Keith White" in _developer_information_text()
    assert __version__ in _version_information_text()
    assert window.save_project_action.shortcut() == QtGui.QKeySequence(
        QtGui.QKeySequence.StandardKey.Save
    )
    assert window.workflow_context_label.text() == "No Project"
    assert not window.load_files_action.isEnabled()
    assert not window.giwaxs_simulation_action.isEnabled()
    assert window.pyfai_calibration_action.isEnabled()
    assert not window.data_tree.import_file_button.isEnabled()


def test_new_project_enables_import_and_save_actions(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("Named Project", True),
    )

    window.new_project()

    assert window.project_active
    assert window.project.name == "Named Project"
    assert window.load_files_action.isEnabled()
    assert window.save_project_action.isEnabled()
    assert window.data_tree.import_file_button.isEnabled()


def test_load_file_prompts_for_display_name_and_preserves_original_filename(
    qtbot, repo_root, monkeypatch
):
    path = next((repo_root / "example").glob("*.tiff"))
    project = ProjectState(name="Import Project")
    window = MainWindow(project=project)
    qtbot.addWidget(window)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(path), "Detector Images (*.tif *.tiff)"),
    )
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("Named Detector Image", True),
    )
    monkeypatch.setattr(
        window,
        "_request_metadata_context",
        lambda import_kind: {
            "metadata_type": "filename",
            "delimiter": "_",
            "metadata_yml": None,
        },
    )
    monkeypatch.setattr(
        window,
        "_review_group_metadata",
        lambda group, files_requiring_metadata_input: None,
    )

    window.load_files()

    data_file = project.data_groups[0].data_files[0]
    assert data_file.data_id == path.stem
    assert data_file.name == "Named Detector Image"
    assert data_file.metadata["original_file_name"] == path.name
    root = window.data_tree.tree.topLevelItem(0)
    file_item = root.child(0)
    assert file_item.text(0) == "Named Detector Image"
    assert window.workflow_context_label.text() == (
        "Data file: Named Detector Image"
    )
    metadata_item = file_item.child(0)
    assert _child_with_text(metadata_item, "Name").text(1) == (
        "Named Detector Image"
    )
    assert (
        _child_with_text(metadata_item, "Original file name").text(1)
        == path.name
    )


def test_data_file_name_prompt_defaults_to_filename(qtbot, monkeypatch):
    window = MainWindow(project=ProjectState(name="Import Project"))
    qtbot.addWidget(window)
    path = Path("/tmp/example_detector.tiff")
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("", True),
    )

    assert window._request_data_file_name(path) == "example_detector"


def test_data_file_name_prompt_cancel_defaults_to_filename(qtbot, monkeypatch):
    window = MainWindow(project=ProjectState(name="Import Project"))
    qtbot.addWidget(window)
    path = Path("/tmp/example_detector.tiff")
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("Ignored", False),
    )

    assert window._request_data_file_name(path) == "example_detector"


def test_new_project_offers_to_save_current_project(
    qtbot, tmp_path, monkeypatch
):
    project = ProjectState(name="Current Project")
    project.add_data_file("current_001.tiff")
    window = MainWindow(project=project)
    qtbot.addWidget(window)
    window.project_path = tmp_path / "current_project.ewld"
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Save,
    )
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("Fresh Project", True),
    )

    window.new_project()

    saved = load_project(tmp_path / "current_project.ewld")
    assert saved.name == "Current Project"
    assert window.project_active
    assert window.project_path is None
    assert window.project.name == "Fresh Project"


def test_new_project_cancel_keeps_current_project(
    qtbot, tmp_path, monkeypatch
):
    project = ProjectState(name="Current Project")
    window = MainWindow(project=project)
    qtbot.addWidget(window)
    window.project_path = tmp_path / "current_project.ewld"
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Cancel,
    )

    window.new_project()

    assert window.project is project
    assert window.project_path == tmp_path / "current_project.ewld"


def test_open_project_offers_to_discard_current_project(
    qtbot, tmp_path, monkeypatch
):
    current = ProjectState(name="Current Project")
    next_project = ProjectState(name="Next Project")
    next_path = write_project(next_project, tmp_path / "next_project")
    window = MainWindow(project=current)
    qtbot.addWidget(window)
    window.project_path = tmp_path / "current_project.ewld"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(next_path), "EWALD Projects (*.ewld)"),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Discard,
    )

    window.open_project()

    assert window.project.name == "Next Project"
    assert window.project_path == next_path


def test_new_project_cancelled_name_keeps_current_state(qtbot, monkeypatch):
    project = ProjectState(name="Current Project")
    window = MainWindow(project=project)
    qtbot.addWidget(window)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Discard,
    )
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("", False),
    )

    window.new_project()

    assert window.project is project
    assert window.project.name == "Current Project"


def test_loaded_project_window_saves_back_to_project_path(qtbot, tmp_path):
    saved_path = write_project(
        ProjectState(name="Reloaded Project"),
        tmp_path / "reloaded_project",
    )
    loaded = load_project(saved_path)
    window = MainWindow(project=loaded, project_path=saved_path)
    qtbot.addWidget(window)

    window.project.name = "Reloaded Project Updated"

    assert window.save_project()
    assert window.project_path == saved_path
    assert load_project(saved_path).name == "Reloaded Project Updated"


def test_save_project_as_defaults_and_remembers_project_folder(
    qtbot, repo_root, tmp_path, monkeypatch
):
    settings_path = tmp_path / "settings.ini"
    settings = QtCore.QSettings(
        str(settings_path),
        QtCore.QSettings.Format.IniFormat,
    )
    window = MainWindow(
        project=ProjectState(name="Default Path Project"),
        settings=settings,
    )
    qtbot.addWidget(window)
    first_capture = {}
    custom_dir = tmp_path / "custom_projects"
    custom_path = custom_dir / "custom_project.ewld"

    def first_save_dialog(*args, **kwargs):
        first_capture["path"] = args[2]
        return str(custom_path), "EWALD Projects (*.ewld)"

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        first_save_dialog,
    )

    assert window.save_project_as()
    assert Path(first_capture["path"]).parent == repo_root / "example/projects"
    assert window.project_path == custom_path

    next_settings = QtCore.QSettings(
        str(settings_path),
        QtCore.QSettings.Format.IniFormat,
    )
    next_window = MainWindow(
        project=ProjectState(name="Next Project"),
        settings=next_settings,
    )
    qtbot.addWidget(next_window)
    second_capture = {}

    def second_save_dialog(*args, **kwargs):
        second_capture["path"] = args[2]
        return "", ""

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        second_save_dialog,
    )

    assert not next_window.save_project_as()
    assert Path(second_capture["path"]).parent == custom_dir


def test_save_project_as_keeps_active_data_viewer_channel_plots(
    qtbot, repo_root, tmp_path, monkeypatch
):
    class DummyPeakIdentificationPane(QtWidgets.QWidget):
        peakSetChanged = QtCore.Signal(str)

        def __init__(self, *_args, **_kwargs):
            super().__init__()

        def apply_image_style(self, *_args):
            pass

    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState()
    project.add_data_group(group)
    mask = project.add_mask(
        repo_root / "example" / "mask.edf", target_ids=[data_id]
    )
    calibrant = project.add_calibrant(
        repo_root / "example" / "calib.poni", target_ids=[data_id]
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            confirmed=True,
        )
    )
    monkeypatch.setattr(
        "ewald.ui.main_window.PeakIdentificationPane",
        DummyPeakIdentificationPane,
    )
    window = MainWindow(project=project)
    qtbot.addWidget(window)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (
            str(tmp_path / "channel_project.ewld"),
            "EWALD Projects (*.ewld)",
        ),
    )

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    viewer = window.tabs.widget(0)
    roi = viewer.add_roi_from_bounds(0.1, 0.4, -1.5, -0.7)
    assert roi is not None
    viewer._toggle_roi_channel(roi.roi_id, 1, True)

    assert viewer.channel_assignments[1] == {roi.roi_id}
    assert len(viewer.channel_panels[1].series) == 1

    assert window.save_project_as()

    assert window.tabs.widget(0) is viewer
    assert viewer.channel_assignments[1] == {roi.roi_id}
    assert len(viewer.channel_panels[1].series) == 1


def test_folder_groups_and_file_nodes_show_different_processing_scope(
    qtbot, repo_root
):
    group, _ = build_data_group_from_folder(repo_root / "example")
    project = ProjectState()
    project.add_data_group(group)

    pane = DataTreePane()
    qtbot.addWidget(pane)
    pane.set_project(project)

    group_item = pane.tree.topLevelItem(0).child(0)
    group_scope = group_item.child(1)
    assert group_scope.child(0).text(1) == "time/temperature workflows"
    assert group_scope.child(2).text(1) == "single-image fits"

    file_item = group_item.child(2)
    file_scope = file_item.child(1)
    assert file_scope.child(0).text(1) == "single-image fits"
    assert file_scope.child(1).text(1) == "lattice determination"


def test_data_tree_renders_masks_calibrants_and_assignments(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    project = ProjectState()
    project.add_data_group(group)
    data_id = group.data_files[0].data_id
    project.add_mask(
        repo_root / "example" / "mask.edf",
        target_ids=[group.group_id],
    )
    project.add_calibrant(
        repo_root / "example" / "calib.poni",
        target_ids=[data_id],
    )

    pane = DataTreePane()
    qtbot.addWidget(pane)
    pane.set_project(project)

    root = pane.tree.topLevelItem(0)
    file_item = root.child(0)
    file_metadata = file_item.child(0)
    assert _child_with_text(file_metadata, "Mask").text(1) == "mask (folder)"
    assert _child_with_text(file_metadata, "Calibrant").text(1) == "calib"

    masks_item = _child_with_text(root, "Masks")
    assert _child_with_text(masks_item, "Name").text(1) == "mask"
    assert _child_with_text(masks_item, "Path").text(1).endswith("mask.edf")
    mask_applied_item = _child_with_text(masks_item, "Applied To")
    assert mask_applied_item.child(0).text(1) == "Example"

    calibrants_item = _child_with_text(root, "Calibrants")
    assert _child_with_text(calibrants_item, "Name").text(1) == "calib"
    assert (
        _child_with_text(calibrants_item, "Path")
        .text(1)
        .endswith("calib.poni")
    )
    calibrant_applied_item = _child_with_text(calibrants_item, "Applied To")
    assert calibrant_applied_item.child(0).text(1) == data_id


def test_main_window_toolbar_tracks_data_tree_selection(
    qtbot, repo_root, monkeypatch
):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    project = ProjectState()
    project.add_data_group(group)
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)

    data_id = group.data_files[0].data_id
    assert window.workflow_context_label.text() == f"Data file: {data_id}"
    assert window.load_mask_action.isEnabled()
    assert window.load_calibrant_action.isEnabled()
    assert window.pyfai_calibration_action.isEnabled()

    launched = {"count": 0}

    def launch_pyfai_calib2():
        launched["count"] += 1
        return True

    monkeypatch.setattr(
        window.pyfai_calib2_launcher,
        "launch",
        launch_pyfai_calib2,
    )

    window.open_pyfai_calibration_tool()

    assert launched["count"] == 1


def test_uncorrected_data_file_shows_raw_viewer_and_corrections_tab(
    qtbot, repo_root
):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    project = ProjectState()
    project.add_data_group(group)
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)

    assert window.tabs.count() == 2
    assert window.tabs.tabText(0) == "Data Viewer"
    assert window.tabs.tabText(1) == "Apply Image Corrections"
    viewer = window.tabs.widget(0)
    assert viewer.coordinate_space == "pixel"
    assert viewer.axis_ranges is None
    assert not viewer.roi_table.isEnabled()
    pane = window.tabs.widget(1)
    assert pane.load_mask_button.defaultAction() is window.load_mask_action
    assert (
        pane.load_mask_button.toolButtonStyle()
        == QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
    )
    assert (
        pane.load_calibrant_button.defaultAction()
        is window.load_calibrant_action
    )
    assert (
        pane.pyfai_calibration_button.defaultAction()
        is window.pyfai_calibration_action
    )
    assert (
        pane.pyfai_calibration_status_label.text()
        == "pyfai-calib2: not launched"
    )
    assert (
        pane.asset_tool_grid.itemAtPosition(0, 0).widget()
        is pane.load_mask_button
    )
    assert (
        pane.asset_tool_grid.itemAtPosition(0, 1).widget()
        is pane.load_calibrant_button
    )
    assert (
        pane.asset_tool_grid.itemAtPosition(1, 0).widget()
        is pane.pyfai_calibration_button
    )


def test_loading_correction_assets_preserves_current_tab(
    qtbot, repo_root, monkeypatch
):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    project = ProjectState()
    project.add_data_group(group)
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    window.tabs.setCurrentIndex(1)

    selected_paths = [
        repo_root / "example" / "mask.edf",
        repo_root / "example" / "calib.poni",
    ]

    def fake_get_open_file_name(*args, **kwargs):
        return str(selected_paths.pop(0)), ""

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        fake_get_open_file_name,
    )

    window.load_mask()

    assert window.tabs.currentIndex() == 1
    assert window.tabs.tabText(window.tabs.currentIndex()) == (
        "Apply Image Corrections"
    )
    pane = window.tabs.currentWidget()
    assert pane.mask_combo.currentData() == project.masks[0].asset_id

    window.load_calibrant()

    assert window.tabs.currentIndex() == 1
    assert window.tabs.tabText(window.tabs.currentIndex()) == (
        "Apply Image Corrections"
    )
    pane = window.tabs.currentWidget()
    assert pane.calibrant_combo.currentData() == project.calibrants[0].asset_id


def test_single_file_project_root_opens_raw_viewer_tabs(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    project = ProjectState()
    project.add_data_group(group)
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    project_item = window.data_tree.tree.topLevelItem(0)
    window.data_tree.tree.setCurrentItem(project_item)

    assert window.tabs.tabText(0) == "Data Viewer"
    assert window.tabs.tabText(1) == "Apply Image Corrections"
    assert window.tabs.widget(0).coordinate_space == "pixel"


def test_raw_preview_orientation_buttons_update_pyfai_state(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    project = ProjectState()
    project.add_data_group(group)
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    data_id = group.data_files[0].data_id
    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)

    viewer = window.tabs.widget(0)
    pane = window.tabs.widget(1)
    viewer.rotate_right_button.click()

    state = project.image_corrections[data_id]
    assert state.image_rotation_deg == 90
    assert not state.image_mirrored_y
    assert state.pyfai_sample_orientation == 8
    assert pane.rotation_combo.currentData() == 90
    assert not pane.mirror_y_check.isChecked()
    assert pane.sample_orientation_combo.currentData() == 8

    viewer.mirror_y_button.click()

    assert state.image_mirrored_y
    assert state.pyfai_sample_orientation == 5
    assert pane.mirror_y_check.isChecked()
    assert pane.sample_orientation_combo.currentData() == 5

    viewer.rotate_left_button.click()

    assert state.image_rotation_deg == 0
    assert state.image_mirrored_y
    assert state.pyfai_sample_orientation == 2
    assert pane.rotation_combo.currentData() == 0
    assert pane.sample_orientation_combo.currentData() == 2


def test_confirming_image_corrections_unlocks_analysis_tabs(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    project = ProjectState()
    project.add_data_group(group)
    project.add_mask(repo_root / "example" / "mask.edf")
    project.add_calibrant(repo_root / "example" / "calib.poni")
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    data_id = group.data_files[0].data_id
    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    pane = window.tabs.widget(1)
    correction_tooltip_widgets = [
        pane.load_mask_button,
        pane.load_calibrant_button,
        pane.pyfai_calibration_button,
        pane.mask_combo,
        pane.calibrant_combo,
        pane.energy_kev,
        pane.solid_angle_check,
        pane.polarization_check,
        pane.polarization_factor,
        pane.normalization_factor,
        pane.dummy_check,
        pane.dummy_value,
        pane.delta_dummy,
        pane.rotation_combo,
        pane.mirror_y_check,
        pane.sample_orientation_combo,
        pane.reflected_x,
        pane.reflected_y,
        pane.critical_angle,
        pane.sample_stoichiometry,
        pane.sample_density,
        pane.film_memory_combo,
        pane.save_film_memory_button,
        pane.load_film_memory_button,
        pane.delete_film_memory_button,
        pane.clear_film_memory_button,
        pane.film_memory_status,
        pane.refractive_delta,
        pane.artifact_table,
        pane.estimate_delta_button,
        pane.estimate_structure_button,
        pane.calculate_critical_button,
        pane.estimate_reflected_button,
        pane.add_identifiers_button,
        pane.add_artifact_button,
        pane.apply_button,
        pane.confirm_button,
    ]
    assert all(widget.toolTip() for widget in correction_tooltip_widgets)
    pane.mask_combo.setCurrentIndex(1)
    pane.calibrant_combo.setCurrentIndex(1)
    pane.energy_kev.setValue(13.5)
    pane.solid_angle_check.setChecked(False)
    pane.polarization_factor.setValue(0.91)
    pane.normalization_factor.setValue(2.5)
    assert "pyFAI dummy" in pane.dummy_check.toolTip()
    pane.dummy_check.setChecked(True)
    pane.dummy_value.setValue(-1.0)
    pane.delta_dummy.setValue(0.2)
    pane.rotation_combo.setCurrentIndex(1)
    pane.mirror_y_check.setChecked(True)
    pane.reflected_x.setValue(100.0)
    pane.reflected_y.setValue(200.0)
    pane.critical_angle.setValue(0.12)
    pane.sample_stoichiometry.setText("CH3NH3PbI3")
    pane.sample_density.setValue(4.16)
    pane.refractive_delta.setValue(3.2e-6)
    pane.add_artifact_region()
    pane.confirm_corrections()

    assert project.image_corrections[data_id].confirmed
    assert project.image_corrections[data_id].critical_angle_deg == 0.12
    assert project.image_corrections[data_id].xray_energy_kev == 13.5
    assert not project.image_corrections[data_id].correct_solid_angle
    assert project.image_corrections[data_id].polarization_factor == 0.91
    assert project.image_corrections[data_id].normalization_factor == 2.5
    assert project.image_corrections[data_id].dummy == -1.0
    assert project.image_corrections[data_id].delta_dummy == 0.2
    assert project.image_corrections[data_id].image_rotation_deg == 90
    assert project.image_corrections[data_id].image_mirrored_y
    assert project.image_corrections[data_id].pyfai_sample_orientation == 5
    assert (
        project.image_corrections[data_id].sample_stoichiometry == "CH3NH3PbI3"
    )
    assert project.image_corrections[data_id].sample_density_g_cm3 == 4.16
    assert project.image_corrections[
        data_id
    ].refractive_index_delta == pytest.approx(3.2e-6)
    assert project.image_corrections[data_id].artifact_regions[0]["label"]
    assert not window.load_mask_action.isEnabled()
    assert window.pyfai_calibration_action.isEnabled()
    viewer = window.tabs.widget(0)
    assert viewer.coordinate_space == "qspace"
    assert viewer.roi_table.isEnabled()
    assert [
        window.tabs.tabText(index) for index in range(window.tabs.count())
    ] == [
        "Data Viewer",
        "Peak Identification",
        "Structure Analysis",
        "GIWAXS Simulation",
    ]


def test_film_material_memory_add_load_delete_clear_persists(
    qtbot,
    repo_root,
    tmp_path,
    monkeypatch,
):
    settings_path = tmp_path / "film_memory_settings.ini"
    settings = QtCore.QSettings(
        str(settings_path),
        QtCore.QSettings.Format.IniFormat,
    )
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    project = ProjectState()
    project.add_data_group(group)
    window = MainWindow(project=project, settings=settings)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    pane = window.tabs.widget(1)

    assert pane.film_memory_combo.currentData() is None
    assert not pane.load_film_memory_button.isEnabled()
    assert not pane.delete_film_memory_button.isEnabled()
    assert not pane.clear_film_memory_button.isEnabled()

    messages = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args: messages.append(_args[2]),
    )
    pane.sample_stoichiometry.setText("!!!")
    pane.save_film_material_memory()
    assert "stoichiometry" in messages[0]
    assert project.film_material_memory == []

    pane.sample_stoichiometry.setText("CH3NH3PbI3")
    pane.sample_density.setValue(4.16)
    pane.save_film_material_memory()

    assert len(project.film_material_memory) == 1
    memory_id = project.film_material_memory[0]["memory_id"]
    assert pane.film_memory_combo.findData(memory_id) >= 0
    assert pane.loaded_film_memory_id == memory_id
    assert "Loaded from memory" in pane.film_memory_status.text()
    assert pane.clear_film_memory_button.isEnabled()
    settings.sync()

    reopened_settings = QtCore.QSettings(
        str(settings_path),
        QtCore.QSettings.Format.IniFormat,
    )
    reopened_group, _ = build_data_group_from_paths(
        [path],
        group_name="Example",
    )
    reopened_project = ProjectState()
    reopened_project.add_data_group(reopened_group)
    reopened = MainWindow(project=reopened_project, settings=reopened_settings)
    qtbot.addWidget(reopened)
    reopened_file_item = reopened.data_tree.tree.topLevelItem(0).child(0)
    reopened.data_tree.tree.setCurrentItem(reopened_file_item)
    reopened_pane = reopened.tabs.widget(1)

    assert len(reopened_project.film_material_memory) == 1
    assert reopened_pane.film_memory_combo.findData(memory_id) >= 0

    reopened_pane.sample_stoichiometry.clear()
    reopened_pane.sample_density.setValue(1.0)
    reopened_pane.film_memory_combo.setCurrentIndex(
        reopened_pane.film_memory_combo.findData(memory_id)
    )
    reopened_pane.load_selected_film_material_memory()

    assert reopened_pane.sample_stoichiometry.text() == "CH3NH3PbI3"
    assert reopened_pane.sample_density.value() == pytest.approx(4.16)
    assert reopened_pane.loaded_film_memory_id == memory_id
    assert "Loaded from memory" in reopened_pane.film_memory_status.text()

    reopened_pane.sample_density.setValue(4.2)
    assert (
        "Edited values from memory" in reopened_pane.film_memory_status.text()
    )

    reopened_pane.sample_density.setValue(4.16)
    reopened_pane.delete_selected_film_material_memory()
    assert reopened_project.film_material_memory == []
    assert reopened_pane.film_memory_combo.currentData() is None
    assert not reopened_pane.clear_film_memory_button.isEnabled()

    reopened_pane.sample_stoichiometry.setText("Si")
    reopened_pane.sample_density.setValue(2.33)
    reopened_pane.save_film_material_memory()
    reopened_pane.sample_stoichiometry.setText("MAPbI3")
    reopened_pane.sample_density.setValue(4.16)
    reopened_pane.save_film_material_memory()

    assert len(reopened_project.film_material_memory) == 2

    reopened_pane.clear_film_material_memory()
    assert reopened_project.film_material_memory == []
    assert reopened_pane.film_memory_combo.currentData() is None

    cleared_settings = QtCore.QSettings(
        str(settings_path),
        QtCore.QSettings.Format.IniFormat,
    )
    cleared_group, _ = build_data_group_from_paths(
        [path],
        group_name="Example",
    )
    cleared_project = ProjectState()
    cleared_project.add_data_group(cleared_group)
    cleared = MainWindow(project=cleared_project, settings=cleared_settings)
    qtbot.addWidget(cleared)
    cleared_file_item = cleared.data_tree.tree.topLevelItem(0).child(0)
    cleared.data_tree.tree.setCurrentItem(cleared_file_item)

    assert cleared_project.film_material_memory == []


def test_low_q_identifiers_are_preserved_and_drawn(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    project = ProjectState()
    project.add_data_group(group)
    project.add_mask(repo_root / "example" / "mask.edf")
    project.add_calibrant(repo_root / "example" / "calib.poni")
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    data_id = group.data_files[0].data_id
    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    pane = window.tabs.widget(1)
    pane.mask_combo.setCurrentIndex(1)
    pane.calibrant_combo.setCurrentIndex(1)
    pane.energy_kev.setValue(12.398419843320026)
    pane.sample_stoichiometry.setText("1MAI1PbI2")
    pane.sample_density.setValue(4.16)
    pane.estimate_delta_from_chemistry()
    pane.reflected_x.setValue(10.0)
    pane.reflected_y.setValue(20.0)
    pane.add_low_q_identifiers()

    state = project.image_corrections[data_id]
    features = state.metadata["low_q_features"]
    assert {feature["kind"] for feature in features} >= {
        "direct_beam",
        "reflected_beam",
        "yoneda_band",
        "effective_beam_center",
    }
    assert state.sample_stoichiometry == "1MAI1PbI2"
    assert state.sample_density_g_cm3 == 4.16
    assert state.refractive_index_delta > 0
    assert state.metadata["refractive_index_estimate"]["normalized_formula"]
    assert any("Yoneda" in feature["label"] for feature in features)

    pane.confirm_corrections()

    assert project.image_corrections[data_id].metadata["low_q_features"]
    viewer = window.tabs.widget(0)
    assert viewer.coordinate_space == "qspace"
    assert len(viewer.low_q_graphics) == len(features)


def test_structure_optics_estimate_populates_film_inputs(
    qtbot,
    repo_root,
    tmp_path,
    monkeypatch,
):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    project = ProjectState()
    project.add_data_group(group)
    window = MainWindow(project=project)
    qtbot.addWidget(window)
    poscar = _write_si_poscar(tmp_path / "POSCAR_Si")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(poscar), ""),
    )
    monkeypatch.setattr(
        "ewald.ui.corrections.StructureOpticsReviewDialog.exec",
        lambda self: QtWidgets.QDialog.DialogCode.Accepted,
    )

    data_id = group.data_files[0].data_id
    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    pane = window.tabs.widget(1)
    pane.energy_kev.setValue(12.398419843320026)
    pane.estimate_delta_from_structure()

    state = project.image_corrections[data_id]
    assert pane.sample_stoichiometry.text() == "Si"
    assert pane.sample_density.value() > 0
    assert pane.refractive_delta.value() > 0
    assert pane.critical_angle.value() > 0
    assert state.sample_stoichiometry == "Si"
    assert state.sample_density_g_cm3 == pane.sample_density.value()
    assert state.refractive_index_delta == pytest.approx(
        pane.refractive_delta.value()
    )
    assert state.metadata["structure_optics_estimate"]["file_format"] == (
        "POSCAR"
    )
    assert state.metadata["structure_optics_estimate"]["composition"][
        "Si"
    ] == (pytest.approx(2.0))


def test_confirmed_file_restores_analysis_tabs(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState()
    project.add_data_group(group)
    mask = project.add_mask(
        repo_root / "example" / "mask.edf", target_ids=[data_id]
    )
    calibrant = project.add_calibrant(
        repo_root / "example" / "calib.poni", target_ids=[data_id]
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            confirmed=True,
        )
    )
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)

    assert window.tabs.tabText(0) == "Data Viewer"
    assert window.tabs.tabText(window.tabs.count() - 1) == "GIWAXS Simulation"
    assert "Integration" not in [
        window.tabs.tabText(index) for index in range(window.tabs.count())
    ]


def test_major_analysis_plots_share_locked_aspect(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState()
    project.add_data_group(group)
    mask = project.add_mask(
        repo_root / "example" / "mask.edf", target_ids=[data_id]
    )
    calibrant = project.add_calibrant(
        repo_root / "example" / "calib.poni", target_ids=[data_id]
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            confirmed=True,
        )
    )
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    viewer = window.tabs.widget(0)
    peak_pane = window.tabs.widget(1)
    structure_pane = window.tabs.widget(2)

    if viewer.view_box is None:
        pytest.skip("pyqtgraph is unavailable")

    assert viewer.view_box.state["aspectLocked"] == 1
    assert peak_pane.view_box.state["aspectLocked"] == 1
    assert structure_pane.plot_widget.getViewBox().state["aspectLocked"] == 1
    assert type(peak_pane.plot_frame) is type(viewer.plot_frame)
    assert type(structure_pane.plot_frame) is type(viewer.plot_frame)
    assert viewer.coordinate_space == "qspace"
    assert peak_pane.coordinate_space == viewer.coordinate_space
    assert structure_pane.coordinate_space == viewer.coordinate_space
    assert peak_pane.axis_ranges == viewer.axis_ranges
    assert structure_pane.axis_ranges == viewer.axis_ranges
    assert peak_pane.image_data.shape == viewer.image_data.shape
    assert structure_pane.image_data.shape == viewer.image_data.shape

    x_min, x_max, y_min, y_max = viewer.axis_ranges
    x_span = x_max - x_min
    y_span = y_max - y_min
    roi = viewer.add_roi_from_bounds(
        x_min + 0.2 * x_span,
        x_min + 0.35 * x_span,
        y_min + 0.2 * y_span,
        y_min + 0.35 * y_span,
    )
    assert roi is not None
    graphic = viewer.roi_graphics[roi.roi_id]
    roi_pos_before = (float(graphic.pos().x()), float(graphic.pos().y()))
    roi_size_before = (float(graphic.size().x()), float(graphic.size().y()))

    qxy = x_min + 0.5 * x_span
    qz = y_min + 0.5 * y_span
    peak_pane.add_peak_at(qxy, qz)
    x_data, y_data = peak_pane.active_peak_scatter.getData()
    assert x_data[0] == pytest.approx(qxy)
    assert y_data[0] == pytest.approx(qz)

    window.resize(1800, 1050)
    qtbot.wait(50)

    assert viewer.view_box.state["aspectLocked"] == 1
    assert peak_pane.view_box.state["aspectLocked"] == 1
    assert structure_pane.plot_widget.getViewBox().state["aspectLocked"] == 1
    assert peak_pane.plot_frame is not None
    assert structure_pane.plot_frame is not None
    assert (
        float(graphic.pos().x()),
        float(graphic.pos().y()),
    ) == pytest.approx(roi_pos_before)
    assert (
        float(graphic.size().x()),
        float(graphic.size().y()),
    ) == pytest.approx(roi_size_before)
    x_data, y_data = peak_pane.active_peak_scatter.getData()
    assert x_data[0] == pytest.approx(qxy)
    assert y_data[0] == pytest.approx(qz)


def test_peak_identification_adds_detects_and_regions_peaks(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState()
    project.add_data_group(group)
    mask = project.add_mask(
        repo_root / "example" / "mask.edf", target_ids=[data_id]
    )
    calibrant = project.add_calibrant(
        repo_root / "example" / "calib.poni", target_ids=[data_id]
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            confirmed=True,
        )
    )
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    viewer = window.tabs.widget(0)
    pane = window.tabs.widget(1)
    structure_pane = window.tabs.widget(2)
    assert type(pane).__name__ == "PeakIdentificationPane"
    assert pane.side_tabs.tabText(0) == "Peak Finder"
    assert pane.side_tabs.tabText(1) == "ROI Selection"
    assert pane.side_tabs.tabText(2) == "Peak Fit"
    peak_finder_tab = pane.side_tabs.widget(0)
    assert peak_finder_tab.layout().itemAt(0).widget() is pane.peak_action_bar
    assert (
        peak_finder_tab.layout().itemAt(1).widget() is pane.peak_finder_subtabs
    )
    assert pane.peak_finder_subtabs.tabText(0) == "Peak Detection"
    assert pane.peak_finder_subtabs.tabText(1) == "Crystal Overlay"
    roi_groups = {
        group.title()
        for group in pane.side_tabs.widget(1).findChildren(QtWidgets.QGroupBox)
    }
    assert "ROI Tools" in roi_groups
    for button in (
        pane.undo_button,
        pane.redo_button,
        pane.clear_peaks_button,
    ):
        assert button.parentWidget() is pane.peak_action_bar
        assert not button.icon().isNull()
        assert (
            button.toolButtonStyle()
            == QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
    assert pane.image_style == viewer.image_display_style()
    assert structure_pane.image_style == viewer.image_display_style()
    assert pane.image_display_style() == viewer.image_display_style()
    assert structure_pane.image_display_style() == viewer.image_display_style()
    assert pane.zoom_fit_button.text() == "Autoscale"
    assert structure_pane.zoom_fit_button.text() == "Autoscale"
    for button in (
        pane.zoom_in_button,
        pane.zoom_out_button,
        pane.zoom_fit_button,
        pane.pan_button,
    ):
        assert button.parentWidget() is pane.plot_toolbar
    for button in (
        structure_pane.zoom_in_button,
        structure_pane.zoom_out_button,
        structure_pane.zoom_fit_button,
        structure_pane.pan_button,
    ):
        assert button.parentWidget() is structure_pane.plot_toolbar

    viewer.colormap_combo.setCurrentIndex(1)
    viewer.quantile_low.setValue(5.0)
    viewer.quantile_high.setValue(95.0)
    assert pane.image_style.colormap == "magma"
    assert pane.image_style.use_quantile
    assert pane.image_style.quantile_low == pytest.approx(5.0)
    assert pane.image_style.quantile_high == pytest.approx(95.0)
    assert structure_pane.image_style.colormap == "magma"
    assert structure_pane.image_style.use_quantile
    assert structure_pane.image_style.quantile_low == pytest.approx(5.0)
    assert structure_pane.image_style.quantile_high == pytest.approx(95.0)

    viewer.quantile_check.setChecked(False)
    viewer.level_min.setValue(2.0)
    viewer.level_max.setValue(40.0)
    assert not pane.image_style.use_quantile
    assert pane.image_style.level_min == pytest.approx(2.0)
    assert pane.image_style.level_max == pytest.approx(40.0)
    assert not structure_pane.image_style.use_quantile
    assert structure_pane.image_style.level_min == pytest.approx(2.0)
    assert structure_pane.image_style.level_max == pytest.approx(40.0)

    pane.colormap_combo.setCurrentIndex(2)
    pane.quantile_check.setChecked(True)
    pane.quantile_low.setValue(3.0)
    pane.quantile_high.setValue(97.0)
    assert pane.image_style.colormap == "turbo"
    assert pane.image_style.use_quantile
    assert pane.image_style.quantile_low == pytest.approx(3.0)
    assert pane.image_style.quantile_high == pytest.approx(97.0)

    structure_pane.colormap_combo.setCurrentIndex(3)
    structure_pane.quantile_check.setChecked(False)
    structure_pane.level_min.setValue(1.0)
    structure_pane.level_max.setValue(11.0)
    assert structure_pane.image_style.colormap == "gray"
    assert not structure_pane.image_style.use_quantile
    assert structure_pane.image_style.level_min == pytest.approx(1.0)
    assert structure_pane.image_style.level_max == pytest.approx(11.0)

    pane.image_data = np.zeros((30, 30), dtype=float)
    pane.image_data[10, 15] = 50.0
    pane.image_data[20, 5] = 30.0
    pane.axis_ranges = (-1.0, 1.0, -2.0, 2.0)
    pane.coordinate_space = "qspace"
    pane.min_qz.setValue(-2.0)
    pane.threshold_percentile.setValue(0.0)
    pane.max_peaks.setValue(5)
    pane.min_distance_px.setValue(4)
    pane.neighborhood_radius_px.setValue(1)
    pane.run_peak_finder()

    assert pane.peak_table.rowCount() == 2
    assert len(project.peak_sets[data_id]) == 2
    assert pane.active_peak_id is not None
    assert pane.active_peak_scatter is not None

    pane.add_peak_at(0.0, 0.0)
    manual_peak_id = project.peak_sets[data_id][-1]["peak_id"]
    assert pane.peak_table.rowCount() == 3
    assert project.peak_sets[data_id][-1]["source"] == "manual"

    pane.roi_width.setValue(0.2)
    pane.roi_height.setValue(0.4)
    pane.apply_roi_to_selected_peak()

    active = [
        record
        for record in project.peak_sets[data_id]
        if record["peak_id"] == pane.active_peak_id
    ][0]
    assert active["roi"]["qxy_min"] == pytest.approx(-0.1)
    assert active["roi"]["qxy_max"] == pytest.approx(0.1)
    assert active["roi"]["qz_min"] == pytest.approx(-0.2)
    assert active["roi"]["qz_max"] == pytest.approx(0.2)

    pane.remove_point_button.setChecked(True)
    pane.active_peak_scatter.peakClicked.emit(manual_peak_id)
    assert len(project.peak_sets[data_id]) == 2
    assert manual_peak_id not in {
        record["peak_id"] for record in project.peak_sets[data_id]
    }


def test_peak_finder_preserves_and_consolidates_existing_peaks(qtbot):
    from ewald.ui.peak_identification import PeakIdentificationPane

    project = ProjectState()
    pane = PeakIdentificationPane(project, "synthetic")
    qtbot.addWidget(pane)

    pane.image_data = np.zeros((20, 20), dtype=float)
    pane.image_data[5, 5] = 60.0
    pane.image_data[14, 14] = 90.0
    pane.image_data[2, 17] = 40.0
    pane.axis_ranges = (0.0, 19.0, 0.0, 19.0)
    pane.coordinate_space = "qspace"
    pane.min_qz.setValue(0.0)
    pane.threshold_percentile.setValue(0.0)
    pane.max_peaks.setValue(10)
    pane.min_distance_px.setValue(3)
    pane.neighborhood_radius_px.setValue(1)

    manual = pane.add_peak_at(5.4, 5.2, record_history=False)
    channel = pane.add_peak_at(
        13.5,
        14.2,
        source="integration-channel",
        record_history=False,
    )
    channel["metadata"] = {"integration_marker_id": "trace-1"}
    existing_auto = pane.add_peak_at(
        17.0,
        2.0,
        source="auto-local-maximum",
        record_history=False,
    )
    existing_ids = {
        manual["peak_id"],
        channel["peak_id"],
        existing_auto["peak_id"],
    }

    pane.run_peak_finder()

    records = {
        record["peak_id"]: record for record in project.peak_sets["synthetic"]
    }
    assert set(records) == existing_ids

    manual_after = records[manual["peak_id"]]
    assert manual_after["source"] == "manual-local-maximum"
    assert manual_after["qxy"] == pytest.approx(5.0)
    assert manual_after["qz"] == pytest.approx(5.0)
    assert manual_after["intensity"] == pytest.approx(60.0)
    assert manual_after["metadata"]["consolidated_by"] == "find-peaks"
    assert manual_after["metadata"]["peak_finder"]["min_snr"] == pytest.approx(
        pane.min_snr.value()
    )

    channel_after = records[channel["peak_id"]]
    assert channel_after["source"] == "integration-channel-local-maximum"
    assert channel_after["qxy"] == pytest.approx(14.0)
    assert channel_after["qz"] == pytest.approx(14.0)
    assert channel_after["intensity"] == pytest.approx(90.0)
    assert channel_after["metadata"]["integration_marker_id"] == "trace-1"

    auto_after = records[existing_auto["peak_id"]]
    assert auto_after["source"] == "auto-local-maximum"
    assert auto_after["qxy"] == pytest.approx(17.0)
    assert auto_after["qz"] == pytest.approx(2.0)
    assert "2 consolidated" in pane.peak_finder_status_label.text()


def test_peak_finder_presets_update_detection_controls(qtbot):
    from ewald.ui.peak_identification import PeakIdentificationPane

    pane = PeakIdentificationPane(ProjectState(), "synthetic")
    qtbot.addWidget(pane)

    peak_finder_controls = [
        pane.threshold_percentile,
        pane.adaptive_peak_threshold_check,
        pane.adaptive_floor_percentile,
        pane.min_snr,
        pane.background_radius_px,
        pane.max_peaks,
        pane.min_distance_px,
        pane.neighborhood_radius_px,
        pane.min_qz,
        pane.ignore_nonpositive_check,
        pane.consolidate_peaks_check,
        pane.find_peaks_button,
    ]
    for control in peak_finder_controls:
        assert control.toolTip().startswith("<qt>")
    assert (
        "image-wide intensity cutoff"
        in pane.global_peak_preset_button.toolTip()
    )
    assert "local background" in pane.adaptive_peak_preset_button.toolTip()
    assert "weak peaks" in pane.sensitive_peak_preset_button.toolTip()
    finder_labels = [
        label
        for label in pane.peak_finder_subtabs.widget(0).findChildren(
            QtWidgets.QLabel
        )
        if label.toolTip().startswith("<qt>")
    ]
    assert len(finder_labels) >= 8
    assert any(label.text() == "Threshold" for label in finder_labels)

    assert pane.max_peaks.value() == 500
    assert not pane.adaptive_peak_threshold_check.isChecked()

    pane.sensitive_peak_preset_button.click()
    assert pane.min_snr.value() == pytest.approx(3.5)
    assert pane.max_peaks.value() == 900
    assert pane.adaptive_peak_threshold_check.isChecked()
    assert "Sensitive" in pane.peak_finder_status_label.text()

    pane.global_peak_preset_button.click()
    assert pane.min_snr.value() == pytest.approx(4.5)
    assert pane.max_peaks.value() == 500
    assert not pane.adaptive_peak_threshold_check.isChecked()

    pane.adaptive_peak_preset_button.click()
    assert pane.min_snr.value() == pytest.approx(4.5)
    assert pane.max_peaks.value() == 600
    assert pane.adaptive_peak_threshold_check.isChecked()


def test_peak_finder_mirrors_selected_missing_peaks(qtbot):
    from ewald.ui.peak_identification import PeakIdentificationPane

    project = ProjectState()
    pane = PeakIdentificationPane(project, "synthetic")
    qtbot.addWidget(pane)

    pane.image_data = np.zeros((41, 41), dtype=float)
    pane.axis_ranges = (-1.0, 1.0, 0.0, 2.0)
    pane.coordinate_space = "qspace"
    pane.symmetry_qxy_tolerance.setValue(0.02)
    pane.symmetry_qz_tolerance.setValue(0.02)

    matched_source = pane.add_peak_at(
        0.35,
        0.8,
        record_history=False,
    )
    missing_source = pane.add_peak_at(
        0.65,
        1.2,
        record_history=False,
    )
    pane.add_peak_at(-0.35, 0.8, record_history=False)

    assert (
        pane.peak_table.selectionMode()
        == QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
    )
    pane.peak_table.clearSelection()
    selection = pane.peak_table.selectionModel()
    flags = (
        QtCore.QItemSelectionModel.SelectionFlag.Select
        | QtCore.QItemSelectionModel.SelectionFlag.Rows
    )
    for row in (0, 1):
        selection.select(pane.peak_table.model().index(row, 0), flags)

    pane.mirror_missing_button.click()

    records = project.peak_sets["synthetic"]
    assert len(records) == 4
    mirrored = [
        record
        for record in records
        if record.get("metadata", {}).get("mirror_source_peak_id")
        == missing_source["peak_id"]
    ]
    assert len(mirrored) == 1
    mirrored_peak = mirrored[0]
    assert mirrored_peak["qxy"] == pytest.approx(-0.65)
    assert mirrored_peak["qz"] == pytest.approx(1.2)
    assert mirrored_peak["source"] == "gap estimate"
    assert mirrored_peak["point_kind"] == PEAK_POINT_KIND_GAP_ESTIMATED
    assert mirrored_peak["gap_estimated"] is True
    assert mirrored_peak["metadata"]["estimate_method"] == (
        "mirror across qz axis"
    )
    assert not any(
        record.get("metadata", {}).get("mirror_source_peak_id")
        == matched_source["peak_id"]
        for record in records
    )
    assert (
        "Added 1 mirrored gap estimate" in pane.symmetry_summary_label.text()
    )

    pane.mirror_missing_button.click()
    assert len(project.peak_sets["synthetic"]) == 4
    assert "No missing mirrored partners" in pane.symmetry_summary_label.text()

    pane.undo_peak_action()
    assert len(project.peak_sets["synthetic"]) == 3


@pytest.mark.parametrize(
    ("gap_axis", "center_qxy", "center_qz"),
    [
        ("qxy", 0.06, 0.24),
        ("qz", 0.34, 0.16),
    ],
)
def test_peak_finder_centers_masked_gap_click_from_side_maxima(
    qtbot,
    gap_axis,
    center_qxy,
    center_qz,
):
    from ewald.ui.peak_identification import PeakIdentificationPane

    project = ProjectState()
    pane = PeakIdentificationPane(project, "synthetic")
    qtbot.addWidget(pane)

    x_axis = np.linspace(-1.0, 1.0, 101)
    y_axis = np.linspace(-1.0, 1.0, 101)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    image = 4.0 + 150.0 * np.exp(
        -0.5
        * (
            ((x_grid - center_qxy) / 0.22) ** 2
            + ((y_grid - center_qz) / 0.16) ** 2
        )
    )
    if gap_axis == "qxy":
        image[
            :, (x_axis >= center_qxy - 0.08) & (x_axis <= center_qxy + 0.08)
        ] = 0.0
    else:
        image[
            (y_axis >= center_qz - 0.08) & (y_axis <= center_qz + 0.08), :
        ] = 0.0
    pane.image_data = image
    pane.axis_ranges = (-1.0, 1.0, -1.0, 1.0)
    pane.coordinate_space = "qspace"

    record = pane.add_peak_at(center_qxy, center_qz)

    assert record["qxy"] == pytest.approx(center_qxy, abs=0.035)
    assert record["qz"] == pytest.approx(center_qz, abs=0.035)
    assert record["source"] == "gap estimate"
    assert record["point_kind"] == PEAK_POINT_KIND_GAP_ESTIMATED
    assert record["gap_estimated"] is True
    assert record["metadata"]["estimate_method"] == "masked gap gaussian"
    assert record["metadata"]["gap_axis"] == gap_axis
    assert np.isfinite(record["intensity"])
    assert "masked-gap peak estimate" in pane.snap_feedback_label.text()


def test_peak_fit_subtab_runs_roi_fit_workflow(qtbot):
    from ewald.ui.peak_identification import PeakIdentificationPane

    project = ProjectState()
    pane = PeakIdentificationPane(project, "synthetic")
    qtbot.addWidget(pane)

    qxy = np.linspace(-1.0, 1.0, 61)
    qz = np.linspace(-1.0, 1.0, 61)
    qxy_grid, qz_grid = np.meshgrid(qxy, qz)
    pane.image_data = 4.0 + 80.0 * np.exp(
        -0.5
        * (((qxy_grid - 0.12) / 0.08) ** 2 + ((qz_grid - 0.18) / 0.11) ** 2)
    )
    pane.axis_ranges = (-1.0, 1.0, -1.0, 1.0)
    pane.coordinate_space = "qspace"

    pane.add_peak_at(0.12, 0.18)
    pane.roi_width.setValue(0.6)
    pane.roi_height.setValue(0.6)
    pane.apply_roi_to_selected_peak()
    pane.run_integrations_for_selected_roi()
    pane.run_all_integration_fits_for_selected_roi()
    pane.run_2d_fit_for_selected_roi()

    assert pane.side_tabs.tabText(2) == "Peak Fit"
    assert pane.peak_fit_scroll_area.verticalScrollBarPolicy() == (
        QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert pane.peak_fit_scroll_up_button.arrowType() == (
        QtCore.Qt.ArrowType.UpArrow
    )
    assert pane.peak_fit_scroll_down_button.arrowType() == (
        QtCore.Qt.ArrowType.DownArrow
    )
    peak_fit_labels = [
        label.text()
        for label in pane.side_tabs.widget(2).findChildren(QtWidgets.QLabel)
    ]
    assert "▲ more Peak Fit controls above" not in peak_fit_labels
    assert "▼ more Peak Fit controls below" not in peak_fit_labels

    pane.peak_fit_scroll_bar.setRange(0, 100)
    pane.peak_fit_scroll_bar.setSingleStep(10)
    pane.peak_fit_scroll_bar.setValue(50)
    pane.peak_fit_scroll_up_button.setEnabled(True)
    pane.peak_fit_scroll_down_button.setEnabled(True)
    pane.peak_fit_scroll_up_button.click()
    assert pane.peak_fit_scroll_bar.value() == 26
    pane.peak_fit_scroll_down_button.click()
    assert pane.peak_fit_scroll_bar.value() == 50

    peak_id = pane.active_peak_id
    fit_store = project.fits["synthetic"]["peak_fit"][peak_id]
    assert set(fit_store["integrations"]) == {"qxy", "qz", "azimuthal"}
    assert set(fit_store["integration_fits"]) == {"qxy", "qz", "azimuthal"}
    assert fit_store["fit_2d"]["center_qxy"] == pytest.approx(0.12, abs=0.04)
    assert fit_store["fit_2d"]["center_qz"] == pytest.approx(0.18, abs=0.04)
    assert pane.peak_table.item(0, 9).text() == "3/3"
    assert pane.peak_table.item(0, 11).text() == "Yes"
    assert pane.fit_detail_tree.topLevelItem(0).text(0) == "Peak"


def test_batch_peak_fit_flags_and_sorts_problem_fits(qtbot):
    from ewald.ui.peak_identification import PeakIdentificationPane

    project = ProjectState()
    pane = PeakIdentificationPane(project, "synthetic")
    qtbot.addWidget(pane)

    qxy = np.linspace(-1.0, 1.0, 61)
    qz = np.linspace(-1.0, 1.0, 61)
    qxy_grid, qz_grid = np.meshgrid(qxy, qz)
    image = 3.0 + 90.0 * np.exp(
        -0.5
        * (((qxy_grid + 0.45) / 0.08) ** 2 + ((qz_grid + 0.35) / 0.1) ** 2)
    )
    image[(qxy_grid > 0.35) & (qz_grid > 0.35)] = np.nan
    pane.image_data = image
    pane.axis_ranges = (-1.0, 1.0, -1.0, 1.0)
    pane.coordinate_space = "qspace"

    good = pane.add_peak_at(-0.45, -0.35)
    pane.roi_width.setValue(0.45)
    pane.roi_height.setValue(0.45)
    pane.apply_roi_to_selected_peak()
    bad = pane.add_peak_at(0.65, 0.65)
    pane.apply_roi_to_selected_peak()

    pane.batch_process_all_peak_fits()

    stores = project.fits["synthetic"]["peak_fit"]
    assert stores[good["peak_id"]]["fit_2d"]["status"] == "fit"
    assert stores[bad["peak_id"]]["fit_2d_failure"]["status"] == "failed"
    bad_row = next(
        row
        for row in range(pane.peak_table.rowCount())
        if pane.peak_table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        == bad["peak_id"]
    )
    assert pane.peak_table.item(
        bad_row, 0
    ).background().color() == QtGui.QColor("#fef3c7")
    assert "No finite ROI pixels" in pane.peak_table.item(bad_row, 0).toolTip()

    pane.fit_issues_first_button.click()

    assert pane.fit_issues_first_button.isChecked()
    assert (
        pane.peak_table.item(0, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        == bad["peak_id"]
    )


def test_peak_identification_crystal_overlay_updates_project(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState()
    project.add_data_group(group)
    mask = project.add_mask(
        repo_root / "example" / "mask.edf", target_ids=[data_id]
    )
    calibrant = project.add_calibrant(
        repo_root / "example" / "calib.poni", target_ids=[data_id]
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            confirmed=True,
        )
    )
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    pane = window.tabs.widget(1)

    orientation_layout = pane.crystal_preview_widget.parentWidget().layout()
    assert isinstance(orientation_layout, QtWidgets.QHBoxLayout)
    assert orientation_layout.itemAt(1).widget() is pane.crystal_preview_widget
    orientation_controls = orientation_layout.itemAt(0).widget()
    assert orientation_controls.objectName() == "CrystalOrientationControls"
    retained_controls = [
        pane.orientation_x_slider,
        pane.orientation_y_slider,
        pane.orientation_z_slider,
        pane.rotation_step,
        pane.rotate_x_neg_button,
        pane.rotate_x_pos_button,
        pane.rotate_y_neg_button,
        pane.rotate_y_pos_button,
        pane.rotate_z_neg_button,
        pane.rotate_z_pos_button,
        pane.reset_orientation_button,
    ]
    assert all(
        orientation_controls.isAncestorOf(control)
        for control in retained_controls
    )
    crystal_groups = {
        group.title()
        for group in pane.peak_finder_subtabs.widget(1).findChildren(
            QtWidgets.QGroupBox
        )
    }
    assert "Lattice & Overlay Peaks" in crystal_groups
    assert "Overlay Peaks" not in crystal_groups
    assert pane.auto_update_crystal_overlay_button.isCheckable()
    assert pane.auto_update_crystal_overlay_button.isChecked()

    pane.crystal_system_combo.setCurrentText("Cubic")
    pane.lattice_a.setValue(10.0)
    pane.h_max.setValue(1)
    pane.k_max.setValue(0)
    pane.l_max.setValue(0)
    pane.positive_qz_check.setChecked(False)
    pane.show_crystal_overlay_check.setChecked(True)
    pane._update_crystal_overlay()

    overlay_state = project.analysis_results["crystal_overlays"][data_id]
    assert overlay_state["parameters"]["crystal_system"] == "Cubic"
    assert overlay_state["parameters"]["b"] == pytest.approx(10.0)
    assert overlay_state["parameters"]["c"] == pytest.approx(10.0)
    assert overlay_state["peak_count"] == 2
    assert pane.crystal_peak_table.rowCount() == 2
    assert pane.show_crystal_overlay_check.isChecked()
    assert pane.show_crystal_hkl_labels_check.isChecked()
    assert overlay_state["show_hkl_labels"] is True

    assert pane.crystal_overlay_scatter is not None
    assert len(pane.crystal_overlay_scatter.points()) == 2
    assert [
        point.data() for point in pane.crystal_overlay_scatter.points()
    ] == [
        "(-1 0 0)",
        "(1 0 0)",
    ]
    assert [item.toPlainText() for item in pane.crystal_overlay_graphics] == [
        "(-1 0 0)",
        "(1 0 0)",
    ]
    qxy_before, _ = pane.crystal_overlay_scatter.getData()
    assert np.max(np.abs(qxy_before)) == pytest.approx(2 * np.pi / 10.0)

    pane.show_crystal_hkl_labels_check.setChecked(False)
    pane._update_crystal_overlay()
    assert pane.crystal_overlay_graphics == []
    assert (
        project.analysis_results["crystal_overlays"][data_id][
            "show_hkl_labels"
        ]
        is False
    )

    pane.auto_update_crystal_overlay_button.setChecked(False)
    pane.lattice_a.setValue(20.0)
    qxy_after, _ = pane.crystal_overlay_scatter.getData()
    assert np.max(np.abs(qxy_after)) == pytest.approx(2 * np.pi / 20.0)
    overlay_state = project.analysis_results["crystal_overlays"][data_id]
    assert overlay_state["parameters"]["a"] == pytest.approx(20.0)
    assert overlay_state["parameters"]["b"] == pytest.approx(20.0)
    assert overlay_state["parameters"]["c"] == pytest.approx(20.0)
    assert overlay_state["auto_update_overlay"] is False

    pane._rotate_crystal((0.0, 1.0, 0.0), 90.0)
    pane._update_crystal_overlay()
    quaternion = project.analysis_results["crystal_overlays"][data_id][
        "parameters"
    ]["orientation_quaternion"]
    assert not np.allclose(quaternion, [0.0, 0.0, 0.0, 1.0])
    assert pane.orientation_y_slider.value() == 900
    assert project.analysis_results["crystal_overlays"][data_id][
        "orientation_angles_deg"
    ][1] == pytest.approx(90.0)

    pane._update_crystal_overlay()
    pane.h_max.setValue(2)
    qtbot.wait(80)
    overlay_state = project.analysis_results["crystal_overlays"][data_id]
    assert overlay_state["parameters"]["h_max"] == 1
    assert overlay_state["auto_update_overlay"] is False

    pane.update_crystal_overlay_button.click()
    overlay_state = project.analysis_results["crystal_overlays"][data_id]
    assert overlay_state["parameters"]["h_max"] == 2

    pane.auto_update_crystal_overlay_button.setChecked(True)
    pane.h_max.setValue(1)
    qtbot.wait(80)
    overlay_state = project.analysis_results["crystal_overlays"][data_id]
    assert overlay_state["parameters"]["h_max"] == 1
    assert overlay_state["auto_update_overlay"] is True


def test_data_viewer_persists_box_and_arch_rois(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState()
    project.add_data_group(group)
    mask = project.add_mask(
        repo_root / "example" / "mask.edf", target_ids=[data_id]
    )
    calibrant = project.add_calibrant(
        repo_root / "example" / "calib.poni", target_ids=[data_id]
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            confirmed=True,
        )
    )
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    viewer = window.tabs.widget(0)
    assert viewer.axis_ranges is not None
    viewer.add_roi_from_bounds(0.1, 0.4, 0.2, 0.7)
    viewer.arch_button.setChecked(True)
    viewer.add_roi_from_bounds(-0.5, 0.5, 0.4, 0.9)

    rois = project.rois_for_target(data_id)
    assert len(rois) == 2
    assert rois[0].kind == "box"
    assert rois[0].integration_axis == "qz"
    assert rois[0].integration_direction == "vertical"
    assert rois[1].kind == "arch"
    assert rois[1].integration_axis == "chi"
    assert rois[1].qr_min == pytest.approx(0.4)
    assert rois[1].qr_max == pytest.approx(0.9)
    assert rois[1].chi_min == pytest.approx(-rois[1].chi_max)
    assert rois[1].metadata["chi_locked"] is True
    assert type(viewer.roi_graphics[rois[1].roi_id]).__name__ == "_ArchROI"
    assert viewer.roi_table.rowCount() == 2
    assert viewer.roi_table.horizontalHeaderItem(1).text() == "Ch 1"
    assert viewer.roi_table.horizontalHeaderItem(4).text() == "Direction"
    assert viewer.roi_table.horizontalHeaderItem(13).text() == "Chi lock"
    assert viewer.roi_table.horizontalHeaderItem(ROI_COL_H).text() == "h"
    assert viewer.roi_table.horizontalHeaderItem(ROI_COL_K).text() == "k"
    assert viewer.roi_table.horizontalHeaderItem(ROI_COL_L).text() == "l"
    assert viewer.roi_table.item(0, 4).text() == "vertical"


def test_data_viewer_edits_hkl_and_emits_pole_figure_context(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    project = ProjectState()
    data_file = project.add_data_file(path)
    data_id = data_file.data_id
    assert data_id is not None
    project.set_image_corrections(
        ImageCorrectionState(target_id=data_id, confirmed=True)
    )
    viewer = DataViewerPane(project, data_id)
    qtbot.addWidget(viewer)
    viewer.image_data = np.arange(100, dtype=float).reshape(10, 10)
    viewer.axis_ranges = (0.0, 9.0, 0.0, 9.0)
    viewer.coordinate_space = "qspace"
    viewer.roi_controls_enabled = True
    viewer._set_roi_controls_enabled(True)

    roi = viewer.add_roi_from_bounds(2.0, 4.0, 3.0, 7.0)
    assert roi is not None

    viewer.roi_table.item(0, ROI_COL_H).setText("1")
    viewer.roi_table.item(0, ROI_COL_K).setText("0")
    viewer.roi_table.item(0, ROI_COL_L).setText("2")
    viewer.roi_table.item(0, ROI_COL_HKL_LABEL).setText("(102)")

    assert roi.metadata["hkl"] == {
        "h": 1,
        "k": 0,
        "l": 2,
        "label": "(102)",
    }

    with qtbot.waitSignal(viewer.poleFigureRequested) as blocker:
        viewer.open_pole_figure_button.click()

    assert blocker.args[0] == data_id
    assert blocker.args[1] is roi
    assert np.asarray(blocker.args[2]).shape == (10, 10)
    assert blocker.args[3] == (0.0, 9.0, 0.0, 9.0)

    record = {
        "background_subtraction": {"method": "none"},
        "generation_parameters": {},
        "generated_at": "2026-05-15T00:00:00+00:00",
    }
    project.set_roi_pole_figure_metadata(data_id, roi.roi_id or "", record)
    viewer.refresh_roi_table()

    assert viewer.roi_table.item(0, ROI_COL_POLE_FIGURE).text() == "Current"

    viewer.roi_table.item(0, ROI_COL_QXY_MAX).setText("5.0")

    assert roi.qxy_max == pytest.approx(5.0)
    assert viewer.roi_table.item(0, ROI_COL_POLE_FIGURE).text() == "Stale"
    if roi.roi_id in viewer.roi_graphics:
        graphic = viewer.roi_graphics[roi.roi_id]
        assert graphic.pos().x() == pytest.approx(2.0)
        assert graphic.size().x() == pytest.approx(3.0)


def test_data_viewer_coupled_box_arch_pair_shares_center_and_decouples(qtbot):
    project = ProjectState()
    data_id = "synthetic"
    project.set_image_corrections(
        ImageCorrectionState(target_id=data_id, confirmed=True)
    )
    viewer = DataViewerPane(project, data_id)
    qtbot.addWidget(viewer)
    viewer.image_data = np.arange(100, dtype=float).reshape(10, 10)
    viewer.axis_ranges = (0.0, 9.0, 0.0, 9.0)
    viewer.coordinate_space = "qspace"
    viewer.roi_controls_enabled = True
    viewer._set_roi_controls_enabled(True)

    pair = viewer.add_coupled_roi_pair_from_bounds(2.0, 4.0, 3.0, 7.0)
    assert pair is not None
    box, arch = pair

    assert box.kind == "box"
    assert arch.kind == "arch"
    assert arch.roi_id in box.metadata["coupled_roi_ids"]
    assert box.roi_id in arch.metadata["coupled_roi_ids"]
    assert viewer.roi_table.item(0, ROI_COL_COUPLED).text()
    assert viewer.roi_table.item(1, ROI_COL_COUPLED).text()
    assert (box.qxy_min + box.qxy_max) / 2.0 == pytest.approx(arch.qxy_center)
    assert (box.qz_min + box.qz_max) / 2.0 == pytest.approx(arch.qz_center)

    original_arch_width = (arch.qr_max or 0.0) - (arch.qr_min or 0.0)
    box_graphic = viewer.roi_graphics[box.roi_id]
    box_graphic.setPos((3.0, 4.0))
    box_graphic.setSize((2.0, 4.0))
    viewer._handle_roi_graphic_changed(box.roi_id, box_graphic)

    assert (box.qxy_min + box.qxy_max) / 2.0 == pytest.approx(4.0)
    assert (box.qz_min + box.qz_max) / 2.0 == pytest.approx(6.0)
    assert arch.qxy_center == pytest.approx(4.0)
    assert arch.qz_center == pytest.approx(6.0)
    assert (arch.qr_max or 0.0) - (arch.qr_min or 0.0) == pytest.approx(
        original_arch_width
    )

    viewer._select_roi(box.roi_id)
    viewer._decouple_selected_roi()

    assert "coupled_roi_ids" not in box.metadata
    assert "coupled_roi_ids" not in arch.metadata
    assert viewer.roi_table.item(0, ROI_COL_COUPLED).text() == ""


def test_data_viewer_roi_graphics_are_draggable_and_resizable(
    qtbot, repo_root
):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState()
    project.add_data_group(group)
    mask = project.add_mask(
        repo_root / "example" / "mask.edf", target_ids=[data_id]
    )
    calibrant = project.add_calibrant(
        repo_root / "example" / "calib.poni", target_ids=[data_id]
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            confirmed=True,
        )
    )
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    viewer = window.tabs.widget(0)
    assert viewer.view_box.state["mouseEnabled"] == [False, False]
    assert viewer.zoom_in_button.text() == "Zoom In"
    assert viewer.zoom_out_button.text() == "Zoom Out"
    assert viewer.zoom_fit_button.text() == "Autoscale"
    assert viewer.pan_button.text() == "Pan"
    for button in (
        viewer.zoom_in_button,
        viewer.zoom_out_button,
        viewer.zoom_fit_button,
        viewer.pan_button,
    ):
        assert button.parentWidget() is viewer.plot_toolbar

    (x_min, x_max), (y_min, y_max) = viewer.view_box.viewRange()
    viewer.zoom_in_button.click()
    (zoom_x_min, zoom_x_max), (zoom_y_min, zoom_y_max) = (
        viewer.view_box.viewRange()
    )
    assert zoom_x_max - zoom_x_min < x_max - x_min
    assert zoom_y_max - zoom_y_min < y_max - y_min

    viewer.zoom_fit_button.click()
    (fit_x_min, fit_x_max), (fit_y_min, fit_y_max) = (
        viewer.view_box.viewRange()
    )
    axis_x_min, axis_x_max, axis_y_min, axis_y_max = viewer.axis_ranges
    assert fit_x_min == pytest.approx(axis_x_min)
    assert fit_x_max == pytest.approx(axis_x_max)
    assert fit_y_min <= axis_y_min
    assert fit_y_max >= axis_y_max

    viewer.pan_button.setChecked(True)
    assert viewer.view_box.state["mouseEnabled"] == [True, True]
    assert not viewer.draw_toggle.isChecked()
    viewer.pan_button.setChecked(False)
    assert viewer.view_box.state["mouseEnabled"] == [False, False]

    viewer.draw_toggle.setChecked(True)
    assert viewer.view_box.drawing_enabled
    viewer.pan_button.setChecked(True)
    assert not viewer.draw_toggle.isChecked()
    assert not viewer.view_box.drawing_enabled
    assert viewer.view_box.state["mouseEnabled"] == [True, True]

    box = viewer.add_roi_from_bounds(0.1, 0.4, 0.2, 0.7)
    assert box is not None
    box_graphic = viewer.roi_graphics[box.roi_id]
    assert box_graphic.translatable
    assert {handle["name"] for handle in box_graphic.handles} == {
        "box-corner",
        "box-left",
        "box-right",
        "box-bottom",
        "box-top",
    }

    box_graphic.setPos((0.2, 0.3))
    box_graphic.setSize((0.5, 0.6))
    viewer._handle_roi_graphic_changed(box.roi_id, box_graphic)

    assert box.qxy_min == pytest.approx(0.2)
    assert box.qxy_max == pytest.approx(0.7)
    assert box.qz_min == pytest.approx(0.3)
    assert box.qz_max == pytest.approx(0.9)

    box_handles = {
        str(handle["name"]): handle["item"] for handle in box_graphic.handles
    }
    box_graphic.movePoint(
        box_handles["box-left"],
        QtCore.QPointF(0.0, 0.6),
    )
    box_graphic.movePoint(
        box_handles["box-right"],
        QtCore.QPointF(0.8, 0.6),
    )
    box_graphic.movePoint(
        box_handles["box-bottom"],
        QtCore.QPointF(0.4, 0.1),
    )
    box_graphic.movePoint(
        box_handles["box-top"],
        QtCore.QPointF(0.4, 1.0),
    )

    assert box.qxy_min == pytest.approx(0.0)
    assert box.qxy_max == pytest.approx(0.8)
    assert box.qz_min == pytest.approx(0.1)
    assert box.qz_max == pytest.approx(1.0)

    viewer.arch_button.setChecked(True)
    arch = viewer.add_roi_from_bounds(-0.4, 0.4, 0.5, 1.0)
    assert arch is not None
    arch_graphic = viewer.roi_graphics[arch.roi_id]
    assert arch_graphic.translatable
    assert len(arch_graphic.getHandles()) == 4
    assert {handle["name"] for handle in arch_graphic.handles} == {
        "arch-radius",
        "arch-thickness",
        "arch-chi-min",
        "arch-chi-max",
    }

    arch_graphic.blockSignals(True)
    arch_graphic.setPos((-0.3, 0.6))
    arch_graphic.blockSignals(False)
    viewer._handle_roi_graphic_changed(arch.roi_id, arch_graphic)

    assert arch.qr_min == pytest.approx(0.5)
    assert arch.qr_max == pytest.approx(1.0)
    assert arch.chi_min == pytest.approx(-arch.chi_max)
    assert arch.chi_max == pytest.approx(28.0725, abs=1.0e-3)
    assert arch.qxy_center == pytest.approx(0.1706, abs=1.0e-3)
    assert arch.qz_center == pytest.approx(0.1588, abs=1.0e-3)
    assert viewer.roi_table.rowCount() == 2

    arch_handles = {
        str(handle["name"]): handle["item"] for handle in arch_graphic.handles
    }
    center_after_drag = (arch.qxy_center, arch.qz_center)
    thickness_handle = arch_handles["arch-thickness"]
    arch_graphic.movePoint(
        thickness_handle,
        QtCore.QPointF(arch.qxy_center, arch.qz_center + 1.2),
    )

    assert arch.qxy_center == pytest.approx(center_after_drag[0])
    assert arch.qz_center == pytest.approx(center_after_drag[1])
    assert arch.qr_min == pytest.approx(0.5)
    assert arch.qr_max == pytest.approx(1.2)

    radius_handle = arch_handles["arch-radius"]
    thickness = (arch.qr_max or 0.0) - (arch.qr_min or 0.0)
    radius_center = ((arch.qr_min or 0.0) + (arch.qr_max or 0.0)) / 2.0
    target_radius_center = radius_center + 0.25
    arch_graphic.movePoint(
        radius_handle,
        QtCore.QPointF(
            arch.qxy_center,
            arch.qz_center + target_radius_center,
        ),
    )

    assert arch.qxy_center == pytest.approx(center_after_drag[0])
    assert arch.qz_center == pytest.approx(center_after_drag[1])
    assert (arch.qr_max or 0.0) - (arch.qr_min or 0.0) == pytest.approx(
        thickness
    )
    radius_center_after_resize = (
        (arch.qr_min or 0.0) + (arch.qr_max or 0.0)
    ) / 2.0
    assert radius_center_after_resize == pytest.approx(target_radius_center)

    chi_max_handle = arch_handles["arch-chi-max"]
    side_radius = ((arch.qr_min or 0.0) + (arch.qr_max or 0.0)) / 2.0
    side_x = arch.qxy_center + side_radius * np.sin(np.radians(45.0))
    side_y = arch.qz_center + side_radius * np.cos(np.radians(45.0))
    arch_graphic.movePoint(chi_max_handle, QtCore.QPointF(side_x, side_y))

    assert arch.chi_min == pytest.approx(-45.0)
    assert arch.chi_max == pytest.approx(45.0)


def test_arch_roi_controls_adjust_thickness_and_chi_without_dragging_shape(
    qtbot, repo_root
):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState()
    project.add_data_group(group)
    mask = project.add_mask(
        repo_root / "example" / "mask.edf", target_ids=[data_id]
    )
    calibrant = project.add_calibrant(
        repo_root / "example" / "calib.poni", target_ids=[data_id]
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            confirmed=True,
        )
    )
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    viewer = window.tabs.widget(0)
    viewer.arch_button.setChecked(True)
    arch = viewer.add_roi_from_bounds(-0.4, 0.4, 0.5, 1.0)
    assert arch is not None

    viewer.arch_thickness.setValue(0.2)
    viewer.arch_chi_min.setValue(-30.0)
    viewer.arch_chi_max.setValue(50.0)
    viewer._apply_arch_adjustments()

    assert arch.qr_min == pytest.approx(0.65)
    assert arch.qr_max == pytest.approx(0.85)
    assert arch.chi_min == pytest.approx(-50.0)
    assert arch.chi_max == pytest.approx(50.0)

    lock_button = viewer.roi_table.cellWidget(0, 13).findChild(
        QtWidgets.QToolButton
    )
    lock_button.setChecked(False)
    assert arch.metadata["chi_locked"] is False

    viewer.arch_chi_min.setValue(-30.0)
    viewer.arch_chi_max.setValue(50.0)
    viewer._apply_arch_adjustments()

    assert arch.chi_min == pytest.approx(-30.0)
    assert arch.chi_max == pytest.approx(50.0)

    graphic = viewer.roi_graphics[arch.roi_id]
    graphic.setPos((0.25, 0.75))
    viewer._handle_roi_graphic_changed(arch.roi_id, graphic)

    assert arch.qr_min == pytest.approx(0.65)
    assert arch.qr_max == pytest.approx(0.85)
    assert arch.chi_min == pytest.approx(-30.0)
    assert arch.chi_max == pytest.approx(50.0)
    assert arch.qxy_center != pytest.approx(0.0)
    assert arch.qz_center != pytest.approx(0.0)


def test_roi_integration_channels_lock_axes_and_clear(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState()
    project.add_data_group(group)
    mask = project.add_mask(
        repo_root / "example" / "mask.edf", target_ids=[data_id]
    )
    calibrant = project.add_calibrant(
        repo_root / "example" / "calib.poni", target_ids=[data_id]
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            confirmed=True,
        )
    )
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    viewer = window.tabs.widget(0)
    assert viewer.channel_panels[1].drag_label.text() == "Channel 1"
    assert viewer.channel_panels[2].drag_label.text() == "Channel 2"

    vertical = viewer.add_roi_from_bounds(0.1, 0.4, -1.5, -0.7)
    assert vertical is not None
    second_vertical = viewer.add_roi_from_bounds(0.2, 0.5, -1.4, -0.6)
    assert second_vertical is not None
    viewer.box_axis_combo.setCurrentIndex(1)
    horizontal = viewer.add_roi_from_bounds(0.1, 0.4, -1.5, -0.7)
    assert horizontal is not None
    viewer.arch_button.setChecked(True)
    arch = viewer.add_roi_from_bounds(-0.4, 0.4, 0.5, 1.0)
    assert arch is not None

    viewer._toggle_roi_channel(vertical.roi_id, 1, True)
    viewer._toggle_roi_channel(second_vertical.roi_id, 1, True)

    assert viewer.channel_modes[1] == "box:qz"
    assert viewer.channel_assignments[1] == {
        vertical.roi_id,
        second_vertical.roi_id,
    }
    assert len(viewer.channel_panels[1].series) == 2
    assert {trace.mode for trace in viewer.channel_panels[1].series} == {
        "box:qz"
    }
    assert (
        viewer.channel_panels[1].drag_label.text()
        == "Channel 1 (Vertical Box)"
    )

    viewer._toggle_roi_channel(horizontal.roi_id, 1, True)
    viewer._toggle_roi_channel(arch.roi_id, 1, True)

    assert horizontal.roi_id not in viewer.channel_assignments[1]
    assert arch.roi_id not in viewer.channel_assignments[1]
    assert viewer.channel_modes[1] == "box:qz"
    assert (
        viewer.channel_panels[1].drag_label.text()
        == "Channel 1 (Vertical Box)"
    )

    viewer._toggle_roi_channel(horizontal.roi_id, 2, True)
    assert viewer.channel_modes[2] == "box:qxy"
    assert viewer.channel_panels[2].series[0].color == "#3da5d9"
    assert (
        viewer.channel_panels[2].drag_label.text()
        == "Channel 2 (Horizontal Box)"
    )

    viewer._toggle_roi_channel(horizontal.roi_id, 2, False)
    assert viewer.channel_modes[2] is None
    assert viewer.channel_assignments[2] == set()
    assert viewer.channel_panels[2].series == []
    assert viewer.channel_panels[2].drag_label.text() == "Channel 2"

    viewer._toggle_roi_channel(arch.roi_id, 2, True)
    assert viewer.channel_modes[2] == "arch"
    assert viewer.channel_assignments[2] == {arch.roi_id}
    assert viewer.channel_panels[2].mode == "arch"
    assert viewer.channel_panels[2].drag_label.text() == "Channel 2 (Arch)"

    viewer._toggle_roi_channel(arch.roi_id, 2, False)
    assert viewer.channel_modes[2] is None
    assert viewer.channel_assignments[2] == set()
    assert viewer.channel_panels[2].series == []
    assert viewer.channel_panels[2].drag_label.text() == "Channel 2"

    viewer._toggle_roi_channel(horizontal.roi_id, 2, True)
    assert viewer.channel_modes[2] == "box:qxy"
    assert viewer.channel_assignments[2] == {horizontal.roi_id}
    assert (
        viewer.channel_panels[2].drag_label.text()
        == "Channel 2 (Horizontal Box)"
    )

    viewer._clear_channel(1)
    assert viewer.channel_modes[1] is None
    assert viewer.channel_assignments[1] == set()
    assert viewer.channel_panels[1].series == []
    assert viewer.channel_panels[1].drag_label.text() == "Channel 1"


def test_integration_channel_peak_markers_push_to_peak_identification(
    qtbot,
    repo_root,
):
    path = next((repo_root / "example").glob("*.tiff"))
    project = ProjectState()
    data_file = project.add_data_file(path)
    data_id = data_file.data_id
    assert data_id is not None
    project.set_image_corrections(
        ImageCorrectionState(target_id=data_id, confirmed=True)
    )
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    viewer = window.tabs.widget(0)
    pane = window.tabs.widget(1)
    assert type(pane).__name__ == "PeakIdentificationPane"
    image = np.arange(100, dtype=float).reshape(10, 10)
    viewer.image_data = image
    viewer.axis_ranges = (0.0, 9.0, 0.0, 9.0)
    viewer.coordinate_space = "qspace"
    viewer.roi_controls_enabled = True
    viewer._set_roi_controls_enabled(True)
    pane.image_data = image
    pane.axis_ranges = viewer.axis_ranges
    pane.coordinate_space = "qspace"

    vertical = viewer.add_roi_from_bounds(2.0, 4.0, 3.0, 7.0)
    assert vertical is not None
    viewer._toggle_roi_channel(vertical.roi_id, 1, True)
    viewer._add_channel_peak_marker(1, vertical.roi_id, 5.0, 123.0)

    marker = viewer.integration_peak_markers[1][0]
    assert marker.qxy == pytest.approx(3.0)
    assert marker.qz == pytest.approx(5.0)
    assert marker.integrated_intensity == pytest.approx(123.0)
    panel = viewer.channel_panels[1]
    assert panel.marker_count_label.text() == "1 mark"
    assert panel.coordinate_readout_label.text() == (
        "Ch 1 active peak: trace qz=5, I=123, qxy=3, qz=5"
    )

    plot_widget = panel.plot_widget
    if plot_widget.axes is None or plot_widget.canvas is None:
        pytest.skip("matplotlib is unavailable")
    trace = viewer.channel_panels[1].series[0]
    drag_index = int(np.argmin(np.abs(trace.x_values - 6.0)))
    drag_x = float(trace.x_values[drag_index])
    drag_y = float(trace.y_values[drag_index])
    plot_widget.canvas.draw()

    def mpl_event(x_value, y_value):
        display_x, display_y = plot_widget.axes.transData.transform(
            (x_value, y_value)
        )
        return type(
            "MplEvent",
            (),
            {
                "button": 1,
                "xdata": x_value,
                "ydata": y_value,
                "x": display_x,
                "y": display_y,
                "inaxes": plot_widget.axes,
            },
        )()

    plot_widget._handle_mouse_press(
        mpl_event(marker.integration_x, marker.integrated_intensity)
    )
    plot_widget._handle_mouse_motion(mpl_event(drag_x, drag_y))
    live_readout = panel.coordinate_readout_label.text()
    assert live_readout.startswith("Ch 1 active peak: trace qz=")
    assert f"I={drag_y:.5g}" in live_readout
    assert "qxy=3" in live_readout
    assert f"qz={drag_x:.5g}" in live_readout
    assert viewer.roi_status_label.text() == live_readout
    plot_widget._handle_mouse_release(mpl_event(drag_x, drag_y))

    marker = viewer.integration_peak_markers[1][0]
    assert marker.integration_x == pytest.approx(drag_x)
    assert marker.integrated_intensity == pytest.approx(drag_y)
    assert marker.qxy == pytest.approx(3.0)
    assert marker.qz == pytest.approx(drag_x)
    assert plot_widget.markers[0].integration_x == pytest.approx(drag_x)

    viewer._push_channel_markers(1)

    assert window.tabs.currentWidget() is pane
    assert len(project.peak_sets[data_id]) == 1
    pushed = project.peak_sets[data_id][0]
    assert pushed["source"] == "integration-channel"
    assert pushed["point_kind"] == PEAK_POINT_KIND_COMMITTED
    assert pushed["qxy"] == pytest.approx(3.0)
    assert pushed["qz"] == pytest.approx(drag_x)
    assert pushed["metadata"]["integration_marker_id"] == marker.marker_id
    assert pushed["metadata"]["integration_channel"] == 1
    assert pane.peak_table.rowCount() == 1

    def mpl_off_trace_event():
        return type(
            "MplEvent",
            (),
            {
                "button": 1,
                "xdata": None,
                "ydata": None,
                "x": -50.0,
                "y": -50.0,
                "inaxes": None,
            },
        )()

    plot_widget._handle_mouse_press(
        mpl_event(marker.integration_x, marker.integrated_intensity)
    )
    plot_widget._handle_mouse_motion(mpl_off_trace_event())
    plot_widget._handle_mouse_release(mpl_off_trace_event())

    assert plot_widget._poof_timers
    assert plot_widget._poof_artists
    assert viewer.integration_peak_markers[1] == []
    assert viewer.channel_panels[1].marker_count_label.text() == "0 marks"
    assert plot_widget.markers == []

    viewer._push_channel_markers(1)

    assert len(project.peak_sets[data_id]) == 1

    graphic = viewer.roi_graphics[vertical.roi_id]
    graphic.setPos((3.0, 4.0))
    viewer._handle_roi_graphic_changed(vertical.roi_id, graphic)

    assert viewer.integration_peak_markers[1] == []
    assert len(project.peak_sets[data_id]) == 1
    assert (
        project.peak_sets[data_id][0]["point_kind"]
        == PEAK_POINT_KIND_COMMITTED
    )


def test_integration_channel_clicks_snap_and_detect_local_maxima(
    qtbot,
    tmp_path,
):
    project = ProjectState()
    data_file = project.add_data_file(tmp_path / "synthetic.tiff")
    data_id = data_file.data_id
    assert data_id is not None
    project.set_image_corrections(
        ImageCorrectionState(target_id=data_id, confirmed=True)
    )
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    viewer = window.tabs.widget(0)
    image = np.zeros((10, 10), dtype=float)
    image[4, 2:5] = 20.0
    image[6, 2:5] = 10.0
    viewer.image_data = image
    viewer.axis_ranges = (0.0, 9.0, 0.0, 9.0)
    viewer.coordinate_space = "qspace"
    viewer.roi_controls_enabled = True
    viewer._set_roi_controls_enabled(True)

    vertical = viewer.add_roi_from_bounds(2.0, 4.0, 3.0, 7.0)
    assert vertical is not None
    viewer._toggle_roi_channel(vertical.roi_id, 1, True)

    panel = viewer.channel_panels[1]
    assert panel.detect_peaks_button.text() == "Detect Peaks"
    assert panel.detect_peaks_button.isEnabled()
    assert panel.autosnap_button.text() == "Autosnap"
    assert panel.autosnap_button.isChecked()
    plot_widget = panel.plot_widget
    if plot_widget.axes is None or plot_widget.canvas is None:
        pytest.skip("matplotlib is unavailable")
    plot_widget.canvas.draw()

    def mpl_event(x_value, y_value):
        display_x, display_y = plot_widget.axes.transData.transform(
            (x_value, y_value)
        )
        return type(
            "MplEvent",
            (),
            {
                "button": 1,
                "xdata": x_value,
                "ydata": y_value,
                "x": display_x,
                "y": display_y,
                "inaxes": plot_widget.axes,
            },
        )()

    plot_widget._handle_mouse_press(mpl_event(4.25, 35.0))

    marker = viewer.integration_peak_markers[1][0]
    assert marker.integration_x == pytest.approx(4.0)
    assert marker.integrated_intensity == pytest.approx(60.0)

    viewer._clear_channel_markers(1)
    panel.autosnap_button.setChecked(False)

    assert viewer.channel_autosnap_enabled[1] is False
    assert plot_widget.autosnap_enabled is False

    plot_widget._handle_mouse_press(mpl_event(5.1, 35.0))

    marker = viewer.integration_peak_markers[1][0]
    assert marker.integration_x == pytest.approx(5.0)
    assert marker.integrated_intensity == pytest.approx(0.0)

    viewer._clear_channel_markers(1)
    panel.autosnap_button.setChecked(True)

    assert viewer.channel_autosnap_enabled[1] is True
    assert plot_widget.autosnap_enabled is True

    panel.detect_peaks_button.click()

    markers = viewer.integration_peak_markers[1]
    assert len(markers) == 2
    assert [marker.integration_x for marker in markers] == pytest.approx(
        [4.0, 6.0]
    )
    assert [
        marker.integrated_intensity for marker in markers
    ] == pytest.approx([60.0, 30.0])

    panel.detect_peaks_button.click()

    assert len(viewer.integration_peak_markers[1]) == 2


def test_integration_peak_coordinate_conversion_covers_channel_modes():
    horizontal = ROIRegion(
        target_id="detector",
        kind="box",
        qxy_min=1.0,
        qxy_max=3.0,
        qz_min=4.0,
        qz_max=8.0,
        integration_axis="qxy",
        integration_direction="horizontal",
    )
    vertical = ROIRegion(
        target_id="detector",
        kind="box",
        qxy_min=1.0,
        qxy_max=3.0,
        qz_min=4.0,
        qz_max=8.0,
        integration_axis="qz",
        integration_direction="vertical",
    )
    arch = ROIRegion(
        target_id="detector",
        kind="arch",
        qxy_center=0.5,
        qz_center=1.0,
        qr_min=2.0,
        qr_max=4.0,
        chi_min=-45.0,
        chi_max=45.0,
        integration_axis="chi",
        integration_direction="azimuthal",
    )

    assert _integration_peak_qspace_coordinate(horizontal, 2.5) == (
        pytest.approx(2.5),
        pytest.approx(6.0),
    )
    assert _integration_peak_qspace_coordinate(vertical, 6.5) == (
        pytest.approx(2.0),
        pytest.approx(6.5),
    )
    arch_qxy, arch_qz = _integration_peak_qspace_coordinate(arch, 30.0)
    assert arch_qxy == pytest.approx(2.0)
    assert arch_qz == pytest.approx(1.0 + 3.0 * np.cos(np.radians(30.0)))


def test_integration_channel_detaches_and_reattaches(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState()
    project.add_data_group(group)
    mask = project.add_mask(
        repo_root / "example" / "mask.edf", target_ids=[data_id]
    )
    calibrant = project.add_calibrant(
        repo_root / "example" / "calib.poni", target_ids=[data_id]
    )
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            confirmed=True,
        )
    )
    window = MainWindow(project=project)
    qtbot.addWidget(window)

    file_item = window.data_tree.tree.topLevelItem(0).child(0)
    window.data_tree.tree.setCurrentItem(file_item)
    viewer = window.tabs.widget(0)
    box = viewer.add_roi_from_bounds(0.1, 0.4, -1.5, -0.7)
    assert box is not None
    viewer._toggle_roi_channel(box.roi_id, 1, True)

    panel = viewer.channel_panels[1]
    assert not panel.plot_widget.isHidden()
    assert panel.placeholder.isHidden()
    panel.autosnap_button.setChecked(False)

    assert viewer.channel_autosnap_enabled[1] is False
    assert not panel.plot_widget.autosnap_enabled

    viewer._detach_channel(1)
    detached_window = viewer.channel_windows[1]
    qtbot.addWidget(detached_window)

    assert panel.detached
    assert panel.plot_widget.isHidden()
    assert not panel.placeholder.isHidden()
    assert len(detached_window.plot_widget.series) == 1
    assert detached_window.drag_label.text() == "Channel 1 (Vertical Box)"
    assert not detached_window.autosnap_button.isChecked()
    assert not detached_window.plot_widget.autosnap_enabled

    detached_window.autosnap_button.setChecked(True)

    assert viewer.channel_autosnap_enabled[1] is True
    assert panel.autosnap_button.isChecked()
    assert panel.plot_widget.autosnap_enabled

    viewer._reattach_channel(1)

    assert 1 not in viewer.channel_windows
    assert not panel.detached
    assert not panel.plot_widget.isHidden()
    assert panel.placeholder.isHidden()


def test_image_rotation_helper_rotates_clockwise():
    import numpy as np

    image = np.array([[1, 2, 3], [4, 5, 6]])

    rotated = _apply_image_rotation(image, 90)

    assert rotated.tolist() == [[4, 1], [5, 2], [6, 3]]


def test_image_orientation_helper_mirrors_after_rotation():
    import numpy as np

    image = np.array([[1, 2, 3], [4, 5, 6]])

    oriented = _apply_image_orientation(image, 90, True)

    assert oriented.tolist() == [[1, 4], [2, 5], [3, 6]]


def test_pyfai_orientation_mapping_covers_mirrored_transforms():
    assert sample_orientation_for_image_transform(0) == 1
    assert sample_orientation_for_image_transform(90) == 8
    assert sample_orientation_for_image_transform(180) == 3
    assert sample_orientation_for_image_transform(270) == 6
    assert sample_orientation_for_image_transform(0, mirrored_y=True) == 2
    assert sample_orientation_for_image_transform(90, mirrored_y=True) == 5
    assert sample_orientation_for_image_transform(180, mirrored_y=True) == 4
    assert sample_orientation_for_image_transform(270, mirrored_y=True) == 7


def test_confirmed_qspace_mapping_uses_raw_detector_frame_for_pyfai(
    qtbot,
    tmp_path,
    monkeypatch,
):
    import tifffile

    image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    mask = np.array([[False, True, False], [False, False, True]])
    image_path = tmp_path / "detector.tiff"
    mask_path = tmp_path / "mask.npy"
    calibrant_path = tmp_path / "calib.poni"
    tifffile.imwrite(image_path, image)
    np.save(mask_path, mask)
    calibrant_path.write_text("placeholder poni")

    group, _ = build_data_group_from_paths([image_path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState()
    project.add_data_group(group)
    mask_asset = project.add_mask(mask_path, target_ids=[data_id])
    calibrant = project.add_calibrant(calibrant_path, target_ids=[data_id])
    project.set_image_corrections(
        ImageCorrectionState(
            target_id=data_id,
            mask_asset_id=mask_asset.asset_id,
            calibrant_asset_id=calibrant.asset_id,
            image_rotation_deg=90,
            image_mirrored_y=False,
            pyfai_sample_orientation=sample_orientation_for_image_transform(
                90
            ),
            confirmed=True,
        )
    )

    captured = {}

    def fake_map(data, poni_file, *, config, mask=None, **kwargs):
        captured["data"] = np.asarray(data)
        captured["mask"] = np.asarray(mask)
        captured["sample_orientation"] = config.sample_orientation
        return xr.DataArray(
            np.ones((2, 2)),
            dims=("q_oop", "q_ip"),
            coords={"q_oop": [0.0, 1.0], "q_ip": [-1.0, 0.0]},
        )

    monkeypatch.setattr(
        "ewald.processing.qspace.map_grazing_incidence_qspace",
        fake_map,
    )

    viewer = DataViewerPane(project, data_id)
    qtbot.addWidget(viewer)

    assert viewer.coordinate_space == "qspace"
    assert captured["data"].tolist() == image.tolist()
    assert captured["mask"].tolist() == mask.tolist()
    assert captured["sample_orientation"] == 8


def test_data_tree_lists_saved_rois(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    data_id = group.data_files[0].data_id
    project = ProjectState()
    project.add_data_group(group)
    project.add_roi_region(
        ROIRegion(
            target_id=data_id,
            kind="box",
            name="Fit window",
            qxy_min=0.1,
            qxy_max=0.4,
            qz_min=0.2,
            qz_max=0.7,
        )
    )
    pane = DataTreePane()
    qtbot.addWidget(pane)
    pane.set_project(project)

    file_item = pane.tree.topLevelItem(0).child(0)
    rois_item = _child_with_text(file_item, "ROIs")

    assert rois_item.text(1) == "1"
    assert rois_item.child(0).text(0) == "Fit window"


def test_metadata_context_options_differ_by_import_kind(qtbot):
    file_dialog = MetadataImportContextDialog("file")
    folder_dialog = MetadataImportContextDialog("folder")
    qtbot.addWidget(file_dialog)
    qtbot.addWidget(folder_dialog)

    assert file_dialog.metadata_type_combo.itemText(0) == "Filename tokens"
    assert (
        folder_dialog.metadata_type_combo.itemText(0)
        == "Filename tokens per file"
    )


def test_manual_metadata_dialog_can_rename_inferred_attribute(
    qtbot, repo_root
):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    dialog = ManualMetadataDialog(group)
    qtbot.addWidget(dialog)

    target_row = None
    for row in range(dialog.table.rowCount()):
        if dialog.table.item(row, 2).text() == "sample_composition":
            target_row = row
            break

    assert target_row is not None
    dialog.table.item(target_row, 2).setText("composition_label")
    dialog.apply_metadata()

    metadata = group.data_files[0].metadata
    assert metadata["composition_label"] == "1MAI1PbI2"
    assert "sample_composition" not in metadata


def test_manual_metadata_dialog_selects_time_candidates(qtbot, repo_root):
    path = next((repo_root / "example").glob("*.tiff"))
    group, _ = build_data_group_from_paths([path], group_name="Example")
    dialog = ManualMetadataDialog(group)
    qtbot.addWidget(dialog)

    assert dialog.time_table.rowCount() == 1
    frame_combo = dialog.time_table.cellWidget(0, 1)
    exposure_combo = dialog.time_table.cellWidget(0, 2)
    assert isinstance(frame_combo, QtWidgets.QComboBox)
    assert isinstance(exposure_combo, QtWidgets.QComboBox)
    assert frame_combo.count() == 2
    assert exposure_combo.count() == 2
    assert frame_combo.currentData()["value"] == 2068.2
    assert exposure_combo.currentData()["value"] == 0.49

    exposure_combo.setCurrentIndex(0)
    dialog.apply_metadata()

    metadata = group.data_files[0].metadata
    assert metadata["frame_timestamp_s"] == 2068.2
    assert metadata["exposure_time_s"] == 2068.2
    fields = {
        field["key"]: field
        for field in metadata["_metadata_fields"]
        if field["key"] in {"frame_timestamp_s", "exposure_time_s"}
    }
    assert fields["frame_timestamp_s"]["raw_token"] == "2068.2s"
    assert fields["exposure_time_s"]["raw_token"] == "2068.2s"
