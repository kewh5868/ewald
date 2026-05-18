from PyQt6.QtWidgets import QToolBar, QFileDialog
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import pyqtSignal
import os


class MainToolBar(QToolBar):
    """Toolbar for GIWAXS main window with icon actions."""

    # Signals for loading external files
    maskLoaded = pyqtSignal(str)
    poniLoaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__('MainToolbar', parent)
        self.window = parent
        self.mask_files = []
        self.poni_files = []
        
        # Base path for icons (relative to this file)
        icons_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'resources', 'icons')
        )
        # Icon filenames
        icons = {
            'load_single': 'load_single.png',
            'load_multiple': 'load_multiple.png',
            'load_poni': 'load_poni.png',
            'load_mask': 'load_mask.png',
            'roi_box': 'roi_box.png',
            'roi_azimuthal': 'roi_azimuthal.png',
            'add_point': 'add_point.png',
            'remove_point': 'remove_point.png',
            'create_group': 'create_group.png',
        }
        # Load QIcons
        self.icons = {
            name: QIcon(os.path.join(icons_dir, filename))
            for name, filename in icons.items()
        }

        # Connect signals to parent handlers
        self.poniLoaded.connect(self.window.onPoniLoaded)
        self.maskLoaded.connect(self.window.onMaskLoaded)

        # Create and connect actions
        self._create_actions()
        # Add actions to the toolbar
        self._populate_toolbar()

    def _create_actions(self):
        # File loading actions
        self.load_single_action = QAction(
            self.icons['load_single'], 'Load Single Image', self
        )
        self.load_multiple_action = QAction(
            self.icons['load_multiple'], 'Load Series Images', self
        )
        self.load_poni_action = QAction(
            self.icons['load_poni'], 'Load PONI', self
        )
        self.load_mask_action = QAction(
            self.icons['load_mask'], 'Load Mask', self
        )

        # ROI and annotation tools
        self.roi_box_action = QAction(
            self.icons['roi_box'], 'Draw ROI (Box)', self
        )
        self.roi_azimuthal_action = QAction(
            self.icons['roi_azimuthal'], 'Draw ROI (Azimuthal)', self
        )
        self.add_point_action = QAction(
            self.icons['add_point'], 'Add Point', self
        )
        self.remove_point_action = QAction(
            self.icons['remove_point'], 'Remove Point', self
        )
        self.create_group_action = QAction(
            self.icons['create_group'], 'Create Group', self
        )

        # Configure and connect file-loading actions
        for action, slot in [
            (self.load_single_action, self.window.openLoadSingleImageDialog),
            (self.load_multiple_action, self.window.openLoadSeriesImageDialog),
            (self.load_poni_action, self._onLoadPoni),
            (self.load_mask_action, self._onLoadMask),
        ]:
            action.setToolTip(action.text())
            action.triggered.connect(slot)

        # Configure tooltips for ROI tools
        for action in [
            self.roi_box_action,
            self.roi_azimuthal_action,
            self.add_point_action,
            self.remove_point_action,
            self.create_group_action,
        ]:
            action.setToolTip(action.text())

    def _populate_toolbar(self):
        # File-load section
        self.addAction(self.load_single_action)
        self.addAction(self.load_multiple_action)
        self.addAction(self.load_poni_action)
        self.addAction(self.load_mask_action)
        self.addSeparator()
        # ROI and annotation tools
        self.addAction(self.roi_box_action)
        self.addAction(self.roi_azimuthal_action)
        self.addAction(self.add_point_action)
        self.addAction(self.remove_point_action)
        self.addAction(self.create_group_action)

    def _onLoadMask(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Select Mask File",
            "",
            "Mask Files (*.edf *.json)"
        )
        if path:
            # Avoid duplicates
            if path not in self.mask_files:
                self.mask_files.append(path)
            # Notify any listeners
            self.maskLoaded.emit(path)
            # Update the dialog dropdown if open
            dlg = getattr(self.window, 'loadSingleDialog', None)
            if dlg:
                dlg.addMaskFile(path)

    def _onLoadPoni(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Select PONI File",
            "",
            "PONI Files (*.poni)"
        )
        if path:
            # Avoid duplicates
            if path not in self.poni_files:
                self.poni_files.append(path)
            # Notify any listeners
            self.poniLoaded.emit(path)
            # Update the dialog dropdown if open
            dlg = getattr(self.window, 'loadSingleDialog', None)
            if dlg:
                dlg.addPoniFile(path)
