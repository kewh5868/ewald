from PyQt6.QtWidgets import QToolBar
from PyQt6.QtGui import QAction, QIcon
import os


class MainToolBar(QToolBar):
    """Toolbar for GIWAXS main window with icon actions."""
    def __init__(self, parent=None):
        super().__init__('MainToolbar', parent)
        # Base path for icons (relative to this file)
        icons_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'resources', 'icons')
        )
        # Icon filenames
        icons = {
            'load_single': 'load_single.png',
            'load_multiple': 'load_multiple.png',
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

        # Create actions and set tooltips
        self._create_actions()
        # Populate toolbar with actions
        self._populate_toolbar()

    def _create_actions(self):
        self.load_single_action = QAction(
            self.icons['load_single'], 'Load Single Image', self
        )
        self.load_multiple_action = QAction(
            self.icons['load_multiple'], 'Load Series Images', self
        )
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
        # Show the text on hover
        for action in [
            self.load_single_action,
            self.load_multiple_action,
            self.roi_box_action,
            self.roi_azimuthal_action,
            self.add_point_action,
            self.remove_point_action,
            self.create_group_action,
        ]:
            action.setToolTip(action.text())

    def _populate_toolbar(self):
        # File/load section
        self.addAction(self.load_single_action)
        self.addAction(self.load_multiple_action)
        self.addSeparator()
        # ROI and annotation tools
        self.addAction(self.roi_box_action)
        self.addAction(self.roi_azimuthal_action)
        self.addAction(self.add_point_action)
        self.addAction(self.remove_point_action)
        self.addAction(self.create_group_action)
