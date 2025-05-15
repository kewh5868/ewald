from PyQt6.QtWidgets import (QMenuBar, QMenu)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

class AppMenuBar(QMenuBar):
    """
    Shared application menu bar with standard menus.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        file_menu = self.addMenu("File")
        load_menu = QMenu("Load Files", self)
        load_action = QAction("Load Files...", self)
        load_menu.addAction(load_action)
        file_menu.addMenu(load_menu)
        self.load_action = load_action
        self.addMenu("Edit")
        view_menu = self.addMenu("View")
        modify_range = QAction("Modify Plot Range...", self)
        view_menu.addAction(modify_range)
        self.modify_range_action = modify_range
        self.addMenu("Tools")
        self.addMenu("Windows")
        self.addMenu("Settings")