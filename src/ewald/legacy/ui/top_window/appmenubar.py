from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

class AppMenuBar(QMenuBar):
    """
    Shared application menu bar for Ewald with standard and custom menus.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- File Menu ---
        file_menu = self.addMenu("File")
        load_menu = QMenu("Load Files", self)
        load_action = QAction("Load Files...", self)
        load_menu.addAction(load_action)
        file_menu.addMenu(load_menu)
        self.load_action = load_action

        # --- Edit Menu ---
        edit_menu = self.addMenu("Edit")
        # (Existing edit actions retained or added here)

        # --- View Menu ---
        view_menu = self.addMenu("View")
        modify_range = QAction("Modify Plot Range...", self)
        view_menu.addAction(modify_range)
        self.modify_range_action = modify_range

        # --- Data Manager Menu ---
        data_manager_menu = self.addMenu("Data Manager")
        self.data_manager_menu = data_manager_menu

        # --- Tools Menu ---
        tools_menu = self.addMenu("Tools")
        self.tools_menu = tools_menu

        # --- Fit Menu ---
        fit_menu = self.addMenu("Fit")
        self.fit_menu = fit_menu

        # --- Windows Menu ---
        windows_menu = self.addMenu("Fit")
        self.windows_menu = windows_menu

        # --- Settings Menu ---
        settings_menu = self.addMenu("Settings")
        # Preferences option in Settings
        preferences_action = QAction("Preferences...", self)
        settings_menu.addAction(preferences_action)
        self.preferences_action = preferences_action
        self.settings_menu = settings_menu
