# File: ewald/ui/right_pane/structure_tree.py

from PyQt6.QtWidgets import QTreeView, QMenu
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import pyqtSignal, Qt, QModelIndex


class StructureTreeView(QTreeView):
    """
    Right‐hand file browser for structure definitions (CIFs and custom unit cells).
    Supports adding custom structures programmatically and deleting via context menu.
    """
    # Emitted when the user selects a structure (double‐click or single click)
    structureSelected = pyqtSignal(str)
    # Emitted when the user deletes a structure via right‐click
    structureDeleted = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Use a QStandardItemModel for flexibility (custom entries + file entries)
        self.model = QStandardItemModel(self)
        self.model.setHorizontalHeaderLabels(["Structures"])
        self.setModel(self.model)

        # Root item under which all structures live
        self._root = self.model.invisibleRootItem()

        # Enable context menu for deletion
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_context_menu)

        # Handle click to emit selection
        self.clicked.connect(self._on_item_clicked)

    def addCustomStructure(self, name: str, a, b, c, alpha, beta, gamma):
        """
        Add a new custom structure to the tree.
        `name` is the user‐given identifier.
        The lattice parameters are stored in the item's userData.
        """
        item = QStandardItem(name)
        item.setData(
            {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma},
            role=Qt.ItemDataRole.UserRole
        )
        item.setEditable(False)
        self._root.appendRow(item)

    def _on_item_clicked(self, index: QModelIndex):
        """
        Emit structureSelected with the name of the clicked structure.
        """
        item = self.model.itemFromIndex(index)
        if item:
            self.structureSelected.emit(item.text())

    def _open_context_menu(self, pos):
        """
        Show a context menu allowing the user to delete the selected structure.
        """
        index = self.indexAt(pos)
        if not index.isValid():
            return

        item = self.model.itemFromIndex(index)
        menu = QMenu(self)
        delete_action = menu.addAction("Delete Structure")
        action = menu.exec(self.viewport().mapToGlobal(pos))

        if action == delete_action:
            name = item.text()
            # Remove from the model
            self._root.removeRow(index.row())
            # Notify anyone interested
            self.structureDeleted.emit(name)
