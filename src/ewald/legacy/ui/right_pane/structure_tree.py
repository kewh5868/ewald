# File: ewald/ui/right_pane/structure_tree.py
"""
StructureTreeView: table of structures showing lattice parameters and source.
Columns: Name | System | a | b | c | alpha | beta | gamma | Source
"""
from PyQt6.QtWidgets import QTreeView, QMenu
from PyQt6.QtGui      import QStandardItemModel, QStandardItem
from PyQt6.QtCore     import pyqtSignal, Qt, QModelIndex

class StructureTreeView(QTreeView):
    # name, system, a, b, c, alpha, beta, gamma
    structureSelected = pyqtSignal(str, str, float, float, float, float, float, float)
    structureDeleted  = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # # 9-column model
        # self.model = QStandardItemModel(0, 9, self)
        # headers = ["Name","System","a","b","c","alpha","beta","gamma","Source"]
        # now a 12-column model, adding rot_ω, rot_χ, rot_φ
        self.model = QStandardItemModel(0, 12, self)
        headers = ["Name","System","a","b","c","alpha","beta","gamma",
                   "rot_ω","rot_χ","rot_φ","Source"]
        self.model.setHorizontalHeaderLabels(headers)
        self.setModel(self.model)
        self.setSelectionBehavior(self.SelectionBehavior.SelectRows)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_context_menu)
        self.clicked.connect(self._on_item_clicked)

    def addCustomStructure(self, name, system, a, b, c, alpha, beta, gamma):
        src = "Custom"
        items = [
            QStandardItem(name),
            QStandardItem(system),
            QStandardItem(f"{a:.4f}"),
            QStandardItem(f"{b:.4f}"),
            QStandardItem(f"{c:.4f}"),
            QStandardItem(f"{alpha:.2f}"),
            QStandardItem(f"{beta:.2f}"),
            QStandardItem(f"{gamma:.2f}"),
            QStandardItem(src)
        ]
        for it in items:
            it.setEditable(False)
        self.model.appendRow(items)

    def _on_item_clicked(self, index: QModelIndex):
        row = index.row()
        vals = [self.model.item(row, col).text() for col in range(9)]
        name, sys_ = vals[0], vals[1]
        a, b, c = map(float, vals[2:5])
        alpha, beta, gamma = map(float, vals[5:8])
        self.structureSelected.emit(name, sys_, a, b, c, alpha, beta, gamma)

    def _open_context_menu(self, pos):
        idx = self.indexAt(pos)
        if not idx.isValid():
            return
        menu = QMenu(self)
        action = menu.addAction("Delete Structure")
        chosen = menu.exec(self.viewport().mapToGlobal(pos))
        if chosen == action:
            row = idx.row()
            name = self.model.item(row, 0).text()
            self.model.removeRow(row)
            self.structureDeleted.emit(name)

    def updateStructureRotation(self, name, rot_omega, rot_chi, rot_phi):
        # find row by matching the Name column
        for row in range(self.model.rowCount()):
            if self.model.item(row, 0).text() == name:
                self.model.item(row, 8).setText(f"{rot_omega:.1f}")
                self.model.item(row, 9).setText(f"{rot_chi:.1f}")
                self.model.item(row,10).setText(f"{rot_phi:.1f}")
                return
