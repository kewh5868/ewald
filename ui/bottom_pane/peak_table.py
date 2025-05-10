# File: ewald/ui/bottom_pane/peak_table.py
"""
PeakTableView: tabbed widget for managing two sets of peaks:
 - Experimental: index, q_xy, q_z, intensity, region, h, k, l
 - Calculated:   index, q_xy, q_z, h, k, l
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableView, QPushButton,
    QHBoxLayout, QTabWidget
)
from PyQt6.QtCore    import Qt, QAbstractTableModel, QModelIndex, QVariant

class ExperimentalPeakModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._headers = ["#","q_xy","q_z","Intensity","Region","h","k","l"]
        self._data = []  # list of tuples: (qxy, qz, intensity, region, h, k, l)

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        r, c = index.row(), index.column()
        if c == 0:
            return r + 1
        val = self._data[r][c-1]
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        return QVariant()

    def add_peak(self, qxy, qz, intensity, region, h, k, l):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self._data.append((qxy, qz, intensity, region, h, k, l))
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self._data.clear()
        self.endResetModel()

class CalculatedPeakModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._headers = ["#","q_xy","q_z","h","k","l"]
        self._data = []  # list of tuples: (qxy, qz, h, k, l)

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        r, c = index.row(), index.column()
        if c == 0:
            return r + 1
        val = self._data[r][c-1]
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        return QVariant()

    def add_peaks(self, peaks):
        """
        peaks: iterable of (qxy, qz, h, k, l)
        """
        self.beginResetModel()
        self._data = list(peaks)
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._data.clear()
        self.endResetModel()

class PeakTableView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        tabs = QTabWidget()
        # Experimental tab
        self.exp_model = ExperimentalPeakModel(self)
        exp_view = QTableView()
        exp_view.setModel(self.exp_model)
        exp_view.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(exp_view, "Experimental")

        # Calculated tab
        self.calc_model = CalculatedPeakModel(self)
        calc_view = QTableView()
        calc_view.setModel(self.calc_model)
        calc_view.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(calc_view, "Calculated")

        # Clear buttons for each
        clear_exp = QPushButton("Clear Exp")
        clear_exp.clicked.connect(self.exp_model.clear)
        clear_calc = QPushButton("Clear Calc")
        clear_calc.clicked.connect(self.calc_model.clear)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(clear_exp)
        btn_layout.addWidget(clear_calc)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addLayout(btn_layout)
