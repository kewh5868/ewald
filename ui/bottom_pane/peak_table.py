# File: ewald/ui/bottom_pane/peak_table.py
"""
PeakTableView: tabbed widget for managing three sets of data:
 - Experimental: index, q_xy, q_z, intensity, region, h, k, l
 - Calculated:   index, q_xy, q_z, h, k, l
 - ROI:          ROI Index, ROI Type, ROI Center q_xy, ROI Center q_z, C1, C2, C3, C4, Linked (hkl)
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableView, QPushButton,
    QHBoxLayout, QTabWidget, QMessageBox
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


class ROITableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._headers = [
            "ROI Index","ROI Type","ROI Center q_xy","ROI Center q_z",
            "C1","C2","C3","C4","Linked (hkl)"
        ]
        self._data = []  # list of tuples: (roi_type, qxy, qz, c1, c2, c3, c4, linked_hkl)

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

    def add_roi(self, roi_type, qxy, qz, c1, c2, c3, c4, linked_hkl):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self._data.append((roi_type, qxy, qz, c1, c2, c3, c4, linked_hkl))
        self.endInsertRows()

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

        # ROI tab
        self.roi_model = ROITableModel(self)
        roi_view = QTableView()
        roi_view.setModel(self.roi_model)
        roi_view.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(roi_view, "ROI")

        # Clear buttons for each
        clear_exp  = QPushButton("Clear Exp")
        clear_exp.clicked.connect(self.exp_model.clear)
        clear_calc = QPushButton("Clear Calc")
        clear_calc.clicked.connect(self.calc_model.clear)
        # -- Clear All ROIs button with confirmation --
        self.clear_roi_button = QPushButton("Clear All ROIs")
        self.clear_roi_button.clicked.connect(self._confirm_clear_rois)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(clear_exp)
        btn_layout.addWidget(clear_calc)
        btn_layout.addWidget(self.clear_roi_button)

        # Main layout
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addLayout(btn_layout)

    def _confirm_clear_rois(self):
        """
        Prompt user before clearing all ROIs from both the table and canvas.
        """
        reply = QMessageBox.question(
             self,
             "Clear All ROIs",
             "Are you sure? This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
         )

        if reply == QMessageBox.StandardButton.Yes:
            # If the model knows its manager, let it clear patches too
            if hasattr(self.roi_model, 'manager'):
                self.roi_model.manager.clear_all()
            else:
                # fallback: just clear the table
                self.roi_model.clear()