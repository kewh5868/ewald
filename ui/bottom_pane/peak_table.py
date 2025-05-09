# -------------------------------------------
# File: ewald/ui/bottom_pane/peak_table.py
"""
PeakTableView: table for entering and editing peak positions.
"""
from PyQt6.QtWidgets import QTableView

class PeakTableView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        # TODO: set model to custom QAbstractTableModel backed by xarray DataArray
