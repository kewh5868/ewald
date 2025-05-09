# File: ewald/ui/main_window.py
"""
MainWindow assembling all UI panes into a QMainWindow with splitters and docks.
"""
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QApplication,
    QDockWidget, QVBoxLayout
)
from PyQt6.QtCore import Qt

# Import pane components
from .left_pane.file_tree import FileTreeView
from .center_pane.image_view import ImageCanvas
from .center_pane.subplots import IntegrationPlot1D, TopographyPlot2D
from .bottom_pane.peak_table import PeakTableView
from .right_pane.unit_cell_view import UnitCellView
from .right_pane.structure_tree import StructureTreeView
from .right_pane.cell_params import CellParamsEditor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EWALD GUI")
        self._init_ui()
        # Open window maximized
        self.showMaximized()

    def _init_ui(self):
        # Top-level horizontal splitter: left (file tree) and main area
        hsplit = QSplitter(Qt.Orientation.Horizontal, self)
        hsplit.setChildrenCollapsible(False)
        self.setCentralWidget(hsplit)

        # Left pane: file tree (1/5 width)
        self.file_tree = FileTreeView()
        hsplit.addWidget(self.file_tree)
        hsplit.setStretchFactor(0, 1)

        # Main area container: vertical splitter for center and bottom
        main_container = QSplitter(Qt.Orientation.Vertical)
        main_container.setChildrenCollapsible(False)
        hsplit.addWidget(main_container)
        hsplit.setStretchFactor(1, 4)

        # Center area: vertical splitter for image + subplots
        center_split = QSplitter(Qt.Orientation.Vertical)
        center_split.setChildrenCollapsible(False)

        # 2D image canvas
        self.image_canvas = ImageCanvas()
        center_split.addWidget(self.image_canvas)

        # Subplots container
        sub_widget = QWidget()
        sub_layout = QVBoxLayout(sub_widget)
        self.int_plot = IntegrationPlot1D()
        self.topo_plot = TopographyPlot2D()
        sub_layout.addWidget(self.int_plot)
        sub_layout.addWidget(self.topo_plot)
        center_split.addWidget(sub_widget)

        main_container.addWidget(center_split)

        # Bottom pane: peak table
        self.peak_table = PeakTableView()
        main_container.addWidget(self.peak_table)
        main_container.setStretchFactor(0, 5)
        main_container.setStretchFactor(1, 1)

        # Right-side control panel dock
        control_widget = QWidget()
        ctrl_layout = QVBoxLayout(control_widget)
        self.unit_view = UnitCellView()
        self.struct_tree = StructureTreeView()
        self.cell_params = CellParamsEditor()
        ctrl_layout.addWidget(self.unit_view)
        ctrl_layout.addWidget(self.struct_tree)
        ctrl_layout.addWidget(self.cell_params)
        control_widget.setLayout(ctrl_layout)

        dock = QDockWidget("Control Panel", self)
        dock.setWidget(control_widget)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())