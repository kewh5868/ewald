# File: ewald/ui/main_window.py

from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QSplitter, QDockWidget,
    QWidget, QVBoxLayout
)
from PyQt6.QtCore import Qt

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
        self.setup_ui()
        self.showMaximized()

    def setup_ui(self):
        # Left pane: image file tree
        self.image_tree = FileTreeView()

        # Center widgets
        self.image_canvas = ImageCanvas()
        self.int_plot = IntegrationPlot1D()
        self.topo_plot = TopographyPlot2D()
        self.peak_table = PeakTableView()

        # Center splitter: image, subplots, table
        center_splitter = QSplitter(Qt.Orientation.Vertical)
        center_splitter.addWidget(self.image_canvas)

        subplots_splitter = QSplitter(Qt.Orientation.Horizontal)
        subplots_splitter.addWidget(self.int_plot)
        subplots_splitter.addWidget(self.topo_plot)
        center_splitter.addWidget(subplots_splitter)

        center_splitter.addWidget(self.peak_table)

        # Main splitter: left (images) | center
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self.image_tree)
        main_splitter.addWidget(center_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 4)
        self.setCentralWidget(main_splitter)

        # Right dock: controls
        dock = QDockWidget("Controls", self)
        right_container = QWidget()
        layout = QVBoxLayout(right_container)

        # 3D unit-cell view
        self.unit_cell_view = UnitCellView()
        layout.addWidget(self.unit_cell_view)

        # Structure file tree
        self.struct_tree = StructureTreeView()
        layout.addWidget(self.struct_tree)

        # Cell parameters editor
        self.cell_params = CellParamsEditor()
        layout.addWidget(self.cell_params)

        dock.setWidget(right_container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # Signal hookups
        self.cell_params.latticeChanged.connect(self.unit_cell_view.setCell)
        self.cell_params.orientationChanged.connect(self.unit_cell_view.setOrientation)
        self.cell_params.customStructureAdded.connect(self.struct_tree.addCustomStructure)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())