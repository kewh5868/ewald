# -------------------------------------------
# File: ewald/ui/right_pane/unit_cell_view.py
"""
UnitCellView: 3D visualization of the currently selected unit cell.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from vispy.scene import SceneCanvas
from vispy.scene import visuals
import numpy as np

class UnitCellView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.canvas = SceneCanvas(keys='interactive', show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        layout.addWidget(self.canvas.native)
        # Draw axes
        visuals.XYZAxis(parent=self.view.scene)
        # Placeholder for cell edges
        self.cell_lines = visuals.Line(color='white', parent=self.view.scene)

    def setCell(self, a, b, c, alpha, beta, gamma):
        # Calculate orthogonal corners of unit cell
        corners = np.array([
            [0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0], [0, 0, 0],
            [0, 0, c], [a, 0, c], [a, b, c], [0, b, c], [0, 0, c]
        ], dtype=float)
        self.cell_lines.set_data(corners, connect='strip')

    def setOrientation(self, rot_matrix):
        # Apply rotation matrix to the cell lines
        transform = visuals.transforms.MatrixTransform()
        transform.matrix = rot_matrix
        self.cell_lines.transform = transform
