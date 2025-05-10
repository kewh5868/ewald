# File: ewald/ui/right_pane/unit_cell_view.py
"""
UnitCellView: 3D visualization of the currently selected unit cell.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from vispy.scene import SceneCanvas, visuals
from vispy.visuals.transforms import MatrixTransform
import numpy as np
import math

class UnitCellView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.canvas = SceneCanvas(keys='interactive', show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        layout.addWidget(self.canvas.native)
        
        # floating overlay label
        self.rot_label = QLabel(self)
        self.rot_label.setStyleSheet("color: white; background: rgba(0,0,0,0);")
        self.rot_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self.rot_label.raise_()

        visuals.XYZAxis(parent=self.view.scene)
        self.cell_lines = visuals.Line(color='white', parent=self.view.scene)

    def setCell(self, a, b, c, alpha, beta, gamma):
        corners = np.array([
            [0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0], [0, 0, 0],
            [0, 0, c], [a, 0, c], [a, b, c], [0, b, c], [0, 0, c]
        ], dtype=float)
        self.cell_lines.set_data(corners, connect='strip')

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # position at top-right, 200px wide by 20px high
        self.rot_label.setGeometry(self.width()-210, 10, 200, 20)

    def setOrientation(self, omega, chi, phi):
        """
        Apply Euler rotations (omega, chi, phi in degrees) to the unit cell wireframe.
        Rotation order: X (omega), then Y (chi), then Z (phi).
        """
        # Convert degrees to radians
        o = math.radians(omega)
        c_ang = math.radians(chi)
        p = math.radians(phi)

        # Rotation matrices (4x4 homogeneous)
        Rx = np.array([
            [1, 0, 0, 0],
            [0, math.cos(o), -math.sin(o), 0],
            [0, math.sin(o),  math.cos(o), 0],
            [0, 0, 0, 1]
        ], dtype=float)
        Ry = np.array([
            [ math.cos(c_ang), 0, math.sin(c_ang), 0],
            [0, 1, 0, 0],
            [-math.sin(c_ang), 0, math.cos(c_ang), 0],
            [0, 0, 0, 1]
        ], dtype=float)
        Rz = np.array([
            [math.cos(p), -math.sin(p), 0, 0],
            [math.sin(p),  math.cos(p), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        # Combined rotation
        matrix = Rz @ Ry @ Rx
        transform = MatrixTransform()
        transform.matrix = matrix
        self.cell_lines.transform = transform

        # update the label text
        self.rot_label.setText(f"ω={omega:.1f}°  χ={chi:.1f}°  φ={phi:.1f}°")
