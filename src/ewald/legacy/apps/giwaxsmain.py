import sys
import os
import numpy as np
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTableView, QSlider, QCheckBox, QLabel, QGroupBox, QFormLayout,
    QFileSystemModel, QTreeView
)
from PyQt6.QtCore import Qt

# Import your existing components
from ewald.ui.right_pane.unit_cell_view import UnitCellView
from ewald.ui.bottom_pane.peak_table import PeakTableView
from ewald.ui.right_pane.structure_tree import StructureTreeView
from ewald.analysis.reciprocal_calculator import ReciprocalCalculator

def rotate_lattice(a, b, c, axis, angle_deg):
    """
    Rotate lattice vectors a, b, c about `axis` by `angle_deg` degrees.
    """
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    theta = np.deg2rad(angle_deg)
    ux, uy, uz = axis
    cth, sth = np.cos(theta), np.sin(theta)
    R = np.array([
        [cth + ux*ux*(1-cth),    ux*uy*(1-cth) - uz*sth, ux*uz*(1-cth) + uy*sth],
        [uy*ux*(1-cth) + uz*sth, cth + uy*uy*(1-cth),    uy*uz*(1-cth) - ux*sth],
        [uz*ux*(1-cth) - uy*sth, uz*uy*(1-cth) + ux*sth, cth + uz*uz*(1-cth)]
    ], dtype=float)
    return R.dot(a), R.dot(b), R.dot(c)

class GIWAXSMainWindow(QMainWindow):
    def __init__(self, img_array, qxy, qz,
                 a_len, b_len, c_len,
                 alpha_deg, beta_deg, gamma_deg,
                 hkl_range=(1,1,1), structure_dir=None):
        super().__init__()
        self.img = img_array
        self.qxy = qxy
        self.qz = qz
        # Initialize reciprocal-space calculator
        self.calc = ReciprocalCalculator(a_len, b_len, c_len,
                                         alpha_deg, beta_deg, gamma_deg)
        self.hkl_range = hkl_range
        # Directory for structure files (defaults to current working directory)
        self.structure_dir = structure_dir or os.getcwd()

        self._init_ui()
        self.update_views()
        self.setWindowTitle("GIWAXS Reciprocal Space Viewer")
        self.showMaximized()

    def _init_ui(self):
        # Central widget and layout
        central = QWidget()
        main_layout = QVBoxLayout(central)

        # Top region: data plot and unit cell + structure tree + file tree
        top_split = QSplitter(Qt.Orientation.Horizontal)

        # Main plot canvas
        self.fig = Figure(facecolor='white')
        self.canvas = FigureCanvas(self.fig)
        self.ax_main = self.fig.add_subplot(111)
        top_split.addWidget(self.canvas)

        # Right pane: vertical split of UnitCellView, StructureTreeView, and FileTree
        right_split = QSplitter(Qt.Orientation.Vertical)
        # Unit cell view
        self.unit_cell_view = UnitCellView()
        self.unit_cell_view.setCell(
            self.calc.a_len, self.calc.b_len, self.calc.c_len,
            self.calc.alpha_deg, self.calc.beta_deg, self.calc.gamma_deg
        )
        right_split.addWidget(self.unit_cell_view)

        # Structure tree view (existing structures)
        self.struct_tree = StructureTreeView()
        right_split.addWidget(self.struct_tree)

        # File system tree view for structure files
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(self.structure_dir)
        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(self.structure_dir))
        right_split.addWidget(self.file_tree)

        # Adjust stretch: unit cell larger than structure and file trees
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 1)
        right_split.setStretchFactor(2, 1)

        top_split.addWidget(right_split)
        top_split.setStretchFactor(0, 4)
        top_split.setStretchFactor(1, 1)
        main_layout.addWidget(top_split)

        # Bottom region: peaks table and orientation sliders
        bottom_split = QSplitter(Qt.Orientation.Horizontal)
        # Peaks table (calculated peaks)
        self.peak_table = PeakTableView()
        # Clear experimental peaks tab
        self.peak_table.exp_model.clear()
        bottom_split.addWidget(self.peak_table)
        # Slider panel
        slider_group = QGroupBox("Orientation Controls")
        form = QFormLayout(slider_group)
        self.slider_omega = QSlider(Qt.Orientation.Horizontal)
        self.slider_omega.setRange(-180, 180)
        self.slider_chi = QSlider(Qt.Orientation.Horizontal)
        self.slider_chi.setRange(-180, 180)
        self.slider_phi = QSlider(Qt.Orientation.Horizontal)
        self.slider_phi.setRange(-180, 180)
        self.checkbox_draw = QCheckBox("Draw Unit Cell")
        self.checkbox_draw.setChecked(True)
        form.addRow("ω (deg)", self.slider_omega)
        form.addRow("χ (deg)", self.slider_chi)
        form.addRow("φ (deg)", self.slider_phi)
        form.addRow(self.checkbox_draw)
        bottom_split.addWidget(slider_group)
        bottom_split.setStretchFactor(0, 3)
        bottom_split.setStretchFactor(1, 1)
        main_layout.addWidget(bottom_split)

        self.setCentralWidget(central)

        # Signal connections
        for slider in (self.slider_omega, self.slider_chi, self.slider_phi):
            slider.valueChanged.connect(self.update_views)
        self.checkbox_draw.toggled.connect(self.unit_cell_view.setVisible)

    def update_views(self):
        # Get orientation angles
        omega = self.slider_omega.value()
        chi = self.slider_chi.value()
        phi = self.slider_phi.value()
        # Rotate real-space vectors
        a_vec, b_vec, c_vec = self.calc.a_vec, self.calc.b_vec, self.calc.c_vec
        a_r, b_r, c_r = rotate_lattice(a_vec, b_vec, c_vec, [1,0,0], omega)
        a_r, b_r, c_r = rotate_lattice(a_r, b_r, c_r, [0,1,0], chi)
        a_r, b_r, c_r = rotate_lattice(a_r, b_r, c_r, [0,0,1], phi)
        # Update reciprocal vectors
        self.calc.update_reciprocal(a_r, b_r, c_r)
        # Compute peaks
        peaks = self.calc.find_peaks(hkl_range=
                                     range(-self.hkl_range[0], self.hkl_range[0]+1),
                                     target_q=None, tol=0.0)
        # Update main plot
        self.ax_main.cla()
        vmin, vmax = np.percentile(self.img, 1), np.percentile(self.img, 99.3)
        self.ax_main.imshow(
            self.img, origin='lower', aspect='auto', cmap='turbo',
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
            extent=(self.qxy.min(), self.qxy.max(), self.qz.min(), self.qz.max())
        )
        # Overlay peaks
        for (h,k,l), q_mag, chi_p in peaks:
            cr = np.deg2rad(chi_p)
            qxy_val = q_mag * np.sin(cr)
            qz_val  = q_mag * np.cos(cr)
            self.ax_main.scatter(qxy_val, qz_val,
                                 s=30, edgecolors='r', facecolors='none')
        self.canvas.draw()
        # Update table
        data = []
        for (h,k,l), q_mag, chi_p in peaks:
            cr = np.deg2rad(chi_p)
            data.append((q_mag*np.sin(cr), q_mag*np.cos(cr), h, k, l))
        self.peak_table.calc_model.clear()
        self.peak_table.calc_model.add_peaks(data)
        # Update unit cell orientation
        self.unit_cell_view.setOrientation(omega, chi, phi)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GIWAXSMainWindow(img_array=np.zeros((100,100)),
                           qxy=np.linspace(0,3,100),
                           qz=np.linspace(0,3,100),
                           a_len=10.0, b_len=10.0, c_len=10.0,
                           alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0)
    win.show()
    sys.exit(app.exec())
