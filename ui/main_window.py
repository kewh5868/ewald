import sys
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QSplitter, QDockWidget,
    QWidget, QVBoxLayout, QMenuBar, QMenu,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
import numpy as np
from ewald.analysis.reciprocal_calculator import ReciprocalCalculator

# UI components
from .left_pane.file_tree import FileTreeView
from .center_pane.image_view import ImageCanvas
from .bottom_pane.peak_table import PeakTableView
from .right_pane.unit_cell_view import UnitCellView
from .right_pane.structure_tree import StructureTreeView
from .right_pane.cell_params import CellParamsEditor
from .top_window.appmenubar import AppMenuBar
from .top_window.maintoolbar import MainToolBar
from .center_pane.roi_manager import ROIManager
from .dialogs.load_single_image_dialog import LoadSingleImageDialog

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.xmin = self.xmax = None
        self.ymin = self.ymax = None
        self.current_lattice = None
        self.current_orientation = (0.0, 0.0, 0.0)
        self.peak_range = (1, 1, 1)
        self.calc = None

        ## Setup the main UI
        self.setup_ui()
        menu = AppMenuBar(self)
        self.setMenuBar(menu)
        menu.load_action.triggered.connect(self.load_files)
        menu.modify_range_action.triggered.connect(self.open_plot_range_dialog)

        ## Add the toolbar
        self.toolbar = MainToolBar(self)
        
        ## Connect single image dialog
        self.toolbar.load_single_action.triggered.connect(
        self.open_load_single_image_dialog
    )
        # allow toggle behavior for ROI
        self.toolbar.roi_box_action.setCheckable(True)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        
        # Initialize dialogs
        self._init_dialogs()

        # Instantiate ROI manager and hook ROI action after UI setup
        # ROI drawing will only activate when the toolbar button is toggled
        self.roi_manager = ROIManager(self.image_canvas, self.peak_table)
        self.toolbar.roi_box_action.toggled.connect(
            self.roi_manager.enable_selector
        )
        # When exiting ROI mode, ensure selector is disabled
        self.toolbar.roi_box_action.toggled.connect(
            lambda checked: self.toolbar.roi_box_action.setChecked(checked)
        )  # reflect state

    def setup_ui(self):
        self.image_tree = FileTreeView()
        self.image_tree.imageSelected.connect(self.on_file_selected)

        self.image_canvas = ImageCanvas()
        self.image_canvas.fig.set_facecolor('white')
        self.peak_table = PeakTableView()

        center_split = QSplitter(Qt.Orientation.Vertical)
        center_split.addWidget(self.image_canvas)
        center_split.addWidget(self.peak_table)
        center_split.setStretchFactor(0, 5)
        center_split.setStretchFactor(1, 1)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.addWidget(self.image_tree)
        main_split.addWidget(center_split)
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 4)
        self.setCentralWidget(main_split)

        dock = QDockWidget("Controls", self)
        right_split = QSplitter(Qt.Orientation.Vertical)
        self.unit_cell_view = UnitCellView()
        self.struct_tree = StructureTreeView()
        self.cell_params = CellParamsEditor()
        right_split.addWidget(self.unit_cell_view)
        right_split.addWidget(self.struct_tree)
        right_split.addWidget(self.cell_params)
        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 2)
        right_split.setStretchFactor(2, 1)
        dock.setWidget(right_split)
        dock.setMinimumWidth(250)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self.cell_params.latticeChanged.connect(self.on_lattice_changed)
        self.cell_params.orientationChanged.connect(self.on_orientation_changed)
        self.cell_params.calculateRequested.connect(self.compute_peaks)
        # also update tree on orientation change
        self.cell_params.orientationChanged.connect(self.update_tree_rotation)

        self.cell_params.peakRangeChanged.connect(self.on_peak_range_changed)
        self.cell_params.customStructureAdded.connect(
            lambda name, sys_, a,b,c,alpha,beta,gamma: self.struct_tree.addCustomStructure(
                name, sys_, a,b,c,alpha,beta,gamma)
        )
        self.struct_tree.structureSelected.connect(self.on_structure_selected)

    def update_tree_rotation(self, omega, chi, phi):
        # push into the tree model
        if hasattr(self, "current_structure_name"):
            self.struct_tree.updateStructureRotation(
                self.current_structure_name,
                omega, chi, phi
            )

    def load_files(self):
        pass

    def on_file_selected(self, path: str):
        self.compute_peaks()

    def open_plot_range_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Modify Plot Range")
        form = QFormLayout(dlg)
        cur_xlim = self.image_canvas.ax_main.get_xlim()
        cur_ylim = self.image_canvas.ax_main.get_ylim()
        self.xmin_edit = QLineEdit(str(cur_xlim[0] if self.xmin is None else self.xmin))
        self.xmax_edit = QLineEdit(str(cur_xlim[1] if self.xmax is None else self.xmax))
        self.ymin_edit = QLineEdit(str(cur_ylim[0] if self.ymin is None else self.ymin))
        self.ymax_edit = QLineEdit(str(cur_ylim[1] if self.ymax is None else self.ymax))
        form.addRow("X min:", self.xmin_edit)
        form.addRow("X max:", self.xmax_edit)
        form.addRow("Y min:", self.ymin_edit)
        form.addRow("Y max:", self.ymax_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addWidget(buttons)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            try:
                self.xmin = float(self.xmin_edit.text())
                self.xmax = float(self.xmax_edit.text())
                self.ymin = float(self.ymin_edit.text())
                self.ymax = float(self.ymax_edit.text())
                self.compute_peaks()
            except ValueError:
                pass

    def on_lattice_changed(self, a, b, c, alpha, beta, gamma):
        self.current_lattice = (a, b, c, alpha, beta, gamma)
        self.unit_cell_view.setCell(a, b, c, alpha, beta, gamma)
        self.calc = ReciprocalCalculator(a, b, c, alpha, beta, gamma)
        self.compute_peaks()

    def on_orientation_changed(self, omega, chi, phi):
        self.current_orientation = (omega, chi, phi)
        self.unit_cell_view.setOrientation(omega, chi, phi)
        self.compute_peaks()

    def on_peak_range_changed(self, hmax, kmax, lmax):
        self.peak_range = (hmax, kmax, lmax)
        self.compute_peaks()

    def on_structure_selected(self, name, sys_, a, b, c, alpha, beta, gamma):
        self.current_structure_name = name
        self.current_lattice = (a, b, c, alpha, beta, gamma)
        self.calc = ReciprocalCalculator(a, b, c, alpha, beta, gamma)
        self.current_orientation = (0.0, 0.0, 0.0)
        self.unit_cell_view.setCell(a, b, c, alpha, beta, gamma)
        self.unit_cell_view.setOrientation(0.0, 0.0, 0.0)
        self.compute_peaks()

    def compute_peaks(self):
        if not self.calc or self.current_lattice is None:
            return
        a, b, c, alpha, beta, gamma = self.current_lattice
        omega, chi, phi = self.current_orientation
        # Rotate real-space cell
        a_vec, b_vec, c_vec = self.calc.a_vec, self.calc.b_vec, self.calc.c_vec
        a_r, b_r, c_r = rotate_lattice(a_vec, b_vec, c_vec, [1,0,0], omega)
        a_r, b_r, c_r = rotate_lattice(a_r, b_r, c_r, [0,1,0], chi)
        a_r, b_r, c_r = rotate_lattice(a_r, b_r, c_r, [0,0,1], phi)
        self.calc.update_reciprocal(a_r, b_r, c_r)
        hmax, kmax, lmax = self.peak_range
        peaks = self.calc.find_peaks(hkl_range=range(-hmax, hmax+1), target_q=None, tol=0.0)
        qxy_vals, qz_vals, data_peaks = [], [], []
        for (h,k,l), q_mag, chi_deg in peaks:
            cr = np.deg2rad(chi_deg)
            qxy_val = q_mag * np.sin(cr)
            qz_val = q_mag * np.cos(cr)
            if qxy_val <= 0 or qz_val <= 0:
                continue
            qxy_vals.append(qxy_val)
            qz_vals.append(qz_val)
            data_peaks.append((qxy_val, qz_val, h, k, l))
        if not qxy_vals:
            return
        ax = self.image_canvas.ax_main
        self.image_canvas.clear()
        if self.xmin is not None:
            ax.set_xlim(self.xmin, self.xmax)
        else:
            ax.set_xlim(min(qxy_vals)*0.9, max(qxy_vals)*1.1)
        if self.ymin is not None:
            ax.set_ylim(self.ymin, self.ymax)
        else:
            ax.set_ylim(min(qz_vals)*0.9, max(qz_vals)*1.1)
        ax.scatter(qxy_vals, qz_vals, s=50, edgecolors='r', facecolors='none')
        self.image_canvas.canvas.draw()
        self.peak_table.calc_model.clear()
        self.peak_table.calc_model.add_peaks(data_peaks)

    def open_load_single_image_dialog(self):
        dlg = LoadSingleImageDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            path = dlg.file_path_edit.text()
            mask = dlg.mask_combo.currentText()
            poni = dlg.poni_combo.currentText()
            incidence = dlg.incidence_spin.value()
            polarization = dlg.polarization_spin.value()
            solid_angle = dlg.solid_angle_chk.isChecked()
            metadata = dlg.get_metadata()
            # Now pass these into your loader, e.g.:
            self.load_single_image(path, mask, poni,
                                incidence, polarization,
                                solid_angle, metadata)

    def _init_dialogs(self):
        # Single image loader dialog
        self.loadSingleDialog = LoadSingleImageDialog(self)
        # Series images loader dialog
        # self.loadSeriesDialog = LoadSeriesImageDialog(self)

    # Open dialog slots
    def openLoadSingleImageDialog(self):
        self.loadSingleDialog.show()

    def openLoadSeriesImageDialog(self):
        self.loadSeriesDialog.show()

    # Handlers for toolbar file-loading signals
    def onPoniLoaded(self, filePath: str):
        """Receive new PONI path and add to single-image dialog dropdown."""
        try:
            self.loadSingleDialog.addPoniFile(filePath)
        except AttributeError:
            # Fallback: reload full list
            self.loadSingleDialog.reloadPoniList()

    def onMaskLoaded(self, filePath: str):
        """Receive new mask path and add to single-image dialog dropdown."""
        try:
            self.loadSingleDialog.addMaskFile(filePath)
        except AttributeError:
            # Fallback: reload full list
            self.loadSingleDialog.reloadMaskList()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
