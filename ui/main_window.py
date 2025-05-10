# File: ewald/ui/main_window.py

from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QSplitter, QDockWidget,
    QWidget, QVBoxLayout
)
from PyQt6.QtCore import Qt

from .left_pane.file_tree       import FileTreeView
from .center_pane.image_view    import ImageCanvas
from .bottom_pane.peak_table    import PeakTableView
from .right_pane.unit_cell_view import UnitCellView
from .right_pane.structure_tree import StructureTreeView
from .right_pane.cell_params    import CellParamsEditor
from ewald.analysis.bragg_calculator import BraggCalculator

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # State for current structure and Bragg settings
        self.current_lattice     = None
        self.current_orientation = None
        self.peak_range          = (1, 1, 1)
        self.bragg_calc          = None

        print("[DEBUG] Initializing MainWindow")
        self.setup_ui()
        self.showMaximized()

    def setup_ui(self):
        print("[DEBUG] Setting up UI components")
        # --- Left: image file tree ---
        self.image_tree = FileTreeView()

        # --- Center: main image + peak table ---
        self.image_canvas = ImageCanvas()
        self.peak_table   = PeakTableView()

        center_split = QSplitter(Qt.Orientation.Vertical)
        center_split.addWidget(self.image_canvas)
        center_split.addWidget(self.peak_table)
        center_split.setStretchFactor(0, 5)
        center_split.setStretchFactor(1, 1)

        # --- Main splitter: left (tree) | center ---
        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.addWidget(self.image_tree)
        main_split.addWidget(center_split)
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 4)
        self.setCentralWidget(main_split)

        # --- Right dock: controls ---
        dock = QDockWidget("Controls", self)
        right_split = QSplitter(Qt.Orientation.Vertical)
        self.unit_cell_view = UnitCellView()
        self.struct_tree    = StructureTreeView()
        self.cell_params    = CellParamsEditor()
        right_split.addWidget(self.unit_cell_view)
        right_split.addWidget(self.struct_tree)
        right_split.addWidget(self.cell_params)
        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 2)
        right_split.setStretchFactor(2, 1)
        dock.setWidget(right_split)
        dock.setMinimumWidth(250)
        # Use Qt.RightDockWidgetArea instead of nested
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # --- Connect signals ---
        print("[DEBUG] Connecting signals")
        # Lattice -> update unit cell and Bragg
        self.cell_params.latticeChanged.connect(self.unit_cell_view.setCell)
        self.cell_params.latticeChanged.connect(self._on_lattice_changed)
        # Orientation -> update unit cell orientation and Bragg
        self.cell_params.orientationChanged.connect(self.unit_cell_view.setOrientation)
        self.cell_params.orientationChanged.connect(self._on_orientation_changed)
        # Peak range and calculate
        self.cell_params.peakRangeChanged.connect(self._on_peak_range_changed)
        self.cell_params.calculateRequested.connect(self._on_calculate_requested)
        # Custom structure -> add to structure tree
        self.cell_params.customStructureAdded.connect(
            lambda name, sys_, a, b, c, alpha, beta, gamma:
                self.struct_tree.addCustomStructure(name, sys_, a, b, c, alpha, beta, gamma)
        )
        # Selecting structure -> load into unit cell view
        self.struct_tree.structureSelected.connect(self._on_structure_selected)

    def _on_lattice_changed(self, a, b, c, alpha, beta, gamma):
        print(f"[DEBUG] Lattice changed: a={a}, b={b}, c={c}, α={alpha}, β={beta}, γ={gamma}")
        self.current_lattice = (a, b, c, alpha, beta, gamma)
        self.bragg_calc      = BraggCalculator(a, b, c, alpha, beta, gamma)
        self._compute_and_overlay_peaks()

    def _on_orientation_changed(self, omega, chi, phi):
        print(f"[DEBUG] Orientation changed: ω={omega}, χ={chi}, φ={phi}")
        self.current_orientation = (omega, chi, phi)
        self._compute_and_overlay_peaks()

    def _on_peak_range_changed(self, hmax, kmax, lmax):
        print(f"[DEBUG] Peak range changed: h_max={hmax}, k_max={kmax}, l_max={lmax}")
        self.peak_range = (hmax, kmax, lmax)
        self._compute_and_overlay_peaks()

    def _on_structure_selected(self, name, system, a, b, c, alpha, beta, gamma):
        print(f"[DEBUG] Structure selected: {name} ({system}), lattice=({a},{b},{c},{alpha},{beta},{gamma})")
        self.current_lattice     = (a, b, c, alpha, beta, gamma)
        self.bragg_calc          = BraggCalculator(a, b, c, alpha, beta, gamma)
        self.current_orientation = (0, 0, 0)
        self.unit_cell_view.setCell(a, b, c, alpha, beta, gamma)
        self.unit_cell_view.setOrientation(0, 0, 0)

    def _on_calculate_requested(self):
        print("[DEBUG] Calculate button pressed")
        if not self.bragg_calc:
            print("[DEBUG] No BraggCalculator available, skipping compute")
            return
        self._compute_and_overlay_peaks()

    def _compute_and_overlay_peaks(self):
        """
        Compute Bragg peaks (q_xy, q_z, hkl) and update image and table,
        only keeping positive q values and auto-scaling axes to fit the peaks.
        """
        print("[DEBUG] Starting Bragg peak computation...")
        if not self.bragg_calc or self.current_orientation is None:
            print("[DEBUG] Missing lattice/orientation, aborting compute")
            return

        # Compute peaks
        qxy, qz, hkl = self.bragg_calc.compute_peaks(
            orientation=self.current_orientation,
            hkl_range=self.peak_range
        )
        print(f"[DEBUG] Computed {qxy.size} raw peaks before filtering")

        # Filter only positive qxy & qz
        mask = (qxy > 0) & (qz > 0)
        qxy_filt = qxy[mask]
        qz_filt  = qz[mask]
        hkl_filt = hkl[mask]
        print(f"[DEBUG] {qxy_filt.size} peaks after filtering positive qxy/qz")

        if qxy_filt.size == 0:
            print("[DEBUG] No peaks to display after filtering")
            return

        # Clear main axes and reset base frame
        self.image_canvas.clear()
        ax = self.image_canvas.ax_main
        # Auto-scale axes to include all peaks
        ax.set_xlim(0, qxy_filt.max() * 1.1)
        ax.set_ylim(0, qz_filt.max() * 1.1)

        # Overlay filtered peaks
        ax.scatter(qxy_filt, qz_filt, s=50, edgecolors='r', facecolors='none')
        self.image_canvas.canvas.draw()
        print("[DEBUG] Peaks overlaid on image")

        # Populate Calculated table with filtered (q_xy, q_z, h, k, l)
        peaks = [
            (float(q), float(z), int(h), int(k), int(l))
            for q, z, (h, k, l) in zip(qxy_filt, qz_filt, hkl_filt)
        ]
        self.peak_table.calc_model.clear()
        self.peak_table.calc_model.add_peaks(peaks)
        print(f"[DEBUG] Populated Calculated peaks table with {len(peaks)} entries")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
