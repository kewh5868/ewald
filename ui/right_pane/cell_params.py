# File: ewald/ui/right_pane/cell_params.py
"""
CellParamsEditor: edit unit cell parameters and orientation.
"""
from PyQt6.QtWidgets import QWidget, QTabWidget, QFormLayout, QDoubleSpinBox, QVBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal

class CellParamsEditor(QWidget):
    # Emitted when lattice or orientation parameters are applied
    latticeChanged = pyqtSignal(float, float, float, float, float, float)
    orientationChanged = pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        tabs = QTabWidget()
        # Lattice parameters tab
        lattice_tab = QWidget()
        form1 = QFormLayout(lattice_tab)
        self.spin_a = QDoubleSpinBox(); self.spin_b = QDoubleSpinBox()
        self.spin_c = QDoubleSpinBox(); self.spin_alpha = QDoubleSpinBox()
        self.spin_beta = QDoubleSpinBox(); self.spin_gamma = QDoubleSpinBox()
        for name, spin in [
            ("a", self.spin_a), ("b", self.spin_b), ("c", self.spin_c),
            ("alpha", self.spin_alpha), ("beta", self.spin_beta), ("gamma", self.spin_gamma)
        ]:
            spin.setRange(0.0, 1000.0)
            spin.setDecimals(4)
            form1.addRow(name, spin)
        tabs.addTab(lattice_tab, "Lattice")

        # Orientation tab
        orient_tab = QWidget()
        form2 = QFormLayout(orient_tab)
        self.spin_omega = QDoubleSpinBox(); self.spin_chi = QDoubleSpinBox(); self.spin_phi = QDoubleSpinBox()
        for name, spin in [("omega", self.spin_omega), ("chi", self.spin_chi), ("phi", self.spin_phi)]:
            spin.setRange(-360.0, 360.0)
            spin.setDecimals(2)
            form2.addRow(name, spin)
        tabs.addTab(orient_tab, "Orientation")

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(tabs)
        main_layout.addWidget(apply_btn)

    def _on_apply(self):
        # Emit both lattice and orientation signals
        self.latticeChanged.emit(
            self.spin_a.value(), self.spin_b.value(), self.spin_c.value(),
            self.spin_alpha.value(), self.spin_beta.value(), self.spin_gamma.value()
        )
        self.orientationChanged.emit(
            self.spin_omega.value(), self.spin_chi.value(), self.spin_phi.value()
        )
