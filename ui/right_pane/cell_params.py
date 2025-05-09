# File: ewald/ui/right_pane/cell_params.py
"""
CellParamsEditor: edit unit cell parameters and orientation with crystal system presets, add custom structure naming.
"""
from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QFormLayout, QDoubleSpinBox,
    QVBoxLayout, QPushButton, QComboBox, QLineEdit
)
from PyQt6.QtCore import pyqtSignal

class CellParamsEditor(QWidget):
    # Emitted when lattice or orientation parameters are applied
    latticeChanged = pyqtSignal(float, float, float, float, float, float)
    orientationChanged = pyqtSignal(float, float, float)
    # Emitted when a new custom structure is defined
    customStructureAdded = pyqtSignal(str, str, float, float, float, float, float, float)


    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._setup_presets()

    def _init_ui(self):
        self.tabs = QTabWidget()

        # Lattice parameters tab
        lattice_tab = QWidget()
        form1 = QFormLayout(lattice_tab)

        # Custom structure name field
        self.name_edit = QLineEdit()
        form1.addRow("Name", self.name_edit)

        # Crystal system selector
        self.combo_system = QComboBox()
        form1.addRow("Crystal System", self.combo_system)

        # Lattice spin boxes
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

        self.tabs.addTab(lattice_tab, "Lattice")

        # Orientation tab
        orient_tab = QWidget()
        form2 = QFormLayout(orient_tab)
        self.spin_omega = QDoubleSpinBox(); self.spin_chi = QDoubleSpinBox(); self.spin_phi = QDoubleSpinBox()
        for name, spin in [("omega", self.spin_omega), ("chi", self.spin_chi), ("phi", self.spin_phi)]:
            spin.setRange(-360.0, 360.0)
            spin.setDecimals(2)
            form2.addRow(name, spin)
        self.tabs.addTab(orient_tab, "Orientation")

        # Apply button
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(apply_btn)

        # Connect preset change
        self.combo_system.currentTextChanged.connect(self._on_system_change)

    def _setup_presets(self):
        # Define crystal system presets: (a,b,c,alpha,beta,gamma)
        self.lattice_presets = {
            "Triclinic":   (1.0, 1.2, 1.3, 100.0, 110.0, 120.0),
            "Monoclinic":  (1.0, 1.1, 1.3, 90.0, 110.0, 90.0),
            "Orthorhombic":(1.0, 1.2, 1.3, 90.0, 90.0, 90.0),
            "Tetragonal":  (1.0, 1.0, 1.2, 90.0, 90.0, 90.0),
            "Trigonal":    (1.0, 1.0, 1.0, 60.0, 60.0, 60.0),
            "Hexagonal":   (1.0, 1.0, 1.5, 90.0, 90.0, 120.0),
            "Cubic":       (1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
            "Custom":      None
        }
        # Fields to disable per crystal system
        self.disable_map = {
            "Triclinic": [],
            "Monoclinic": ["alpha", "gamma"],
            "Orthorhombic": ["alpha", "beta", "gamma"],
            "Tetragonal": ["b", "alpha", "beta", "gamma"],
            "Trigonal": ["b", "c", "alpha", "beta"],
            "Hexagonal": ["b", "alpha", "beta"],
            "Cubic": ["b", "c", "alpha", "beta", "gamma"],
            "Custom": []
        }
        # Populate combo box
        self.combo_system.addItems(list(self.lattice_presets.keys()))
        self.combo_system.setCurrentText("Custom")

    def _on_system_change(self, system):
        # Apply preset values and enable/disable fields
        preset = self.lattice_presets.get(system)
        field_map = {
            "a": self.spin_a, "b": self.spin_b, "c": self.spin_c,
            "alpha": self.spin_alpha, "beta": self.spin_beta, "gamma": self.spin_gamma
        }
        if preset:
            for spin in field_map.values(): spin.blockSignals(True)
            a, b, c, alpha, beta, gamma = preset
            self.spin_a.setValue(a); self.spin_b.setValue(b)
            self.spin_c.setValue(c); self.spin_alpha.setValue(alpha)
            self.spin_beta.setValue(beta); self.spin_gamma.setValue(gamma)
            for spin in field_map.values(): spin.blockSignals(False)
        disabled = set(self.disable_map.get(system, []))
        for name, spin in field_map.items():
            spin.setEnabled(name not in disabled)

    def _on_apply(self):
        a, b, c = self.spin_a.value(), self.spin_b.value(), self.spin_c.value()
        alpha, beta, gamma = (self.spin_alpha.value(), self.spin_beta.value(), self.spin_gamma.value())
        # Emit lattice and orientation
        self.latticeChanged.emit(a, b, c, alpha, beta, gamma)
        self.orientationChanged.emit(
            self.spin_omega.value(), self.spin_chi.value(), self.spin_phi.value()
        )
        # Emit custom structure with correct args
        name = self.name_edit.text().strip()
        if name:
            system = self.combo_system.currentText()
            self.customStructureAdded.emit(
                name, system,
                a, b, c,
                alpha, beta, gamma
            )
