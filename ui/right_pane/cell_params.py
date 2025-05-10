# File: ewald/ui/right_pane/cell_params.py
"""
CellParamsEditor: edit unit cell parameters, orientation, and Bragg peak calculation settings.
Tabs: "Lattice", "Orientation", "Bragg".
"""
from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QFormLayout, QDoubleSpinBox,
    QSpinBox, QVBoxLayout, QPushButton, QComboBox, QLineEdit,
    QSlider, QLabel
)
from PyQt6.QtCore import pyqtSignal, Qt


class CellParamsEditor(QWidget):
    # Signals
    latticeChanged          = pyqtSignal(float, float, float, float, float, float)
    orientationChanged      = pyqtSignal(float, float, float)
    peakRangeChanged        = pyqtSignal(int, int, int)
    calculateRequested      = pyqtSignal()
    customStructureAdded    = pyqtSignal(str, str, float, float, float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._setup_presets()

    def _init_ui(self):
        tabs = QTabWidget()

        # --- Lattice Tab ---
        lattice_tab = QWidget()
        form_latt = QFormLayout(lattice_tab)
        self.name_edit = QLineEdit()
        form_latt.addRow("Name", self.name_edit)
        self.combo_system = QComboBox()
        form_latt.addRow("Crystal System", self.combo_system)
        self.spin_a = QDoubleSpinBox(); self.spin_b = QDoubleSpinBox(); self.spin_c = QDoubleSpinBox()
        self.spin_alpha = QDoubleSpinBox(); self.spin_beta = QDoubleSpinBox(); self.spin_gamma = QDoubleSpinBox()
        for label, spin in [("a", self.spin_a),("b", self.spin_b),("c", self.spin_c),
                             ("alpha", self.spin_alpha),("beta", self.spin_beta),("gamma", self.spin_gamma)]:
            spin.setRange(0.0, 1000.0)
            spin.setDecimals(4)
            form_latt.addRow(label, spin)
        tabs.addTab(lattice_tab, "Lattice")

        # --- Orientation Tab ---
        orient_tab = QWidget()
        v_orient = QVBoxLayout(orient_tab)

        # 1) Spinboxes
        form_orient = QFormLayout()
        self.spin_omega = QDoubleSpinBox(); self.spin_chi = QDoubleSpinBox(); self.spin_phi = QDoubleSpinBox()
        for label, spin in [("ω", self.spin_omega), ("χ", self.spin_chi), ("φ", self.spin_phi)]:
            spin.setRange(-360.0, 360.0)
            spin.setDecimals(1)
            form_orient.addRow(label, spin)
        v_orient.addLayout(form_orient)

        # 2) Sliders
        # map unicode to spinboxes
        orient_map = {"ω": self.spin_omega, "χ": self.spin_chi, "φ": self.spin_phi}
        for label_sym, slider in [("ω", QSlider(Qt.Orientation.Horizontal)),
                                  ("χ", QSlider(Qt.Orientation.Horizontal)),
                                  ("φ", QSlider(Qt.Orientation.Horizontal))]:
            slider.setRange(-360, 360)
            v_orient.addWidget(QLabel(f"{label_sym} slider"))
            v_orient.addWidget(slider)

            # keep spinbox and slider in sync
            sb = orient_map[label_sym]
            slider.valueChanged.connect(sb.setValue)
            sb.valueChanged.connect(slider.setValue)

            # broadcast orientationChanged on any change
            sb.valueChanged.connect(lambda _, s=self: s.orientationChanged.emit(
                s.spin_omega.value(),
                s.spin_chi.value(),
                s.spin_phi.value()
            ))

        tabs.addTab(orient_tab, "Orientation")

        # --- Bragg Tab ---
        bragg_tab = QWidget()
        form_bragg = QFormLayout(bragg_tab)
        self.spin_h = QSpinBox(); self.spin_k = QSpinBox(); self.spin_l = QSpinBox()
        for label, spin in [("h_max", self.spin_h),("k_max", self.spin_k),("l_max", self.spin_l)]:
            spin.setRange(0, 10)
            form_bragg.addRow(label, spin)
        # Calculate button
        self.calc_btn = QPushButton("Calculate")
        self.calc_btn.clicked.connect(self.calculateRequested)
        form_bragg.addRow(self.calc_btn)
        tabs.addTab(bragg_tab, "Bragg")

        # Apply button for lattice/orientation and adding custom
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(tabs)
        main_layout.addWidget(apply_btn)

        # Connect crystal system change
        self.combo_system.currentTextChanged.connect(self._on_system_change)

    def _setup_presets(self):
        self.lattice_presets = {
            "Triclinic":    (1.0,1.2,1.3,100.0,110.0,120.0),
            "Monoclinic":   (1.0,1.1,1.3, 90.0,110.0, 90.0),
            "Orthorhombic": (1.0,1.2,1.3, 90.0, 90.0, 90.0),
            "Tetragonal":   (1.0,1.0,1.2, 90.0, 90.0, 90.0),
            "Trigonal":     (1.0,1.0,1.0, 60.0, 60.0, 60.0),
            "Hexagonal":    (1.0,1.0,1.5, 90.0, 90.0,120.0),
            "Cubic":        (1.0,1.0,1.0, 90.0, 90.0, 90.0),
            "Custom":       None
        }
        self.disable_map = {
            "Triclinic": [],
            "Monoclinic": ["alpha","gamma"],
            "Orthorhombic":["alpha","beta","gamma"],
            "Tetragonal": ["b","alpha","beta","gamma"],
            "Trigonal":   ["b","c","alpha","beta"],
            "Hexagonal":  ["b","alpha","beta"],
            "Cubic":      ["b","c","alpha","beta","gamma"],
            "Custom":     []
        }
        self.combo_system.addItems(self.lattice_presets.keys())
        self.combo_system.setCurrentText("Custom")

    def _on_system_change(self, system):
        preset = self.lattice_presets.get(system)
        fields = {"a":self.spin_a,"b":self.spin_b,"c":self.spin_c,
                  "alpha":self.spin_alpha,"beta":self.spin_beta,"gamma":self.spin_gamma}
        if preset:
            for spin in fields.values(): spin.blockSignals(True)
            for val, spin in zip(preset, fields.values()): spin.setValue(val)
            for spin in fields.values(): spin.blockSignals(False)
        disabled = set(self.disable_map.get(system, []))
        for name, spin in fields.items(): spin.setEnabled(name not in disabled)

    def _on_apply(self):
        # Emit lattice
        a,b,c = self.spin_a.value(), self.spin_b.value(), self.spin_c.value()
        alpha,beta,gamma = (self.spin_alpha.value(), self.spin_beta.value(), self.spin_gamma.value())
        self.latticeChanged.emit(a,b,c,alpha,beta,gamma)
        # Emit orientation
        omega,chi,phi = self.spin_omega.value(), self.spin_chi.value(), self.spin_phi.value()
        # self.orientationChanged.emit(omega,chi,phi)
        # Emit Bragg range
        h,k,l = self.spin_h.value(), self.spin_k.value(), self.spin_l.value()
        self.peakRangeChanged.emit(h,k,l)
        # Emit custom structure
        name = self.name_edit.text().strip()
        if name:
            system = self.combo_system.currentText()
            self.customStructureAdded.emit(name, system, a,b,c,alpha,beta,gamma)
