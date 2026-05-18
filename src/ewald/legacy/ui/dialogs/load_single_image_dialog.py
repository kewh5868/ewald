from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QPushButton, QComboBox,
    QDoubleSpinBox, QCheckBox, QDialogButtonBox, QVBoxLayout,
    QHBoxLayout, QFileDialog, QLabel, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
from ...dataclass.single_image import SingleImage

class LoadSingleImageDialog(QDialog):
    """
    Dialog for loading a single GIWAXS image with options for mask, PONI,
    incidence angle, polarization, solid angle toggle, and custom metadata.
    """
    single_image_loaded = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize file lists for dropdowns
        self.mask_files = []
        self.poni_files = []
        self.setWindowTitle("Load Single Image")
        self.meta_rows = []  # list of (key_widget, value_widget, type_dropdown)

        self._init_ui()

    def _init_ui(self):
        # Main vertical layout
        layout = QVBoxLayout(self)
        # Form layout for rows
        form = QFormLayout()

        # --- Data Object Name ---
        self.data_object_name_edit = QLineEdit()
        self.data_object_name_edit.setPlaceholderText("Enter data object name")
        form.addRow("Data Object Name:", self.data_object_name_edit)

        # --- File selector ---
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_file)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_btn)
        form.addRow("Image File:", file_layout)

        # --- Mask dropdown ---
        self.maskCombo = QComboBox()
        self.maskCombo.addItem("None")
        form.addRow("Mask File:", self.maskCombo)

        # --- PONI dropdown ---
        self.poniCombo = QComboBox()
        self.poniCombo.addItem("None")
        form.addRow("PONI File:", self.poniCombo)

        # --- Incidence angle ---
        self.incidence_spin = QDoubleSpinBox()
        self.incidence_spin.setRange(0.0, 90.0)
        self.incidence_spin.setDecimals(4)
        self.incidence_spin.setValue(0.3)  # Default for GIWAXS
        form.addRow("Incidence Angle (°):", self.incidence_spin)

        # --- Tilt angle ---
        self.tilt_spin = QDoubleSpinBox()
        self.tilt_spin.setRange(-90.0, 90.0)
        self.tilt_spin.setDecimals(4)
        self.tilt_spin.setValue(0.0)  # Default for GIWAXS
        form.addRow("Tilt Angle (°):", self.tilt_spin)

        # --- Sample orientation ---
        self.sample_orientation_combo = QComboBox()
        self.sample_orientation_combo.addItems([str(i) for i in range(1, 9)])
        self.sample_orientation_combo.setCurrentText("4")
        form.addRow("Sample Orientation:", self.sample_orientation_combo)
        
        # --- Split pixels, output space, and solid angle toggle ---
        self.split_pixels_chk = QCheckBox()
        self.split_pixels_chk.setChecked(True)  # Default to split pixels
        form.addRow("Split Pixels:", self.split_pixels_chk)
        self.output_space_combo = QComboBox()
        self.output_space_combo.addItems(["Reciprocal Space", "Polar (Azimuthal)", "Both"])
        self.output_space_combo.setCurrentText("Reciprocal Space")
        form.addRow("Output Space:", self.output_space_combo)

        # --- Polarization ---
        self.polarization_spin = QDoubleSpinBox()
        self.polarization_spin.setRange(0.0, 1.0)
        self.polarization_spin.setDecimals(4)
        self.polarization_spin.setValue(0.95)  # Default for synchrotron data
        form.addRow("Polarization:", self.polarization_spin)

        # --- Solid angle toggle ---
        self.solid_angle_chk = QCheckBox()
        form.addRow("Apply Solid Angle:", self.solid_angle_chk)

        # Add form to main layout
        layout.addLayout(form)

        # --- Metadata section ---
        meta_header = QHBoxLayout()
        meta_label = QLabel("Metadata Attributes:")
        add_meta_btn = QPushButton("+")
        add_meta_btn.setFixedWidth(24)
        add_meta_btn.clicked.connect(self._add_meta_row)
        meta_header.addWidget(meta_label)
        meta_header.addStretch()
        meta_header.addWidget(add_meta_btn)
        layout.addLayout(meta_header)

        self.meta_rows_layout = QVBoxLayout()
        layout.addLayout(self.meta_rows_layout)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "TIFF Files (*.tif *.tiff)"
        )
        if path:
            self.file_path_edit.setText(path)

    def addMaskFile(self, filePath: str):
        """
        Add a new mask file to the internal list and dropdown.
        """
        if not hasattr(self, 'mask_files'):
            self.mask_files = []
        if filePath not in self.mask_files:
            self.mask_files.append(filePath)
            self.maskCombo.addItem(filePath)

    def reloadMaskList(self):
        """
        Reload the mask dropdown from the stored list of mask files.
        """
        self.maskCombo.clear()
        self.maskCombo.addItem("None")
        for f in self.mask_files:
            self.maskCombo.addItem(f)

    def addPoniFile(self, filePath: str):
        """
        Add a new PONI file to the internal list and dropdown.
        """
        if not hasattr(self, 'poni_files'):
            self.poni_files = []
        if filePath not in self.poni_files:
            self.poni_files.append(filePath)
            self.poniCombo.addItem(filePath)

    def reloadPoniList(self):
        """
        Reload the PONI dropdown from the stored list of PONI files.
        """
        self.poniCombo.clear()
        self.poniCombo.addItem("None")
        for f in self.poni_files:
            self.poniCombo.addItem(f)

    def getSelectedMask(self) -> str:
        """
        Return the currently selected mask file, or None if "None" is chosen.
        """
        text = self.maskCombo.currentText()
        return text if text != "None" else None

    def getSelectedPoni(self) -> str:
        """
        Return the currently selected PONI file, or None if "None" is chosen.
        """
        text = self.poniCombo.currentText()
        return text if text != "None" else None

    def _add_meta_row(self):
        row_widget = QWidget()
        hl = QHBoxLayout(row_widget)
        name_edit = QLineEdit()
        value_edit = QLineEdit()
        type_combo = QComboBox()
        type_combo.addItems(["Value", "String"])
        remove_btn = QPushButton("-")
        remove_btn.setFixedWidth(24)
        remove_btn.clicked.connect(lambda: self._remove_meta_row(row_widget))
        hl.addWidget(name_edit)
        hl.addWidget(value_edit)
        hl.addWidget(type_combo)
        hl.addWidget(remove_btn)
        self.meta_rows_layout.addWidget(row_widget)

    def _remove_meta_row(self, row_widget):
        row_widget.setParent(None)

    def get_metadata(self):
        data = []
        for i in range(self.meta_rows_layout.count()):
            row = self.meta_rows_layout.itemAt(i).widget()
            edits = row.findChildren(QLineEdit)
            combo = row.findChildren(QComboBox)[0]
            key = edits[0].text().strip()
            value = edits[1].text().strip()
            typ = combo.currentText()
            if key:
                data.append((key, value, typ))
        return data
    
    def _on_accept(self):
        # Gather values
        data_name = self.data_object_name_edit.text().strip()
        file_path = self.file_path_edit.text().strip()
        mask_file = self.getSelectedMask()
        poni_file = self.getSelectedPoni()
        incident_angle = float(self.incidence_spin.value())
        tilt_angle = float(self.tilt_spin.value())
        sample_orientation = int(self.sample_orientation_combo.currentText())
        split_pixels = self.split_pixels_chk.isChecked()
        output_space = self.output_space_combo.currentText()
        if output_space == "Reciprocal Space":
            output_space = "recip"
        elif output_space == "Polar (Azimuthal)":
            output_space = "polar"
        else:
            output_space = "both"
        polarization = float(self.polarization_spin.value())
        solid_angle = self.solid_angle_chk.isChecked()
        metadata_entries = self.get_metadata()
        metadata = {}
        for key, value, typ in metadata_entries:
            if typ == "Value":
                try:
                    metadata[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    metadata[key] = value
            else:
                metadata[key] = value

        single_image = SingleImage(
            data_name=data_name,
            file_path=file_path,
            mask_file=mask_file,
            poni_file=poni_file,
            incident_angle=incident_angle,
            tilt_angle=tilt_angle,
            sample_orientation=sample_orientation,
            split_pixels=split_pixels,
            output_space=output_space,
            polarization=polarization,
            solid_angle=solid_angle,
            metadata_attributes=metadata
        )
        self.single_image_loaded.emit(single_image)
        super().accept()