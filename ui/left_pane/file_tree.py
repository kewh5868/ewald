# File: ewald/ui/left_pane/file_tree.py

from PyQt6.QtWidgets import QTreeView
from PyQt6.QtGui     import QFileSystemModel
from PyQt6.QtCore    import pyqtSignal

class FileTreeView(QTreeView):
    """
    Left‐hand file browser for image stacks.
    Emits `imageSelected(str)` when a .tiff/.tif file is clicked.
    """
    imageSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = QFileSystemModel(self)
        # show only TIFF files by default
        self.model.setNameFilters(["*.tiff", "*.tif"])
        self.model.setNameFilterDisables(False)
        self.setModel(self.model)
        self.clicked.connect(self.on_clicked)

    def set_root_path(self, path: str):
        """
        Set the folder to browse.
        """
        idx = self.model.setRootPath(path)
        self.setRootIndex(idx)

    def on_clicked(self, index):
        """
        Slot for click events—emit the full file path.
        """
        file_path = self.model.filePath(index)
        self.imageSelected.emit(file_path)
