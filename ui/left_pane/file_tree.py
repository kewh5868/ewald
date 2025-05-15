# File: ewald/ui/left_pane/file_tree.py

from PyQt6.QtWidgets import QTreeView
from PyQt6.QtGui     import QFileSystemModel
from PyQt6.QtCore    import pyqtSignal
from PyQt6.QtGui import QStandardItemModel, QStandardItem

class FileTreeView(QTreeView):
    """
    Left‐hand file browser for image stacks.
    Emits `imageSelected(str)` when a .tiff/.tif file is clicked.
    """
    imageSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["Data Objects"])
        self.setModel(self.model)
        # keep references to items by name
        self._items = {}
        self.clicked.connect(self.on_clicked)

    def add_data_object(self, data_object):
        # root node shows name and type
        label = f"{data_object.data_name} ({data_object.type})"
        root_item = QStandardItem(label)
        # add metadata as children
        for key, val in data_object.metadata_attributes.items():
            child = QStandardItem(f"{key}: {val}")
            root_item.appendRow(child)
        self.model.appendRow(root_item)
        self._items[data_object.data_name] = root_item

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
