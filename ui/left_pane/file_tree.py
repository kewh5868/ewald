# File: ewald/ui/left_pane/file_tree.py

from PyQt6.QtWidgets import QTreeView
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import pyqtSignal

class FileTreeView(QTreeView):
    """
    Left-hand browser for SingleImage objects.
    Emits `imageSelected(object)` with the clicked SingleImage.
    """
    # generic Python object signal
    imageSelected = pyqtSignal("PyQt_PyObject")

    def __init__(self, parent=None):
        super().__init__(parent)
        # set up the in-memory model
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["Data Objects"])
        self.setModel(self.model)
        # mappings for lookup
        self._data_objects = {}
        self._items = {}
        # connect click
        self.clicked.connect(self.on_clicked)

    def add_data_object(self, data_object):
        name = data_object.data_name
        print(f"[FileTreeView] Adding data_object: {name}")
        self._data_objects[name] = data_object

        label = f"{name} ({data_object.type})"
        root_item = QStandardItem(label)
        for key, val in data_object.metadata_attributes.items():
            child = QStandardItem(f"{key}: {val}")
            root_item.appendRow(child)

        self.model.appendRow(root_item)
        self._items[name] = root_item

    def on_clicked(self, index):
        print(f"[FileTreeView] on_clicked at row={index.row()}, col={index.column()}")
        item = self.model.itemFromIndex(index)
        if not item:
            print("[FileTreeView] No item at this index")
            return
        if item.parent():
            print("[FileTreeView] Ignoring non-top-level click")
            return

        name = item.text().split(' (', 1)[0]
        print(f"[FileTreeView] Clicked top-level: {name}")
        single_image = self._data_objects.get(name)
        if single_image is None:
            print(f"[FileTreeView] No SingleImage found for {name}")
            return

        print(f"[FileTreeView] Emitting imageSelected with {single_image}")
        # emit the SingleImage instance for external handling
        self.imageSelected.emit(single_image)
