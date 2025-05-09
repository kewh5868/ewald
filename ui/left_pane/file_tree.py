# File: ewald/ui/left_pane/file_tree.py
"""
FileTreeView: browse project TIFF files and select images.
"""
from PyQt6.QtWidgets import QTreeView
from PyQt6.QtCore import QModelIndex

class FileTreeView(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)
        # TODO: set model to QFileSystemModel, configure root path
        # self.model = QFileSystemModel(self)
        # self.setModel(self.model)