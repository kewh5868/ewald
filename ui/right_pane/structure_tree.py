# -------------------------------------------
# File: ewald/ui/right_pane/structure_tree.py
"""
StructureTreeView: browse loaded CIF or project files.
"""
from PyQt6.QtWidgets import QTreeView

class StructureTreeView(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)
        # TODO: configure QFileSystemModel for CIF/.ewld files