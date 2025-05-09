# -------------------------------------------
# File: ewald/ui/center_pane/image_view.py
"""
ImageCanvas: display 2D GIWAXS images and handle ROI selection.
"""
from PyQt6.QtWidgets import QWidget

class ImageCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # TODO: initialize graphics view or pyqtgraph ImageView
        # self.view = ImageView(self)