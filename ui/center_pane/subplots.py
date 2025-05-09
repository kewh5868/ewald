# -------------------------------------------
# File: ewald/ui/center_pane/subplots.py
"""
IntegrationPlot1D & TopographyPlot2D: show 1D/2D plots of selected region.
"""
from PyQt6.QtWidgets import QWidget

class IntegrationPlot1D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # TODO: embed a Matplotlib FigureCanvas for 1D plot

class TopographyPlot2D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # TODO: embed a Matplotlib FigureCanvas for 2D topography plot