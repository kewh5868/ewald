# File: ewald/ui/center_pane/image_view.py
"""
ImageCanvas: composite widget displaying a main 2D image frame with four subplots below.
The main frame has default q_xy vs q_z axes; subplots:
 - q_xy vs Intensity
 - q_z vs Intensity
 - q_r vs Intensity
 - q_xy vs q_z (small)
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np

class ImageCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Create figure with dark gray background
        self.fig = Figure(facecolor='darkgray')
        self.canvas = FigureCanvas(self.fig)
        # Create gridspec: 2 rows, 4 cols; top row spans all columns
        # Increase height ratio for main image to make it larger
        gs = GridSpec(2, 4, figure=self.fig,
                      height_ratios=[4, 1],  # main 4:1 subplots
                      hspace=0.4, wspace=0.5)
        # Main large axis
        self.ax_main = self.fig.add_subplot(gs[0, :])
        # Four small subplots
        self.ax_qxy      = self.fig.add_subplot(gs[1, 0])
        self.ax_qz       = self.fig.add_subplot(gs[1, 1])
        self.ax_qr       = self.fig.add_subplot(gs[1, 2])
        self.ax_small2d  = self.fig.add_subplot(gs[1, 3])

        # Configure axes
        self._setup_main()
        self._setup_subplots()

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def _setup_main(self):
        ax = self.ax_main
        ax.set_facecolor('darkgray')
        ax.set_xlabel(r'$q_{xy}\,\mathrm{(\AA^{-1})}$')
        ax.set_ylabel(r'$q_{z}\,\mathrm{(\AA^{-1})}$')
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)

    def _setup_subplots(self):
        # q_xy vs Intensity
        self.ax_qxy.set_facecolor('white')
        self.ax_qxy.set_xlabel(r'$q_{xy}\,\mathrm{(\AA^{-1})}$')
        self.ax_qxy.set_ylabel('Intensity')
        # q_z vs Intensity
        self.ax_qz.set_facecolor('white')
        self.ax_qz.set_xlabel(r'$q_{z}\,\mathrm{(\AA^{-1})}$')
        self.ax_qz.set_ylabel('Intensity')
        # q_r vs Intensity
        self.ax_qr.set_facecolor('white')
        self.ax_qr.set_xlabel(r'$q_{r}\,\mathrm{(\AA^{-1})}$')
        self.ax_qr.set_ylabel('Intensity')
        # small 2D
        self.ax_small2d.set_facecolor('darkgray')
        self.ax_small2d.set_xlabel(r'$q_{xy}\,\mathrm{(\AA^{-1})}$')
        self.ax_small2d.set_ylabel(r'$q_{z}\,\mathrm{(\AA^{-1})}$')
        self.ax_small2d.set_xlim(0, 3)
        self.ax_small2d.set_ylim(0, 3)

    def clear(self):
        """Clear all axes to initial state."""
        self.ax_main.cla()
        self._setup_main()
        for ax in [self.ax_qxy, self.ax_qz, self.ax_qr, self.ax_small2d]:
            ax.cla()
        self._setup_subplots()
        self.canvas.draw()

    def displayImage(self, data, extent=None, cmap='viridis'):
        """Display data on the main axis and reset limits."""
        self.ax_main.cla()
        self._setup_main()
        if extent is None:
            extent = [0, data.shape[1] * (3.0/data.shape[1]),
                      0, data.shape[0] * (3.0/data.shape[0])]
        self.ax_main.imshow(
            data, origin='lower', extent=extent,
            aspect='auto', cmap=cmap
        )
        self.ax_main.set_xlim(extent[0], extent[1])
        self.ax_main.set_ylim(extent[2], extent[3])
        self.canvas.draw()

    def overlayPeaks(self, qxy, qz, **kwargs):
        """Overlay Bragg peaks on the main axis."""
        self.ax_main.scatter(qxy, qz, **kwargs)
        self.canvas.draw()

    def update1D(self, axis, x, y, **kwargs):
        """Update one of the 1D subplots: 'qxy','qz','qr'."""
        mapping = {
            'qxy': self.ax_qxy,
            'qz':  self.ax_qz,
            'qr':  self.ax_qr
        }
        ax = mapping.get(axis)
        if ax is None:
            return
        ax.cla()
        if axis == 'qxy':
            ax.set_xlabel(r'$q_{xy}\,\mathrm{(\AA^{-1})}$')
        elif axis == 'qz':
            ax.set_xlabel(r'$q_{z}\,\mathrm{(\AA^{-1})}$')
        elif axis == 'qr':
            ax.set_xlabel(r'$q_{r}\,\mathrm{(\AA^{-1})}$')
        ax.set_ylabel('Intensity')
        ax.plot(x, y, **kwargs)
        self.canvas.draw()

    def update2Dsmall(self, data, extent=None, cmap='viridis'):
        """Display data on the small 2D subplot."""
        self.ax_small2d.cla()
        self.ax_small2d.set_facecolor('darkgray')
        if extent is None:
            extent = [0, data.shape[1] * (3.0/data.shape[1]),
                      0, data.shape[0] * (3.0/data.shape[0])]
        self.ax_small2d.imshow(
            data, origin='lower', extent=extent,
            aspect='auto', cmap=cmap
        )
        self.ax_small2d.set_xlim(extent[0], extent[1])
        self.ax_small2d.set_ylim(extent[2], extent[3])
        self.canvas.draw()